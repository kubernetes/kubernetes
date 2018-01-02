// Package tagmanager provides access to the Tag Manager API.
//
// See https://developers.google.com/tag-manager/api/v2/
//
// Usage example:
//
//   import "google.golang.org/api/tagmanager/v2"
//   ...
//   tagmanagerService, err := tagmanager.New(oauthHttpClient)
package tagmanager // import "google.golang.org/api/tagmanager/v2"

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

const apiId = "tagmanager:v2"
const apiName = "tagmanager"
const apiVersion = "v2"
const basePath = "https://www.googleapis.com/tagmanager/v2/"

// OAuth2 scopes used by this API.
const (
	// Delete your Google Tag Manager containers
	TagmanagerDeleteContainersScope = "https://www.googleapis.com/auth/tagmanager.delete.containers"

	// Manage your Google Tag Manager container and its subcomponents,
	// excluding versioning and publishing
	TagmanagerEditContainersScope = "https://www.googleapis.com/auth/tagmanager.edit.containers"

	// Manage your Google Tag Manager container versions
	TagmanagerEditContainerversionsScope = "https://www.googleapis.com/auth/tagmanager.edit.containerversions"

	// View and manage your Google Tag Manager accounts
	TagmanagerManageAccountsScope = "https://www.googleapis.com/auth/tagmanager.manage.accounts"

	// Manage user permissions of your Google Tag Manager account and
	// container
	TagmanagerManageUsersScope = "https://www.googleapis.com/auth/tagmanager.manage.users"

	// Publish your Google Tag Manager container versions
	TagmanagerPublishScope = "https://www.googleapis.com/auth/tagmanager.publish"

	// View your Google Tag Manager container and its subcomponents
	TagmanagerReadonlyScope = "https://www.googleapis.com/auth/tagmanager.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Accounts = NewAccountsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Accounts *AccountsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewAccountsService(s *Service) *AccountsService {
	rs := &AccountsService{s: s}
	rs.Containers = NewAccountsContainersService(s)
	rs.UserPermissions = NewAccountsUserPermissionsService(s)
	return rs
}

type AccountsService struct {
	s *Service

	Containers *AccountsContainersService

	UserPermissions *AccountsUserPermissionsService
}

func NewAccountsContainersService(s *Service) *AccountsContainersService {
	rs := &AccountsContainersService{s: s}
	rs.Environments = NewAccountsContainersEnvironmentsService(s)
	rs.VersionHeaders = NewAccountsContainersVersionHeadersService(s)
	rs.Versions = NewAccountsContainersVersionsService(s)
	rs.Workspaces = NewAccountsContainersWorkspacesService(s)
	return rs
}

type AccountsContainersService struct {
	s *Service

	Environments *AccountsContainersEnvironmentsService

	VersionHeaders *AccountsContainersVersionHeadersService

	Versions *AccountsContainersVersionsService

	Workspaces *AccountsContainersWorkspacesService
}

func NewAccountsContainersEnvironmentsService(s *Service) *AccountsContainersEnvironmentsService {
	rs := &AccountsContainersEnvironmentsService{s: s}
	return rs
}

type AccountsContainersEnvironmentsService struct {
	s *Service
}

func NewAccountsContainersVersionHeadersService(s *Service) *AccountsContainersVersionHeadersService {
	rs := &AccountsContainersVersionHeadersService{s: s}
	return rs
}

type AccountsContainersVersionHeadersService struct {
	s *Service
}

func NewAccountsContainersVersionsService(s *Service) *AccountsContainersVersionsService {
	rs := &AccountsContainersVersionsService{s: s}
	return rs
}

type AccountsContainersVersionsService struct {
	s *Service
}

func NewAccountsContainersWorkspacesService(s *Service) *AccountsContainersWorkspacesService {
	rs := &AccountsContainersWorkspacesService{s: s}
	rs.BuiltInVariables = NewAccountsContainersWorkspacesBuiltInVariablesService(s)
	rs.Folders = NewAccountsContainersWorkspacesFoldersService(s)
	rs.Proposal = NewAccountsContainersWorkspacesProposalService(s)
	rs.Tags = NewAccountsContainersWorkspacesTagsService(s)
	rs.Triggers = NewAccountsContainersWorkspacesTriggersService(s)
	rs.Variables = NewAccountsContainersWorkspacesVariablesService(s)
	return rs
}

type AccountsContainersWorkspacesService struct {
	s *Service

	BuiltInVariables *AccountsContainersWorkspacesBuiltInVariablesService

	Folders *AccountsContainersWorkspacesFoldersService

	Proposal *AccountsContainersWorkspacesProposalService

	Tags *AccountsContainersWorkspacesTagsService

	Triggers *AccountsContainersWorkspacesTriggersService

	Variables *AccountsContainersWorkspacesVariablesService
}

func NewAccountsContainersWorkspacesBuiltInVariablesService(s *Service) *AccountsContainersWorkspacesBuiltInVariablesService {
	rs := &AccountsContainersWorkspacesBuiltInVariablesService{s: s}
	return rs
}

type AccountsContainersWorkspacesBuiltInVariablesService struct {
	s *Service
}

func NewAccountsContainersWorkspacesFoldersService(s *Service) *AccountsContainersWorkspacesFoldersService {
	rs := &AccountsContainersWorkspacesFoldersService{s: s}
	return rs
}

type AccountsContainersWorkspacesFoldersService struct {
	s *Service
}

func NewAccountsContainersWorkspacesProposalService(s *Service) *AccountsContainersWorkspacesProposalService {
	rs := &AccountsContainersWorkspacesProposalService{s: s}
	return rs
}

type AccountsContainersWorkspacesProposalService struct {
	s *Service
}

func NewAccountsContainersWorkspacesTagsService(s *Service) *AccountsContainersWorkspacesTagsService {
	rs := &AccountsContainersWorkspacesTagsService{s: s}
	return rs
}

type AccountsContainersWorkspacesTagsService struct {
	s *Service
}

func NewAccountsContainersWorkspacesTriggersService(s *Service) *AccountsContainersWorkspacesTriggersService {
	rs := &AccountsContainersWorkspacesTriggersService{s: s}
	return rs
}

type AccountsContainersWorkspacesTriggersService struct {
	s *Service
}

func NewAccountsContainersWorkspacesVariablesService(s *Service) *AccountsContainersWorkspacesVariablesService {
	rs := &AccountsContainersWorkspacesVariablesService{s: s}
	return rs
}

type AccountsContainersWorkspacesVariablesService struct {
	s *Service
}

func NewAccountsUserPermissionsService(s *Service) *AccountsUserPermissionsService {
	rs := &AccountsUserPermissionsService{s: s}
	return rs
}

type AccountsUserPermissionsService struct {
	s *Service
}

// Account: Represents a Google Tag Manager Account.
type Account struct {
	// AccountId: The Account ID uniquely identifies the GTM Account.
	AccountId string `json:"accountId,omitempty"`

	// Fingerprint: The fingerprint of the GTM Account as computed at
	// storage time. This value is recomputed whenever the account is
	// modified.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Name: Account display name.
	Name string `json:"name,omitempty"`

	// Path: GTM Account's API relative path.
	Path string `json:"path,omitempty"`

	// ShareData: Whether the account shares data anonymously with Google
	// and others. This flag enables benchmarking by sharing your data in an
	// anonymous form. Google will remove all identifiable information about
	// your website, combine the data with hundreds of other anonymous sites
	// and report aggregate trends in the benchmarking service.
	ShareData bool `json:"shareData,omitempty"`

	// TagManagerUrl: Auto generated link to the tag manager UI
	TagManagerUrl string `json:"tagManagerUrl,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Account) MarshalJSON() ([]byte, error) {
	type noMethod Account
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AccountAccess: Defines the Google Tag Manager Account access
// permissions.
type AccountAccess struct {
	// Permission: Whether the user has no access, user access, or admin
	// access to an account.
	//
	// Possible values:
	//   "accountPermissionUnspecified"
	//   "admin"
	//   "noAccess"
	//   "user"
	Permission string `json:"permission,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Permission") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Permission") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AccountAccess) MarshalJSON() ([]byte, error) {
	type noMethod AccountAccess
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BuiltInVariable: Built-in variables are a special category of
// variables that are pre-created and non-customizable. They provide
// common functionality like accessing propeties of the gtm data layer,
// monitoring clicks, or accessing elements of a page URL.
type BuiltInVariable struct {
	// AccountId: GTM Account ID.
	AccountId string `json:"accountId,omitempty"`

	// ContainerId: GTM Container ID.
	ContainerId string `json:"containerId,omitempty"`

	// Name: Name of the built-in variable to be used to refer to the
	// built-in variable.
	Name string `json:"name,omitempty"`

	// Path: GTM BuiltInVariable's API relative path.
	Path string `json:"path,omitempty"`

	// Type: Type of built-in variable.
	//
	// Possible values:
	//   "advertiserId"
	//   "advertisingTrackingEnabled"
	//   "ampBrowserLanguage"
	//   "ampCanonicalHost"
	//   "ampCanonicalPath"
	//   "ampCanonicalUrl"
	//   "ampClientId"
	//   "ampClientMaxScrollX"
	//   "ampClientMaxScrollY"
	//   "ampClientScreenHeight"
	//   "ampClientScreenWidth"
	//   "ampClientScrollX"
	//   "ampClientScrollY"
	//   "ampClientTimestamp"
	//   "ampClientTimezone"
	//   "ampGtmEvent"
	//   "ampPageDownloadTime"
	//   "ampPageLoadTime"
	//   "ampPageViewId"
	//   "ampReferrer"
	//   "ampTitle"
	//   "ampTotalEngagedTime"
	//   "appId"
	//   "appName"
	//   "appVersionCode"
	//   "appVersionName"
	//   "builtInVariableTypeUnspecified"
	//   "clickClasses"
	//   "clickElement"
	//   "clickId"
	//   "clickTarget"
	//   "clickText"
	//   "clickUrl"
	//   "containerId"
	//   "containerVersion"
	//   "debugMode"
	//   "deviceName"
	//   "environmentName"
	//   "errorLine"
	//   "errorMessage"
	//   "errorUrl"
	//   "event"
	//   "eventName"
	//   "firebaseEventParameterCampaign"
	//   "firebaseEventParameterCampaignAclid"
	//   "firebaseEventParameterCampaignAnid"
	//   "firebaseEventParameterCampaignClickTimestamp"
	//   "firebaseEventParameterCampaignContent"
	//   "firebaseEventParameterCampaignCp1"
	//   "firebaseEventParameterCampaignGclid"
	//   "firebaseEventParameterCampaignSource"
	//   "firebaseEventParameterCampaignTerm"
	//   "firebaseEventParameterCurrency"
	//   "firebaseEventParameterDynamicLinkAcceptTime"
	//   "firebaseEventParameterDynamicLinkLinkid"
	//   "firebaseEventParameterNotificationMessageDeviceTime"
	//   "firebaseEventParameterNotificationMessageId"
	//   "firebaseEventParameterNotificationMessageName"
	//   "firebaseEventParameterNotificationMessageTime"
	//   "firebaseEventParameterNotificationTopic"
	//   "firebaseEventParameterPreviousAppVersion"
	//   "firebaseEventParameterPreviousOsVersion"
	//   "firebaseEventParameterPrice"
	//   "firebaseEventParameterProductId"
	//   "firebaseEventParameterQuantity"
	//   "firebaseEventParameterValue"
	//   "formClasses"
	//   "formElement"
	//   "formId"
	//   "formTarget"
	//   "formText"
	//   "formUrl"
	//   "historySource"
	//   "htmlId"
	//   "language"
	//   "newHistoryFragment"
	//   "newHistoryState"
	//   "oldHistoryFragment"
	//   "oldHistoryState"
	//   "osVersion"
	//   "pageHostname"
	//   "pagePath"
	//   "pageUrl"
	//   "platform"
	//   "randomNumber"
	//   "referrer"
	//   "resolution"
	//   "sdkVersion"
	Type string `json:"type,omitempty"`

	// WorkspaceId: GTM Workspace ID.
	WorkspaceId string `json:"workspaceId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BuiltInVariable) MarshalJSON() ([]byte, error) {
	type noMethod BuiltInVariable
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Condition: Represents a predicate.
type Condition struct {
	// Parameter: A list of named parameters (key/value), depending on the
	// condition's type. Notes:
	// - For binary operators, include parameters named arg0 and arg1 for
	// specifying the left and right operands, respectively.
	// - At this time, the left operand (arg0) must be a reference to a
	// variable.
	// - For case-insensitive Regex matching, include a boolean parameter
	// named ignore_case that is set to true. If not specified or set to any
	// other value, the matching will be case sensitive.
	// - To negate an operator, include a boolean parameter named negate
	// boolean parameter that is set to true.
	Parameter []*Parameter `json:"parameter,omitempty"`

	// Type: The type of operator for this condition.
	//
	// Possible values:
	//   "conditionTypeUnspecified"
	//   "contains"
	//   "cssSelector"
	//   "endsWith"
	//   "equals"
	//   "greater"
	//   "greaterOrEquals"
	//   "less"
	//   "lessOrEquals"
	//   "matchRegex"
	//   "startsWith"
	//   "urlMatches"
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Parameter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Parameter") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Condition) MarshalJSON() ([]byte, error) {
	type noMethod Condition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Container: Represents a Google Tag Manager Container, which specifies
// the platform tags will run on, manages workspaces, and retains
// container versions.
type Container struct {
	// AccountId: GTM Account ID.
	AccountId string `json:"accountId,omitempty"`

	// ContainerId: The Container ID uniquely identifies the GTM Container.
	ContainerId string `json:"containerId,omitempty"`

	// DomainName: List of domain names associated with the Container.
	DomainName []string `json:"domainName,omitempty"`

	// Fingerprint: The fingerprint of the GTM Container as computed at
	// storage time. This value is recomputed whenever the account is
	// modified.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Name: Container display name.
	Name string `json:"name,omitempty"`

	// Notes: Container Notes.
	Notes string `json:"notes,omitempty"`

	// Path: GTM Container's API relative path.
	Path string `json:"path,omitempty"`

	// PublicId: Container Public ID.
	PublicId string `json:"publicId,omitempty"`

	// TagManagerUrl: Auto generated link to the tag manager UI
	TagManagerUrl string `json:"tagManagerUrl,omitempty"`

	// UsageContext: List of Usage Contexts for the Container. Valid values
	// include: web, android, or ios.
	//
	// Possible values:
	//   "amp"
	//   "android"
	//   "androidSdk5"
	//   "ios"
	//   "iosSdk5"
	//   "usageContextUnspecified"
	//   "web"
	UsageContext []string `json:"usageContext,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Container) MarshalJSON() ([]byte, error) {
	type noMethod Container
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ContainerAccess: Defines the Google Tag Manager Container access
// permissions.
type ContainerAccess struct {
	// ContainerId: GTM Container ID.
	ContainerId string `json:"containerId,omitempty"`

	// Permission: List of Container permissions.
	//
	// Possible values:
	//   "approve"
	//   "containerPermissionUnspecified"
	//   "edit"
	//   "noAccess"
	//   "publish"
	//   "read"
	Permission string `json:"permission,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ContainerId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ContainerId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ContainerAccess) MarshalJSON() ([]byte, error) {
	type noMethod ContainerAccess
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ContainerVersion: Represents a Google Tag Manager Container Version.
type ContainerVersion struct {
	// AccountId: GTM Account ID.
	AccountId string `json:"accountId,omitempty"`

	// BuiltInVariable: The built-in variables in the container that this
	// version was taken from.
	BuiltInVariable []*BuiltInVariable `json:"builtInVariable,omitempty"`

	// Container: The container that this version was taken from.
	Container *Container `json:"container,omitempty"`

	// ContainerId: GTM Container ID.
	ContainerId string `json:"containerId,omitempty"`

	// ContainerVersionId: The Container Version ID uniquely identifies the
	// GTM Container Version.
	ContainerVersionId string `json:"containerVersionId,omitempty"`

	// Deleted: A value of true indicates this container version has been
	// deleted.
	Deleted bool `json:"deleted,omitempty"`

	// Description: Container version description.
	Description string `json:"description,omitempty"`

	// Fingerprint: The fingerprint of the GTM Container Version as computed
	// at storage time. This value is recomputed whenever the container
	// version is modified.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Folder: The folders in the container that this version was taken
	// from.
	Folder []*Folder `json:"folder,omitempty"`

	// Name: Container version display name.
	Name string `json:"name,omitempty"`

	// Path: GTM ContainerVersions's API relative path.
	Path string `json:"path,omitempty"`

	// Tag: The tags in the container that this version was taken from.
	Tag []*Tag `json:"tag,omitempty"`

	// TagManagerUrl: Auto generated link to the tag manager UI
	TagManagerUrl string `json:"tagManagerUrl,omitempty"`

	// Trigger: The triggers in the container that this version was taken
	// from.
	Trigger []*Trigger `json:"trigger,omitempty"`

	// Variable: The variables in the container that this version was taken
	// from.
	Variable []*Variable `json:"variable,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ContainerVersion) MarshalJSON() ([]byte, error) {
	type noMethod ContainerVersion
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ContainerVersionHeader: Represents a Google Tag Manager Container
// Version Header.
type ContainerVersionHeader struct {
	// AccountId: GTM Account ID.
	AccountId string `json:"accountId,omitempty"`

	// ContainerId: GTM Container ID.
	ContainerId string `json:"containerId,omitempty"`

	// ContainerVersionId: The Container Version ID uniquely identifies the
	// GTM Container Version.
	ContainerVersionId string `json:"containerVersionId,omitempty"`

	// Deleted: A value of true indicates this container version has been
	// deleted.
	Deleted bool `json:"deleted,omitempty"`

	// Name: Container version display name.
	Name string `json:"name,omitempty"`

	// NumMacros: Number of macros in the container version.
	NumMacros string `json:"numMacros,omitempty"`

	// NumRules: Number of rules in the container version.
	NumRules string `json:"numRules,omitempty"`

	// NumTags: Number of tags in the container version.
	NumTags string `json:"numTags,omitempty"`

	// NumTriggers: Number of triggers in the container version.
	NumTriggers string `json:"numTriggers,omitempty"`

	// NumVariables: Number of variables in the container version.
	NumVariables string `json:"numVariables,omitempty"`

	// Path: GTM Container Versions's API relative path.
	Path string `json:"path,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ContainerVersionHeader) MarshalJSON() ([]byte, error) {
	type noMethod ContainerVersionHeader
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type CreateBuiltInVariableResponse struct {
	// BuiltInVariable: List of created built-in variables.
	BuiltInVariable []*BuiltInVariable `json:"builtInVariable,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "BuiltInVariable") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BuiltInVariable") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *CreateBuiltInVariableResponse) MarshalJSON() ([]byte, error) {
	type noMethod CreateBuiltInVariableResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateContainerVersionRequestVersionOptions: Options for new
// container versions.
type CreateContainerVersionRequestVersionOptions struct {
	// Name: The name of the container version to be created.
	Name string `json:"name,omitempty"`

	// Notes: The notes of the container version to be created.
	Notes string `json:"notes,omitempty"`

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

func (s *CreateContainerVersionRequestVersionOptions) MarshalJSON() ([]byte, error) {
	type noMethod CreateContainerVersionRequestVersionOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateContainerVersionResponse: Create container versions response.
type CreateContainerVersionResponse struct {
	// CompilerError: Compiler errors or not.
	CompilerError bool `json:"compilerError,omitempty"`

	// ContainerVersion: The container version created.
	ContainerVersion *ContainerVersion `json:"containerVersion,omitempty"`

	// NewWorkspacePath: Auto generated workspace path created as a result
	// of version creation. This field should only be populated if the
	// created version was not a quick preview.
	NewWorkspacePath string `json:"newWorkspacePath,omitempty"`

	// SyncStatus: Whether version creation failed when syncing the
	// workspace to the latest container version.
	SyncStatus *SyncStatus `json:"syncStatus,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CompilerError") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CompilerError") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateContainerVersionResponse) MarshalJSON() ([]byte, error) {
	type noMethod CreateContainerVersionResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateWorkspaceProposalRequest: Creates a workspace proposal to start
// a review of a workspace.
type CreateWorkspaceProposalRequest struct {
	// InitialComment: If present, an initial comment to associate with the
	// workspace proposal.
	InitialComment *WorkspaceProposalHistoryComment `json:"initialComment,omitempty"`

	// Reviewers: List of users to review the workspace proposal.
	Reviewers []*WorkspaceProposalUser `json:"reviewers,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InitialComment") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InitialComment") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *CreateWorkspaceProposalRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateWorkspaceProposalRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Entity: A workspace entity that may represent a tag, trigger,
// variable, or folder in addition to its status in the workspace.
type Entity struct {
	// ChangeStatus: Represents how the entity has been changed in the
	// workspace.
	//
	// Possible values:
	//   "added"
	//   "changeStatusUnspecified"
	//   "deleted"
	//   "none"
	//   "updated"
	ChangeStatus string `json:"changeStatus,omitempty"`

	// Folder: The Folder being represented by the entity.
	Folder *Folder `json:"folder,omitempty"`

	// Tag: The tag being represented by the entity.
	Tag *Tag `json:"tag,omitempty"`

	// Trigger: The trigger being represented by the entity.
	Trigger *Trigger `json:"trigger,omitempty"`

	// Variable: The variable being represented by the entity.
	Variable *Variable `json:"variable,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ChangeStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ChangeStatus") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Entity) MarshalJSON() ([]byte, error) {
	type noMethod Entity
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Environment: Represents a Google Tag Manager Environment. Note that a
// user can create, delete and update environments of type USER, but can
// only update the enable_debug and url fields of environments of other
// types.
type Environment struct {
	// AccountId: GTM Account ID.
	AccountId string `json:"accountId,omitempty"`

	// AuthorizationCode: The environment authorization code.
	AuthorizationCode string `json:"authorizationCode,omitempty"`

	// AuthorizationTimestamp: The last update time-stamp for the
	// authorization code.
	AuthorizationTimestamp *Timestamp `json:"authorizationTimestamp,omitempty"`

	// ContainerId: GTM Container ID.
	ContainerId string `json:"containerId,omitempty"`

	// ContainerVersionId: Represents a link to a container version.
	ContainerVersionId string `json:"containerVersionId,omitempty"`

	// Description: The environment description. Can be set or changed only
	// on USER type environments.
	Description string `json:"description,omitempty"`

	// EnableDebug: Whether or not to enable debug by default for the
	// environment.
	EnableDebug bool `json:"enableDebug,omitempty"`

	// EnvironmentId: GTM Environment ID uniquely identifies the GTM
	// Environment.
	EnvironmentId string `json:"environmentId,omitempty"`

	// Fingerprint: The fingerprint of the GTM environment as computed at
	// storage time. This value is recomputed whenever the environment is
	// modified.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Name: The environment display name. Can be set or changed only on
	// USER type environments.
	Name string `json:"name,omitempty"`

	// Path: GTM Environment's API relative path.
	Path string `json:"path,omitempty"`

	// TagManagerUrl: Auto generated link to the tag manager UI
	TagManagerUrl string `json:"tagManagerUrl,omitempty"`

	// Type: The type of this environment.
	//
	// Possible values:
	//   "latest"
	//   "live"
	//   "user"
	//   "workspace"
	Type string `json:"type,omitempty"`

	// Url: Default preview page url for the environment.
	Url string `json:"url,omitempty"`

	// WorkspaceId: Represents a link to a quick preview of a workspace.
	WorkspaceId string `json:"workspaceId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Environment) MarshalJSON() ([]byte, error) {
	type noMethod Environment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Folder: Represents a Google Tag Manager Folder.
type Folder struct {
	// AccountId: GTM Account ID.
	AccountId string `json:"accountId,omitempty"`

	// ContainerId: GTM Container ID.
	ContainerId string `json:"containerId,omitempty"`

	// Fingerprint: The fingerprint of the GTM Folder as computed at storage
	// time. This value is recomputed whenever the folder is modified.
	Fingerprint string `json:"fingerprint,omitempty"`

	// FolderId: The Folder ID uniquely identifies the GTM Folder.
	FolderId string `json:"folderId,omitempty"`

	// Name: Folder display name.
	Name string `json:"name,omitempty"`

	// Notes: User notes on how to apply this folder in the container.
	Notes string `json:"notes,omitempty"`

	// Path: GTM Folder's API relative path.
	Path string `json:"path,omitempty"`

	// TagManagerUrl: Auto generated link to the tag manager UI
	TagManagerUrl string `json:"tagManagerUrl,omitempty"`

	// WorkspaceId: GTM Workspace ID.
	WorkspaceId string `json:"workspaceId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Folder) MarshalJSON() ([]byte, error) {
	type noMethod Folder
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FolderEntities: Represents a Google Tag Manager Folder's contents.
type FolderEntities struct {
	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Tag: The list of tags inside the folder.
	Tag []*Tag `json:"tag,omitempty"`

	// Trigger: The list of triggers inside the folder.
	Trigger []*Trigger `json:"trigger,omitempty"`

	// Variable: The list of variables inside the folder.
	Variable []*Variable `json:"variable,omitempty"`

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

func (s *FolderEntities) MarshalJSON() ([]byte, error) {
	type noMethod FolderEntities
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GetWorkspaceStatusResponse: The changes that have occurred in the
// workspace since the base container version.
type GetWorkspaceStatusResponse struct {
	// MergeConflict: The merge conflict after sync.
	MergeConflict []*MergeConflict `json:"mergeConflict,omitempty"`

	// WorkspaceChange: Entities that have been changed in the workspace.
	WorkspaceChange []*Entity `json:"workspaceChange,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "MergeConflict") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MergeConflict") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GetWorkspaceStatusResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetWorkspaceStatusResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListAccountsResponse: List Accounts Response.
type ListAccountsResponse struct {
	// Account: List of GTM Accounts that a user has access to.
	Account []*Account `json:"account,omitempty"`

	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Account") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Account") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListAccountsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListAccountsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListContainerVersionsResponse: List container versions response.
type ListContainerVersionsResponse struct {
	// ContainerVersionHeader: All container version headers of a GTM
	// Container.
	ContainerVersionHeader []*ContainerVersionHeader `json:"containerVersionHeader,omitempty"`

	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "ContainerVersionHeader") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ContainerVersionHeader")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ListContainerVersionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListContainerVersionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListContainersResponse: List Containers Response.
type ListContainersResponse struct {
	// Container: All Containers of a GTM Account.
	Container []*Container `json:"container,omitempty"`

	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Container") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Container") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListContainersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListContainersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListEnabledBuiltInVariablesResponse: A list of enabled built-in
// variables.
type ListEnabledBuiltInVariablesResponse struct {
	// BuiltInVariable: All GTM BuiltInVariables of a GTM container.
	BuiltInVariable []*BuiltInVariable `json:"builtInVariable,omitempty"`

	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "BuiltInVariable") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BuiltInVariable") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ListEnabledBuiltInVariablesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListEnabledBuiltInVariablesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListEnvironmentsResponse: List Environments Response.
type ListEnvironmentsResponse struct {
	// Environment: All Environments of a GTM Container.
	Environment []*Environment `json:"environment,omitempty"`

	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *ListEnvironmentsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListEnvironmentsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListFoldersResponse: List Folders Response.
type ListFoldersResponse struct {
	// Folder: All GTM Folders of a GTM Container.
	Folder []*Folder `json:"folder,omitempty"`

	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Folder") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Folder") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListFoldersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListFoldersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListTagsResponse: List Tags Response.
type ListTagsResponse struct {
	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Tag: All GTM Tags of a GTM Container.
	Tag []*Tag `json:"tag,omitempty"`

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

func (s *ListTagsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListTagsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListTriggersResponse: List triggers response.
type ListTriggersResponse struct {
	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Trigger: All GTM Triggers of a GTM Container.
	Trigger []*Trigger `json:"trigger,omitempty"`

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

func (s *ListTriggersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListTriggersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListUserPermissionsResponse: List user permissions response.
type ListUserPermissionsResponse struct {
	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// UserPermission: All GTM UserPermissions of a GTM Account.
	UserPermission []*UserPermission `json:"userPermission,omitempty"`

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

func (s *ListUserPermissionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListUserPermissionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListVariablesResponse: List Variables Response.
type ListVariablesResponse struct {
	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Variable: All GTM Variables of a GTM Container.
	Variable []*Variable `json:"variable,omitempty"`

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

func (s *ListVariablesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListVariablesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListWorkspacesResponse: A list of workspaces in a container.
type ListWorkspacesResponse struct {
	// NextPageToken: Continuation token for fetching the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Workspace: All Workspaces of a GTM Container.
	Workspace []*Workspace `json:"workspace,omitempty"`

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

func (s *ListWorkspacesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListWorkspacesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MergeConflict: Represents a merge conflict.
type MergeConflict struct {
	// EntityInBaseVersion: The base version entity (since the latest sync
	// operation) that has conflicting changes compared to the workspace. If
	// this field is missing, it means the workspace entity is deleted from
	// the base version.
	EntityInBaseVersion *Entity `json:"entityInBaseVersion,omitempty"`

	// EntityInWorkspace: The workspace entity that has conflicting changes
	// compared to the base version. If an entity is deleted in a workspace,
	// it will still appear with a deleted change status.
	EntityInWorkspace *Entity `json:"entityInWorkspace,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EntityInBaseVersion")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EntityInBaseVersion") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *MergeConflict) MarshalJSON() ([]byte, error) {
	type noMethod MergeConflict
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Parameter: Represents a Google Tag Manager Parameter.
type Parameter struct {
	// Key: The named key that uniquely identifies a parameter. Required for
	// top-level parameters, as well as map values. Ignored for list values.
	Key string `json:"key,omitempty"`

	// List: This list parameter's parameters (keys will be ignored).
	List []*Parameter `json:"list,omitempty"`

	// Map: This map parameter's parameters (must have keys; keys must be
	// unique).
	Map []*Parameter `json:"map,omitempty"`

	// Type: The parameter type. Valid values are:
	// - boolean: The value represents a boolean, represented as 'true' or
	// 'false'
	// - integer: The value represents a 64-bit signed integer value, in
	// base 10
	// - list: A list of parameters should be specified
	// - map: A map of parameters should be specified
	// - template: The value represents any text; this can include variable
	// references (even variable references that might return non-string
	// types)
	//
	// Possible values:
	//   "boolean"
	//   "integer"
	//   "list"
	//   "map"
	//   "template"
	//   "typeUnspecified"
	Type string `json:"type,omitempty"`

	// Value: A parameter's value (may contain variable references such as
	// "{{myVariable}}") as appropriate to the specified type.
	Value string `json:"value,omitempty"`

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

func (s *Parameter) MarshalJSON() ([]byte, error) {
	type noMethod Parameter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PublishContainerVersionResponse: Publish container version response.
type PublishContainerVersionResponse struct {
	// CompilerError: Compiler errors or not.
	CompilerError bool `json:"compilerError,omitempty"`

	// ContainerVersion: The container version created.
	ContainerVersion *ContainerVersion `json:"containerVersion,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CompilerError") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CompilerError") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PublishContainerVersionResponse) MarshalJSON() ([]byte, error) {
	type noMethod PublishContainerVersionResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QuickPreviewResponse: Response to quick previewing a workspace.
type QuickPreviewResponse struct {
	// CompilerError: Were there compiler errors or not.
	CompilerError bool `json:"compilerError,omitempty"`

	// ContainerVersion: The quick previewed container version.
	ContainerVersion *ContainerVersion `json:"containerVersion,omitempty"`

	// SyncStatus: Whether quick previewing failed when syncing the
	// workspace to the latest container version.
	SyncStatus *SyncStatus `json:"syncStatus,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CompilerError") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CompilerError") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QuickPreviewResponse) MarshalJSON() ([]byte, error) {
	type noMethod QuickPreviewResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RevertBuiltInVariableResponse: The result of reverting a built-in
// variable in a workspace.
type RevertBuiltInVariableResponse struct {
	// Enabled: Whether the built-in variable is enabled after reversion.
	Enabled bool `json:"enabled,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Enabled") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Enabled") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RevertBuiltInVariableResponse) MarshalJSON() ([]byte, error) {
	type noMethod RevertBuiltInVariableResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RevertFolderResponse: The result of reverting folder changes in a
// workspace.
type RevertFolderResponse struct {
	// Folder: Folder as it appears in the latest container version since
	// the last workspace synchronization operation. If no folder is
	// present, that means the folder was deleted in the latest container
	// version.
	Folder *Folder `json:"folder,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Folder") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Folder") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RevertFolderResponse) MarshalJSON() ([]byte, error) {
	type noMethod RevertFolderResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RevertTagResponse: The result of reverting a tag in a workspace.
type RevertTagResponse struct {
	// Tag: Tag as it appears in the latest container version since the last
	// workspace synchronization operation. If no tag is present, that means
	// the tag was deleted in the latest container version.
	Tag *Tag `json:"tag,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Tag") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Tag") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RevertTagResponse) MarshalJSON() ([]byte, error) {
	type noMethod RevertTagResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RevertTriggerResponse: The result of reverting a trigger in a
// workspace.
type RevertTriggerResponse struct {
	// Trigger: Trigger as it appears in the latest container version since
	// the last workspace synchronization operation. If no trigger is
	// present, that means the trigger was deleted in the latest container
	// version.
	Trigger *Trigger `json:"trigger,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Trigger") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Trigger") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RevertTriggerResponse) MarshalJSON() ([]byte, error) {
	type noMethod RevertTriggerResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RevertVariableResponse: The result of reverting a variable in a
// workspace.
type RevertVariableResponse struct {
	// Variable: Variable as it appears in the latest container version
	// since the last workspace synchronization operation. If no variable is
	// present, that means the variable was deleted in the latest container
	// version.
	Variable *Variable `json:"variable,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Variable") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Variable") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RevertVariableResponse) MarshalJSON() ([]byte, error) {
	type noMethod RevertVariableResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SetupTag: Represents a reference to atag that fires before another
// tag in order to set up dependencies.
type SetupTag struct {
	// StopOnSetupFailure: If true, fire the main tag if and only if the
	// setup tag fires successfully. If false, fire the main tag regardless
	// of setup tag firing status.
	StopOnSetupFailure bool `json:"stopOnSetupFailure,omitempty"`

	// TagName: The name of the setup tag.
	TagName string `json:"tagName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "StopOnSetupFailure")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "StopOnSetupFailure") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SetupTag) MarshalJSON() ([]byte, error) {
	type noMethod SetupTag
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SyncStatus: The status of a workspace after synchronization.
type SyncStatus struct {
	// MergeConflict: Synchornization operation detected a merge conflict.
	MergeConflict bool `json:"mergeConflict,omitempty"`

	// SyncError: An error occurred during the synchronization operation.
	SyncError bool `json:"syncError,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MergeConflict") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MergeConflict") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SyncStatus) MarshalJSON() ([]byte, error) {
	type noMethod SyncStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SyncWorkspaceResponse: A response after synchronizing the workspace
// to the latest container version.
type SyncWorkspaceResponse struct {
	// MergeConflict: The merge conflict after sync. If this field is not
	// empty, the sync is still treated as successful. But a version cannot
	// be created until all conflicts are resolved.
	MergeConflict []*MergeConflict `json:"mergeConflict,omitempty"`

	// SyncStatus: Indicates whether synchronization caused a merge conflict
	// or sync error.
	SyncStatus *SyncStatus `json:"syncStatus,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "MergeConflict") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MergeConflict") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SyncWorkspaceResponse) MarshalJSON() ([]byte, error) {
	type noMethod SyncWorkspaceResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Tag: Represents a Google Tag Manager Tag.
type Tag struct {
	// AccountId: GTM Account ID.
	AccountId string `json:"accountId,omitempty"`

	// BlockingRuleId: Blocking rule IDs. If any of the listed rules
	// evaluate to true, the tag will not fire.
	BlockingRuleId []string `json:"blockingRuleId,omitempty"`

	// BlockingTriggerId: Blocking trigger IDs. If any of the listed
	// triggers evaluate to true, the tag will not fire.
	BlockingTriggerId []string `json:"blockingTriggerId,omitempty"`

	// ContainerId: GTM Container ID.
	ContainerId string `json:"containerId,omitempty"`

	// Fingerprint: The fingerprint of the GTM Tag as computed at storage
	// time. This value is recomputed whenever the tag is modified.
	Fingerprint string `json:"fingerprint,omitempty"`

	// FiringRuleId: Firing rule IDs. A tag will fire when any of the listed
	// rules are true and all of its blockingRuleIds (if any specified) are
	// false.
	FiringRuleId []string `json:"firingRuleId,omitempty"`

	// FiringTriggerId: Firing trigger IDs. A tag will fire when any of the
	// listed triggers are true and all of its blockingTriggerIds (if any
	// specified) are false.
	FiringTriggerId []string `json:"firingTriggerId,omitempty"`

	// LiveOnly: If set to true, this tag will only fire in the live
	// environment (e.g. not in preview or debug mode).
	LiveOnly bool `json:"liveOnly,omitempty"`

	// Name: Tag display name.
	Name string `json:"name,omitempty"`

	// Notes: User notes on how to apply this tag in the container.
	Notes string `json:"notes,omitempty"`

	// Parameter: The tag's parameters.
	Parameter []*Parameter `json:"parameter,omitempty"`

	// ParentFolderId: Parent folder id.
	ParentFolderId string `json:"parentFolderId,omitempty"`

	// Path: GTM Tag's API relative path.
	Path string `json:"path,omitempty"`

	// Priority: User defined numeric priority of the tag. Tags are fired
	// asynchronously in order of priority. Tags with higher numeric value
	// fire first. A tag's priority can be a positive or negative value. The
	// default value is 0.
	Priority *Parameter `json:"priority,omitempty"`

	// ScheduleEndMs: The end timestamp in milliseconds to schedule a tag.
	ScheduleEndMs int64 `json:"scheduleEndMs,omitempty,string"`

	// ScheduleStartMs: The start timestamp in milliseconds to schedule a
	// tag.
	ScheduleStartMs int64 `json:"scheduleStartMs,omitempty,string"`

	// SetupTag: The list of setup tags. Currently we only allow one.
	SetupTag []*SetupTag `json:"setupTag,omitempty"`

	// TagFiringOption: Option to fire this tag.
	//
	// Possible values:
	//   "oncePerEvent"
	//   "oncePerLoad"
	//   "tagFiringOptionUnspecified"
	//   "unlimited"
	TagFiringOption string `json:"tagFiringOption,omitempty"`

	// TagId: The Tag ID uniquely identifies the GTM Tag.
	TagId string `json:"tagId,omitempty"`

	// TagManagerUrl: Auto generated link to the tag manager UI
	TagManagerUrl string `json:"tagManagerUrl,omitempty"`

	// TeardownTag: The list of teardown tags. Currently we only allow one.
	TeardownTag []*TeardownTag `json:"teardownTag,omitempty"`

	// Type: GTM Tag Type.
	Type string `json:"type,omitempty"`

	// WorkspaceId: GTM Workspace ID.
	WorkspaceId string `json:"workspaceId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Tag) MarshalJSON() ([]byte, error) {
	type noMethod Tag
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TeardownTag: Represents a tag that fires after another tag in order
// to tear down dependencies.
type TeardownTag struct {
	// StopTeardownOnFailure: If true, fire the teardown tag if and only if
	// the main tag fires successfully. If false, fire the teardown tag
	// regardless of main tag firing status.
	StopTeardownOnFailure bool `json:"stopTeardownOnFailure,omitempty"`

	// TagName: The name of the teardown tag.
	TagName string `json:"tagName,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "StopTeardownOnFailure") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "StopTeardownOnFailure") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *TeardownTag) MarshalJSON() ([]byte, error) {
	type noMethod TeardownTag
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Timestamp: A Timestamp represents a point in time independent of any
// time zone or calendar, represented as seconds and fractions of
// seconds at nanosecond resolution in UTC Epoch time. It is encoded
// using the Proleptic Gregorian Calendar which extends the Gregorian
// calendar backwards to year one. It is encoded assuming all minutes
// are 60 seconds long, i.e. leap seconds are "smeared" so that no leap
// second table is needed for interpretation. Range is from
// 0001-01-01T00:00:00Z to 9999-12-31T23:59:59.999999999Z. By
// restricting to that range, we ensure that we can convert to and from
// RFC 3339 date strings. See
// [https://www.ietf.org/rfc/rfc3339.txt](https://www.ietf.org/rfc/rfc333
// 9.txt).
//
// # Examples
//
// Example 1: Compute Timestamp from POSIX `time()`.
//
// Timestamp timestamp; timestamp.set_seconds(time(NULL));
// timestamp.set_nanos(0);
//
// Example 2: Compute Timestamp from POSIX `gettimeofday()`.
//
// struct timeval tv; gettimeofday(&tv, NULL);
//
// Timestamp timestamp; timestamp.set_seconds(tv.tv_sec);
// timestamp.set_nanos(tv.tv_usec * 1000);
//
// Example 3: Compute Timestamp from Win32
// `GetSystemTimeAsFileTime()`.
//
// FILETIME ft; GetSystemTimeAsFileTime(&ft); UINT64 ticks =
// (((UINT64)ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
//
// // A Windows tick is 100 nanoseconds. Windows epoch
// 1601-01-01T00:00:00Z // is 11644473600 seconds before Unix epoch
// 1970-01-01T00:00:00Z. Timestamp timestamp;
// timestamp.set_seconds((INT64) ((ticks / 10000000) - 11644473600LL));
// timestamp.set_nanos((INT32) ((ticks % 10000000) * 100));
//
// Example 4: Compute Timestamp from Java
// `System.currentTimeMillis()`.
//
// long millis = System.currentTimeMillis();
//
// Timestamp timestamp = Timestamp.newBuilder().setSeconds(millis /
// 1000) .setNanos((int) ((millis % 1000) *
// 1000000)).build();
//
//
//
// Example 5: Compute Timestamp from current time in Python.
//
// timestamp = Timestamp() timestamp.GetCurrentTime()
//
// # JSON Mapping
//
// In JSON format, the Timestamp type is encoded as a string in the [RFC
// 3339](https://www.ietf.org/rfc/rfc3339.txt) format. That is, the
// format is "{year}-{month}-{day}T{hour}:{min}:{sec}[.{frac_sec}]Z"
// where {year} is always expressed using four digits while {month},
// {day}, {hour}, {min}, and {sec} are zero-padded to two digits each.
// The fractional seconds, which can go up to 9 digits (i.e. up to 1
// nanosecond resolution), are optional. The "Z" suffix indicates the
// timezone ("UTC"); the timezone is required, though only UTC (as
// indicated by "Z") is presently supported.
//
// For example, "2017-01-15T01:30:15.01Z" encodes 15.01 seconds past
// 01:30 UTC on January 15, 2017.
//
// In JavaScript, one can convert a Date object to this format using the
// standard
// [toISOString()](https://developer.mozilla.org/en-US/docs/Web/JavaScrip
// t/Reference/Global_Objects/Date/toISOString] method. In Python, a
// standard `datetime.datetime` object can be converted to this format
// using
// [`strftime`](https://docs.python.org/2/library/time.html#time.strftime
// ) with the time format spec '%Y-%m-%dT%H:%M:%S.%fZ'. Likewise, in
// Java, one can use the Joda Time's [`ISODateTimeFormat.dateTime()`](
// http://joda-time.sourceforge.net/apidocs/org/joda/time/format/ISODateTimeFormat.html#dateTime()) to obtain a formatter capable of generating timestamps in this
// format.
type Timestamp struct {
	// Nanos: Non-negative fractions of a second at nanosecond resolution.
	// Negative second values with fractions must still have non-negative
	// nanos values that count forward in time. Must be from 0 to
	// 999,999,999 inclusive.
	Nanos int64 `json:"nanos,omitempty"`

	// Seconds: Represents seconds of UTC time since Unix epoch
	// 1970-01-01T00:00:00Z. Must be from 0001-01-01T00:00:00Z to
	// 9999-12-31T23:59:59Z inclusive.
	Seconds int64 `json:"seconds,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Nanos") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Nanos") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Timestamp) MarshalJSON() ([]byte, error) {
	type noMethod Timestamp
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Trigger: Represents a Google Tag Manager Trigger
type Trigger struct {
	// AccountId: GTM Account ID.
	AccountId string `json:"accountId,omitempty"`

	// AutoEventFilter: Used in the case of auto event tracking.
	AutoEventFilter []*Condition `json:"autoEventFilter,omitempty"`

	// CheckValidation: Whether or not we should only fire tags if the form
	// submit or link click event is not cancelled by some other event
	// handler (e.g. because of validation). Only valid for Form Submission
	// and Link Click triggers.
	CheckValidation *Parameter `json:"checkValidation,omitempty"`

	// ContainerId: GTM Container ID.
	ContainerId string `json:"containerId,omitempty"`

	// ContinuousTimeMinMilliseconds: A visibility trigger minimum
	// continuous visible time (in milliseconds). Only valid for AMP
	// Visibility trigger.
	ContinuousTimeMinMilliseconds *Parameter `json:"continuousTimeMinMilliseconds,omitempty"`

	// CustomEventFilter: Used in the case of custom event, which is fired
	// iff all Conditions are true.
	CustomEventFilter []*Condition `json:"customEventFilter,omitempty"`

	// EventName: Name of the GTM event that is fired. Only valid for Timer
	// triggers.
	EventName *Parameter `json:"eventName,omitempty"`

	// Filter: The trigger will only fire iff all Conditions are true.
	Filter []*Condition `json:"filter,omitempty"`

	// Fingerprint: The fingerprint of the GTM Trigger as computed at
	// storage time. This value is recomputed whenever the trigger is
	// modified.
	Fingerprint string `json:"fingerprint,omitempty"`

	// HorizontalScrollPercentageList: List of integer percentage values for
	// scroll triggers. The trigger will fire when each percentage is
	// reached when the view is scrolled horizontally. Only valid for AMP
	// scroll triggers.
	HorizontalScrollPercentageList *Parameter `json:"horizontalScrollPercentageList,omitempty"`

	// Interval: Time between triggering recurring Timer Events (in
	// milliseconds). Only valid for Timer triggers.
	Interval *Parameter `json:"interval,omitempty"`

	// IntervalSeconds: Time between Timer Events to fire (in seconds). Only
	// valid for AMP Timer trigger.
	IntervalSeconds *Parameter `json:"intervalSeconds,omitempty"`

	// Limit: Limit of the number of GTM events this Timer Trigger will
	// fire. If no limit is set, we will continue to fire GTM events until
	// the user leaves the page. Only valid for Timer triggers.
	Limit *Parameter `json:"limit,omitempty"`

	// MaxTimerLengthSeconds: Max time to fire Timer Events (in seconds).
	// Only valid for AMP Timer trigger.
	MaxTimerLengthSeconds *Parameter `json:"maxTimerLengthSeconds,omitempty"`

	// Name: Trigger display name.
	Name string `json:"name,omitempty"`

	// Notes: User notes on how to apply this trigger in the container.
	Notes string `json:"notes,omitempty"`

	// ParentFolderId: Parent folder id.
	ParentFolderId string `json:"parentFolderId,omitempty"`

	// Path: GTM Trigger's API relative path.
	Path string `json:"path,omitempty"`

	// Selector: A click trigger CSS selector (i.e. "a", "button" etc.).
	// Only valid for AMP Click trigger.
	Selector *Parameter `json:"selector,omitempty"`

	// TagManagerUrl: Auto generated link to the tag manager UI
	TagManagerUrl string `json:"tagManagerUrl,omitempty"`

	// TotalTimeMinMilliseconds: A visibility trigger minimum total visible
	// time (in milliseconds). Only valid for AMP Visibility trigger.
	TotalTimeMinMilliseconds *Parameter `json:"totalTimeMinMilliseconds,omitempty"`

	// TriggerId: The Trigger ID uniquely identifies the GTM Trigger.
	TriggerId string `json:"triggerId,omitempty"`

	// Type: Defines the data layer event that causes this trigger.
	//
	// Possible values:
	//   "always"
	//   "ampClick"
	//   "ampScroll"
	//   "ampTimer"
	//   "ampVisibility"
	//   "click"
	//   "customEvent"
	//   "domReady"
	//   "eventTypeUnspecified"
	//   "firebaseAppException"
	//   "firebaseAppUpdate"
	//   "firebaseCampaign"
	//   "firebaseFirstOpen"
	//   "firebaseInAppPurchase"
	//   "firebaseNotificationDismiss"
	//   "firebaseNotificationForeground"
	//   "firebaseNotificationOpen"
	//   "firebaseNotificationReceive"
	//   "firebaseOsUpdate"
	//   "firebaseSessionStart"
	//   "firebaseUserEngagement"
	//   "formSubmission"
	//   "historyChange"
	//   "jsError"
	//   "linkClick"
	//   "pageview"
	//   "timer"
	//   "windowLoaded"
	Type string `json:"type,omitempty"`

	// UniqueTriggerId: Globally unique id of the trigger that
	// auto-generates this (a Form Submit, Link Click or Timer listener) if
	// any. Used to make incompatible auto-events work together with trigger
	// filtering based on trigger ids. This value is populated during output
	// generation since the tags implied by triggers don't exist until then.
	// Only valid for Form Submit, Link Click and Timer triggers.
	UniqueTriggerId *Parameter `json:"uniqueTriggerId,omitempty"`

	// VerticalScrollPercentageList: List of integer percentage values for
	// scroll triggers. The trigger will fire when each percentage is
	// reached when the view is scrolled vertically. Only valid for AMP
	// scroll triggers.
	VerticalScrollPercentageList *Parameter `json:"verticalScrollPercentageList,omitempty"`

	// VisibilitySelector: A visibility trigger CSS selector (i.e. "#id").
	// Only valid for AMP Visibility trigger.
	VisibilitySelector *Parameter `json:"visibilitySelector,omitempty"`

	// VisiblePercentageMax: A visibility trigger maximum percent
	// visibility. Only valid for AMP Visibility trigger.
	VisiblePercentageMax *Parameter `json:"visiblePercentageMax,omitempty"`

	// VisiblePercentageMin: A visibility trigger minimum percent
	// visibility. Only valid for AMP Visibility trigger.
	VisiblePercentageMin *Parameter `json:"visiblePercentageMin,omitempty"`

	// WaitForTags: Whether or not we should delay the form submissions or
	// link opening until all of the tags have fired (by preventing the
	// default action and later simulating the default action). Only valid
	// for Form Submission and Link Click triggers.
	WaitForTags *Parameter `json:"waitForTags,omitempty"`

	// WaitForTagsTimeout: How long to wait (in milliseconds) for tags to
	// fire when 'waits_for_tags' above evaluates to true. Only valid for
	// Form Submission and Link Click triggers.
	WaitForTagsTimeout *Parameter `json:"waitForTagsTimeout,omitempty"`

	// WorkspaceId: GTM Workspace ID.
	WorkspaceId string `json:"workspaceId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Trigger) MarshalJSON() ([]byte, error) {
	type noMethod Trigger
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateWorkspaceProposalRequest: Updates a workspace proposal with
// patch-like semantics.
type UpdateWorkspaceProposalRequest struct {
	// Fingerprint: When provided, this fingerprint must match the
	// fingerprint of the proposal in storage.
	Fingerprint string `json:"fingerprint,omitempty"`

	// NewComment: If present, a new comment is added to the workspace
	// proposal history.
	NewComment *WorkspaceProposalHistoryComment `json:"newComment,omitempty"`

	// Reviewers: If present, the list of reviewers of the workspace
	// proposal is updated.
	Reviewers []*WorkspaceProposalUser `json:"reviewers,omitempty"`

	// Status: If present, the status of the workspace proposal is updated.
	//
	// Possible values:
	//   "approved"
	//   "cancelled"
	//   "completed"
	//   "requested"
	//   "reviewed"
	//   "statusUnspecified"
	Status string `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fingerprint") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fingerprint") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateWorkspaceProposalRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateWorkspaceProposalRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UserPermission: Represents a user's permissions to an account and its
// container.
type UserPermission struct {
	// AccountAccess: GTM Account access permissions.
	AccountAccess *AccountAccess `json:"accountAccess,omitempty"`

	// AccountId: The Account ID uniquely identifies the GTM Account.
	AccountId string `json:"accountId,omitempty"`

	// ContainerAccess: GTM Container access permissions.
	ContainerAccess []*ContainerAccess `json:"containerAccess,omitempty"`

	// EmailAddress: User's email address.
	EmailAddress string `json:"emailAddress,omitempty"`

	// Path: GTM UserPermission's API relative path.
	Path string `json:"path,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountAccess") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountAccess") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UserPermission) MarshalJSON() ([]byte, error) {
	type noMethod UserPermission
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Variable: Represents a Google Tag Manager Variable.
type Variable struct {
	// AccountId: GTM Account ID.
	AccountId string `json:"accountId,omitempty"`

	// ContainerId: GTM Container ID.
	ContainerId string `json:"containerId,omitempty"`

	// DisablingTriggerId: For mobile containers only: A list of trigger IDs
	// for disabling conditional variables; the variable is enabled if one
	// of the enabling trigger is true while all the disabling trigger are
	// false. Treated as an unordered set.
	DisablingTriggerId []string `json:"disablingTriggerId,omitempty"`

	// EnablingTriggerId: For mobile containers only: A list of trigger IDs
	// for enabling conditional variables; the variable is enabled if one of
	// the enabling triggers is true while all the disabling triggers are
	// false. Treated as an unordered set.
	EnablingTriggerId []string `json:"enablingTriggerId,omitempty"`

	// Fingerprint: The fingerprint of the GTM Variable as computed at
	// storage time. This value is recomputed whenever the variable is
	// modified.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Name: Variable display name.
	Name string `json:"name,omitempty"`

	// Notes: User notes on how to apply this variable in the container.
	Notes string `json:"notes,omitempty"`

	// Parameter: The variable's parameters.
	Parameter []*Parameter `json:"parameter,omitempty"`

	// ParentFolderId: Parent folder id.
	ParentFolderId string `json:"parentFolderId,omitempty"`

	// Path: GTM Variable's API relative path.
	Path string `json:"path,omitempty"`

	// ScheduleEndMs: The end timestamp in milliseconds to schedule a
	// variable.
	ScheduleEndMs int64 `json:"scheduleEndMs,omitempty,string"`

	// ScheduleStartMs: The start timestamp in milliseconds to schedule a
	// variable.
	ScheduleStartMs int64 `json:"scheduleStartMs,omitempty,string"`

	// TagManagerUrl: Auto generated link to the tag manager UI
	TagManagerUrl string `json:"tagManagerUrl,omitempty"`

	// Type: GTM Variable Type.
	Type string `json:"type,omitempty"`

	// VariableId: The Variable ID uniquely identifies the GTM Variable.
	VariableId string `json:"variableId,omitempty"`

	// WorkspaceId: GTM Workspace ID.
	WorkspaceId string `json:"workspaceId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Variable) MarshalJSON() ([]byte, error) {
	type noMethod Variable
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Workspace: Represents a Google Tag Manager Container Workspace.
type Workspace struct {
	// AccountId: GTM Account ID.
	AccountId string `json:"accountId,omitempty"`

	// ContainerId: GTM Container ID.
	ContainerId string `json:"containerId,omitempty"`

	// Description: Workspace description.
	Description string `json:"description,omitempty"`

	// Fingerprint: The fingerprint of the GTM Workspace as computed at
	// storage time. This value is recomputed whenever the workspace is
	// modified.
	Fingerprint string `json:"fingerprint,omitempty"`

	// Name: Workspace display name.
	Name string `json:"name,omitempty"`

	// Path: GTM Workspace's API relative path.
	Path string `json:"path,omitempty"`

	// TagManagerUrl: Auto generated link to the tag manager UI
	TagManagerUrl string `json:"tagManagerUrl,omitempty"`

	// WorkspaceId: The Workspace ID uniquely identifies the GTM Workspace.
	WorkspaceId string `json:"workspaceId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AccountId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AccountId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Workspace) MarshalJSON() ([]byte, error) {
	type noMethod Workspace
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WorkspaceProposal: A workspace proposal represents an ongoing review
// of workspace changes in an effort to gain approval for container
// version creation.
type WorkspaceProposal struct {
	// Authors: List of authors for the workspace proposal.
	Authors []*WorkspaceProposalUser `json:"authors,omitempty"`

	// Fingerprint: The fingerprint of the GTM workspace proposal as
	// computed at storage time. This value is recomputed whenever the
	// proposal is modified.
	Fingerprint string `json:"fingerprint,omitempty"`

	// History: Records the history of comments and status changes.
	History []*WorkspaceProposalHistory `json:"history,omitempty"`

	// Path: GTM workspace proposal's relative path.
	Path string `json:"path,omitempty"`

	// Reviewers: Lists of reviewers for the workspace proposal.
	Reviewers []*WorkspaceProposalUser `json:"reviewers,omitempty"`

	// Status: The status of the workspace proposal as it goes through
	// review.
	//
	// Possible values:
	//   "approved"
	//   "cancelled"
	//   "completed"
	//   "requested"
	//   "reviewed"
	//   "statusUnspecified"
	Status string `json:"status,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Authors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Authors") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *WorkspaceProposal) MarshalJSON() ([]byte, error) {
	type noMethod WorkspaceProposal
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WorkspaceProposalHistory: A history event that represents a comment
// or status change in the proposal.
type WorkspaceProposalHistory struct {
	// Comment: A user or reviewer comment.
	Comment *WorkspaceProposalHistoryComment `json:"comment,omitempty"`

	// CreatedBy: The party responsible for the change in history.
	CreatedBy *WorkspaceProposalUser `json:"createdBy,omitempty"`

	// CreatedTimestamp: When this history event was added to the workspace
	// proposal.
	CreatedTimestamp *Timestamp `json:"createdTimestamp,omitempty"`

	// StatusChange: A change in the proposal's status.
	StatusChange *WorkspaceProposalHistoryStatusChange `json:"statusChange,omitempty"`

	// Type: The history type distinguishing between comments and status
	// changes.
	//
	// Possible values:
	//   "comment"
	//   "statusChange"
	//   "unspecified"
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Comment") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Comment") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *WorkspaceProposalHistory) MarshalJSON() ([]byte, error) {
	type noMethod WorkspaceProposalHistory
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WorkspaceProposalHistoryComment: A comment from the reviewer or
// author.
type WorkspaceProposalHistoryComment struct {
	// Content: The contents of the reviewer or author comment.
	Content string `json:"content,omitempty"`

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

func (s *WorkspaceProposalHistoryComment) MarshalJSON() ([]byte, error) {
	type noMethod WorkspaceProposalHistoryComment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WorkspaceProposalHistoryStatusChange: A change in the proposal's
// status.
type WorkspaceProposalHistoryStatusChange struct {
	// NewStatus: The new proposal status after that status change.
	//
	// Possible values:
	//   "approved"
	//   "cancelled"
	//   "completed"
	//   "requested"
	//   "reviewed"
	//   "statusUnspecified"
	NewStatus string `json:"newStatus,omitempty"`

	// OldStatus: The old proposal status before the status change.
	//
	// Possible values:
	//   "approved"
	//   "cancelled"
	//   "completed"
	//   "requested"
	//   "reviewed"
	//   "statusUnspecified"
	OldStatus string `json:"oldStatus,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NewStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NewStatus") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *WorkspaceProposalHistoryStatusChange) MarshalJSON() ([]byte, error) {
	type noMethod WorkspaceProposalHistoryStatusChange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WorkspaceProposalUser: Represents an external user or internal Google
// Tag Manager system.
type WorkspaceProposalUser struct {
	// GaiaId: Gaia id associated with a user, absent for the Google Tag
	// Manager system.
	GaiaId int64 `json:"gaiaId,omitempty,string"`

	// Type: User type distinguishes between a user and the Google Tag
	// Manager system.
	//
	// Possible values:
	//   "gaiaId"
	//   "system"
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "GaiaId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "GaiaId") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *WorkspaceProposalUser) MarshalJSON() ([]byte, error) {
	type noMethod WorkspaceProposalUser
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "tagmanager.accounts.get":

type AccountsGetCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a GTM Account.
func (r *AccountsService) Get(path string) *AccountsGetCall {
	c := &AccountsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsGetCall) Fields(s ...googleapi.Field) *AccountsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsGetCall) IfNoneMatch(entityTag string) *AccountsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsGetCall) Context(ctx context.Context) *AccountsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.get" call.
// Exactly one of *Account or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Account.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsGetCall) Do(opts ...googleapi.CallOption) (*Account, error) {
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
	ret := &Account{
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
	//   "description": "Gets a GTM Account.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.get",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Accounts's API relative path. Example: accounts/{account_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "response": {
	//     "$ref": "Account"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.manage.accounts",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.list":

type AccountsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all GTM Accounts that a user has access to.
func (r *AccountsService) List() *AccountsListCall {
	c := &AccountsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsListCall) PageToken(pageToken string) *AccountsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsListCall) Fields(s ...googleapi.Field) *AccountsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsListCall) IfNoneMatch(entityTag string) *AccountsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsListCall) Context(ctx context.Context) *AccountsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.list" call.
// Exactly one of *ListAccountsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListAccountsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsListCall) Do(opts ...googleapi.CallOption) (*ListAccountsResponse, error) {
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
	ret := &ListAccountsResponse{
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
	//   "description": "Lists all GTM Accounts that a user has access to.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.list",
	//   "parameters": {
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts",
	//   "response": {
	//     "$ref": "ListAccountsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.manage.accounts",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsListCall) Pages(ctx context.Context, f func(*ListAccountsResponse) error) error {
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

// method id "tagmanager.accounts.update":

type AccountsUpdateCall struct {
	s          *Service
	path       string
	account    *Account
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a GTM Account.
func (r *AccountsService) Update(path string, account *Account) *AccountsUpdateCall {
	c := &AccountsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.account = account
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the account in
// storage.
func (c *AccountsUpdateCall) Fingerprint(fingerprint string) *AccountsUpdateCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsUpdateCall) Fields(s ...googleapi.Field) *AccountsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsUpdateCall) Context(ctx context.Context) *AccountsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.account)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.update" call.
// Exactly one of *Account or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Account.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsUpdateCall) Do(opts ...googleapi.CallOption) (*Account, error) {
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
	ret := &Account{
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
	//   "description": "Updates a GTM Account.",
	//   "httpMethod": "PUT",
	//   "id": "tagmanager.accounts.update",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the account in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Accounts's API relative path. Example: accounts/{account_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "Account"
	//   },
	//   "response": {
	//     "$ref": "Account"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.manage.accounts"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.create":

type AccountsContainersCreateCall struct {
	s          *Service
	parent     string
	container  *Container
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a Container.
func (r *AccountsContainersService) Create(parent string, container *Container) *AccountsContainersCreateCall {
	c := &AccountsContainersCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.container = container
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersCreateCall) Fields(s ...googleapi.Field) *AccountsContainersCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersCreateCall) Context(ctx context.Context) *AccountsContainersCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.container)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/containers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.create" call.
// Exactly one of *Container or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Container.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersCreateCall) Do(opts ...googleapi.CallOption) (*Container, error) {
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
	ret := &Container{
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
	//   "description": "Creates a Container.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM Account's API relative path. Example: accounts/{account_id}.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/containers",
	//   "request": {
	//     "$ref": "Container"
	//   },
	//   "response": {
	//     "$ref": "Container"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.delete":

type AccountsContainersDeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a Container.
func (r *AccountsContainersService) Delete(path string) *AccountsContainersDeleteCall {
	c := &AccountsContainersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersDeleteCall) Fields(s ...googleapi.Field) *AccountsContainersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersDeleteCall) Context(ctx context.Context) *AccountsContainersDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.delete" call.
func (c *AccountsContainersDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes a Container.",
	//   "httpMethod": "DELETE",
	//   "id": "tagmanager.accounts.containers.delete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Container's API relative path. Example: accounts/{account_id}/containers/{container_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.delete.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.get":

type AccountsContainersGetCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a Container.
func (r *AccountsContainersService) Get(path string) *AccountsContainersGetCall {
	c := &AccountsContainersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersGetCall) Fields(s ...googleapi.Field) *AccountsContainersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersGetCall) IfNoneMatch(entityTag string) *AccountsContainersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersGetCall) Context(ctx context.Context) *AccountsContainersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.get" call.
// Exactly one of *Container or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Container.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersGetCall) Do(opts ...googleapi.CallOption) (*Container, error) {
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
	ret := &Container{
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
	//   "description": "Gets a Container.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.get",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Container's API relative path. Example: accounts/{account_id}/containers/{container_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "response": {
	//     "$ref": "Container"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.list":

type AccountsContainersListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all Containers that belongs to a GTM Account.
func (r *AccountsContainersService) List(parent string) *AccountsContainersListCall {
	c := &AccountsContainersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsContainersListCall) PageToken(pageToken string) *AccountsContainersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersListCall) Fields(s ...googleapi.Field) *AccountsContainersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersListCall) IfNoneMatch(entityTag string) *AccountsContainersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersListCall) Context(ctx context.Context) *AccountsContainersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/containers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.list" call.
// Exactly one of *ListContainersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListContainersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersListCall) Do(opts ...googleapi.CallOption) (*ListContainersResponse, error) {
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
	ret := &ListContainersResponse{
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
	//   "description": "Lists all Containers that belongs to a GTM Account.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "GTM Accounts's API relative path. Example: accounts/{account_id}.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/containers",
	//   "response": {
	//     "$ref": "ListContainersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsContainersListCall) Pages(ctx context.Context, f func(*ListContainersResponse) error) error {
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

// method id "tagmanager.accounts.containers.update":

type AccountsContainersUpdateCall struct {
	s          *Service
	path       string
	container  *Container
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a Container.
func (r *AccountsContainersService) Update(path string, container *Container) *AccountsContainersUpdateCall {
	c := &AccountsContainersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.container = container
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the container in
// storage.
func (c *AccountsContainersUpdateCall) Fingerprint(fingerprint string) *AccountsContainersUpdateCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersUpdateCall) Fields(s ...googleapi.Field) *AccountsContainersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersUpdateCall) Context(ctx context.Context) *AccountsContainersUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.container)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.update" call.
// Exactly one of *Container or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Container.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersUpdateCall) Do(opts ...googleapi.CallOption) (*Container, error) {
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
	ret := &Container{
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
	//   "description": "Updates a Container.",
	//   "httpMethod": "PUT",
	//   "id": "tagmanager.accounts.containers.update",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the container in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Container's API relative path. Example: accounts/{account_id}/containers/{container_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "Container"
	//   },
	//   "response": {
	//     "$ref": "Container"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.environments.create":

type AccountsContainersEnvironmentsCreateCall struct {
	s           *Service
	parent      string
	environment *Environment
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Create: Creates a GTM Environment.
func (r *AccountsContainersEnvironmentsService) Create(parent string, environment *Environment) *AccountsContainersEnvironmentsCreateCall {
	c := &AccountsContainersEnvironmentsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.environment = environment
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersEnvironmentsCreateCall) Fields(s ...googleapi.Field) *AccountsContainersEnvironmentsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersEnvironmentsCreateCall) Context(ctx context.Context) *AccountsContainersEnvironmentsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersEnvironmentsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersEnvironmentsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.environment)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/environments")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.environments.create" call.
// Exactly one of *Environment or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Environment.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersEnvironmentsCreateCall) Do(opts ...googleapi.CallOption) (*Environment, error) {
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
	ret := &Environment{
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
	//   "description": "Creates a GTM Environment.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.environments.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM Container's API relative path. Example: accounts/{account_id}/containers/{container_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/environments",
	//   "request": {
	//     "$ref": "Environment"
	//   },
	//   "response": {
	//     "$ref": "Environment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.environments.delete":

type AccountsContainersEnvironmentsDeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a GTM Environment.
func (r *AccountsContainersEnvironmentsService) Delete(path string) *AccountsContainersEnvironmentsDeleteCall {
	c := &AccountsContainersEnvironmentsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersEnvironmentsDeleteCall) Fields(s ...googleapi.Field) *AccountsContainersEnvironmentsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersEnvironmentsDeleteCall) Context(ctx context.Context) *AccountsContainersEnvironmentsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersEnvironmentsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersEnvironmentsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.environments.delete" call.
func (c *AccountsContainersEnvironmentsDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes a GTM Environment.",
	//   "httpMethod": "DELETE",
	//   "id": "tagmanager.accounts.containers.environments.delete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Environment's API relative path. Example: accounts/{account_id}/containers/{container_id}/environments/{environment_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.environments.get":

type AccountsContainersEnvironmentsGetCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a GTM Environment.
func (r *AccountsContainersEnvironmentsService) Get(path string) *AccountsContainersEnvironmentsGetCall {
	c := &AccountsContainersEnvironmentsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersEnvironmentsGetCall) Fields(s ...googleapi.Field) *AccountsContainersEnvironmentsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersEnvironmentsGetCall) IfNoneMatch(entityTag string) *AccountsContainersEnvironmentsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersEnvironmentsGetCall) Context(ctx context.Context) *AccountsContainersEnvironmentsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersEnvironmentsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersEnvironmentsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.environments.get" call.
// Exactly one of *Environment or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Environment.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersEnvironmentsGetCall) Do(opts ...googleapi.CallOption) (*Environment, error) {
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
	ret := &Environment{
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
	//   "description": "Gets a GTM Environment.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.environments.get",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Environment's API relative path. Example: accounts/{account_id}/containers/{container_id}/environments/{environment_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "response": {
	//     "$ref": "Environment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.environments.list":

type AccountsContainersEnvironmentsListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all GTM Environments of a GTM Container.
func (r *AccountsContainersEnvironmentsService) List(parent string) *AccountsContainersEnvironmentsListCall {
	c := &AccountsContainersEnvironmentsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsContainersEnvironmentsListCall) PageToken(pageToken string) *AccountsContainersEnvironmentsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersEnvironmentsListCall) Fields(s ...googleapi.Field) *AccountsContainersEnvironmentsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersEnvironmentsListCall) IfNoneMatch(entityTag string) *AccountsContainersEnvironmentsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersEnvironmentsListCall) Context(ctx context.Context) *AccountsContainersEnvironmentsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersEnvironmentsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersEnvironmentsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/environments")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.environments.list" call.
// Exactly one of *ListEnvironmentsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListEnvironmentsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersEnvironmentsListCall) Do(opts ...googleapi.CallOption) (*ListEnvironmentsResponse, error) {
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
	ret := &ListEnvironmentsResponse{
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
	//   "description": "Lists all GTM Environments of a GTM Container.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.environments.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "GTM Container's API relative path. Example: accounts/{account_id}/containers/{container_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/environments",
	//   "response": {
	//     "$ref": "ListEnvironmentsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsContainersEnvironmentsListCall) Pages(ctx context.Context, f func(*ListEnvironmentsResponse) error) error {
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

// method id "tagmanager.accounts.containers.environments.patch":

type AccountsContainersEnvironmentsPatchCall struct {
	s           *Service
	path        string
	environment *Environment
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Patch: Updates a GTM Environment. This method supports patch
// semantics.
func (r *AccountsContainersEnvironmentsService) Patch(path string, environment *Environment) *AccountsContainersEnvironmentsPatchCall {
	c := &AccountsContainersEnvironmentsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.environment = environment
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the environment in
// storage.
func (c *AccountsContainersEnvironmentsPatchCall) Fingerprint(fingerprint string) *AccountsContainersEnvironmentsPatchCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersEnvironmentsPatchCall) Fields(s ...googleapi.Field) *AccountsContainersEnvironmentsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersEnvironmentsPatchCall) Context(ctx context.Context) *AccountsContainersEnvironmentsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersEnvironmentsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersEnvironmentsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.environment)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.environments.patch" call.
// Exactly one of *Environment or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Environment.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersEnvironmentsPatchCall) Do(opts ...googleapi.CallOption) (*Environment, error) {
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
	ret := &Environment{
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
	//   "description": "Updates a GTM Environment. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "tagmanager.accounts.containers.environments.patch",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the environment in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Environment's API relative path. Example: accounts/{account_id}/containers/{container_id}/environments/{environment_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "Environment"
	//   },
	//   "response": {
	//     "$ref": "Environment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.environments.reauthorize":

type AccountsContainersEnvironmentsReauthorizeCall struct {
	s           *Service
	path        string
	environment *Environment
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Reauthorize: Re-generates the authorization code for a GTM
// Environment.
func (r *AccountsContainersEnvironmentsService) Reauthorize(path string, environment *Environment) *AccountsContainersEnvironmentsReauthorizeCall {
	c := &AccountsContainersEnvironmentsReauthorizeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.environment = environment
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersEnvironmentsReauthorizeCall) Fields(s ...googleapi.Field) *AccountsContainersEnvironmentsReauthorizeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersEnvironmentsReauthorizeCall) Context(ctx context.Context) *AccountsContainersEnvironmentsReauthorizeCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersEnvironmentsReauthorizeCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersEnvironmentsReauthorizeCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.environment)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:reauthorize")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.environments.reauthorize" call.
// Exactly one of *Environment or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Environment.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersEnvironmentsReauthorizeCall) Do(opts ...googleapi.CallOption) (*Environment, error) {
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
	ret := &Environment{
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
	//   "description": "Re-generates the authorization code for a GTM Environment.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.environments.reauthorize",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Environment's API relative path. Example: accounts/{account_id}/containers/{container_id}/environments/{environment_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:reauthorize",
	//   "request": {
	//     "$ref": "Environment"
	//   },
	//   "response": {
	//     "$ref": "Environment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.publish"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.environments.update":

type AccountsContainersEnvironmentsUpdateCall struct {
	s           *Service
	path        string
	environment *Environment
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Update: Updates a GTM Environment.
func (r *AccountsContainersEnvironmentsService) Update(path string, environment *Environment) *AccountsContainersEnvironmentsUpdateCall {
	c := &AccountsContainersEnvironmentsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.environment = environment
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the environment in
// storage.
func (c *AccountsContainersEnvironmentsUpdateCall) Fingerprint(fingerprint string) *AccountsContainersEnvironmentsUpdateCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersEnvironmentsUpdateCall) Fields(s ...googleapi.Field) *AccountsContainersEnvironmentsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersEnvironmentsUpdateCall) Context(ctx context.Context) *AccountsContainersEnvironmentsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersEnvironmentsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersEnvironmentsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.environment)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.environments.update" call.
// Exactly one of *Environment or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Environment.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersEnvironmentsUpdateCall) Do(opts ...googleapi.CallOption) (*Environment, error) {
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
	ret := &Environment{
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
	//   "description": "Updates a GTM Environment.",
	//   "httpMethod": "PUT",
	//   "id": "tagmanager.accounts.containers.environments.update",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the environment in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Environment's API relative path. Example: accounts/{account_id}/containers/{container_id}/environments/{environment_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "Environment"
	//   },
	//   "response": {
	//     "$ref": "Environment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.version_headers.latest":

type AccountsContainersVersionHeadersLatestCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Latest: Gets the latest container version header
func (r *AccountsContainersVersionHeadersService) Latest(parent string) *AccountsContainersVersionHeadersLatestCall {
	c := &AccountsContainersVersionHeadersLatestCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersVersionHeadersLatestCall) Fields(s ...googleapi.Field) *AccountsContainersVersionHeadersLatestCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersVersionHeadersLatestCall) IfNoneMatch(entityTag string) *AccountsContainersVersionHeadersLatestCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersVersionHeadersLatestCall) Context(ctx context.Context) *AccountsContainersVersionHeadersLatestCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersVersionHeadersLatestCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersVersionHeadersLatestCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/version_headers:latest")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.version_headers.latest" call.
// Exactly one of *ContainerVersionHeader or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ContainerVersionHeader.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersVersionHeadersLatestCall) Do(opts ...googleapi.CallOption) (*ContainerVersionHeader, error) {
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
	ret := &ContainerVersionHeader{
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
	//   "description": "Gets the latest container version header",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.version_headers.latest",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM Container's API relative path. Example: accounts/{account_id}/containers/{container_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/version_headers:latest",
	//   "response": {
	//     "$ref": "ContainerVersionHeader"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.version_headers.list":

type AccountsContainersVersionHeadersListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all Container Versions of a GTM Container.
func (r *AccountsContainersVersionHeadersService) List(parent string) *AccountsContainersVersionHeadersListCall {
	c := &AccountsContainersVersionHeadersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// IncludeDeleted sets the optional parameter "includeDeleted": Also
// retrieve deleted (archived) versions when true.
func (c *AccountsContainersVersionHeadersListCall) IncludeDeleted(includeDeleted bool) *AccountsContainersVersionHeadersListCall {
	c.urlParams_.Set("includeDeleted", fmt.Sprint(includeDeleted))
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsContainersVersionHeadersListCall) PageToken(pageToken string) *AccountsContainersVersionHeadersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersVersionHeadersListCall) Fields(s ...googleapi.Field) *AccountsContainersVersionHeadersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersVersionHeadersListCall) IfNoneMatch(entityTag string) *AccountsContainersVersionHeadersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersVersionHeadersListCall) Context(ctx context.Context) *AccountsContainersVersionHeadersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersVersionHeadersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersVersionHeadersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/version_headers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.version_headers.list" call.
// Exactly one of *ListContainerVersionsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListContainerVersionsResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersVersionHeadersListCall) Do(opts ...googleapi.CallOption) (*ListContainerVersionsResponse, error) {
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
	ret := &ListContainerVersionsResponse{
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
	//   "description": "Lists all Container Versions of a GTM Container.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.version_headers.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "includeDeleted": {
	//       "description": "Also retrieve deleted (archived) versions when true.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "GTM Container's API relative path. Example: accounts/{account_id}/containers/{container_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/version_headers",
	//   "response": {
	//     "$ref": "ListContainerVersionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.edit.containerversions",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsContainersVersionHeadersListCall) Pages(ctx context.Context, f func(*ListContainerVersionsResponse) error) error {
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

// method id "tagmanager.accounts.containers.versions.delete":

type AccountsContainersVersionsDeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a Container Version.
func (r *AccountsContainersVersionsService) Delete(path string) *AccountsContainersVersionsDeleteCall {
	c := &AccountsContainersVersionsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersVersionsDeleteCall) Fields(s ...googleapi.Field) *AccountsContainersVersionsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersVersionsDeleteCall) Context(ctx context.Context) *AccountsContainersVersionsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersVersionsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersVersionsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.versions.delete" call.
func (c *AccountsContainersVersionsDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes a Container Version.",
	//   "httpMethod": "DELETE",
	//   "id": "tagmanager.accounts.containers.versions.delete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM ContainerVersion's API relative path. Example: accounts/{account_id}/containers/{container_id}/versions/{version_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containerversions"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.versions.get":

type AccountsContainersVersionsGetCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a Container Version.
func (r *AccountsContainersVersionsService) Get(path string) *AccountsContainersVersionsGetCall {
	c := &AccountsContainersVersionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// ContainerVersionId sets the optional parameter "containerVersionId":
// The GTM ContainerVersion ID. Specify published to retrieve the
// currently published version.
func (c *AccountsContainersVersionsGetCall) ContainerVersionId(containerVersionId string) *AccountsContainersVersionsGetCall {
	c.urlParams_.Set("containerVersionId", containerVersionId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersVersionsGetCall) Fields(s ...googleapi.Field) *AccountsContainersVersionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersVersionsGetCall) IfNoneMatch(entityTag string) *AccountsContainersVersionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersVersionsGetCall) Context(ctx context.Context) *AccountsContainersVersionsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersVersionsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersVersionsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.versions.get" call.
// Exactly one of *ContainerVersion or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ContainerVersion.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersVersionsGetCall) Do(opts ...googleapi.CallOption) (*ContainerVersion, error) {
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
	ret := &ContainerVersion{
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
	//   "description": "Gets a Container Version.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.versions.get",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "containerVersionId": {
	//       "description": "The GTM ContainerVersion ID. Specify published to retrieve the currently published version.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM ContainerVersion's API relative path. Example: accounts/{account_id}/containers/{container_id}/versions/{version_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "response": {
	//     "$ref": "ContainerVersion"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.edit.containerversions",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.versions.live":

type AccountsContainersVersionsLiveCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Live: Gets the live (i.e. published) container version
func (r *AccountsContainersVersionsService) Live(parent string) *AccountsContainersVersionsLiveCall {
	c := &AccountsContainersVersionsLiveCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersVersionsLiveCall) Fields(s ...googleapi.Field) *AccountsContainersVersionsLiveCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersVersionsLiveCall) IfNoneMatch(entityTag string) *AccountsContainersVersionsLiveCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersVersionsLiveCall) Context(ctx context.Context) *AccountsContainersVersionsLiveCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersVersionsLiveCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersVersionsLiveCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/versions:live")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.versions.live" call.
// Exactly one of *ContainerVersion or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ContainerVersion.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersVersionsLiveCall) Do(opts ...googleapi.CallOption) (*ContainerVersion, error) {
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
	ret := &ContainerVersion{
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
	//   "description": "Gets the live (i.e. published) container version",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.versions.live",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM Container's API relative path. Example: accounts/{account_id}/containers/{container_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/versions:live",
	//   "response": {
	//     "$ref": "ContainerVersion"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.versions.publish":

type AccountsContainersVersionsPublishCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Publish: Publishes a Container Version.
func (r *AccountsContainersVersionsService) Publish(path string) *AccountsContainersVersionsPublishCall {
	c := &AccountsContainersVersionsPublishCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the container version
// in storage.
func (c *AccountsContainersVersionsPublishCall) Fingerprint(fingerprint string) *AccountsContainersVersionsPublishCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersVersionsPublishCall) Fields(s ...googleapi.Field) *AccountsContainersVersionsPublishCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersVersionsPublishCall) Context(ctx context.Context) *AccountsContainersVersionsPublishCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersVersionsPublishCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersVersionsPublishCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:publish")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.versions.publish" call.
// Exactly one of *PublishContainerVersionResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *PublishContainerVersionResponse.ServerResponse.Header or (if
// a response was returned at all) in error.(*googleapi.Error).Header.
// Use googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersVersionsPublishCall) Do(opts ...googleapi.CallOption) (*PublishContainerVersionResponse, error) {
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
	ret := &PublishContainerVersionResponse{
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
	//   "description": "Publishes a Container Version.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.versions.publish",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the container version in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM ContainerVersion's API relative path. Example: accounts/{account_id}/containers/{container_id}/versions/{version_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:publish",
	//   "response": {
	//     "$ref": "PublishContainerVersionResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.publish"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.versions.set_latest":

type AccountsContainersVersionsSetLatestCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// SetLatest: Sets the latest version used for synchronization of
// workspaces when detecting conflicts and errors.
func (r *AccountsContainersVersionsService) SetLatest(path string) *AccountsContainersVersionsSetLatestCall {
	c := &AccountsContainersVersionsSetLatestCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersVersionsSetLatestCall) Fields(s ...googleapi.Field) *AccountsContainersVersionsSetLatestCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersVersionsSetLatestCall) Context(ctx context.Context) *AccountsContainersVersionsSetLatestCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersVersionsSetLatestCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersVersionsSetLatestCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:set_latest")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.versions.set_latest" call.
// Exactly one of *ContainerVersion or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ContainerVersion.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersVersionsSetLatestCall) Do(opts ...googleapi.CallOption) (*ContainerVersion, error) {
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
	ret := &ContainerVersion{
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
	//   "description": "Sets the latest version used for synchronization of workspaces when detecting conflicts and errors.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.versions.set_latest",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM ContainerVersion's API relative path. Example: accounts/{account_id}/containers/{container_id}/versions/{version_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:set_latest",
	//   "response": {
	//     "$ref": "ContainerVersion"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.versions.undelete":

type AccountsContainersVersionsUndeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Undelete: Undeletes a Container Version.
func (r *AccountsContainersVersionsService) Undelete(path string) *AccountsContainersVersionsUndeleteCall {
	c := &AccountsContainersVersionsUndeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersVersionsUndeleteCall) Fields(s ...googleapi.Field) *AccountsContainersVersionsUndeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersVersionsUndeleteCall) Context(ctx context.Context) *AccountsContainersVersionsUndeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersVersionsUndeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersVersionsUndeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:undelete")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.versions.undelete" call.
// Exactly one of *ContainerVersion or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ContainerVersion.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersVersionsUndeleteCall) Do(opts ...googleapi.CallOption) (*ContainerVersion, error) {
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
	ret := &ContainerVersion{
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
	//   "description": "Undeletes a Container Version.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.versions.undelete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM ContainerVersion's API relative path. Example: accounts/{account_id}/containers/{container_id}/versions/{version_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:undelete",
	//   "response": {
	//     "$ref": "ContainerVersion"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containerversions"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.versions.update":

type AccountsContainersVersionsUpdateCall struct {
	s                *Service
	path             string
	containerversion *ContainerVersion
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Update: Updates a Container Version.
func (r *AccountsContainersVersionsService) Update(path string, containerversion *ContainerVersion) *AccountsContainersVersionsUpdateCall {
	c := &AccountsContainersVersionsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.containerversion = containerversion
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the container version
// in storage.
func (c *AccountsContainersVersionsUpdateCall) Fingerprint(fingerprint string) *AccountsContainersVersionsUpdateCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersVersionsUpdateCall) Fields(s ...googleapi.Field) *AccountsContainersVersionsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersVersionsUpdateCall) Context(ctx context.Context) *AccountsContainersVersionsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersVersionsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersVersionsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.containerversion)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.versions.update" call.
// Exactly one of *ContainerVersion or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ContainerVersion.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersVersionsUpdateCall) Do(opts ...googleapi.CallOption) (*ContainerVersion, error) {
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
	ret := &ContainerVersion{
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
	//   "description": "Updates a Container Version.",
	//   "httpMethod": "PUT",
	//   "id": "tagmanager.accounts.containers.versions.update",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the container version in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM ContainerVersion's API relative path. Example: accounts/{account_id}/containers/{container_id}/versions/{version_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "ContainerVersion"
	//   },
	//   "response": {
	//     "$ref": "ContainerVersion"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containerversions"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.create":

type AccountsContainersWorkspacesCreateCall struct {
	s          *Service
	parent     string
	workspace  *Workspace
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a Workspace.
func (r *AccountsContainersWorkspacesService) Create(parent string, workspace *Workspace) *AccountsContainersWorkspacesCreateCall {
	c := &AccountsContainersWorkspacesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.workspace = workspace
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesCreateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesCreateCall) Context(ctx context.Context) *AccountsContainersWorkspacesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.workspace)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/workspaces")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.create" call.
// Exactly one of *Workspace or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Workspace.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesCreateCall) Do(opts ...googleapi.CallOption) (*Workspace, error) {
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
	ret := &Workspace{
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
	//   "description": "Creates a Workspace.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM parent Container's API relative path. Example: accounts/{account_id}/containers/{container_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/workspaces",
	//   "request": {
	//     "$ref": "Workspace"
	//   },
	//   "response": {
	//     "$ref": "Workspace"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.create_version":

type AccountsContainersWorkspacesCreateVersionCall struct {
	s                                           *Service
	path                                        string
	createcontainerversionrequestversionoptions *CreateContainerVersionRequestVersionOptions
	urlParams_                                  gensupport.URLParams
	ctx_                                        context.Context
	header_                                     http.Header
}

// CreateVersion: Creates a Container Version from the entities present
// in the workspace, deletes the workspace, and sets the base container
// version to the newly created version.
func (r *AccountsContainersWorkspacesService) CreateVersion(path string, createcontainerversionrequestversionoptions *CreateContainerVersionRequestVersionOptions) *AccountsContainersWorkspacesCreateVersionCall {
	c := &AccountsContainersWorkspacesCreateVersionCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.createcontainerversionrequestversionoptions = createcontainerversionrequestversionoptions
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesCreateVersionCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesCreateVersionCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesCreateVersionCall) Context(ctx context.Context) *AccountsContainersWorkspacesCreateVersionCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesCreateVersionCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesCreateVersionCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createcontainerversionrequestversionoptions)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:create_version")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.create_version" call.
// Exactly one of *CreateContainerVersionResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *CreateContainerVersionResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesCreateVersionCall) Do(opts ...googleapi.CallOption) (*CreateContainerVersionResponse, error) {
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
	ret := &CreateContainerVersionResponse{
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
	//   "description": "Creates a Container Version from the entities present in the workspace, deletes the workspace, and sets the base container version to the newly created version.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.create_version",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:create_version",
	//   "request": {
	//     "$ref": "CreateContainerVersionRequestVersionOptions"
	//   },
	//   "response": {
	//     "$ref": "CreateContainerVersionResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containerversions"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.delete":

type AccountsContainersWorkspacesDeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a Workspace.
func (r *AccountsContainersWorkspacesService) Delete(path string) *AccountsContainersWorkspacesDeleteCall {
	c := &AccountsContainersWorkspacesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesDeleteCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesDeleteCall) Context(ctx context.Context) *AccountsContainersWorkspacesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.delete" call.
func (c *AccountsContainersWorkspacesDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes a Workspace.",
	//   "httpMethod": "DELETE",
	//   "id": "tagmanager.accounts.containers.workspaces.delete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.delete.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.get":

type AccountsContainersWorkspacesGetCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a Workspace.
func (r *AccountsContainersWorkspacesService) Get(path string) *AccountsContainersWorkspacesGetCall {
	c := &AccountsContainersWorkspacesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesGetCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesGetCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesGetCall) Context(ctx context.Context) *AccountsContainersWorkspacesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.get" call.
// Exactly one of *Workspace or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Workspace.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesGetCall) Do(opts ...googleapi.CallOption) (*Workspace, error) {
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
	ret := &Workspace{
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
	//   "description": "Gets a Workspace.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.get",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "response": {
	//     "$ref": "Workspace"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.getProposal":

type AccountsContainersWorkspacesGetProposalCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetProposal: Gets a GTM Workspace Proposal.
func (r *AccountsContainersWorkspacesService) GetProposal(path string) *AccountsContainersWorkspacesGetProposalCall {
	c := &AccountsContainersWorkspacesGetProposalCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesGetProposalCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesGetProposalCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesGetProposalCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesGetProposalCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesGetProposalCall) Context(ctx context.Context) *AccountsContainersWorkspacesGetProposalCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesGetProposalCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesGetProposalCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.getProposal" call.
// Exactly one of *WorkspaceProposal or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *WorkspaceProposal.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesGetProposalCall) Do(opts ...googleapi.CallOption) (*WorkspaceProposal, error) {
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
	ret := &WorkspaceProposal{
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
	//   "description": "Gets a GTM Workspace Proposal.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.getProposal",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM workspace proposal's relative path: Example: accounts/{aid}/containers/{cid}/workspace/{wid}/workspace_proposal",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "response": {
	//     "$ref": "WorkspaceProposal"
	//   }
	// }

}

// method id "tagmanager.accounts.containers.workspaces.getStatus":

type AccountsContainersWorkspacesGetStatusCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetStatus: Finds conflicting and modified entities in the workspace.
func (r *AccountsContainersWorkspacesService) GetStatus(path string) *AccountsContainersWorkspacesGetStatusCall {
	c := &AccountsContainersWorkspacesGetStatusCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesGetStatusCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesGetStatusCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesGetStatusCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesGetStatusCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesGetStatusCall) Context(ctx context.Context) *AccountsContainersWorkspacesGetStatusCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesGetStatusCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesGetStatusCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}/status")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.getStatus" call.
// Exactly one of *GetWorkspaceStatusResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GetWorkspaceStatusResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesGetStatusCall) Do(opts ...googleapi.CallOption) (*GetWorkspaceStatusResponse, error) {
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
	ret := &GetWorkspaceStatusResponse{
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
	//   "description": "Finds conflicting and modified entities in the workspace.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.getStatus",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}/status",
	//   "response": {
	//     "$ref": "GetWorkspaceStatusResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.list":

type AccountsContainersWorkspacesListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all Workspaces that belong to a GTM Container.
func (r *AccountsContainersWorkspacesService) List(parent string) *AccountsContainersWorkspacesListCall {
	c := &AccountsContainersWorkspacesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsContainersWorkspacesListCall) PageToken(pageToken string) *AccountsContainersWorkspacesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesListCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesListCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesListCall) Context(ctx context.Context) *AccountsContainersWorkspacesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/workspaces")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.list" call.
// Exactly one of *ListWorkspacesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListWorkspacesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesListCall) Do(opts ...googleapi.CallOption) (*ListWorkspacesResponse, error) {
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
	ret := &ListWorkspacesResponse{
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
	//   "description": "Lists all Workspaces that belong to a GTM Container.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "GTM parent Container's API relative path. Example: accounts/{account_id}/containers/{container_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/workspaces",
	//   "response": {
	//     "$ref": "ListWorkspacesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsContainersWorkspacesListCall) Pages(ctx context.Context, f func(*ListWorkspacesResponse) error) error {
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

// method id "tagmanager.accounts.containers.workspaces.quick_preview":

type AccountsContainersWorkspacesQuickPreviewCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// QuickPreview: Quick previews a workspace by creating a fake container
// version from all entities in the provided workspace.
func (r *AccountsContainersWorkspacesService) QuickPreview(path string) *AccountsContainersWorkspacesQuickPreviewCall {
	c := &AccountsContainersWorkspacesQuickPreviewCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesQuickPreviewCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesQuickPreviewCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesQuickPreviewCall) Context(ctx context.Context) *AccountsContainersWorkspacesQuickPreviewCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesQuickPreviewCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesQuickPreviewCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:quick_preview")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.quick_preview" call.
// Exactly one of *QuickPreviewResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *QuickPreviewResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesQuickPreviewCall) Do(opts ...googleapi.CallOption) (*QuickPreviewResponse, error) {
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
	ret := &QuickPreviewResponse{
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
	//   "description": "Quick previews a workspace by creating a fake container version from all entities in the provided workspace.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.quick_preview",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:quick_preview",
	//   "response": {
	//     "$ref": "QuickPreviewResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containerversions"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.resolve_conflict":

type AccountsContainersWorkspacesResolveConflictCall struct {
	s          *Service
	path       string
	entity     *Entity
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// ResolveConflict: Resolves a merge conflict for a workspace entity by
// updating it to the resolved entity passed in the request.
func (r *AccountsContainersWorkspacesService) ResolveConflict(path string, entity *Entity) *AccountsContainersWorkspacesResolveConflictCall {
	c := &AccountsContainersWorkspacesResolveConflictCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.entity = entity
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the
// entity_in_workspace in the merge conflict.
func (c *AccountsContainersWorkspacesResolveConflictCall) Fingerprint(fingerprint string) *AccountsContainersWorkspacesResolveConflictCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesResolveConflictCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesResolveConflictCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesResolveConflictCall) Context(ctx context.Context) *AccountsContainersWorkspacesResolveConflictCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesResolveConflictCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesResolveConflictCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.entity)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:resolve_conflict")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.resolve_conflict" call.
func (c *AccountsContainersWorkspacesResolveConflictCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Resolves a merge conflict for a workspace entity by updating it to the resolved entity passed in the request.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.resolve_conflict",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the entity_in_workspace in the merge conflict.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:resolve_conflict",
	//   "request": {
	//     "$ref": "Entity"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.sync":

type AccountsContainersWorkspacesSyncCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Sync: Syncs a workspace to the latest container version by updating
// all unmodified workspace entities and displaying conflicts for
// modified entities.
func (r *AccountsContainersWorkspacesService) Sync(path string) *AccountsContainersWorkspacesSyncCall {
	c := &AccountsContainersWorkspacesSyncCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesSyncCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesSyncCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesSyncCall) Context(ctx context.Context) *AccountsContainersWorkspacesSyncCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesSyncCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesSyncCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:sync")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.sync" call.
// Exactly one of *SyncWorkspaceResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *SyncWorkspaceResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesSyncCall) Do(opts ...googleapi.CallOption) (*SyncWorkspaceResponse, error) {
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
	ret := &SyncWorkspaceResponse{
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
	//   "description": "Syncs a workspace to the latest container version by updating all unmodified workspace entities and displaying conflicts for modified entities.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.sync",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:sync",
	//   "response": {
	//     "$ref": "SyncWorkspaceResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.update":

type AccountsContainersWorkspacesUpdateCall struct {
	s          *Service
	path       string
	workspace  *Workspace
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a Workspace.
func (r *AccountsContainersWorkspacesService) Update(path string, workspace *Workspace) *AccountsContainersWorkspacesUpdateCall {
	c := &AccountsContainersWorkspacesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.workspace = workspace
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the workspace in
// storage.
func (c *AccountsContainersWorkspacesUpdateCall) Fingerprint(fingerprint string) *AccountsContainersWorkspacesUpdateCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesUpdateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesUpdateCall) Context(ctx context.Context) *AccountsContainersWorkspacesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.workspace)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.update" call.
// Exactly one of *Workspace or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Workspace.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesUpdateCall) Do(opts ...googleapi.CallOption) (*Workspace, error) {
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
	ret := &Workspace{
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
	//   "description": "Updates a Workspace.",
	//   "httpMethod": "PUT",
	//   "id": "tagmanager.accounts.containers.workspaces.update",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the workspace in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "Workspace"
	//   },
	//   "response": {
	//     "$ref": "Workspace"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.updateProposal":

type AccountsContainersWorkspacesUpdateProposalCall struct {
	s                              *Service
	path                           string
	updateworkspaceproposalrequest *UpdateWorkspaceProposalRequest
	urlParams_                     gensupport.URLParams
	ctx_                           context.Context
	header_                        http.Header
}

// UpdateProposal: Updates a GTM Workspace Proposal.
func (r *AccountsContainersWorkspacesService) UpdateProposal(path string, updateworkspaceproposalrequest *UpdateWorkspaceProposalRequest) *AccountsContainersWorkspacesUpdateProposalCall {
	c := &AccountsContainersWorkspacesUpdateProposalCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.updateworkspaceproposalrequest = updateworkspaceproposalrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesUpdateProposalCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesUpdateProposalCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesUpdateProposalCall) Context(ctx context.Context) *AccountsContainersWorkspacesUpdateProposalCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesUpdateProposalCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesUpdateProposalCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.updateworkspaceproposalrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.updateProposal" call.
// Exactly one of *WorkspaceProposal or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *WorkspaceProposal.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesUpdateProposalCall) Do(opts ...googleapi.CallOption) (*WorkspaceProposal, error) {
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
	ret := &WorkspaceProposal{
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
	//   "description": "Updates a GTM Workspace Proposal.",
	//   "httpMethod": "PUT",
	//   "id": "tagmanager.accounts.containers.workspaces.updateProposal",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM workspace proposal's relative path: Example: accounts/{aid}/containers/{cid}/workspace/{wid}/workspace_proposal",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "UpdateWorkspaceProposalRequest"
	//   },
	//   "response": {
	//     "$ref": "WorkspaceProposal"
	//   }
	// }

}

// method id "tagmanager.accounts.containers.workspaces.built_in_variables.create":

type AccountsContainersWorkspacesBuiltInVariablesCreateCall struct {
	s          *Service
	parent     string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates one or more GTM Built-In Variables.
func (r *AccountsContainersWorkspacesBuiltInVariablesService) Create(parent string) *AccountsContainersWorkspacesBuiltInVariablesCreateCall {
	c := &AccountsContainersWorkspacesBuiltInVariablesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// Type sets the optional parameter "type": The types of built-in
// variables to enable.
//
// Possible values:
//   "advertiserId"
//   "advertisingTrackingEnabled"
//   "ampBrowserLanguage"
//   "ampCanonicalHost"
//   "ampCanonicalPath"
//   "ampCanonicalUrl"
//   "ampClientId"
//   "ampClientMaxScrollX"
//   "ampClientMaxScrollY"
//   "ampClientScreenHeight"
//   "ampClientScreenWidth"
//   "ampClientScrollX"
//   "ampClientScrollY"
//   "ampClientTimestamp"
//   "ampClientTimezone"
//   "ampGtmEvent"
//   "ampPageDownloadTime"
//   "ampPageLoadTime"
//   "ampPageViewId"
//   "ampReferrer"
//   "ampTitle"
//   "ampTotalEngagedTime"
//   "appId"
//   "appName"
//   "appVersionCode"
//   "appVersionName"
//   "builtInVariableTypeUnspecified"
//   "clickClasses"
//   "clickElement"
//   "clickId"
//   "clickTarget"
//   "clickText"
//   "clickUrl"
//   "containerId"
//   "containerVersion"
//   "debugMode"
//   "deviceName"
//   "environmentName"
//   "errorLine"
//   "errorMessage"
//   "errorUrl"
//   "event"
//   "eventName"
//   "firebaseEventParameterCampaign"
//   "firebaseEventParameterCampaignAclid"
//   "firebaseEventParameterCampaignAnid"
//   "firebaseEventParameterCampaignClickTimestamp"
//   "firebaseEventParameterCampaignContent"
//   "firebaseEventParameterCampaignCp1"
//   "firebaseEventParameterCampaignGclid"
//   "firebaseEventParameterCampaignSource"
//   "firebaseEventParameterCampaignTerm"
//   "firebaseEventParameterCurrency"
//   "firebaseEventParameterDynamicLinkAcceptTime"
//   "firebaseEventParameterDynamicLinkLinkid"
//   "firebaseEventParameterNotificationMessageDeviceTime"
//   "firebaseEventParameterNotificationMessageId"
//   "firebaseEventParameterNotificationMessageName"
//   "firebaseEventParameterNotificationMessageTime"
//   "firebaseEventParameterNotificationTopic"
//   "firebaseEventParameterPreviousAppVersion"
//   "firebaseEventParameterPreviousOsVersion"
//   "firebaseEventParameterPrice"
//   "firebaseEventParameterProductId"
//   "firebaseEventParameterQuantity"
//   "firebaseEventParameterValue"
//   "formClasses"
//   "formElement"
//   "formId"
//   "formTarget"
//   "formText"
//   "formUrl"
//   "historySource"
//   "htmlId"
//   "language"
//   "newHistoryFragment"
//   "newHistoryState"
//   "oldHistoryFragment"
//   "oldHistoryState"
//   "osVersion"
//   "pageHostname"
//   "pagePath"
//   "pageUrl"
//   "platform"
//   "randomNumber"
//   "referrer"
//   "resolution"
//   "sdkVersion"
func (c *AccountsContainersWorkspacesBuiltInVariablesCreateCall) Type(type_ ...string) *AccountsContainersWorkspacesBuiltInVariablesCreateCall {
	c.urlParams_.SetMulti("type", append([]string{}, type_...))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesBuiltInVariablesCreateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesBuiltInVariablesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesBuiltInVariablesCreateCall) Context(ctx context.Context) *AccountsContainersWorkspacesBuiltInVariablesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesBuiltInVariablesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesBuiltInVariablesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/built_in_variables")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.built_in_variables.create" call.
// Exactly one of *CreateBuiltInVariableResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *CreateBuiltInVariableResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesBuiltInVariablesCreateCall) Do(opts ...googleapi.CallOption) (*CreateBuiltInVariableResponse, error) {
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
	ret := &CreateBuiltInVariableResponse{
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
	//   "description": "Creates one or more GTM Built-In Variables.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.built_in_variables.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "type": {
	//       "description": "The types of built-in variables to enable.",
	//       "enum": [
	//         "advertiserId",
	//         "advertisingTrackingEnabled",
	//         "ampBrowserLanguage",
	//         "ampCanonicalHost",
	//         "ampCanonicalPath",
	//         "ampCanonicalUrl",
	//         "ampClientId",
	//         "ampClientMaxScrollX",
	//         "ampClientMaxScrollY",
	//         "ampClientScreenHeight",
	//         "ampClientScreenWidth",
	//         "ampClientScrollX",
	//         "ampClientScrollY",
	//         "ampClientTimestamp",
	//         "ampClientTimezone",
	//         "ampGtmEvent",
	//         "ampPageDownloadTime",
	//         "ampPageLoadTime",
	//         "ampPageViewId",
	//         "ampReferrer",
	//         "ampTitle",
	//         "ampTotalEngagedTime",
	//         "appId",
	//         "appName",
	//         "appVersionCode",
	//         "appVersionName",
	//         "builtInVariableTypeUnspecified",
	//         "clickClasses",
	//         "clickElement",
	//         "clickId",
	//         "clickTarget",
	//         "clickText",
	//         "clickUrl",
	//         "containerId",
	//         "containerVersion",
	//         "debugMode",
	//         "deviceName",
	//         "environmentName",
	//         "errorLine",
	//         "errorMessage",
	//         "errorUrl",
	//         "event",
	//         "eventName",
	//         "firebaseEventParameterCampaign",
	//         "firebaseEventParameterCampaignAclid",
	//         "firebaseEventParameterCampaignAnid",
	//         "firebaseEventParameterCampaignClickTimestamp",
	//         "firebaseEventParameterCampaignContent",
	//         "firebaseEventParameterCampaignCp1",
	//         "firebaseEventParameterCampaignGclid",
	//         "firebaseEventParameterCampaignSource",
	//         "firebaseEventParameterCampaignTerm",
	//         "firebaseEventParameterCurrency",
	//         "firebaseEventParameterDynamicLinkAcceptTime",
	//         "firebaseEventParameterDynamicLinkLinkid",
	//         "firebaseEventParameterNotificationMessageDeviceTime",
	//         "firebaseEventParameterNotificationMessageId",
	//         "firebaseEventParameterNotificationMessageName",
	//         "firebaseEventParameterNotificationMessageTime",
	//         "firebaseEventParameterNotificationTopic",
	//         "firebaseEventParameterPreviousAppVersion",
	//         "firebaseEventParameterPreviousOsVersion",
	//         "firebaseEventParameterPrice",
	//         "firebaseEventParameterProductId",
	//         "firebaseEventParameterQuantity",
	//         "firebaseEventParameterValue",
	//         "formClasses",
	//         "formElement",
	//         "formId",
	//         "formTarget",
	//         "formText",
	//         "formUrl",
	//         "historySource",
	//         "htmlId",
	//         "language",
	//         "newHistoryFragment",
	//         "newHistoryState",
	//         "oldHistoryFragment",
	//         "oldHistoryState",
	//         "osVersion",
	//         "pageHostname",
	//         "pagePath",
	//         "pageUrl",
	//         "platform",
	//         "randomNumber",
	//         "referrer",
	//         "resolution",
	//         "sdkVersion"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/built_in_variables",
	//   "response": {
	//     "$ref": "CreateBuiltInVariableResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.built_in_variables.delete":

type AccountsContainersWorkspacesBuiltInVariablesDeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes one or more GTM Built-In Variables.
func (r *AccountsContainersWorkspacesBuiltInVariablesService) Delete(path string) *AccountsContainersWorkspacesBuiltInVariablesDeleteCall {
	c := &AccountsContainersWorkspacesBuiltInVariablesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Type sets the optional parameter "type": The types of built-in
// variables to delete.
//
// Possible values:
//   "advertiserId"
//   "advertisingTrackingEnabled"
//   "ampBrowserLanguage"
//   "ampCanonicalHost"
//   "ampCanonicalPath"
//   "ampCanonicalUrl"
//   "ampClientId"
//   "ampClientMaxScrollX"
//   "ampClientMaxScrollY"
//   "ampClientScreenHeight"
//   "ampClientScreenWidth"
//   "ampClientScrollX"
//   "ampClientScrollY"
//   "ampClientTimestamp"
//   "ampClientTimezone"
//   "ampGtmEvent"
//   "ampPageDownloadTime"
//   "ampPageLoadTime"
//   "ampPageViewId"
//   "ampReferrer"
//   "ampTitle"
//   "ampTotalEngagedTime"
//   "appId"
//   "appName"
//   "appVersionCode"
//   "appVersionName"
//   "builtInVariableTypeUnspecified"
//   "clickClasses"
//   "clickElement"
//   "clickId"
//   "clickTarget"
//   "clickText"
//   "clickUrl"
//   "containerId"
//   "containerVersion"
//   "debugMode"
//   "deviceName"
//   "environmentName"
//   "errorLine"
//   "errorMessage"
//   "errorUrl"
//   "event"
//   "eventName"
//   "firebaseEventParameterCampaign"
//   "firebaseEventParameterCampaignAclid"
//   "firebaseEventParameterCampaignAnid"
//   "firebaseEventParameterCampaignClickTimestamp"
//   "firebaseEventParameterCampaignContent"
//   "firebaseEventParameterCampaignCp1"
//   "firebaseEventParameterCampaignGclid"
//   "firebaseEventParameterCampaignSource"
//   "firebaseEventParameterCampaignTerm"
//   "firebaseEventParameterCurrency"
//   "firebaseEventParameterDynamicLinkAcceptTime"
//   "firebaseEventParameterDynamicLinkLinkid"
//   "firebaseEventParameterNotificationMessageDeviceTime"
//   "firebaseEventParameterNotificationMessageId"
//   "firebaseEventParameterNotificationMessageName"
//   "firebaseEventParameterNotificationMessageTime"
//   "firebaseEventParameterNotificationTopic"
//   "firebaseEventParameterPreviousAppVersion"
//   "firebaseEventParameterPreviousOsVersion"
//   "firebaseEventParameterPrice"
//   "firebaseEventParameterProductId"
//   "firebaseEventParameterQuantity"
//   "firebaseEventParameterValue"
//   "formClasses"
//   "formElement"
//   "formId"
//   "formTarget"
//   "formText"
//   "formUrl"
//   "historySource"
//   "htmlId"
//   "language"
//   "newHistoryFragment"
//   "newHistoryState"
//   "oldHistoryFragment"
//   "oldHistoryState"
//   "osVersion"
//   "pageHostname"
//   "pagePath"
//   "pageUrl"
//   "platform"
//   "randomNumber"
//   "referrer"
//   "resolution"
//   "sdkVersion"
func (c *AccountsContainersWorkspacesBuiltInVariablesDeleteCall) Type(type_ ...string) *AccountsContainersWorkspacesBuiltInVariablesDeleteCall {
	c.urlParams_.SetMulti("type", append([]string{}, type_...))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesBuiltInVariablesDeleteCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesBuiltInVariablesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesBuiltInVariablesDeleteCall) Context(ctx context.Context) *AccountsContainersWorkspacesBuiltInVariablesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesBuiltInVariablesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesBuiltInVariablesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.built_in_variables.delete" call.
func (c *AccountsContainersWorkspacesBuiltInVariablesDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes one or more GTM Built-In Variables.",
	//   "httpMethod": "DELETE",
	//   "id": "tagmanager.accounts.containers.workspaces.built_in_variables.delete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM BuiltInVariable's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/built_in_variables",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "type": {
	//       "description": "The types of built-in variables to delete.",
	//       "enum": [
	//         "advertiserId",
	//         "advertisingTrackingEnabled",
	//         "ampBrowserLanguage",
	//         "ampCanonicalHost",
	//         "ampCanonicalPath",
	//         "ampCanonicalUrl",
	//         "ampClientId",
	//         "ampClientMaxScrollX",
	//         "ampClientMaxScrollY",
	//         "ampClientScreenHeight",
	//         "ampClientScreenWidth",
	//         "ampClientScrollX",
	//         "ampClientScrollY",
	//         "ampClientTimestamp",
	//         "ampClientTimezone",
	//         "ampGtmEvent",
	//         "ampPageDownloadTime",
	//         "ampPageLoadTime",
	//         "ampPageViewId",
	//         "ampReferrer",
	//         "ampTitle",
	//         "ampTotalEngagedTime",
	//         "appId",
	//         "appName",
	//         "appVersionCode",
	//         "appVersionName",
	//         "builtInVariableTypeUnspecified",
	//         "clickClasses",
	//         "clickElement",
	//         "clickId",
	//         "clickTarget",
	//         "clickText",
	//         "clickUrl",
	//         "containerId",
	//         "containerVersion",
	//         "debugMode",
	//         "deviceName",
	//         "environmentName",
	//         "errorLine",
	//         "errorMessage",
	//         "errorUrl",
	//         "event",
	//         "eventName",
	//         "firebaseEventParameterCampaign",
	//         "firebaseEventParameterCampaignAclid",
	//         "firebaseEventParameterCampaignAnid",
	//         "firebaseEventParameterCampaignClickTimestamp",
	//         "firebaseEventParameterCampaignContent",
	//         "firebaseEventParameterCampaignCp1",
	//         "firebaseEventParameterCampaignGclid",
	//         "firebaseEventParameterCampaignSource",
	//         "firebaseEventParameterCampaignTerm",
	//         "firebaseEventParameterCurrency",
	//         "firebaseEventParameterDynamicLinkAcceptTime",
	//         "firebaseEventParameterDynamicLinkLinkid",
	//         "firebaseEventParameterNotificationMessageDeviceTime",
	//         "firebaseEventParameterNotificationMessageId",
	//         "firebaseEventParameterNotificationMessageName",
	//         "firebaseEventParameterNotificationMessageTime",
	//         "firebaseEventParameterNotificationTopic",
	//         "firebaseEventParameterPreviousAppVersion",
	//         "firebaseEventParameterPreviousOsVersion",
	//         "firebaseEventParameterPrice",
	//         "firebaseEventParameterProductId",
	//         "firebaseEventParameterQuantity",
	//         "firebaseEventParameterValue",
	//         "formClasses",
	//         "formElement",
	//         "formId",
	//         "formTarget",
	//         "formText",
	//         "formUrl",
	//         "historySource",
	//         "htmlId",
	//         "language",
	//         "newHistoryFragment",
	//         "newHistoryState",
	//         "oldHistoryFragment",
	//         "oldHistoryState",
	//         "osVersion",
	//         "pageHostname",
	//         "pagePath",
	//         "pageUrl",
	//         "platform",
	//         "randomNumber",
	//         "referrer",
	//         "resolution",
	//         "sdkVersion"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.built_in_variables.list":

type AccountsContainersWorkspacesBuiltInVariablesListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all the enabled Built-In Variables of a GTM Container.
func (r *AccountsContainersWorkspacesBuiltInVariablesService) List(parent string) *AccountsContainersWorkspacesBuiltInVariablesListCall {
	c := &AccountsContainersWorkspacesBuiltInVariablesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsContainersWorkspacesBuiltInVariablesListCall) PageToken(pageToken string) *AccountsContainersWorkspacesBuiltInVariablesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesBuiltInVariablesListCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesBuiltInVariablesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesBuiltInVariablesListCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesBuiltInVariablesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesBuiltInVariablesListCall) Context(ctx context.Context) *AccountsContainersWorkspacesBuiltInVariablesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesBuiltInVariablesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesBuiltInVariablesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/built_in_variables")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.built_in_variables.list" call.
// Exactly one of *ListEnabledBuiltInVariablesResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListEnabledBuiltInVariablesResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountsContainersWorkspacesBuiltInVariablesListCall) Do(opts ...googleapi.CallOption) (*ListEnabledBuiltInVariablesResponse, error) {
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
	ret := &ListEnabledBuiltInVariablesResponse{
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
	//   "description": "Lists all the enabled Built-In Variables of a GTM Container.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.built_in_variables.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/built_in_variables",
	//   "response": {
	//     "$ref": "ListEnabledBuiltInVariablesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsContainersWorkspacesBuiltInVariablesListCall) Pages(ctx context.Context, f func(*ListEnabledBuiltInVariablesResponse) error) error {
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

// method id "tagmanager.accounts.containers.workspaces.built_in_variables.revert":

type AccountsContainersWorkspacesBuiltInVariablesRevertCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Revert: Reverts changes to a GTM Built-In Variables in a GTM
// Workspace.
func (r *AccountsContainersWorkspacesBuiltInVariablesService) Revert(path string) *AccountsContainersWorkspacesBuiltInVariablesRevertCall {
	c := &AccountsContainersWorkspacesBuiltInVariablesRevertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Type sets the optional parameter "type": The type of built-in
// variable to revert.
//
// Possible values:
//   "advertiserId"
//   "advertisingTrackingEnabled"
//   "ampBrowserLanguage"
//   "ampCanonicalHost"
//   "ampCanonicalPath"
//   "ampCanonicalUrl"
//   "ampClientId"
//   "ampClientMaxScrollX"
//   "ampClientMaxScrollY"
//   "ampClientScreenHeight"
//   "ampClientScreenWidth"
//   "ampClientScrollX"
//   "ampClientScrollY"
//   "ampClientTimestamp"
//   "ampClientTimezone"
//   "ampGtmEvent"
//   "ampPageDownloadTime"
//   "ampPageLoadTime"
//   "ampPageViewId"
//   "ampReferrer"
//   "ampTitle"
//   "ampTotalEngagedTime"
//   "appId"
//   "appName"
//   "appVersionCode"
//   "appVersionName"
//   "builtInVariableTypeUnspecified"
//   "clickClasses"
//   "clickElement"
//   "clickId"
//   "clickTarget"
//   "clickText"
//   "clickUrl"
//   "containerId"
//   "containerVersion"
//   "debugMode"
//   "deviceName"
//   "environmentName"
//   "errorLine"
//   "errorMessage"
//   "errorUrl"
//   "event"
//   "eventName"
//   "firebaseEventParameterCampaign"
//   "firebaseEventParameterCampaignAclid"
//   "firebaseEventParameterCampaignAnid"
//   "firebaseEventParameterCampaignClickTimestamp"
//   "firebaseEventParameterCampaignContent"
//   "firebaseEventParameterCampaignCp1"
//   "firebaseEventParameterCampaignGclid"
//   "firebaseEventParameterCampaignSource"
//   "firebaseEventParameterCampaignTerm"
//   "firebaseEventParameterCurrency"
//   "firebaseEventParameterDynamicLinkAcceptTime"
//   "firebaseEventParameterDynamicLinkLinkid"
//   "firebaseEventParameterNotificationMessageDeviceTime"
//   "firebaseEventParameterNotificationMessageId"
//   "firebaseEventParameterNotificationMessageName"
//   "firebaseEventParameterNotificationMessageTime"
//   "firebaseEventParameterNotificationTopic"
//   "firebaseEventParameterPreviousAppVersion"
//   "firebaseEventParameterPreviousOsVersion"
//   "firebaseEventParameterPrice"
//   "firebaseEventParameterProductId"
//   "firebaseEventParameterQuantity"
//   "firebaseEventParameterValue"
//   "formClasses"
//   "formElement"
//   "formId"
//   "formTarget"
//   "formText"
//   "formUrl"
//   "historySource"
//   "htmlId"
//   "language"
//   "newHistoryFragment"
//   "newHistoryState"
//   "oldHistoryFragment"
//   "oldHistoryState"
//   "osVersion"
//   "pageHostname"
//   "pagePath"
//   "pageUrl"
//   "platform"
//   "randomNumber"
//   "referrer"
//   "resolution"
//   "sdkVersion"
func (c *AccountsContainersWorkspacesBuiltInVariablesRevertCall) Type(type_ string) *AccountsContainersWorkspacesBuiltInVariablesRevertCall {
	c.urlParams_.Set("type", type_)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesBuiltInVariablesRevertCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesBuiltInVariablesRevertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesBuiltInVariablesRevertCall) Context(ctx context.Context) *AccountsContainersWorkspacesBuiltInVariablesRevertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesBuiltInVariablesRevertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesBuiltInVariablesRevertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}/built_in_variables:revert")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.built_in_variables.revert" call.
// Exactly one of *RevertBuiltInVariableResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *RevertBuiltInVariableResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesBuiltInVariablesRevertCall) Do(opts ...googleapi.CallOption) (*RevertBuiltInVariableResponse, error) {
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
	ret := &RevertBuiltInVariableResponse{
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
	//   "description": "Reverts changes to a GTM Built-In Variables in a GTM Workspace.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.built_in_variables.revert",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM BuiltInVariable's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/built_in_variables",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "type": {
	//       "description": "The type of built-in variable to revert.",
	//       "enum": [
	//         "advertiserId",
	//         "advertisingTrackingEnabled",
	//         "ampBrowserLanguage",
	//         "ampCanonicalHost",
	//         "ampCanonicalPath",
	//         "ampCanonicalUrl",
	//         "ampClientId",
	//         "ampClientMaxScrollX",
	//         "ampClientMaxScrollY",
	//         "ampClientScreenHeight",
	//         "ampClientScreenWidth",
	//         "ampClientScrollX",
	//         "ampClientScrollY",
	//         "ampClientTimestamp",
	//         "ampClientTimezone",
	//         "ampGtmEvent",
	//         "ampPageDownloadTime",
	//         "ampPageLoadTime",
	//         "ampPageViewId",
	//         "ampReferrer",
	//         "ampTitle",
	//         "ampTotalEngagedTime",
	//         "appId",
	//         "appName",
	//         "appVersionCode",
	//         "appVersionName",
	//         "builtInVariableTypeUnspecified",
	//         "clickClasses",
	//         "clickElement",
	//         "clickId",
	//         "clickTarget",
	//         "clickText",
	//         "clickUrl",
	//         "containerId",
	//         "containerVersion",
	//         "debugMode",
	//         "deviceName",
	//         "environmentName",
	//         "errorLine",
	//         "errorMessage",
	//         "errorUrl",
	//         "event",
	//         "eventName",
	//         "firebaseEventParameterCampaign",
	//         "firebaseEventParameterCampaignAclid",
	//         "firebaseEventParameterCampaignAnid",
	//         "firebaseEventParameterCampaignClickTimestamp",
	//         "firebaseEventParameterCampaignContent",
	//         "firebaseEventParameterCampaignCp1",
	//         "firebaseEventParameterCampaignGclid",
	//         "firebaseEventParameterCampaignSource",
	//         "firebaseEventParameterCampaignTerm",
	//         "firebaseEventParameterCurrency",
	//         "firebaseEventParameterDynamicLinkAcceptTime",
	//         "firebaseEventParameterDynamicLinkLinkid",
	//         "firebaseEventParameterNotificationMessageDeviceTime",
	//         "firebaseEventParameterNotificationMessageId",
	//         "firebaseEventParameterNotificationMessageName",
	//         "firebaseEventParameterNotificationMessageTime",
	//         "firebaseEventParameterNotificationTopic",
	//         "firebaseEventParameterPreviousAppVersion",
	//         "firebaseEventParameterPreviousOsVersion",
	//         "firebaseEventParameterPrice",
	//         "firebaseEventParameterProductId",
	//         "firebaseEventParameterQuantity",
	//         "firebaseEventParameterValue",
	//         "formClasses",
	//         "formElement",
	//         "formId",
	//         "formTarget",
	//         "formText",
	//         "formUrl",
	//         "historySource",
	//         "htmlId",
	//         "language",
	//         "newHistoryFragment",
	//         "newHistoryState",
	//         "oldHistoryFragment",
	//         "oldHistoryState",
	//         "osVersion",
	//         "pageHostname",
	//         "pagePath",
	//         "pageUrl",
	//         "platform",
	//         "randomNumber",
	//         "referrer",
	//         "resolution",
	//         "sdkVersion"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}/built_in_variables:revert",
	//   "response": {
	//     "$ref": "RevertBuiltInVariableResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.folders.create":

type AccountsContainersWorkspacesFoldersCreateCall struct {
	s          *Service
	parent     string
	folder     *Folder
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a GTM Folder.
func (r *AccountsContainersWorkspacesFoldersService) Create(parent string, folder *Folder) *AccountsContainersWorkspacesFoldersCreateCall {
	c := &AccountsContainersWorkspacesFoldersCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.folder = folder
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesFoldersCreateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesFoldersCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesFoldersCreateCall) Context(ctx context.Context) *AccountsContainersWorkspacesFoldersCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesFoldersCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesFoldersCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.folder)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/folders")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.folders.create" call.
// Exactly one of *Folder or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Folder.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsContainersWorkspacesFoldersCreateCall) Do(opts ...googleapi.CallOption) (*Folder, error) {
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
	ret := &Folder{
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
	//   "description": "Creates a GTM Folder.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.folders.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/folders",
	//   "request": {
	//     "$ref": "Folder"
	//   },
	//   "response": {
	//     "$ref": "Folder"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.folders.delete":

type AccountsContainersWorkspacesFoldersDeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a GTM Folder.
func (r *AccountsContainersWorkspacesFoldersService) Delete(path string) *AccountsContainersWorkspacesFoldersDeleteCall {
	c := &AccountsContainersWorkspacesFoldersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesFoldersDeleteCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesFoldersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesFoldersDeleteCall) Context(ctx context.Context) *AccountsContainersWorkspacesFoldersDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesFoldersDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesFoldersDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.folders.delete" call.
func (c *AccountsContainersWorkspacesFoldersDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes a GTM Folder.",
	//   "httpMethod": "DELETE",
	//   "id": "tagmanager.accounts.containers.workspaces.folders.delete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Folder's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/folders/{folder_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.folders.entities":

type AccountsContainersWorkspacesFoldersEntitiesCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Entities: List all entities in a GTM Folder.
func (r *AccountsContainersWorkspacesFoldersService) Entities(path string) *AccountsContainersWorkspacesFoldersEntitiesCall {
	c := &AccountsContainersWorkspacesFoldersEntitiesCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsContainersWorkspacesFoldersEntitiesCall) PageToken(pageToken string) *AccountsContainersWorkspacesFoldersEntitiesCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesFoldersEntitiesCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesFoldersEntitiesCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesFoldersEntitiesCall) Context(ctx context.Context) *AccountsContainersWorkspacesFoldersEntitiesCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesFoldersEntitiesCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesFoldersEntitiesCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:entities")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.folders.entities" call.
// Exactly one of *FolderEntities or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *FolderEntities.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesFoldersEntitiesCall) Do(opts ...googleapi.CallOption) (*FolderEntities, error) {
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
	ret := &FolderEntities{
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
	//   "description": "List all entities in a GTM Folder.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.folders.entities",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Folder's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/folders/{folder_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:entities",
	//   "response": {
	//     "$ref": "FolderEntities"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsContainersWorkspacesFoldersEntitiesCall) Pages(ctx context.Context, f func(*FolderEntities) error) error {
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

// method id "tagmanager.accounts.containers.workspaces.folders.get":

type AccountsContainersWorkspacesFoldersGetCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a GTM Folder.
func (r *AccountsContainersWorkspacesFoldersService) Get(path string) *AccountsContainersWorkspacesFoldersGetCall {
	c := &AccountsContainersWorkspacesFoldersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesFoldersGetCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesFoldersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesFoldersGetCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesFoldersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesFoldersGetCall) Context(ctx context.Context) *AccountsContainersWorkspacesFoldersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesFoldersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesFoldersGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.folders.get" call.
// Exactly one of *Folder or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Folder.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsContainersWorkspacesFoldersGetCall) Do(opts ...googleapi.CallOption) (*Folder, error) {
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
	ret := &Folder{
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
	//   "description": "Gets a GTM Folder.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.folders.get",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Folder's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/folders/{folder_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "response": {
	//     "$ref": "Folder"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.folders.list":

type AccountsContainersWorkspacesFoldersListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all GTM Folders of a Container.
func (r *AccountsContainersWorkspacesFoldersService) List(parent string) *AccountsContainersWorkspacesFoldersListCall {
	c := &AccountsContainersWorkspacesFoldersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsContainersWorkspacesFoldersListCall) PageToken(pageToken string) *AccountsContainersWorkspacesFoldersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesFoldersListCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesFoldersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesFoldersListCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesFoldersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesFoldersListCall) Context(ctx context.Context) *AccountsContainersWorkspacesFoldersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesFoldersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesFoldersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/folders")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.folders.list" call.
// Exactly one of *ListFoldersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListFoldersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesFoldersListCall) Do(opts ...googleapi.CallOption) (*ListFoldersResponse, error) {
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
	ret := &ListFoldersResponse{
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
	//   "description": "Lists all GTM Folders of a Container.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.folders.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/folders",
	//   "response": {
	//     "$ref": "ListFoldersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsContainersWorkspacesFoldersListCall) Pages(ctx context.Context, f func(*ListFoldersResponse) error) error {
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

// method id "tagmanager.accounts.containers.workspaces.folders.move_entities_to_folder":

type AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall struct {
	s          *Service
	path       string
	folder     *Folder
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// MoveEntitiesToFolder: Moves entities to a GTM Folder.
func (r *AccountsContainersWorkspacesFoldersService) MoveEntitiesToFolder(path string, folder *Folder) *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall {
	c := &AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.folder = folder
	return c
}

// TagId sets the optional parameter "tagId": The tags to be moved to
// the folder.
func (c *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall) TagId(tagId ...string) *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall {
	c.urlParams_.SetMulti("tagId", append([]string{}, tagId...))
	return c
}

// TriggerId sets the optional parameter "triggerId": The triggers to be
// moved to the folder.
func (c *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall) TriggerId(triggerId ...string) *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall {
	c.urlParams_.SetMulti("triggerId", append([]string{}, triggerId...))
	return c
}

// VariableId sets the optional parameter "variableId": The variables to
// be moved to the folder.
func (c *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall) VariableId(variableId ...string) *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall {
	c.urlParams_.SetMulti("variableId", append([]string{}, variableId...))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall) Context(ctx context.Context) *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.folder)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:move_entities_to_folder")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.folders.move_entities_to_folder" call.
func (c *AccountsContainersWorkspacesFoldersMoveEntitiesToFolderCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Moves entities to a GTM Folder.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.folders.move_entities_to_folder",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Folder's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/folders/{folder_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tagId": {
	//       "description": "The tags to be moved to the folder.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "triggerId": {
	//       "description": "The triggers to be moved to the folder.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "variableId": {
	//       "description": "The variables to be moved to the folder.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:move_entities_to_folder",
	//   "request": {
	//     "$ref": "Folder"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.folders.revert":

type AccountsContainersWorkspacesFoldersRevertCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Revert: Reverts changes to a GTM Folder in a GTM Workspace.
func (r *AccountsContainersWorkspacesFoldersService) Revert(path string) *AccountsContainersWorkspacesFoldersRevertCall {
	c := &AccountsContainersWorkspacesFoldersRevertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the tag in storage.
func (c *AccountsContainersWorkspacesFoldersRevertCall) Fingerprint(fingerprint string) *AccountsContainersWorkspacesFoldersRevertCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesFoldersRevertCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesFoldersRevertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesFoldersRevertCall) Context(ctx context.Context) *AccountsContainersWorkspacesFoldersRevertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesFoldersRevertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesFoldersRevertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:revert")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.folders.revert" call.
// Exactly one of *RevertFolderResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *RevertFolderResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesFoldersRevertCall) Do(opts ...googleapi.CallOption) (*RevertFolderResponse, error) {
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
	ret := &RevertFolderResponse{
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
	//   "description": "Reverts changes to a GTM Folder in a GTM Workspace.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.folders.revert",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the tag in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Folder's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/folders/{folder_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:revert",
	//   "response": {
	//     "$ref": "RevertFolderResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.folders.update":

type AccountsContainersWorkspacesFoldersUpdateCall struct {
	s          *Service
	path       string
	folder     *Folder
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a GTM Folder.
func (r *AccountsContainersWorkspacesFoldersService) Update(path string, folder *Folder) *AccountsContainersWorkspacesFoldersUpdateCall {
	c := &AccountsContainersWorkspacesFoldersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.folder = folder
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the folder in storage.
func (c *AccountsContainersWorkspacesFoldersUpdateCall) Fingerprint(fingerprint string) *AccountsContainersWorkspacesFoldersUpdateCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesFoldersUpdateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesFoldersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesFoldersUpdateCall) Context(ctx context.Context) *AccountsContainersWorkspacesFoldersUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesFoldersUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesFoldersUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.folder)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.folders.update" call.
// Exactly one of *Folder or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Folder.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsContainersWorkspacesFoldersUpdateCall) Do(opts ...googleapi.CallOption) (*Folder, error) {
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
	ret := &Folder{
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
	//   "description": "Updates a GTM Folder.",
	//   "httpMethod": "PUT",
	//   "id": "tagmanager.accounts.containers.workspaces.folders.update",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the folder in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Folder's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/folders/{folder_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "Folder"
	//   },
	//   "response": {
	//     "$ref": "Folder"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.proposal.create":

type AccountsContainersWorkspacesProposalCreateCall struct {
	s                              *Service
	parent                         string
	createworkspaceproposalrequest *CreateWorkspaceProposalRequest
	urlParams_                     gensupport.URLParams
	ctx_                           context.Context
	header_                        http.Header
}

// Create: Creates a GTM Workspace Proposal.
func (r *AccountsContainersWorkspacesProposalService) Create(parent string, createworkspaceproposalrequest *CreateWorkspaceProposalRequest) *AccountsContainersWorkspacesProposalCreateCall {
	c := &AccountsContainersWorkspacesProposalCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.createworkspaceproposalrequest = createworkspaceproposalrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesProposalCreateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesProposalCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesProposalCreateCall) Context(ctx context.Context) *AccountsContainersWorkspacesProposalCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesProposalCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesProposalCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createworkspaceproposalrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/proposal")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.proposal.create" call.
// Exactly one of *WorkspaceProposal or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *WorkspaceProposal.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesProposalCreateCall) Do(opts ...googleapi.CallOption) (*WorkspaceProposal, error) {
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
	ret := &WorkspaceProposal{
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
	//   "description": "Creates a GTM Workspace Proposal.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.proposal.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{aid}/containers/{cid}/workspace/{wid}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/proposal",
	//   "request": {
	//     "$ref": "CreateWorkspaceProposalRequest"
	//   },
	//   "response": {
	//     "$ref": "WorkspaceProposal"
	//   }
	// }

}

// method id "tagmanager.accounts.containers.workspaces.proposal.delete":

type AccountsContainersWorkspacesProposalDeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a GTM Workspace Proposal.
func (r *AccountsContainersWorkspacesProposalService) Delete(path string) *AccountsContainersWorkspacesProposalDeleteCall {
	c := &AccountsContainersWorkspacesProposalDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesProposalDeleteCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesProposalDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesProposalDeleteCall) Context(ctx context.Context) *AccountsContainersWorkspacesProposalDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesProposalDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesProposalDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.proposal.delete" call.
func (c *AccountsContainersWorkspacesProposalDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes a GTM Workspace Proposal.",
	//   "httpMethod": "DELETE",
	//   "id": "tagmanager.accounts.containers.workspaces.proposal.delete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM workspace proposal's relative path: Example: accounts/{aid}/containers/{cid}/workspace/{wid}/workspace_proposal",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}"
	// }

}

// method id "tagmanager.accounts.containers.workspaces.tags.create":

type AccountsContainersWorkspacesTagsCreateCall struct {
	s          *Service
	parent     string
	tag        *Tag
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a GTM Tag.
func (r *AccountsContainersWorkspacesTagsService) Create(parent string, tag *Tag) *AccountsContainersWorkspacesTagsCreateCall {
	c := &AccountsContainersWorkspacesTagsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.tag = tag
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTagsCreateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTagsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTagsCreateCall) Context(ctx context.Context) *AccountsContainersWorkspacesTagsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTagsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTagsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tag)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/tags")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.tags.create" call.
// Exactly one of *Tag or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Tag.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountsContainersWorkspacesTagsCreateCall) Do(opts ...googleapi.CallOption) (*Tag, error) {
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
	ret := &Tag{
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
	//   "description": "Creates a GTM Tag.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.tags.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/tags",
	//   "request": {
	//     "$ref": "Tag"
	//   },
	//   "response": {
	//     "$ref": "Tag"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.tags.delete":

type AccountsContainersWorkspacesTagsDeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a GTM Tag.
func (r *AccountsContainersWorkspacesTagsService) Delete(path string) *AccountsContainersWorkspacesTagsDeleteCall {
	c := &AccountsContainersWorkspacesTagsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTagsDeleteCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTagsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTagsDeleteCall) Context(ctx context.Context) *AccountsContainersWorkspacesTagsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTagsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTagsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.tags.delete" call.
func (c *AccountsContainersWorkspacesTagsDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes a GTM Tag.",
	//   "httpMethod": "DELETE",
	//   "id": "tagmanager.accounts.containers.workspaces.tags.delete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Tag's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/tags/{tag_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.tags.get":

type AccountsContainersWorkspacesTagsGetCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a GTM Tag.
func (r *AccountsContainersWorkspacesTagsService) Get(path string) *AccountsContainersWorkspacesTagsGetCall {
	c := &AccountsContainersWorkspacesTagsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTagsGetCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTagsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesTagsGetCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesTagsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTagsGetCall) Context(ctx context.Context) *AccountsContainersWorkspacesTagsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTagsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTagsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.tags.get" call.
// Exactly one of *Tag or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Tag.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountsContainersWorkspacesTagsGetCall) Do(opts ...googleapi.CallOption) (*Tag, error) {
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
	ret := &Tag{
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
	//   "description": "Gets a GTM Tag.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.tags.get",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Tag's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/tags/{tag_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "response": {
	//     "$ref": "Tag"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.tags.list":

type AccountsContainersWorkspacesTagsListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all GTM Tags of a Container.
func (r *AccountsContainersWorkspacesTagsService) List(parent string) *AccountsContainersWorkspacesTagsListCall {
	c := &AccountsContainersWorkspacesTagsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsContainersWorkspacesTagsListCall) PageToken(pageToken string) *AccountsContainersWorkspacesTagsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTagsListCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTagsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesTagsListCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesTagsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTagsListCall) Context(ctx context.Context) *AccountsContainersWorkspacesTagsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTagsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTagsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/tags")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.tags.list" call.
// Exactly one of *ListTagsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListTagsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesTagsListCall) Do(opts ...googleapi.CallOption) (*ListTagsResponse, error) {
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
	ret := &ListTagsResponse{
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
	//   "description": "Lists all GTM Tags of a Container.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.tags.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/tags",
	//   "response": {
	//     "$ref": "ListTagsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsContainersWorkspacesTagsListCall) Pages(ctx context.Context, f func(*ListTagsResponse) error) error {
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

// method id "tagmanager.accounts.containers.workspaces.tags.revert":

type AccountsContainersWorkspacesTagsRevertCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Revert: Reverts changes to a GTM Tag in a GTM Workspace.
func (r *AccountsContainersWorkspacesTagsService) Revert(path string) *AccountsContainersWorkspacesTagsRevertCall {
	c := &AccountsContainersWorkspacesTagsRevertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of thetag in storage.
func (c *AccountsContainersWorkspacesTagsRevertCall) Fingerprint(fingerprint string) *AccountsContainersWorkspacesTagsRevertCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTagsRevertCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTagsRevertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTagsRevertCall) Context(ctx context.Context) *AccountsContainersWorkspacesTagsRevertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTagsRevertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTagsRevertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:revert")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.tags.revert" call.
// Exactly one of *RevertTagResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *RevertTagResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesTagsRevertCall) Do(opts ...googleapi.CallOption) (*RevertTagResponse, error) {
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
	ret := &RevertTagResponse{
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
	//   "description": "Reverts changes to a GTM Tag in a GTM Workspace.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.tags.revert",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of thetag in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Tag's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/tags/{tag_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:revert",
	//   "response": {
	//     "$ref": "RevertTagResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.tags.update":

type AccountsContainersWorkspacesTagsUpdateCall struct {
	s          *Service
	path       string
	tag        *Tag
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a GTM Tag.
func (r *AccountsContainersWorkspacesTagsService) Update(path string, tag *Tag) *AccountsContainersWorkspacesTagsUpdateCall {
	c := &AccountsContainersWorkspacesTagsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.tag = tag
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the tag in storage.
func (c *AccountsContainersWorkspacesTagsUpdateCall) Fingerprint(fingerprint string) *AccountsContainersWorkspacesTagsUpdateCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTagsUpdateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTagsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTagsUpdateCall) Context(ctx context.Context) *AccountsContainersWorkspacesTagsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTagsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTagsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tag)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.tags.update" call.
// Exactly one of *Tag or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Tag.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *AccountsContainersWorkspacesTagsUpdateCall) Do(opts ...googleapi.CallOption) (*Tag, error) {
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
	ret := &Tag{
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
	//   "description": "Updates a GTM Tag.",
	//   "httpMethod": "PUT",
	//   "id": "tagmanager.accounts.containers.workspaces.tags.update",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the tag in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Tag's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/tags/{tag_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "Tag"
	//   },
	//   "response": {
	//     "$ref": "Tag"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.triggers.create":

type AccountsContainersWorkspacesTriggersCreateCall struct {
	s          *Service
	parent     string
	trigger    *Trigger
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a GTM Trigger.
func (r *AccountsContainersWorkspacesTriggersService) Create(parent string, trigger *Trigger) *AccountsContainersWorkspacesTriggersCreateCall {
	c := &AccountsContainersWorkspacesTriggersCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.trigger = trigger
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTriggersCreateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTriggersCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTriggersCreateCall) Context(ctx context.Context) *AccountsContainersWorkspacesTriggersCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTriggersCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTriggersCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.trigger)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/triggers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.triggers.create" call.
// Exactly one of *Trigger or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Trigger.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsContainersWorkspacesTriggersCreateCall) Do(opts ...googleapi.CallOption) (*Trigger, error) {
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
	ret := &Trigger{
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
	//   "description": "Creates a GTM Trigger.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.triggers.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM Workspaces's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/triggers",
	//   "request": {
	//     "$ref": "Trigger"
	//   },
	//   "response": {
	//     "$ref": "Trigger"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.triggers.delete":

type AccountsContainersWorkspacesTriggersDeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a GTM Trigger.
func (r *AccountsContainersWorkspacesTriggersService) Delete(path string) *AccountsContainersWorkspacesTriggersDeleteCall {
	c := &AccountsContainersWorkspacesTriggersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTriggersDeleteCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTriggersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTriggersDeleteCall) Context(ctx context.Context) *AccountsContainersWorkspacesTriggersDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTriggersDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTriggersDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.triggers.delete" call.
func (c *AccountsContainersWorkspacesTriggersDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes a GTM Trigger.",
	//   "httpMethod": "DELETE",
	//   "id": "tagmanager.accounts.containers.workspaces.triggers.delete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Trigger's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/triggers/{trigger_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.triggers.get":

type AccountsContainersWorkspacesTriggersGetCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a GTM Trigger.
func (r *AccountsContainersWorkspacesTriggersService) Get(path string) *AccountsContainersWorkspacesTriggersGetCall {
	c := &AccountsContainersWorkspacesTriggersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTriggersGetCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTriggersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesTriggersGetCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesTriggersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTriggersGetCall) Context(ctx context.Context) *AccountsContainersWorkspacesTriggersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTriggersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTriggersGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.triggers.get" call.
// Exactly one of *Trigger or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Trigger.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsContainersWorkspacesTriggersGetCall) Do(opts ...googleapi.CallOption) (*Trigger, error) {
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
	ret := &Trigger{
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
	//   "description": "Gets a GTM Trigger.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.triggers.get",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Trigger's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/triggers/{trigger_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "response": {
	//     "$ref": "Trigger"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.triggers.list":

type AccountsContainersWorkspacesTriggersListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all GTM Triggers of a Container.
func (r *AccountsContainersWorkspacesTriggersService) List(parent string) *AccountsContainersWorkspacesTriggersListCall {
	c := &AccountsContainersWorkspacesTriggersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsContainersWorkspacesTriggersListCall) PageToken(pageToken string) *AccountsContainersWorkspacesTriggersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTriggersListCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTriggersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesTriggersListCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesTriggersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTriggersListCall) Context(ctx context.Context) *AccountsContainersWorkspacesTriggersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTriggersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTriggersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/triggers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.triggers.list" call.
// Exactly one of *ListTriggersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListTriggersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesTriggersListCall) Do(opts ...googleapi.CallOption) (*ListTriggersResponse, error) {
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
	ret := &ListTriggersResponse{
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
	//   "description": "Lists all GTM Triggers of a Container.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.triggers.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "GTM Workspaces's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/triggers",
	//   "response": {
	//     "$ref": "ListTriggersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsContainersWorkspacesTriggersListCall) Pages(ctx context.Context, f func(*ListTriggersResponse) error) error {
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

// method id "tagmanager.accounts.containers.workspaces.triggers.revert":

type AccountsContainersWorkspacesTriggersRevertCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Revert: Reverts changes to a GTM Trigger in a GTM Workspace.
func (r *AccountsContainersWorkspacesTriggersService) Revert(path string) *AccountsContainersWorkspacesTriggersRevertCall {
	c := &AccountsContainersWorkspacesTriggersRevertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the trigger in
// storage.
func (c *AccountsContainersWorkspacesTriggersRevertCall) Fingerprint(fingerprint string) *AccountsContainersWorkspacesTriggersRevertCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTriggersRevertCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTriggersRevertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTriggersRevertCall) Context(ctx context.Context) *AccountsContainersWorkspacesTriggersRevertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTriggersRevertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTriggersRevertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:revert")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.triggers.revert" call.
// Exactly one of *RevertTriggerResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *RevertTriggerResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesTriggersRevertCall) Do(opts ...googleapi.CallOption) (*RevertTriggerResponse, error) {
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
	ret := &RevertTriggerResponse{
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
	//   "description": "Reverts changes to a GTM Trigger in a GTM Workspace.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.triggers.revert",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the trigger in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Trigger's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/triggers/{trigger_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:revert",
	//   "response": {
	//     "$ref": "RevertTriggerResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.triggers.update":

type AccountsContainersWorkspacesTriggersUpdateCall struct {
	s          *Service
	path       string
	trigger    *Trigger
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a GTM Trigger.
func (r *AccountsContainersWorkspacesTriggersService) Update(path string, trigger *Trigger) *AccountsContainersWorkspacesTriggersUpdateCall {
	c := &AccountsContainersWorkspacesTriggersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.trigger = trigger
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the trigger in
// storage.
func (c *AccountsContainersWorkspacesTriggersUpdateCall) Fingerprint(fingerprint string) *AccountsContainersWorkspacesTriggersUpdateCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesTriggersUpdateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesTriggersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesTriggersUpdateCall) Context(ctx context.Context) *AccountsContainersWorkspacesTriggersUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesTriggersUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesTriggersUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.trigger)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.triggers.update" call.
// Exactly one of *Trigger or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Trigger.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *AccountsContainersWorkspacesTriggersUpdateCall) Do(opts ...googleapi.CallOption) (*Trigger, error) {
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
	ret := &Trigger{
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
	//   "description": "Updates a GTM Trigger.",
	//   "httpMethod": "PUT",
	//   "id": "tagmanager.accounts.containers.workspaces.triggers.update",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the trigger in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Trigger's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/triggers/{trigger_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "Trigger"
	//   },
	//   "response": {
	//     "$ref": "Trigger"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.variables.create":

type AccountsContainersWorkspacesVariablesCreateCall struct {
	s          *Service
	parent     string
	variable   *Variable
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a GTM Variable.
func (r *AccountsContainersWorkspacesVariablesService) Create(parent string, variable *Variable) *AccountsContainersWorkspacesVariablesCreateCall {
	c := &AccountsContainersWorkspacesVariablesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.variable = variable
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesVariablesCreateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesVariablesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesVariablesCreateCall) Context(ctx context.Context) *AccountsContainersWorkspacesVariablesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesVariablesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesVariablesCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.variable)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/variables")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.variables.create" call.
// Exactly one of *Variable or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Variable.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesVariablesCreateCall) Do(opts ...googleapi.CallOption) (*Variable, error) {
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
	ret := &Variable{
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
	//   "description": "Creates a GTM Variable.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.variables.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/variables",
	//   "request": {
	//     "$ref": "Variable"
	//   },
	//   "response": {
	//     "$ref": "Variable"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.variables.delete":

type AccountsContainersWorkspacesVariablesDeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a GTM Variable.
func (r *AccountsContainersWorkspacesVariablesService) Delete(path string) *AccountsContainersWorkspacesVariablesDeleteCall {
	c := &AccountsContainersWorkspacesVariablesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesVariablesDeleteCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesVariablesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesVariablesDeleteCall) Context(ctx context.Context) *AccountsContainersWorkspacesVariablesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesVariablesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesVariablesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.variables.delete" call.
func (c *AccountsContainersWorkspacesVariablesDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Deletes a GTM Variable.",
	//   "httpMethod": "DELETE",
	//   "id": "tagmanager.accounts.containers.workspaces.variables.delete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Variable's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/variables/{variable_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.variables.get":

type AccountsContainersWorkspacesVariablesGetCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a GTM Variable.
func (r *AccountsContainersWorkspacesVariablesService) Get(path string) *AccountsContainersWorkspacesVariablesGetCall {
	c := &AccountsContainersWorkspacesVariablesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesVariablesGetCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesVariablesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesVariablesGetCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesVariablesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesVariablesGetCall) Context(ctx context.Context) *AccountsContainersWorkspacesVariablesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesVariablesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesVariablesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.variables.get" call.
// Exactly one of *Variable or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Variable.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesVariablesGetCall) Do(opts ...googleapi.CallOption) (*Variable, error) {
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
	ret := &Variable{
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
	//   "description": "Gets a GTM Variable.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.variables.get",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM Variable's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/variables/{variable_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "response": {
	//     "$ref": "Variable"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.variables.list":

type AccountsContainersWorkspacesVariablesListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all GTM Variables of a Container.
func (r *AccountsContainersWorkspacesVariablesService) List(parent string) *AccountsContainersWorkspacesVariablesListCall {
	c := &AccountsContainersWorkspacesVariablesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsContainersWorkspacesVariablesListCall) PageToken(pageToken string) *AccountsContainersWorkspacesVariablesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesVariablesListCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesVariablesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsContainersWorkspacesVariablesListCall) IfNoneMatch(entityTag string) *AccountsContainersWorkspacesVariablesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesVariablesListCall) Context(ctx context.Context) *AccountsContainersWorkspacesVariablesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesVariablesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesVariablesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/variables")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.variables.list" call.
// Exactly one of *ListVariablesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListVariablesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesVariablesListCall) Do(opts ...googleapi.CallOption) (*ListVariablesResponse, error) {
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
	ret := &ListVariablesResponse{
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
	//   "description": "Lists all GTM Variables of a Container.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.containers.workspaces.variables.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "GTM Workspace's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/variables",
	//   "response": {
	//     "$ref": "ListVariablesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers",
	//     "https://www.googleapis.com/auth/tagmanager.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsContainersWorkspacesVariablesListCall) Pages(ctx context.Context, f func(*ListVariablesResponse) error) error {
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

// method id "tagmanager.accounts.containers.workspaces.variables.revert":

type AccountsContainersWorkspacesVariablesRevertCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Revert: Reverts changes to a GTM Variable in a GTM Workspace.
func (r *AccountsContainersWorkspacesVariablesService) Revert(path string) *AccountsContainersWorkspacesVariablesRevertCall {
	c := &AccountsContainersWorkspacesVariablesRevertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the variable in
// storage.
func (c *AccountsContainersWorkspacesVariablesRevertCall) Fingerprint(fingerprint string) *AccountsContainersWorkspacesVariablesRevertCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesVariablesRevertCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesVariablesRevertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesVariablesRevertCall) Context(ctx context.Context) *AccountsContainersWorkspacesVariablesRevertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesVariablesRevertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesVariablesRevertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}:revert")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.variables.revert" call.
// Exactly one of *RevertVariableResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *RevertVariableResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesVariablesRevertCall) Do(opts ...googleapi.CallOption) (*RevertVariableResponse, error) {
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
	ret := &RevertVariableResponse{
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
	//   "description": "Reverts changes to a GTM Variable in a GTM Workspace.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.containers.workspaces.variables.revert",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the variable in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Variable's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/variables/{variable_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}:revert",
	//   "response": {
	//     "$ref": "RevertVariableResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.containers.workspaces.variables.update":

type AccountsContainersWorkspacesVariablesUpdateCall struct {
	s          *Service
	path       string
	variable   *Variable
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a GTM Variable.
func (r *AccountsContainersWorkspacesVariablesService) Update(path string, variable *Variable) *AccountsContainersWorkspacesVariablesUpdateCall {
	c := &AccountsContainersWorkspacesVariablesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.variable = variable
	return c
}

// Fingerprint sets the optional parameter "fingerprint": When provided,
// this fingerprint must match the fingerprint of the variable in
// storage.
func (c *AccountsContainersWorkspacesVariablesUpdateCall) Fingerprint(fingerprint string) *AccountsContainersWorkspacesVariablesUpdateCall {
	c.urlParams_.Set("fingerprint", fingerprint)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsContainersWorkspacesVariablesUpdateCall) Fields(s ...googleapi.Field) *AccountsContainersWorkspacesVariablesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsContainersWorkspacesVariablesUpdateCall) Context(ctx context.Context) *AccountsContainersWorkspacesVariablesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsContainersWorkspacesVariablesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsContainersWorkspacesVariablesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.variable)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.containers.workspaces.variables.update" call.
// Exactly one of *Variable or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Variable.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *AccountsContainersWorkspacesVariablesUpdateCall) Do(opts ...googleapi.CallOption) (*Variable, error) {
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
	ret := &Variable{
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
	//   "description": "Updates a GTM Variable.",
	//   "httpMethod": "PUT",
	//   "id": "tagmanager.accounts.containers.workspaces.variables.update",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "fingerprint": {
	//       "description": "When provided, this fingerprint must match the fingerprint of the variable in storage.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "path": {
	//       "description": "GTM Variable's API relative path. Example: accounts/{account_id}/containers/{container_id}/workspaces/{workspace_id}/variables/{variable_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "Variable"
	//   },
	//   "response": {
	//     "$ref": "Variable"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.edit.containers"
	//   ]
	// }

}

// method id "tagmanager.accounts.user_permissions.create":

type AccountsUserPermissionsCreateCall struct {
	s              *Service
	parent         string
	userpermission *UserPermission
	urlParams_     gensupport.URLParams
	ctx_           context.Context
	header_        http.Header
}

// Create: Creates a user's Account & Container access.
func (r *AccountsUserPermissionsService) Create(parent string, userpermission *UserPermission) *AccountsUserPermissionsCreateCall {
	c := &AccountsUserPermissionsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.userpermission = userpermission
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsUserPermissionsCreateCall) Fields(s ...googleapi.Field) *AccountsUserPermissionsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsUserPermissionsCreateCall) Context(ctx context.Context) *AccountsUserPermissionsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsUserPermissionsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsUserPermissionsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.userpermission)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/user_permissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.user_permissions.create" call.
// Exactly one of *UserPermission or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *UserPermission.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsUserPermissionsCreateCall) Do(opts ...googleapi.CallOption) (*UserPermission, error) {
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
	ret := &UserPermission{
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
	//   "description": "Creates a user's Account \u0026 Container access.",
	//   "httpMethod": "POST",
	//   "id": "tagmanager.accounts.user_permissions.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "GTM Account's API relative path. Example: accounts/{account_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/user_permissions",
	//   "request": {
	//     "$ref": "UserPermission"
	//   },
	//   "response": {
	//     "$ref": "UserPermission"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.manage.users"
	//   ]
	// }

}

// method id "tagmanager.accounts.user_permissions.delete":

type AccountsUserPermissionsDeleteCall struct {
	s          *Service
	path       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Removes a user from the account, revoking access to it and
// all of its containers.
func (r *AccountsUserPermissionsService) Delete(path string) *AccountsUserPermissionsDeleteCall {
	c := &AccountsUserPermissionsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsUserPermissionsDeleteCall) Fields(s ...googleapi.Field) *AccountsUserPermissionsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsUserPermissionsDeleteCall) Context(ctx context.Context) *AccountsUserPermissionsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsUserPermissionsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsUserPermissionsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.user_permissions.delete" call.
func (c *AccountsUserPermissionsDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	//   "description": "Removes a user from the account, revoking access to it and all of its containers.",
	//   "httpMethod": "DELETE",
	//   "id": "tagmanager.accounts.user_permissions.delete",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM UserPermission's API relative path. Example: accounts/{account_id}/user_permissions/{user_permission_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.manage.users"
	//   ]
	// }

}

// method id "tagmanager.accounts.user_permissions.get":

type AccountsUserPermissionsGetCall struct {
	s            *Service
	path         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a user's Account & Container access.
func (r *AccountsUserPermissionsService) Get(path string) *AccountsUserPermissionsGetCall {
	c := &AccountsUserPermissionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsUserPermissionsGetCall) Fields(s ...googleapi.Field) *AccountsUserPermissionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsUserPermissionsGetCall) IfNoneMatch(entityTag string) *AccountsUserPermissionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsUserPermissionsGetCall) Context(ctx context.Context) *AccountsUserPermissionsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsUserPermissionsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsUserPermissionsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.user_permissions.get" call.
// Exactly one of *UserPermission or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *UserPermission.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsUserPermissionsGetCall) Do(opts ...googleapi.CallOption) (*UserPermission, error) {
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
	ret := &UserPermission{
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
	//   "description": "Gets a user's Account \u0026 Container access.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.user_permissions.get",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM UserPermission's API relative path. Example: accounts/{account_id}/user_permissions/{user_permission_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "response": {
	//     "$ref": "UserPermission"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.manage.users"
	//   ]
	// }

}

// method id "tagmanager.accounts.user_permissions.list":

type AccountsUserPermissionsListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List all users that have access to the account along with
// Account and Container user access granted to each of them.
func (r *AccountsUserPermissionsService) List(parent string) *AccountsUserPermissionsListCall {
	c := &AccountsUserPermissionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// for fetching the next page of results.
func (c *AccountsUserPermissionsListCall) PageToken(pageToken string) *AccountsUserPermissionsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsUserPermissionsListCall) Fields(s ...googleapi.Field) *AccountsUserPermissionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *AccountsUserPermissionsListCall) IfNoneMatch(entityTag string) *AccountsUserPermissionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsUserPermissionsListCall) Context(ctx context.Context) *AccountsUserPermissionsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsUserPermissionsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsUserPermissionsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+parent}/user_permissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.user_permissions.list" call.
// Exactly one of *ListUserPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListUserPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsUserPermissionsListCall) Do(opts ...googleapi.CallOption) (*ListUserPermissionsResponse, error) {
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
	ret := &ListUserPermissionsResponse{
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
	//   "description": "List all users that have access to the account along with Account and Container user access granted to each of them.",
	//   "httpMethod": "GET",
	//   "id": "tagmanager.accounts.user_permissions.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageToken": {
	//       "description": "Continuation token for fetching the next page of results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "GTM Accounts's API relative path. Example: accounts/{account_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+parent}/user_permissions",
	//   "response": {
	//     "$ref": "ListUserPermissionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.manage.users"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *AccountsUserPermissionsListCall) Pages(ctx context.Context, f func(*ListUserPermissionsResponse) error) error {
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

// method id "tagmanager.accounts.user_permissions.update":

type AccountsUserPermissionsUpdateCall struct {
	s              *Service
	path           string
	userpermission *UserPermission
	urlParams_     gensupport.URLParams
	ctx_           context.Context
	header_        http.Header
}

// Update: Updates a user's Account & Container access.
func (r *AccountsUserPermissionsService) Update(path string, userpermission *UserPermission) *AccountsUserPermissionsUpdateCall {
	c := &AccountsUserPermissionsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.path = path
	c.userpermission = userpermission
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *AccountsUserPermissionsUpdateCall) Fields(s ...googleapi.Field) *AccountsUserPermissionsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *AccountsUserPermissionsUpdateCall) Context(ctx context.Context) *AccountsUserPermissionsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *AccountsUserPermissionsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *AccountsUserPermissionsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.userpermission)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{+path}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"path": c.path,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tagmanager.accounts.user_permissions.update" call.
// Exactly one of *UserPermission or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *UserPermission.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *AccountsUserPermissionsUpdateCall) Do(opts ...googleapi.CallOption) (*UserPermission, error) {
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
	ret := &UserPermission{
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
	//   "description": "Updates a user's Account \u0026 Container access.",
	//   "httpMethod": "PUT",
	//   "id": "tagmanager.accounts.user_permissions.update",
	//   "parameterOrder": [
	//     "path"
	//   ],
	//   "parameters": {
	//     "path": {
	//       "description": "GTM UserPermission's API relative path. Example: accounts/{account_id}/user_permissions/{user_permission_id}",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{+path}",
	//   "request": {
	//     "$ref": "UserPermission"
	//   },
	//   "response": {
	//     "$ref": "UserPermission"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tagmanager.manage.users"
	//   ]
	// }

}
