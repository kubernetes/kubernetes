// Package androidmanagement provides access to the Android Management API.
//
// See https://developers.google.com/android/management
//
// Usage example:
//
//   import "google.golang.org/api/androidmanagement/v1"
//   ...
//   androidmanagementService, err := androidmanagement.New(oauthHttpClient)
package androidmanagement // import "google.golang.org/api/androidmanagement/v1"

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

const apiId = "androidmanagement:v1"
const apiName = "androidmanagement"
const apiVersion = "v1"
const basePath = "https://androidmanagement.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// Manage Android devices and apps for your customers
	AndroidmanagementScope = "https://www.googleapis.com/auth/androidmanagement"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Enterprises = NewEnterprisesService(s)
	s.SignupUrls = NewSignupUrlsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Enterprises *EnterprisesService

	SignupUrls *SignupUrlsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewEnterprisesService(s *Service) *EnterprisesService {
	rs := &EnterprisesService{s: s}
	rs.Applications = NewEnterprisesApplicationsService(s)
	rs.Devices = NewEnterprisesDevicesService(s)
	rs.EnrollmentTokens = NewEnterprisesEnrollmentTokensService(s)
	rs.Policies = NewEnterprisesPoliciesService(s)
	rs.WebTokens = NewEnterprisesWebTokensService(s)
	return rs
}

type EnterprisesService struct {
	s *Service

	Applications *EnterprisesApplicationsService

	Devices *EnterprisesDevicesService

	EnrollmentTokens *EnterprisesEnrollmentTokensService

	Policies *EnterprisesPoliciesService

	WebTokens *EnterprisesWebTokensService
}

func NewEnterprisesApplicationsService(s *Service) *EnterprisesApplicationsService {
	rs := &EnterprisesApplicationsService{s: s}
	return rs
}

type EnterprisesApplicationsService struct {
	s *Service
}

func NewEnterprisesDevicesService(s *Service) *EnterprisesDevicesService {
	rs := &EnterprisesDevicesService{s: s}
	rs.Operations = NewEnterprisesDevicesOperationsService(s)
	return rs
}

type EnterprisesDevicesService struct {
	s *Service

	Operations *EnterprisesDevicesOperationsService
}

func NewEnterprisesDevicesOperationsService(s *Service) *EnterprisesDevicesOperationsService {
	rs := &EnterprisesDevicesOperationsService{s: s}
	return rs
}

type EnterprisesDevicesOperationsService struct {
	s *Service
}

func NewEnterprisesEnrollmentTokensService(s *Service) *EnterprisesEnrollmentTokensService {
	rs := &EnterprisesEnrollmentTokensService{s: s}
	return rs
}

type EnterprisesEnrollmentTokensService struct {
	s *Service
}

func NewEnterprisesPoliciesService(s *Service) *EnterprisesPoliciesService {
	rs := &EnterprisesPoliciesService{s: s}
	return rs
}

type EnterprisesPoliciesService struct {
	s *Service
}

func NewEnterprisesWebTokensService(s *Service) *EnterprisesWebTokensService {
	rs := &EnterprisesWebTokensService{s: s}
	return rs
}

type EnterprisesWebTokensService struct {
	s *Service
}

func NewSignupUrlsService(s *Service) *SignupUrlsService {
	rs := &SignupUrlsService{s: s}
	return rs
}

type SignupUrlsService struct {
	s *Service
}

// ApiLevelCondition: A compliance rule condition which is satisfied if
// the Android Framework API level on the device does not meet a minimum
// requirement. There can only be one rule with this type of condition
// per policy.
type ApiLevelCondition struct {
	// MinApiLevel: The minimum desired Android Framework API level. If the
	// device does not meet the minimum requirement, this condition is
	// satisfied. Must be greater than zero.
	MinApiLevel int64 `json:"minApiLevel,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MinApiLevel") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MinApiLevel") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ApiLevelCondition) MarshalJSON() ([]byte, error) {
	type noMethod ApiLevelCondition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Application: Application information.
type Application struct {
	// ManagedProperties: The set of managed properties available to be
	// pre-configured for the application.
	ManagedProperties []*ManagedProperty `json:"managedProperties,omitempty"`

	// Name: The name of the application in the form
	// enterprises/{enterpriseId}/applications/{package_name}
	Name string `json:"name,omitempty"`

	// Permissions: The permissions required by the app.
	Permissions []*ApplicationPermission `json:"permissions,omitempty"`

	// Title: The title of the application. Localized.
	Title string `json:"title,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ManagedProperties")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ManagedProperties") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Application) MarshalJSON() ([]byte, error) {
	type noMethod Application
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ApplicationPermission: Application permission.
type ApplicationPermission struct {
	// Description: A longer description of the permission, giving more
	// details of what it affects. Localized.
	Description string `json:"description,omitempty"`

	// Name: The name of the permission. Localized.
	Name string `json:"name,omitempty"`

	// PermissionId: An opaque string uniquely identifying the permission.
	// Not localized.
	PermissionId string `json:"permissionId,omitempty"`

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

func (s *ApplicationPermission) MarshalJSON() ([]byte, error) {
	type noMethod ApplicationPermission
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ApplicationPolicy: Policy for an individual app.
type ApplicationPolicy struct {
	// DefaultPermissionPolicy: The default policy for all permissions
	// requested by the app. If specified, this overrides the policy-level
	// default_permission_policy which applies to all apps.
	//
	// Possible values:
	//   "PERMISSION_POLICY_UNSPECIFIED" - Policy not specified. If no
	// policy is specified for a permission at any level, then the PROMPT
	// behavior is used by default.
	//   "PROMPT" - Prompt the user to grant a permission.
	//   "GRANT" - Automatically grant a permission.
	//   "DENY" - Automatically deny a permission.
	DefaultPermissionPolicy string `json:"defaultPermissionPolicy,omitempty"`

	// InstallType: The type of installation to perform.
	//
	// Possible values:
	//   "INSTALL_TYPE_UNSPECIFIED" - No automatic installation is
	// performed. Any other app policies will be applied if the user
	// installs the app.
	//   "PREINSTALLED" - The application is automatically installed and can
	// be removed by the user.
	//   "FORCE_INSTALLED" - The application is automatically installed and
	// cannot be removed by the user.
	InstallType string `json:"installType,omitempty"`

	// LockTaskAllowed: Whether the application is allowed to lock itself in
	// full-screen mode.
	LockTaskAllowed bool `json:"lockTaskAllowed,omitempty"`

	// ManagedConfiguration: Managed configuration applied to the app. The
	// format for the configuration is dictated by the ManagedProperty
	// values supported by the app. Each field name in the managed
	// configuration must match the key field of the ManagedProperty. The
	// field value must be compatible with the type of the ManagedProperty:
	// <table> <tr><td><i>type</i></td><td><i>JSON value</i></td></tr>
	// <tr><td>BOOL</td><td>true or false</td></tr>
	// <tr><td>STRING</td><td>string</td></tr>
	// <tr><td>INTEGER</td><td>number</td></tr>
	// <tr><td>CHOICE</td><td>string</td></tr>
	// <tr><td>MULTISELECT</td><td>array of strings</td></tr>
	// <tr><td>HIDDEN</td><td>string</td></tr>
	// <tr><td>BUNDLE_ARRAY</td><td>array of objects</td></tr> </table>
	ManagedConfiguration googleapi.RawMessage `json:"managedConfiguration,omitempty"`

	// PackageName: The package name of the app, e.g.
	// com.google.android.youtube for the YouTube app.
	PackageName string `json:"packageName,omitempty"`

	// PermissionGrants: Explicit permission grants or denials for the app.
	// These values override the default_permission_policy.
	PermissionGrants []*PermissionGrant `json:"permissionGrants,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "DefaultPermissionPolicy") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DefaultPermissionPolicy")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ApplicationPolicy) MarshalJSON() ([]byte, error) {
	type noMethod ApplicationPolicy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Command: A command.
type Command struct {
	// CreateTime: The timestamp at which the command was created. The
	// timestamp is automatically generated by the server.
	CreateTime string `json:"createTime,omitempty"`

	// Duration: The duration for which the command is valid. The command
	// will expire if not executed by the device during this time. The
	// default duration if unspecified is ten minutes. There is no maximum
	// duration.
	Duration string `json:"duration,omitempty"`

	// NewPassword: For commands of type RESET_PASSWORD, optionally
	// specifies the new password.
	NewPassword string `json:"newPassword,omitempty"`

	// ResetPasswordFlags: For commands of type RESET_PASSWORD, optionally
	// specifies flags.
	//
	// Possible values:
	//   "RESET_PASSWORD_FLAG_UNSPECIFIED" - This value is ignored.
	//   "REQUIRE_ENTRY" - Don't allow other admins to change the password
	// again until the user has entered it.
	//   "DO_NOT_ASK_CREDENTIALS_ON_BOOT" - Don't ask for user credentials
	// on device boot.
	ResetPasswordFlags []string `json:"resetPasswordFlags,omitempty"`

	// Type: The type of the command.
	//
	// Possible values:
	//   "COMMAND_TYPE_UNSPECIFIED" - This value is disallowed.
	//   "LOCK" - Lock the device, as if the lock screen timeout had
	// expired.
	//   "RESET_PASSWORD" - Reset the user's password.
	//   "REBOOT" - Reboot the device. Only supported on API level 24+.
	Type string `json:"type,omitempty"`

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

func (s *Command) MarshalJSON() ([]byte, error) {
	type noMethod Command
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ComplianceRule: A rule declaring which mitigating actions to take
// when a device is not compliant with its policy. For every rule, there
// is always an implicit mitigating action to set policy_compliant to
// false for the Device resource, and display a message on the device
// indicating that the device is not compliant with its policy. Other
// mitigating actions may optionally be taken as well, depending on the
// field values in the rule.
type ComplianceRule struct {
	// ApiLevelCondition: A condition which is satisfied if the Android
	// Framework API level on the device does not meet a minimum
	// requirement.
	ApiLevelCondition *ApiLevelCondition `json:"apiLevelCondition,omitempty"`

	// DisableApps: If set to true, the rule includes a mitigating action to
	// disable applications so that the device is effectively disabled, but
	// application data is preserved. If the device is running an app in
	// locked task mode, the app will be closed and a UI showing the reason
	// for non-compliance will be displayed.
	DisableApps bool `json:"disableApps,omitempty"`

	// NonComplianceDetailCondition: A condition which is satisfied if there
	// exists any matching NonComplianceDetail for the device.
	NonComplianceDetailCondition *NonComplianceDetailCondition `json:"nonComplianceDetailCondition,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ApiLevelCondition")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ApiLevelCondition") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ComplianceRule) MarshalJSON() ([]byte, error) {
	type noMethod ComplianceRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Device: A device owned by an enterprise. Unless otherwise noted, all
// fields are read-only and cannot be modified by an update device
// request.
type Device struct {
	// ApiLevel: The API level of the Android platform version running on
	// the device.
	ApiLevel int64 `json:"apiLevel,omitempty"`

	// AppliedPolicyName: The name of the policy that is currently applied
	// by the device.
	AppliedPolicyName string `json:"appliedPolicyName,omitempty"`

	// AppliedPolicyVersion: The version of the policy that is currently
	// applied by the device.
	AppliedPolicyVersion int64 `json:"appliedPolicyVersion,omitempty,string"`

	// AppliedState: The state that is currently applied by the device.
	//
	// Possible values:
	//   "DEVICE_STATE_UNSPECIFIED" - This value is disallowed.
	//   "ACTIVE" - The device is active.
	//   "DISABLED" - The device is disabled.
	//   "DELETED" - The device was deleted. This state will never be
	// returned by an API call, but will be used in the final policy
	// compliance report published to Cloud Pub/Sub when the device
	// acknowledges the deletion.
	//   "PROVISIONING" - The device is being provisioned. Newly enrolled
	// devices will be in this state until they have applied policy.
	AppliedState string `json:"appliedState,omitempty"`

	// DisabledReason: If the device state is DISABLED, an optional message
	// that is displayed on the device indicating the reason the device is
	// disabled. This field may be modified by an update request.
	DisabledReason *UserFacingMessage `json:"disabledReason,omitempty"`

	// Displays: Displays on the device. This information is only available
	// when displayInfoEnabled is true in the device's policy.
	Displays []*Display `json:"displays,omitempty"`

	// EnrollmentTime: The time of device enrollment.
	EnrollmentTime string `json:"enrollmentTime,omitempty"`

	// EnrollmentTokenData: If this device was enrolled with an enrollment
	// token with additional data provided, this field contains that data.
	EnrollmentTokenData string `json:"enrollmentTokenData,omitempty"`

	// EnrollmentTokenName: If this device was enrolled with an enrollment
	// token, this field contains the name of the token.
	EnrollmentTokenName string `json:"enrollmentTokenName,omitempty"`

	// HardwareInfo: Detailed information about the device hardware.
	HardwareInfo *HardwareInfo `json:"hardwareInfo,omitempty"`

	// HardwareStatusSamples: Hardware status samples in chronological
	// order. This information is only available when hardwareStatusEnabled
	// is true in the device's policy.
	HardwareStatusSamples []*HardwareStatus `json:"hardwareStatusSamples,omitempty"`

	// LastPolicyComplianceReportTime: The last time the device sent a
	// policy compliance report.
	LastPolicyComplianceReportTime string `json:"lastPolicyComplianceReportTime,omitempty"`

	// LastPolicySyncTime: The last time the device fetched its policy.
	LastPolicySyncTime string `json:"lastPolicySyncTime,omitempty"`

	// LastStatusReportTime: The last time the device sent a status report.
	LastStatusReportTime string `json:"lastStatusReportTime,omitempty"`

	// MemoryEvents: Events related to memory and storage measurements in
	// chronological order. This information is only available when
	// memoryInfoEnabled is true in the device's policy.
	MemoryEvents []*MemoryEvent `json:"memoryEvents,omitempty"`

	// MemoryInfo: Memory information. This information is only available
	// when memoryInfoEnabled is true in the device's policy.
	MemoryInfo *MemoryInfo `json:"memoryInfo,omitempty"`

	// Name: The name of the device in the form
	// enterprises/{enterpriseId}/devices/{deviceId}
	Name string `json:"name,omitempty"`

	// NetworkInfo: Device network information. This information is only
	// available when networkInfoEnabled is true in the device's policy.
	NetworkInfo *NetworkInfo `json:"networkInfo,omitempty"`

	// NonComplianceDetails: Details about policy settings for which the
	// device is not in compliance.
	NonComplianceDetails []*NonComplianceDetail `json:"nonComplianceDetails,omitempty"`

	// PolicyCompliant: Whether the device is compliant with its policy.
	PolicyCompliant bool `json:"policyCompliant,omitempty"`

	// PolicyName: The name of the policy that is intended to be applied to
	// the device. If empty, the policy with id default is applied. This
	// field may be modified by an update request. The name of the policy is
	// in the form enterprises/{enterpriseId}/policies/{policyId}. It is
	// also permissible to only specify the policyId when updating this
	// field as long as the policyId contains no slashes since the rest of
	// the policy name can be inferred from context.
	PolicyName string `json:"policyName,omitempty"`

	// PowerManagementEvents: Power management events on the device in
	// chronological order. This information is only available when
	// powerManagementEventsEnabled is true in the device's policy.
	PowerManagementEvents []*PowerManagementEvent `json:"powerManagementEvents,omitempty"`

	// PreviousDeviceNames: The previous device names used for the same
	// physical device when it has been enrolled multiple times. The serial
	// number is used as the unique identifier to determine if the same
	// physical device has enrolled previously. The names are in
	// chronological order.
	PreviousDeviceNames []string `json:"previousDeviceNames,omitempty"`

	// SoftwareInfo: Detailed information about the device software. This
	// information is only available when softwareInfoEnabled is true in the
	// device's policy.
	SoftwareInfo *SoftwareInfo `json:"softwareInfo,omitempty"`

	// State: The state that is intended to be applied to the device. This
	// field may be modified by an update request. Note that UpdateDevice
	// only handles toggling between ACTIVE and DISABLED states. Use the
	// delete device method to cause the device to enter the DELETED state.
	//
	// Possible values:
	//   "DEVICE_STATE_UNSPECIFIED" - This value is disallowed.
	//   "ACTIVE" - The device is active.
	//   "DISABLED" - The device is disabled.
	//   "DELETED" - The device was deleted. This state will never be
	// returned by an API call, but will be used in the final policy
	// compliance report published to Cloud Pub/Sub when the device
	// acknowledges the deletion.
	//   "PROVISIONING" - The device is being provisioned. Newly enrolled
	// devices will be in this state until they have applied policy.
	State string `json:"state,omitempty"`

	// UserName: The resource name of the user of the device in the form
	// enterprises/{enterpriseId}/users/{userId}. This is the name of the
	// device account automatically created for this device.
	UserName string `json:"userName,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ApiLevel") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ApiLevel") to include in
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

// Display: Device display information.
type Display struct {
	// Density: Display density expressed as dots-per-inch.
	Density int64 `json:"density,omitempty"`

	// DisplayId: Unique display id.
	DisplayId int64 `json:"displayId,omitempty"`

	// Height: Display height in pixels.
	Height int64 `json:"height,omitempty"`

	// Name: Name of the display.
	Name string `json:"name,omitempty"`

	// RefreshRate: Refresh rate of the display in frames per second.
	RefreshRate int64 `json:"refreshRate,omitempty"`

	// State: State of the display.
	//
	// Possible values:
	//   "DISPLAY_STATE_UNSPECIFIED" - This value is disallowed.
	//   "OFF" - Display is off.
	//   "ON" - Display is on.
	//   "DOZE" - Display is dozing in a low power state
	//   "SUSPENDED" - Display is dozing in a suspended low power state.
	State string `json:"state,omitempty"`

	// Width: Display width in pixels.
	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Density") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Density") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Display) MarshalJSON() ([]byte, error) {
	type noMethod Display
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Empty: A generic empty message that you can re-use to avoid defining
// duplicated empty messages in your APIs. A typical example is to use
// it as the request or the response type of an API method. For
// instance:
// service Foo {
//   rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty);
// }
// The JSON representation for Empty is empty JSON object {}.
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// EnrollmentToken: An enrollment token.
type EnrollmentToken struct {
	// AdditionalData: Optional, arbitrary data associated with the
	// enrollment token. This could contain, for example, the id of an org
	// unit to which the device is assigned after enrollment. After a device
	// enrolls with the token, this data will be exposed in the
	// enrollment_token_data field of the Device resource. The data must be
	// 1024 characters or less; otherwise, the creation request will fail.
	AdditionalData string `json:"additionalData,omitempty"`

	// Duration: The duration of the token. If not specified, the duration
	// will be 1 hour. The allowed range is 1 minute to 30 days.
	Duration string `json:"duration,omitempty"`

	// ExpirationTimestamp: The expiration time of the token. This is a
	// read-only field generated by the server.
	ExpirationTimestamp string `json:"expirationTimestamp,omitempty"`

	// Name: The name of the enrollment token, which is generated by the
	// server during creation, in the form
	// enterprises/{enterpriseId}/enrollmentTokens/{enrollmentTokenId}
	Name string `json:"name,omitempty"`

	// PolicyName: The name of the policy that will be initially applied to
	// the enrolled device in the form
	// enterprises/{enterpriseId}/policies/{policyId}. If not specified, the
	// policy with id default is applied. It is permissible to only specify
	// the policyId when updating this field as long as the policyId
	// contains no slashes since the rest of the policy name can be inferred
	// from context.
	PolicyName string `json:"policyName,omitempty"`

	// QrCode: A JSON string whose UTF-8 representation can be used to
	// generate a QR code to enroll a device with this enrollment token. To
	// enroll a device using NFC, the NFC record must contain a serialized
	// java.util.Properties representation of the properties in the JSON.
	QrCode string `json:"qrCode,omitempty"`

	// Value: The token value which is passed to the device and authorizes
	// the device to enroll. This is a read-only field generated by the
	// server.
	Value string `json:"value,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AdditionalData") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AdditionalData") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *EnrollmentToken) MarshalJSON() ([]byte, error) {
	type noMethod EnrollmentToken
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Enterprise: The configuration applied to an enterprise.
type Enterprise struct {
	// AppAutoApprovalEnabled: Whether app auto-approval is enabled. When
	// enabled, apps installed via policy for this enterprise have all
	// permissions automatically approved. When enabled, it is the caller's
	// responsibility to display the permissions required by an app to the
	// enterprise admin before setting the app to be installed in a policy.
	AppAutoApprovalEnabled bool `json:"appAutoApprovalEnabled,omitempty"`

	// EnabledNotificationTypes: The notification types to enable via Google
	// Cloud Pub/Sub.
	//
	// Possible values:
	//   "NOTIFICATION_TYPE_UNSPECIFIED" - This value is ignored.
	//   "ENROLLMENT" - A notification sent when a device enrolls.
	//   "COMPLIANCE_REPORT" - A notification sent when a device issues a
	// policy compliance report.
	//   "STATUS_REPORT" - A notification sent when a device issues a status
	// report.
	//   "COMMAND" - A notification sent when a device command has
	// completed.
	EnabledNotificationTypes []string `json:"enabledNotificationTypes,omitempty"`

	// EnterpriseDisplayName: The name of the enterprise as it will appear
	// to users.
	EnterpriseDisplayName string `json:"enterpriseDisplayName,omitempty"`

	// Logo: An image displayed as a logo during device provisioning.
	// Supported types are: image/bmp, image/gif, image/x-ico, image/jpeg,
	// image/png, image/webp, image/vnd.wap.wbmp, image/x-adobe-dng.
	Logo *ExternalData `json:"logo,omitempty"`

	// Name: The name of the enterprise which is generated by the server
	// during creation, in the form enterprises/{enterpriseId}
	Name string `json:"name,omitempty"`

	// PrimaryColor: A color in RGB format indicating the predominant color
	// to display in the device management app UI. The color components are
	// stored as follows: (red << 16) | (green << 8) | blue, where each
	// component may take a value between 0 and 255 inclusive.
	PrimaryColor int64 `json:"primaryColor,omitempty"`

	// PubsubTopic: When Cloud Pub/Sub notifications are enabled, this field
	// is required to indicate the topic to which the notifications will be
	// published. The format of this field is
	// projects/{project}/topics/{topic}. You must have granted the publish
	// permission on this topic to
	// android-cloud-policy@system.gserviceaccount.com
	PubsubTopic string `json:"pubsubTopic,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "AppAutoApprovalEnabled") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AppAutoApprovalEnabled")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Enterprise) MarshalJSON() ([]byte, error) {
	type noMethod Enterprise
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ExternalData: Data hosted at an external location. The data is to be
// downloaded by Android Device Policy and verified against the hash.
type ExternalData struct {
	// Sha256Hash: The base-64 encoded SHA-256 hash of the content hosted at
	// url. If the content does not match this hash, Android Device Policy
	// will not use the data.
	Sha256Hash string `json:"sha256Hash,omitempty"`

	// Url: The absolute URL to the data, which must use either the http or
	// https scheme. Android Device Policy does not provide any credentials
	// in the GET request, so the URL must be publicly accessible. Including
	// a long, random component in the URL may be used to prevent attackers
	// from discovering the URL.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Sha256Hash") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Sha256Hash") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ExternalData) MarshalJSON() ([]byte, error) {
	type noMethod ExternalData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// HardwareInfo: Information about device hardware. The fields related
// to temperature thresholds are only available when
// hardwareStatusEnabled is true in the device's policy.
type HardwareInfo struct {
	// BatteryShutdownTemperatures: Battery shutdown temperature thresholds
	// in Celsius for each battery on the device.
	BatteryShutdownTemperatures []float64 `json:"batteryShutdownTemperatures,omitempty"`

	// BatteryThrottlingTemperatures: Battery throttling temperature
	// thresholds in Celsius for each battery on the device.
	BatteryThrottlingTemperatures []float64 `json:"batteryThrottlingTemperatures,omitempty"`

	// Brand: Brand of the device, e.g. Google.
	Brand string `json:"brand,omitempty"`

	// CpuShutdownTemperatures: CPU shutdown temperature thresholds in
	// Celsius for each CPU on the device.
	CpuShutdownTemperatures []float64 `json:"cpuShutdownTemperatures,omitempty"`

	// CpuThrottlingTemperatures: CPU throttling temperature thresholds in
	// Celsius for each CPU on the device.
	CpuThrottlingTemperatures []float64 `json:"cpuThrottlingTemperatures,omitempty"`

	// DeviceBasebandVersion: Baseband version, e.g.
	// MDM9625_104662.22.05.34p.
	DeviceBasebandVersion string `json:"deviceBasebandVersion,omitempty"`

	// GpuShutdownTemperatures: GPU shutdown temperature thresholds in
	// Celsius for each GPU on the device.
	GpuShutdownTemperatures []float64 `json:"gpuShutdownTemperatures,omitempty"`

	// GpuThrottlingTemperatures: GPU throttling temperature thresholds in
	// Celsius for each GPU on the device.
	GpuThrottlingTemperatures []float64 `json:"gpuThrottlingTemperatures,omitempty"`

	// Hardware: Name of the hardware, e.g. Angler.
	Hardware string `json:"hardware,omitempty"`

	// Manufacturer: Manufacturer, e.g. Motorola.
	Manufacturer string `json:"manufacturer,omitempty"`

	// Model: The model of the device, e.g. Asus Nexus 7.
	Model string `json:"model,omitempty"`

	// SerialNumber: The device serial number.
	SerialNumber string `json:"serialNumber,omitempty"`

	// SkinShutdownTemperatures: Device skin shutdown temperature thresholds
	// in Celsius.
	SkinShutdownTemperatures []float64 `json:"skinShutdownTemperatures,omitempty"`

	// SkinThrottlingTemperatures: Device skin throttling temperature
	// thresholds in Celsius.
	SkinThrottlingTemperatures []float64 `json:"skinThrottlingTemperatures,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "BatteryShutdownTemperatures") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "BatteryShutdownTemperatures") to include in API requests with the
	// JSON null value. By default, fields with empty values are omitted
	// from API requests. However, any field with an empty value appearing
	// in NullFields will be sent to the server as null. It is an error if a
	// field in this list has a non-empty value. This may be used to include
	// null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *HardwareInfo) MarshalJSON() ([]byte, error) {
	type noMethod HardwareInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// HardwareStatus: Hardware status. Temperatures may be compared to the
// temperature thresholds available in hardwareInfo to determine
// hardware health.
type HardwareStatus struct {
	// BatteryTemperatures: Current battery temperatures in Celsius for each
	// battery on the device.
	BatteryTemperatures []float64 `json:"batteryTemperatures,omitempty"`

	// CpuTemperatures: Current CPU temperatures in Celsius for each CPU on
	// the device.
	CpuTemperatures []float64 `json:"cpuTemperatures,omitempty"`

	// CpuUsages: CPU usages in percentage for each core available on the
	// device. Usage is 0 for each unplugged core. Empty array implies that
	// CPU usage is not supported in the system.
	CpuUsages []float64 `json:"cpuUsages,omitempty"`

	// CreateTime: The time the measurements were taken.
	CreateTime string `json:"createTime,omitempty"`

	// FanSpeeds: Fan speeds in RPM for each fan on the device. Empty array
	// means that there are no fans or fan speed is not supported on the
	// system.
	FanSpeeds []float64 `json:"fanSpeeds,omitempty"`

	// GpuTemperatures: Current GPU temperatures in Celsius for each GPU on
	// the device.
	GpuTemperatures []float64 `json:"gpuTemperatures,omitempty"`

	// SkinTemperatures: Current device skin temperatures in Celsius.
	SkinTemperatures []float64 `json:"skinTemperatures,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatteryTemperatures")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatteryTemperatures") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *HardwareStatus) MarshalJSON() ([]byte, error) {
	type noMethod HardwareStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListDevicesResponse: Response to a request to list devices for a
// given enterprise.
type ListDevicesResponse struct {
	// Devices: The list of devices.
	Devices []*Device `json:"devices,omitempty"`

	// NextPageToken: If there are more results, a token to retrieve next
	// page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Devices") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Devices") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListDevicesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListDevicesResponse
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

// ListPoliciesResponse: Response to a request to list policies for a
// given enterprise.
type ListPoliciesResponse struct {
	// NextPageToken: If there are more results, a token to retrieve next
	// page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Policies: The list of policies.
	Policies []*Policy `json:"policies,omitempty"`

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

func (s *ListPoliciesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListPoliciesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ManagedProperty: Managed property.
type ManagedProperty struct {
	// DefaultValue: The default value of the properties. BUNDLE_ARRAY
	// properties never have a default value.
	DefaultValue interface{} `json:"defaultValue,omitempty"`

	// Description: A longer description of the property, giving more detail
	// of what it affects. Localized.
	Description string `json:"description,omitempty"`

	// Entries: For CHOICE or MULTISELECT properties, the list of possible
	// entries.
	Entries []*ManagedPropertyEntry `json:"entries,omitempty"`

	// Key: The unique key that the application uses to identify the
	// property, e.g. "com.google.android.gm.fieldname".
	Key string `json:"key,omitempty"`

	// NestedProperties: For BUNDLE_ARRAY properties, the list of nested
	// properties. A BUNDLE_ARRAY property is at most two levels deep.
	NestedProperties []*ManagedProperty `json:"nestedProperties,omitempty"`

	// Title: The name of the property. Localized.
	Title string `json:"title,omitempty"`

	// Type: The type of the property.
	//
	// Possible values:
	//   "MANAGED_PROPERTY_TYPE_UNSPECIFIED" - Not used.
	//   "BOOL" - A property of boolean type.
	//   "STRING" - A property of string type.
	//   "INTEGER" - A property of integer type.
	//   "CHOICE" - A choice of one item from a set.
	//   "MULTISELECT" - A choice of multiple items from a set.
	//   "HIDDEN" - A hidden restriction of string type (the default value
	// can be used to pass along information that cannot be modified, such
	// as a version code).
	//   "BUNDLE_ARRAY" - An array of property bundles.
	Type string `json:"type,omitempty"`

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

func (s *ManagedProperty) MarshalJSON() ([]byte, error) {
	type noMethod ManagedProperty
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ManagedPropertyEntry: An entry of a managed property.
type ManagedPropertyEntry struct {
	// Name: The human-readable name of the value. Localized.
	Name string `json:"name,omitempty"`

	// Value: The machine-readable value of the entry, which should be used
	// in the configuration. Not localized.
	Value string `json:"value,omitempty"`

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

func (s *ManagedPropertyEntry) MarshalJSON() ([]byte, error) {
	type noMethod ManagedPropertyEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MemoryEvent: An event related to memory and storage measurements.
type MemoryEvent struct {
	// ByteCount: The number of free bytes in the medium, or for
	// EXTERNAL_STORAGE_DETECTED, the total capacity in bytes of the storage
	// medium.
	ByteCount int64 `json:"byteCount,omitempty,string"`

	// CreateTime: The creation time of the event.
	CreateTime string `json:"createTime,omitempty"`

	// EventType: Event type.
	//
	// Possible values:
	//   "MEMORY_EVENT_TYPE_UNSPECIFIED" - Unspecified. No events have this
	// type.
	//   "RAM_MEASURED" - Free space in RAM was measured.
	//   "INTERNAL_STORAGE_MEASURED" - Free space in internal storage was
	// measured.
	//   "EXTERNAL_STORAGE_DETECTED" - A new external storage medium was
	// detected. The reported byte count is the total capacity of the
	// storage medium.
	//   "EXTERNAL_STORAGE_REMOVED" - An external storage medium was
	// removed. The reported byte count is zero.
	//   "EXTERNAL_STORAGE_MEASURED" - Free space in an external storage
	// medium was measured.
	EventType string `json:"eventType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ByteCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ByteCount") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MemoryEvent) MarshalJSON() ([]byte, error) {
	type noMethod MemoryEvent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MemoryInfo: Information about device memory and storage.
type MemoryInfo struct {
	// TotalInternalStorage: Total internal storage on device in bytes.
	TotalInternalStorage int64 `json:"totalInternalStorage,omitempty,string"`

	// TotalRam: Total RAM on device in bytes.
	TotalRam int64 `json:"totalRam,omitempty,string"`

	// ForceSendFields is a list of field names (e.g.
	// "TotalInternalStorage") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "TotalInternalStorage") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *MemoryInfo) MarshalJSON() ([]byte, error) {
	type noMethod MemoryInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NetworkInfo: Device network info.
type NetworkInfo struct {
	// Imei: IMEI number of the GSM device, e.g. A1000031212.
	Imei string `json:"imei,omitempty"`

	// Meid: MEID number of the CDMA device, e.g. A00000292788E1.
	Meid string `json:"meid,omitempty"`

	// WifiMacAddress: WiFi MAC address of the device, e.g.
	// 7c:11:11:11:11:11.
	WifiMacAddress string `json:"wifiMacAddress,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Imei") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Imei") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *NetworkInfo) MarshalJSON() ([]byte, error) {
	type noMethod NetworkInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NonComplianceDetail: Provides detail about non-compliance with a
// policy setting.
type NonComplianceDetail struct {
	// CurrentValue: If the policy setting could not be applied, the current
	// value of the setting on the device.
	CurrentValue interface{} `json:"currentValue,omitempty"`

	// FieldPath: For settings with nested fields, if a particular nested
	// field is out of compliance, this specifies the full path to the
	// offending field. The path is formatted in the same way the policy
	// JSON field would be referenced in JavaScript, that is: 1) For
	// object-typed fields, the field name is followed by a dot then by a
	// subfield name. 2) For array-typed fields, the field name is followed
	// by the array index  enclosed in brackets. For example, to indicate a
	// problem with the url field in the externalData field in the 3rd
	// application, the path would be applications[2].externalData.url
	FieldPath string `json:"fieldPath,omitempty"`

	// InstallationFailureReason: If package_name is set and the
	// non-compliance reason is APP_NOT_INSTALLED, the detailed reason the
	// app cannot be installed.
	//
	// Possible values:
	//   "INSTALLATION_FAILURE_REASON_UNSPECIFIED" - This value is
	// disallowed.
	//   "INSTALLATION_FAILURE_REASON_UNKNOWN" - An unknown condition is
	// preventing the app from being installed. Some potential reaons are
	// that the device does not have enough storage, the device network
	// connection is unreliable, or the installation is taking longer than
	// expected. The installation will be retried automatically.
	//   "IN_PROGRESS" - The installation is still in progress.
	//   "NOT_FOUND" - The app was not found in Play.
	//   "NOT_COMPATIBLE_WITH_DEVICE" - The app is incompatible with the
	// device.
	//   "NOT_APPROVED" - The app has not been approved by the admin.
	//   "PERMISSIONS_NOT_ACCEPTED" - The app has new permissions that have
	// not been accepted by the admin.
	//   "NOT_AVAILABLE_IN_COUNTRY" - The app is not available in the user's
	// country.
	//   "NO_LICENSES_REMAINING" - There are no more licenses to assign to
	// the user.
	//   "NOT_ENROLLED" - The enterprise is no longer enrolled with Play for
	// Work or Android Device Policy is not enabled for the enterprise.
	//   "USER_INVALID" - The user is no longer valid. The user may have
	// been deleted or disabled.
	InstallationFailureReason string `json:"installationFailureReason,omitempty"`

	// NonComplianceReason: The reason the device is not in compliance with
	// the setting.
	//
	// Possible values:
	//   "NON_COMPLIANCE_REASON_UNSPECIFIED" - This value is disallowed.
	//   "API_LEVEL" - The setting is not supported in the API level of
	// Android OS version the device is running.
	//   "ADMIN_TYPE" - The admin type (profile owner, device owner, etc.)
	// does not support the setting.
	//   "USER_ACTION" - The user has not taken required action to comply
	// with the setting.
	//   "INVALID_VALUE" - The setting has an invalid value.
	//   "APP_NOT_INSTALLED" - The application required to implement the
	// policy is not installed.
	//   "UNSUPPORTED" - The policy is not supported by the version of
	// Android Device Policy on the device.
	//   "APP_INSTALLED" - A blocked application is installed.
	//   "PENDING" - The setting was not applied yet at the time of the
	// report, but is expected to be applied shortly.
	//   "APP_INCOMPATIBLE" - The setting cannot be applied to the
	// application because its target SDK version is not high enough.
	//   "APP_NOT_UPDATED" - The application is installed but not updated to
	// the minimum version code specified by policy
	NonComplianceReason string `json:"nonComplianceReason,omitempty"`

	// PackageName: The package name indicating which application is out of
	// compliance, if applicable.
	PackageName string `json:"packageName,omitempty"`

	// SettingName: The name of the policy setting. This is the JSON field
	// name of a top-level Policy  field.
	SettingName string `json:"settingName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CurrentValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CurrentValue") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *NonComplianceDetail) MarshalJSON() ([]byte, error) {
	type noMethod NonComplianceDetail
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NonComplianceDetailCondition: A compliance rule condition which is
// satisfied if there exists any matching NonComplianceDetail for the
// device. A NonComplianceDetail matches a NonComplianceDetailCondition
// if all the fields which are set within the
// NonComplianceDetailCondition match the corresponding
// NonComplianceDetail fields.
type NonComplianceDetailCondition struct {
	// NonComplianceReason: The reason the device is not in compliance with
	// the setting. If not set, then this condition matches any reason.
	//
	// Possible values:
	//   "NON_COMPLIANCE_REASON_UNSPECIFIED" - This value is disallowed.
	//   "API_LEVEL" - The setting is not supported in the API level of
	// Android OS version the device is running.
	//   "ADMIN_TYPE" - The admin type (profile owner, device owner, etc.)
	// does not support the setting.
	//   "USER_ACTION" - The user has not taken required action to comply
	// with the setting.
	//   "INVALID_VALUE" - The setting has an invalid value.
	//   "APP_NOT_INSTALLED" - The application required to implement the
	// policy is not installed.
	//   "UNSUPPORTED" - The policy is not supported by the version of
	// Android Device Policy on the device.
	//   "APP_INSTALLED" - A blocked application is installed.
	//   "PENDING" - The setting was not applied yet at the time of the
	// report, but is expected to be applied shortly.
	//   "APP_INCOMPATIBLE" - The setting cannot be applied to the
	// application because its target SDK version is not high enough.
	//   "APP_NOT_UPDATED" - The application is installed but not updated to
	// the minimum version code specified by policy
	NonComplianceReason string `json:"nonComplianceReason,omitempty"`

	// PackageName: The package name indicating which application is out of
	// compliance. If not set, then this condition matches any package name.
	// If this field is set, then setting_name must be unset or set to
	// applications; otherwise, the condition would never be satisfied.
	PackageName string `json:"packageName,omitempty"`

	// SettingName: The name of the policy setting. This is the JSON field
	// name of a top-level Policy field. If not set, then this condition
	// matches any setting name.
	SettingName string `json:"settingName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NonComplianceReason")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NonComplianceReason") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *NonComplianceDetailCondition) MarshalJSON() ([]byte, error) {
	type noMethod NonComplianceDetailCondition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: This resource represents a long-running operation that is
// the result of a network API call.
type Operation struct {
	// Done: If the value is false, it means the operation is still in
	// progress. If true, the operation is completed, and either error or
	// response is available.
	Done bool `json:"done,omitempty"`

	// Error: The error result of the operation in case of failure or
	// cancellation.
	Error *Status `json:"error,omitempty"`

	// Metadata: Service-specific metadata associated with the operation. It
	// typically contains progress information and common metadata such as
	// create time. Some services might not provide such metadata. Any
	// method that returns a long-running operation should document the
	// metadata type, if any.
	Metadata googleapi.RawMessage `json:"metadata,omitempty"`

	// Name: The server-assigned name, which is only unique within the same
	// service that originally returns it. If you use the default HTTP
	// mapping, the name should have the format of
	// operations/some/unique/name.
	Name string `json:"name,omitempty"`

	// Response: The normal response of the operation in case of success. If
	// the original method returns no data on success, such as Delete, the
	// response is google.protobuf.Empty. If the original method is standard
	// Get/Create/Update, the response should be the resource. For other
	// methods, the response should have the type XxxResponse, where Xxx is
	// the original method name. For example, if the original method name is
	// TakeSnapshot(), the inferred response type is TakeSnapshotResponse.
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

// PasswordRequirements: Requirements for the password used to unlock a
// device.
type PasswordRequirements struct {
	// MaximumFailedPasswordsForWipe: A device will be wiped after too many
	// incorrect device-unlock passwords have been entered. A value of 0
	// means there is no restriction.
	MaximumFailedPasswordsForWipe int64 `json:"maximumFailedPasswordsForWipe,omitempty"`

	// PasswordExpirationTimeout: Password expiration timeout.
	PasswordExpirationTimeout string `json:"passwordExpirationTimeout,omitempty"`

	// PasswordHistoryLength: The length of the password history. After
	// setting this, the user will not be able to enter a new password that
	// is the same as any password in the history. A value of 0 means there
	// is no restriction.
	PasswordHistoryLength int64 `json:"passwordHistoryLength,omitempty"`

	// PasswordMinimumLength: The minimum allowed password length. A value
	// of 0 means there is no restriction. Only enforced when
	// password_quality is NUMERIC, NUMERIC_COMPLEX, ALPHABETIC,
	// ALPHANUMERIC, or COMPLEX.
	PasswordMinimumLength int64 `json:"passwordMinimumLength,omitempty"`

	// PasswordMinimumLetters: Minimum number of letters required in the
	// password. Only enforced when password_quality is COMPLEX.
	PasswordMinimumLetters int64 `json:"passwordMinimumLetters,omitempty"`

	// PasswordMinimumLowerCase: Minimum number of lower case letters
	// required in the password. Only enforced when password_quality is
	// COMPLEX.
	PasswordMinimumLowerCase int64 `json:"passwordMinimumLowerCase,omitempty"`

	// PasswordMinimumNonLetter: Minimum number of non-letter characters
	// (numerical digits or symbols) required in the password. Only enforced
	// when password_quality is COMPLEX.
	PasswordMinimumNonLetter int64 `json:"passwordMinimumNonLetter,omitempty"`

	// PasswordMinimumNumeric: Minimum number of numerical digits required
	// in the password. Only enforced when password_quality is COMPLEX.
	PasswordMinimumNumeric int64 `json:"passwordMinimumNumeric,omitempty"`

	// PasswordMinimumSymbols: Minimum number of symbols required in the
	// password. Only enforced when password_quality is COMPLEX.
	PasswordMinimumSymbols int64 `json:"passwordMinimumSymbols,omitempty"`

	// PasswordMinimumUpperCase: Minimum number of upper case letters
	// required in the password. Only enforced when password_quality is
	// COMPLEX.
	PasswordMinimumUpperCase int64 `json:"passwordMinimumUpperCase,omitempty"`

	// PasswordQuality: The required password quality.
	//
	// Possible values:
	//   "PASSWORD_QUALITY_UNSPECIFIED" - There are no requirements for the
	// password.
	//   "SOMETHING" - There must be a password, but there are no
	// restrictions on its characters.
	//   "NUMERIC" - The password must contain numeric characters.
	//   "NUMERIC_COMPLEX" - The password must contain numeric characters
	// with no repeating (4444) or ordered (1234, 4321, 2468) sequences.
	//   "ALPHABETIC" - The password must contain alphabetic (or symbol)
	// characters.
	//   "ALPHANUMERIC" - The password must contain at both numeric and
	// alphabetic (or symbol) characters.
	//   "COMPLEX" - The password must contain at least a letter, a
	// numerical digit and a special symbol. Other password constraints, for
	// example, password_minimum_letters are enforced.
	PasswordQuality string `json:"passwordQuality,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "MaximumFailedPasswordsForWipe") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "MaximumFailedPasswordsForWipe") to include in API requests with the
	// JSON null value. By default, fields with empty values are omitted
	// from API requests. However, any field with an empty value appearing
	// in NullFields will be sent to the server as null. It is an error if a
	// field in this list has a non-empty value. This may be used to include
	// null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PasswordRequirements) MarshalJSON() ([]byte, error) {
	type noMethod PasswordRequirements
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PermissionGrant: Configuration for an Android permission and its
// grant state.
type PermissionGrant struct {
	// Permission: The android permission, e.g.
	// android.permission.READ_CALENDAR.
	Permission string `json:"permission,omitempty"`

	// Policy: The policy for granting the permission.
	//
	// Possible values:
	//   "PERMISSION_POLICY_UNSPECIFIED" - Policy not specified. If no
	// policy is specified for a permission at any level, then the PROMPT
	// behavior is used by default.
	//   "PROMPT" - Prompt the user to grant a permission.
	//   "GRANT" - Automatically grant a permission.
	//   "DENY" - Automatically deny a permission.
	Policy string `json:"policy,omitempty"`

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

func (s *PermissionGrant) MarshalJSON() ([]byte, error) {
	type noMethod PermissionGrant
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PersistentPreferredActivity: A default activity for handling intents
// that match a particular intent filter.
type PersistentPreferredActivity struct {
	// Actions: The intent actions to match in the filter. If any actions
	// are included in the filter, then an intent's action must be one of
	// those values for it to match. If no actions are included, the intent
	// action is ignored.
	Actions []string `json:"actions,omitempty"`

	// Categories: The intent categories to match in the filter. An intent
	// includes the categories that it requires, all of which must be
	// included in the filter in order to match. In other words, adding a
	// category to the filter has no impact on matching unless that category
	// is specified in the intent.
	Categories []string `json:"categories,omitempty"`

	// ReceiverActivity: The activity that should be the default intent
	// handler. This should be an Android component name, e.g.
	// com.android.enterprise.app/.MainActivity. Alternatively, the value
	// may be the package name of an app, which causes Android Device Policy
	// to choose an appropriate activity from the app to handle the intent.
	ReceiverActivity string `json:"receiverActivity,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Actions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Actions") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PersistentPreferredActivity) MarshalJSON() ([]byte, error) {
	type noMethod PersistentPreferredActivity
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Policy: A policy, which governs behavior for a device.
type Policy struct {
	// AddUserDisabled: Whether adding new users and profiles is disabled.
	AddUserDisabled bool `json:"addUserDisabled,omitempty"`

	// AdjustVolumeDisabled: Whether adjusting the master volume is
	// disabled.
	AdjustVolumeDisabled bool `json:"adjustVolumeDisabled,omitempty"`

	// Applications: Policy applied to apps.
	Applications []*ApplicationPolicy `json:"applications,omitempty"`

	// AutoTimeRequired: Whether auto time is required, which prevents the
	// user from manually setting the date and time.
	AutoTimeRequired bool `json:"autoTimeRequired,omitempty"`

	// BlockApplicationsEnabled: Whether applications other than the ones
	// configured in applications are blocked from being installed. When
	// set, applications that were installed under a previous policy but no
	// longer appear in the policy are automatically uninstalled.
	BlockApplicationsEnabled bool `json:"blockApplicationsEnabled,omitempty"`

	// CameraDisabled: Whether all cameras on the device are disabled.
	CameraDisabled bool `json:"cameraDisabled,omitempty"`

	// ComplianceRules: Rules declaring which mitigating actions to take
	// when a device is not compliant with its policy. When the conditions
	// for multiple rules are satisfied, all of the mitigating actions for
	// the rules are taken. There is a maximum limit of 100 rules.
	ComplianceRules []*ComplianceRule `json:"complianceRules,omitempty"`

	// DebuggingFeaturesAllowed: Whether the user is allowed to enable
	// debugging features.
	DebuggingFeaturesAllowed bool `json:"debuggingFeaturesAllowed,omitempty"`

	// DefaultPermissionPolicy: The default permission policy for requests
	// for runtime permissions.
	//
	// Possible values:
	//   "PERMISSION_POLICY_UNSPECIFIED" - Policy not specified. If no
	// policy is specified for a permission at any level, then the PROMPT
	// behavior is used by default.
	//   "PROMPT" - Prompt the user to grant a permission.
	//   "GRANT" - Automatically grant a permission.
	//   "DENY" - Automatically deny a permission.
	DefaultPermissionPolicy string `json:"defaultPermissionPolicy,omitempty"`

	// FactoryResetDisabled: Whether factory resetting from settings is
	// disabled.
	FactoryResetDisabled bool `json:"factoryResetDisabled,omitempty"`

	// FrpAdminEmails: Email addresses of device administrators for factory
	// reset protection. When the device is factory reset, it will require
	// one of these admins to log in with the Google account email and
	// password to unlock the device. If no admins are specified, the device
	// will not provide factory reset protection.
	FrpAdminEmails []string `json:"frpAdminEmails,omitempty"`

	// FunDisabled: Whether the user is allowed to have fun. Controls
	// whether the Easter egg game in Settings is disabled.
	FunDisabled bool `json:"funDisabled,omitempty"`

	// InstallUnknownSourcesAllowed: Whether the user is allowed to enable
	// the "Unknown Sources" setting, which allows installation of apps from
	// unknown sources.
	InstallUnknownSourcesAllowed bool `json:"installUnknownSourcesAllowed,omitempty"`

	// KeyguardDisabled: Whether the keyguard is disabled.
	KeyguardDisabled bool `json:"keyguardDisabled,omitempty"`

	// MaximumTimeToLock: Maximum time in milliseconds for user activity
	// until the device will lock. A value of 0 means there is no
	// restriction.
	MaximumTimeToLock int64 `json:"maximumTimeToLock,omitempty,string"`

	// ModifyAccountsDisabled: Whether adding or removing accounts is
	// disabled.
	ModifyAccountsDisabled bool `json:"modifyAccountsDisabled,omitempty"`

	// Name: The name of the policy in the form
	// enterprises/{enterpriseId}/policies/{policyId}
	Name string `json:"name,omitempty"`

	// NetworkEscapeHatchEnabled: Flag to specify if network escape hatch is
	// enabled. If this flag has been enabled then upon device boot if
	// device has no network connection, then an activity will be shown that
	// allows the user to temporarily connect to a network to fetch the
	// latest policy. The launched activity will time out if no network has
	// been connected for a given while and will return to the previous
	// activity that was shown.
	NetworkEscapeHatchEnabled bool `json:"networkEscapeHatchEnabled,omitempty"`

	// OpenNetworkConfiguration: Network configuration for the device. See
	// configure networks for more information.
	OpenNetworkConfiguration googleapi.RawMessage `json:"openNetworkConfiguration,omitempty"`

	// PasswordRequirements: Password requirements.
	PasswordRequirements *PasswordRequirements `json:"passwordRequirements,omitempty"`

	// PersistentPreferredActivities: Default intent handler activities.
	PersistentPreferredActivities []*PersistentPreferredActivity `json:"persistentPreferredActivities,omitempty"`

	// RemoveUserDisabled: Whether removing other users is disabled.
	RemoveUserDisabled bool `json:"removeUserDisabled,omitempty"`

	// SafeBootDisabled: Whether rebooting the device into safe boot is
	// disabled.
	SafeBootDisabled bool `json:"safeBootDisabled,omitempty"`

	// ScreenCaptureDisabled: Whether screen capture is disabled.
	ScreenCaptureDisabled bool `json:"screenCaptureDisabled,omitempty"`

	// StatusBarDisabled: Whether the status bar is disabled. This disables
	// notifications, quick settings and other screen overlays that allow
	// escape from full-screen mode.
	StatusBarDisabled bool `json:"statusBarDisabled,omitempty"`

	// StatusReportingSettings: Status reporting settings
	StatusReportingSettings *StatusReportingSettings `json:"statusReportingSettings,omitempty"`

	// StayOnPluggedModes: The battery plugged in modes for which the device
	// stays on. When using this setting, it is recommended to clear
	// maximum_time_to_lock so that the device doesn't lock itself while it
	// stays on.
	//
	// Possible values:
	//   "BATTERY_PLUGGED_MODE_UNSPECIFIED" - This value is ignored.
	//   "AC" - Power source is an AC charger.
	//   "USB" - Power source is a USB port.
	//   "WIRELESS" - Power source is wireless.
	StayOnPluggedModes []string `json:"stayOnPluggedModes,omitempty"`

	// SystemUpdate: The system update policy, which controls how OS updates
	// are applied. If the update type is WINDOWED and the device has a
	// device account, the update window will automatically apply to Play
	// app updates as well.
	SystemUpdate *SystemUpdate `json:"systemUpdate,omitempty"`

	// UnmuteMicrophoneDisabled: Whether the microphone is muted and
	// adjusting microphone volume is disabled.
	UnmuteMicrophoneDisabled bool `json:"unmuteMicrophoneDisabled,omitempty"`

	// Version: The version of the policy. This is a read-only field. The
	// version is incremented each time the policy is updated.
	Version int64 `json:"version,omitempty,string"`

	// WifiConfigDisabled: Whether configuring WiFi access points is
	// disabled.
	WifiConfigDisabled bool `json:"wifiConfigDisabled,omitempty"`

	// WifiConfigsLockdownEnabled: Whether WiFi networks defined in Open
	// Network Configuration are locked so they cannot be edited by the
	// user.
	WifiConfigsLockdownEnabled bool `json:"wifiConfigsLockdownEnabled,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AddUserDisabled") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AddUserDisabled") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Policy) MarshalJSON() ([]byte, error) {
	type noMethod Policy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PowerManagementEvent: A power management event.
type PowerManagementEvent struct {
	// BatteryLevel: For BATTERY_LEVEL_COLLECTED events, the battery level
	// as a percentage.
	BatteryLevel float64 `json:"batteryLevel,omitempty"`

	// CreateTime: The creation time of the event.
	CreateTime string `json:"createTime,omitempty"`

	// EventType: Event type.
	//
	// Possible values:
	//   "POWER_MANAGEMENT_EVENT_TYPE_UNSPECIFIED" - Unspecified. No events
	// have this type.
	//   "BATTERY_LEVEL_COLLECTED" - Battery level was measured.
	//   "POWER_CONNECTED" - The device started charging.
	//   "POWER_DISCONNECTED" - The device stopped charging.
	//   "BATTERY_LOW" - The device entered low-power mode.
	//   "BATTERY_OKAY" - The device exited low-power mode.
	//   "BOOT_COMPLETED" - The device booted.
	//   "SHUTDOWN" - The device shut down.
	EventType string `json:"eventType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BatteryLevel") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BatteryLevel") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PowerManagementEvent) MarshalJSON() ([]byte, error) {
	type noMethod PowerManagementEvent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *PowerManagementEvent) UnmarshalJSON(data []byte) error {
	type noMethod PowerManagementEvent
	var s1 struct {
		BatteryLevel gensupport.JSONFloat64 `json:"batteryLevel"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.BatteryLevel = float64(s1.BatteryLevel)
	return nil
}

// SignupUrl: An enterprise signup URL.
type SignupUrl struct {
	// Name: The name of the resource. This must be included in the create
	// enterprise request at the end of the signup flow.
	Name string `json:"name,omitempty"`

	// Url: A URL under which the Admin can sign up for an enterprise. The
	// page pointed to cannot be rendered in an iframe.
	Url string `json:"url,omitempty"`

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

func (s *SignupUrl) MarshalJSON() ([]byte, error) {
	type noMethod SignupUrl
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SoftwareInfo: Information about device software.
type SoftwareInfo struct {
	// AndroidBuildNumber: Android build Id string meant for displaying to
	// the user, e.g. shamu-userdebug 6.0.1 MOB30I 2756745 dev-keys.
	AndroidBuildNumber string `json:"androidBuildNumber,omitempty"`

	// AndroidBuildTime: Build time.
	AndroidBuildTime string `json:"androidBuildTime,omitempty"`

	// AndroidVersion: The user visible Android version string, e.g. 6.0.1.
	AndroidVersion string `json:"androidVersion,omitempty"`

	// BootloaderVersion: The system bootloader version number, e.g. 0.6.7.
	BootloaderVersion string `json:"bootloaderVersion,omitempty"`

	// DeviceKernelVersion: Kernel version, e.g. 2.6.32.9-g103d848.
	DeviceKernelVersion string `json:"deviceKernelVersion,omitempty"`

	// SecurityPatchLevel: Security patch level, e.g. 2016-05-01.
	SecurityPatchLevel string `json:"securityPatchLevel,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AndroidBuildNumber")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AndroidBuildNumber") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SoftwareInfo) MarshalJSON() ([]byte, error) {
	type noMethod SoftwareInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Status: The Status type defines a logical error model that is
// suitable for different programming environments, including REST APIs
// and RPC APIs. It is used by gRPC (https://github.com/grpc). The error
// model is designed to be:
// Simple to use and understand for most users
// Flexible enough to meet unexpected needsOverviewThe Status message
// contains three pieces of data: error code, error message, and error
// details. The error code should be an enum value of google.rpc.Code,
// but it may accept additional error codes if needed. The error message
// should be a developer-facing English message that helps developers
// understand and resolve the error. If a localized user-facing error
// message is needed, put the localized message in the error details or
// localize it in the client. The optional error details may contain
// arbitrary information about the error. There is a predefined set of
// error detail types in the package google.rpc that can be used for
// common error conditions.Language mappingThe Status message is the
// logical representation of the error model, but it is not necessarily
// the actual wire format. When the Status message is exposed in
// different client libraries and different wire protocols, it can be
// mapped differently. For example, it will likely be mapped to some
// exceptions in Java, but more likely mapped to some error codes in
// C.Other usesThe error model and the Status message can be used in a
// variety of environments, either with or without APIs, to provide a
// consistent developer experience across different environments.Example
// uses of this error model include:
// Partial errors. If a service needs to return partial errors to the
// client, it may embed the Status in the normal response to indicate
// the partial errors.
// Workflow errors. A typical workflow has multiple steps. Each step may
// have a Status message for error reporting.
// Batch operations. If a client uses batch request and batch response,
// the Status message should be used directly inside batch response, one
// for each error sub-response.
// Asynchronous operations. If an API call embeds asynchronous operation
// results in its response, the status of those operations should be
// represented directly using the Status message.
// Logging. If some API errors are stored in logs, the message Status
// could be used directly after any stripping needed for
// security/privacy reasons.
type Status struct {
	// Code: The status code, which should be an enum value of
	// google.rpc.Code.
	Code int64 `json:"code,omitempty"`

	// Details: A list of messages that carry the error details. There is a
	// common set of message types for APIs to use.
	Details []googleapi.RawMessage `json:"details,omitempty"`

	// Message: A developer-facing error message, which should be in
	// English. Any user-facing error message should be localized and sent
	// in the google.rpc.Status.details field, or localized by the client.
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

// StatusReportingSettings: Settings controlling the behavior of status
// reports.
type StatusReportingSettings struct {
	// DisplayInfoEnabled: Whether displays reporting is enabled.
	DisplayInfoEnabled bool `json:"displayInfoEnabled,omitempty"`

	// HardwareStatusEnabled: Whether hardware status reporting is enabled.
	HardwareStatusEnabled bool `json:"hardwareStatusEnabled,omitempty"`

	// MemoryInfoEnabled: Whether memory info reporting is enabled.
	MemoryInfoEnabled bool `json:"memoryInfoEnabled,omitempty"`

	// NetworkInfoEnabled: Whether network info reporting is enabled.
	NetworkInfoEnabled bool `json:"networkInfoEnabled,omitempty"`

	// PowerManagementEventsEnabled: Whether power management event
	// reporting is enabled.
	PowerManagementEventsEnabled bool `json:"powerManagementEventsEnabled,omitempty"`

	// SoftwareInfoEnabled: Whether software info reporting is enabled.
	SoftwareInfoEnabled bool `json:"softwareInfoEnabled,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayInfoEnabled")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DisplayInfoEnabled") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *StatusReportingSettings) MarshalJSON() ([]byte, error) {
	type noMethod StatusReportingSettings
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SystemUpdate: Configuration for managing system updates
type SystemUpdate struct {
	// EndMinutes: If the type is WINDOWED, the end of the maintenance
	// window, measured as the number of minutes after midnight in device
	// local time. This value must be between 0 and 1439, inclusive. If this
	// value is less than start_minutes, then the maintenance window spans
	// midnight. If the maintenance window specified is smaller than 30
	// minutes, the actual window is extended to 30 minutes beyond the start
	// time.
	EndMinutes int64 `json:"endMinutes,omitempty"`

	// StartMinutes: If the type is WINDOWED, the start of the maintenance
	// window, measured as the number of minutes after midnight in device
	// local time. This value must be between 0 and 1439, inclusive.
	StartMinutes int64 `json:"startMinutes,omitempty"`

	// Type: The type of system update to configure.
	//
	// Possible values:
	//   "SYSTEM_UPDATE_TYPE_UNSPECIFIED" - Follow the default update
	// behavior for the device, which typically requires the user to accept
	// system updates.
	//   "AUTOMATIC" - Install automatically as soon as an update is
	// available.
	//   "WINDOWED" - Install automatically within a daily maintenance
	// window. If the device has a device account, this also configures Play
	// apps to be updated within the window. This is strongly recommended
	// for kiosk devices because this is the only way apps persistently
	// pinned to the foreground can be updated by Play.
	//   "POSTPONE" - Postpone automatic install up to a maximum of 30 days.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndMinutes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndMinutes") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SystemUpdate) MarshalJSON() ([]byte, error) {
	type noMethod SystemUpdate
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UserFacingMessage: Provides user facing message with locale info. The
// maximum message length is 4096 characters.
type UserFacingMessage struct {
	// DefaultMessage: The default message that gets displayed if no
	// localized message is specified, or the user's locale does not match
	// with any of the localized messages. A default message must be
	// provided if any localized messages are provided.
	DefaultMessage string `json:"defaultMessage,omitempty"`

	// LocalizedMessages: A map which contains <locale, message> pairs. The
	// locale is a BCP 47 language code, e.g. en-US, es-ES, fr.
	LocalizedMessages map[string]string `json:"localizedMessages,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DefaultMessage") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DefaultMessage") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *UserFacingMessage) MarshalJSON() ([]byte, error) {
	type noMethod UserFacingMessage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// WebToken: A web token used to access an embeddable managed Google
// Play web UI.
type WebToken struct {
	// Name: The name of the web token, which is generated by the server
	// during creation, in the form
	// enterprises/{enterpriseId}/webTokens/{webTokenId}.
	Name string `json:"name,omitempty"`

	// ParentFrameUrl: The URL of the parent frame hosting the iframe with
	// the embedded UI. To prevent XSS, the iframe may not be hosted at
	// other URLs. The URL must use the https scheme.
	ParentFrameUrl string `json:"parentFrameUrl,omitempty"`

	// Permissions: Permissions the admin may exercise in the embedded UI.
	// The admin must have all of these permissions in order to view the UI.
	//
	// Possible values:
	//   "WEB_TOKEN_PERMISSION_UNSPECIFIED" - This value is ignored.
	//   "APPROVE_APPS" - The permission to approve apps for the enterprise.
	Permissions []string `json:"permissions,omitempty"`

	// Value: The token value which is used in the hosting page to generate
	// the iframe with the embedded UI. This is a read-only field generated
	// by the server.
	Value string `json:"value,omitempty"`

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

func (s *WebToken) MarshalJSON() ([]byte, error) {
	type noMethod WebToken
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "androidmanagement.enterprises.create":

type EnterprisesCreateCall struct {
	s          *Service
	enterprise *Enterprise
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates an enterprise by completing the enterprise signup
// flow.
func (r *EnterprisesService) Create(enterprise *Enterprise) *EnterprisesCreateCall {
	c := &EnterprisesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.enterprise = enterprise
	return c
}

// EnterpriseToken sets the optional parameter "enterpriseToken": The
// enterprise token appended to the callback URL.
func (c *EnterprisesCreateCall) EnterpriseToken(enterpriseToken string) *EnterprisesCreateCall {
	c.urlParams_.Set("enterpriseToken", enterpriseToken)
	return c
}

// ProjectId sets the optional parameter "projectId": The id of the
// Google Cloud Platform project which will own the enterprise.
func (c *EnterprisesCreateCall) ProjectId(projectId string) *EnterprisesCreateCall {
	c.urlParams_.Set("projectId", projectId)
	return c
}

// SignupUrlName sets the optional parameter "signupUrlName": The name
// of the SignupUrl used to sign up for the enterprise.
func (c *EnterprisesCreateCall) SignupUrlName(signupUrlName string) *EnterprisesCreateCall {
	c.urlParams_.Set("signupUrlName", signupUrlName)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesCreateCall) Fields(s ...googleapi.Field) *EnterprisesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesCreateCall) Context(ctx context.Context) *EnterprisesCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesCreateCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/enterprises")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidmanagement.enterprises.create" call.
// Exactly one of *Enterprise or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Enterprise.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesCreateCall) Do(opts ...googleapi.CallOption) (*Enterprise, error) {
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
	//   "description": "Creates an enterprise by completing the enterprise signup flow.",
	//   "flatPath": "v1/enterprises",
	//   "httpMethod": "POST",
	//   "id": "androidmanagement.enterprises.create",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "enterpriseToken": {
	//       "description": "The enterprise token appended to the callback URL.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The id of the Google Cloud Platform project which will own the enterprise.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "signupUrlName": {
	//       "description": "The name of the SignupUrl used to sign up for the enterprise.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/enterprises",
	//   "request": {
	//     "$ref": "Enterprise"
	//   },
	//   "response": {
	//     "$ref": "Enterprise"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.get":

type EnterprisesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets an enterprise.
func (r *EnterprisesService) Get(name string) *EnterprisesGetCall {
	c := &EnterprisesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidmanagement.enterprises.get" call.
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
	//   "description": "Gets an enterprise.",
	//   "flatPath": "v1/enterprises/{enterprisesId}",
	//   "httpMethod": "GET",
	//   "id": "androidmanagement.enterprises.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the enterprise in the form enterprises/{enterpriseId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Enterprise"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.patch":

type EnterprisesPatchCall struct {
	s          *Service
	name       string
	enterprise *Enterprise
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates an enterprise.
func (r *EnterprisesService) Patch(name string, enterprise *Enterprise) *EnterprisesPatchCall {
	c := &EnterprisesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.enterprise = enterprise
	return c
}

// UpdateMask sets the optional parameter "updateMask": The field mask
// indicating the fields to update. If not set, all modifiable fields
// will be modified.
func (c *EnterprisesPatchCall) UpdateMask(updateMask string) *EnterprisesPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesPatchCall) Fields(s ...googleapi.Field) *EnterprisesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesPatchCall) Context(ctx context.Context) *EnterprisesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesPatchCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidmanagement.enterprises.patch" call.
// Exactly one of *Enterprise or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Enterprise.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesPatchCall) Do(opts ...googleapi.CallOption) (*Enterprise, error) {
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
	//   "description": "Updates an enterprise.",
	//   "flatPath": "v1/enterprises/{enterprisesId}",
	//   "httpMethod": "PATCH",
	//   "id": "androidmanagement.enterprises.patch",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the enterprise in the form enterprises/{enterpriseId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "The field mask indicating the fields to update. If not set, all modifiable fields will be modified.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "request": {
	//     "$ref": "Enterprise"
	//   },
	//   "response": {
	//     "$ref": "Enterprise"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.applications.get":

type EnterprisesApplicationsGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets info about an application.
func (r *EnterprisesApplicationsService) Get(name string) *EnterprisesApplicationsGetCall {
	c := &EnterprisesApplicationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// LanguageCode sets the optional parameter "languageCode": The
// preferred language for localized application info, as a BCP47 tag
// (e.g. "en-US", "de"). If not specified the default language of the
// application will be used.
func (c *EnterprisesApplicationsGetCall) LanguageCode(languageCode string) *EnterprisesApplicationsGetCall {
	c.urlParams_.Set("languageCode", languageCode)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesApplicationsGetCall) Fields(s ...googleapi.Field) *EnterprisesApplicationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesApplicationsGetCall) IfNoneMatch(entityTag string) *EnterprisesApplicationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesApplicationsGetCall) Context(ctx context.Context) *EnterprisesApplicationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesApplicationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesApplicationsGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "androidmanagement.enterprises.applications.get" call.
// Exactly one of *Application or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Application.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesApplicationsGetCall) Do(opts ...googleapi.CallOption) (*Application, error) {
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
	ret := &Application{
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
	//   "description": "Gets info about an application.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/applications/{applicationsId}",
	//   "httpMethod": "GET",
	//   "id": "androidmanagement.enterprises.applications.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "languageCode": {
	//       "description": "The preferred language for localized application info, as a BCP47 tag (e.g. \"en-US\", \"de\"). If not specified the default language of the application will be used.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "The name of the application in the form enterprises/{enterpriseId}/applications/{package_name}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/applications/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Application"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.devices.delete":

type EnterprisesDevicesDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a device, which causes the device to be wiped.
func (r *EnterprisesDevicesService) Delete(name string) *EnterprisesDevicesDeleteCall {
	c := &EnterprisesDevicesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesDevicesDeleteCall) Fields(s ...googleapi.Field) *EnterprisesDevicesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesDevicesDeleteCall) Context(ctx context.Context) *EnterprisesDevicesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesDevicesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesDevicesDeleteCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "androidmanagement.enterprises.devices.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EnterprisesDevicesDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes a device, which causes the device to be wiped.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/devices/{devicesId}",
	//   "httpMethod": "DELETE",
	//   "id": "androidmanagement.enterprises.devices.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the device in the form enterprises/{enterpriseId}/devices/{deviceId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/devices/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.devices.get":

type EnterprisesDevicesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a device.
func (r *EnterprisesDevicesService) Get(name string) *EnterprisesDevicesGetCall {
	c := &EnterprisesDevicesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesDevicesGetCall) Fields(s ...googleapi.Field) *EnterprisesDevicesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesDevicesGetCall) IfNoneMatch(entityTag string) *EnterprisesDevicesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesDevicesGetCall) Context(ctx context.Context) *EnterprisesDevicesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesDevicesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesDevicesGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "androidmanagement.enterprises.devices.get" call.
// Exactly one of *Device or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Device.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EnterprisesDevicesGetCall) Do(opts ...googleapi.CallOption) (*Device, error) {
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
	//   "description": "Gets a device.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/devices/{devicesId}",
	//   "httpMethod": "GET",
	//   "id": "androidmanagement.enterprises.devices.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the device in the form enterprises/{enterpriseId}/devices/{deviceId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/devices/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Device"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.devices.issueCommand":

type EnterprisesDevicesIssueCommandCall struct {
	s          *Service
	name       string
	command    *Command
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// IssueCommand: Issues a command to a device. The Operation resource
// returned contains a Command in its metadata field. Use the get
// operation method to get the status of the command.
func (r *EnterprisesDevicesService) IssueCommand(name string, command *Command) *EnterprisesDevicesIssueCommandCall {
	c := &EnterprisesDevicesIssueCommandCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.command = command
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesDevicesIssueCommandCall) Fields(s ...googleapi.Field) *EnterprisesDevicesIssueCommandCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesDevicesIssueCommandCall) Context(ctx context.Context) *EnterprisesDevicesIssueCommandCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesDevicesIssueCommandCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesDevicesIssueCommandCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.command)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+name}:issueCommand")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidmanagement.enterprises.devices.issueCommand" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesDevicesIssueCommandCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Issues a command to a device. The Operation resource returned contains a Command in its metadata field. Use the get operation method to get the status of the command.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/devices/{devicesId}:issueCommand",
	//   "httpMethod": "POST",
	//   "id": "androidmanagement.enterprises.devices.issueCommand",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the device in the form enterprises/{enterpriseId}/devices/{deviceId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/devices/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:issueCommand",
	//   "request": {
	//     "$ref": "Command"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.devices.list":

type EnterprisesDevicesListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists devices for a given enterprise.
func (r *EnterprisesDevicesService) List(parent string) *EnterprisesDevicesListCall {
	c := &EnterprisesDevicesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageSize sets the optional parameter "pageSize": The requested page
// size. The actual page size may be fixed to a min or max value.
func (c *EnterprisesDevicesListCall) PageSize(pageSize int64) *EnterprisesDevicesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
func (c *EnterprisesDevicesListCall) PageToken(pageToken string) *EnterprisesDevicesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesDevicesListCall) Fields(s ...googleapi.Field) *EnterprisesDevicesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesDevicesListCall) IfNoneMatch(entityTag string) *EnterprisesDevicesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesDevicesListCall) Context(ctx context.Context) *EnterprisesDevicesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesDevicesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesDevicesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/devices")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidmanagement.enterprises.devices.list" call.
// Exactly one of *ListDevicesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListDevicesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesDevicesListCall) Do(opts ...googleapi.CallOption) (*ListDevicesResponse, error) {
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
	ret := &ListDevicesResponse{
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
	//   "description": "Lists devices for a given enterprise.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/devices",
	//   "httpMethod": "GET",
	//   "id": "androidmanagement.enterprises.devices.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "The requested page size. The actual page size may be fixed to a min or max value.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "The name of the enterprise in the form enterprises/{enterpriseId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/devices",
	//   "response": {
	//     "$ref": "ListDevicesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *EnterprisesDevicesListCall) Pages(ctx context.Context, f func(*ListDevicesResponse) error) error {
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

// method id "androidmanagement.enterprises.devices.patch":

type EnterprisesDevicesPatchCall struct {
	s          *Service
	name       string
	device     *Device
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a device.
func (r *EnterprisesDevicesService) Patch(name string, device *Device) *EnterprisesDevicesPatchCall {
	c := &EnterprisesDevicesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.device = device
	return c
}

// UpdateMask sets the optional parameter "updateMask": The field mask
// indicating the fields to update. If not set, all modifiable fields
// will be modified.
func (c *EnterprisesDevicesPatchCall) UpdateMask(updateMask string) *EnterprisesDevicesPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesDevicesPatchCall) Fields(s ...googleapi.Field) *EnterprisesDevicesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesDevicesPatchCall) Context(ctx context.Context) *EnterprisesDevicesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesDevicesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesDevicesPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.device)
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
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidmanagement.enterprises.devices.patch" call.
// Exactly one of *Device or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Device.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EnterprisesDevicesPatchCall) Do(opts ...googleapi.CallOption) (*Device, error) {
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
	//   "description": "Updates a device.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/devices/{devicesId}",
	//   "httpMethod": "PATCH",
	//   "id": "androidmanagement.enterprises.devices.patch",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the device in the form enterprises/{enterpriseId}/devices/{deviceId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/devices/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "The field mask indicating the fields to update. If not set, all modifiable fields will be modified.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "request": {
	//     "$ref": "Device"
	//   },
	//   "response": {
	//     "$ref": "Device"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.devices.operations.cancel":

type EnterprisesDevicesOperationsCancelCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Cancel: Starts asynchronous cancellation on a long-running operation.
// The server makes a best effort to cancel the operation, but success
// is not guaranteed. If the server doesn't support this method, it
// returns google.rpc.Code.UNIMPLEMENTED. Clients can use
// Operations.GetOperation or other methods to check whether the
// cancellation succeeded or whether the operation completed despite
// cancellation. On successful cancellation, the operation is not
// deleted; instead, it becomes an operation with an Operation.error
// value with a google.rpc.Status.code of 1, corresponding to
// Code.CANCELLED.
func (r *EnterprisesDevicesOperationsService) Cancel(name string) *EnterprisesDevicesOperationsCancelCall {
	c := &EnterprisesDevicesOperationsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesDevicesOperationsCancelCall) Fields(s ...googleapi.Field) *EnterprisesDevicesOperationsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesDevicesOperationsCancelCall) Context(ctx context.Context) *EnterprisesDevicesOperationsCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesDevicesOperationsCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesDevicesOperationsCancelCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "androidmanagement.enterprises.devices.operations.cancel" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EnterprisesDevicesOperationsCancelCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns google.rpc.Code.UNIMPLEMENTED. Clients can use Operations.GetOperation or other methods to check whether the cancellation succeeded or whether the operation completed despite cancellation. On successful cancellation, the operation is not deleted; instead, it becomes an operation with an Operation.error value with a google.rpc.Status.code of 1, corresponding to Code.CANCELLED.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/devices/{devicesId}/operations/{operationsId}:cancel",
	//   "httpMethod": "POST",
	//   "id": "androidmanagement.enterprises.devices.operations.cancel",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be cancelled.",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/devices/[^/]+/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}:cancel",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.devices.operations.delete":

type EnterprisesDevicesOperationsDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a long-running operation. This method indicates that
// the client is no longer interested in the operation result. It does
// not cancel the operation. If the server doesn't support this method,
// it returns google.rpc.Code.UNIMPLEMENTED.
func (r *EnterprisesDevicesOperationsService) Delete(name string) *EnterprisesDevicesOperationsDeleteCall {
	c := &EnterprisesDevicesOperationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesDevicesOperationsDeleteCall) Fields(s ...googleapi.Field) *EnterprisesDevicesOperationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesDevicesOperationsDeleteCall) Context(ctx context.Context) *EnterprisesDevicesOperationsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesDevicesOperationsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesDevicesOperationsDeleteCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "androidmanagement.enterprises.devices.operations.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EnterprisesDevicesOperationsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes a long-running operation. This method indicates that the client is no longer interested in the operation result. It does not cancel the operation. If the server doesn't support this method, it returns google.rpc.Code.UNIMPLEMENTED.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/devices/{devicesId}/operations/{operationsId}",
	//   "httpMethod": "DELETE",
	//   "id": "androidmanagement.enterprises.devices.operations.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be deleted.",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/devices/[^/]+/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.devices.operations.get":

type EnterprisesDevicesOperationsGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the latest state of a long-running operation. Clients can
// use this method to poll the operation result at intervals as
// recommended by the API service.
func (r *EnterprisesDevicesOperationsService) Get(name string) *EnterprisesDevicesOperationsGetCall {
	c := &EnterprisesDevicesOperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesDevicesOperationsGetCall) Fields(s ...googleapi.Field) *EnterprisesDevicesOperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesDevicesOperationsGetCall) IfNoneMatch(entityTag string) *EnterprisesDevicesOperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesDevicesOperationsGetCall) Context(ctx context.Context) *EnterprisesDevicesOperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesDevicesOperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesDevicesOperationsGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "androidmanagement.enterprises.devices.operations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesDevicesOperationsGetCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/devices/{devicesId}/operations/{operationsId}",
	//   "httpMethod": "GET",
	//   "id": "androidmanagement.enterprises.devices.operations.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource.",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/devices/[^/]+/operations/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.devices.operations.list":

type EnterprisesDevicesOperationsListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists operations that match the specified filter in the
// request. If the server doesn't support this method, it returns
// UNIMPLEMENTED.NOTE: the name binding allows API services to override
// the binding to use different resource name schemes, such as
// users/*/operations. To override the binding, API services can add a
// binding such as "/v1/{name=users/*}/operations" to their service
// configuration. For backwards compatibility, the default name includes
// the operations collection id, however overriding users must ensure
// the name binding is the parent resource, without the operations
// collection id.
func (r *EnterprisesDevicesOperationsService) List(name string) *EnterprisesDevicesOperationsListCall {
	c := &EnterprisesDevicesOperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": The standard list
// filter.
func (c *EnterprisesDevicesOperationsListCall) Filter(filter string) *EnterprisesDevicesOperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The standard list
// page size.
func (c *EnterprisesDevicesOperationsListCall) PageSize(pageSize int64) *EnterprisesDevicesOperationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard list
// page token.
func (c *EnterprisesDevicesOperationsListCall) PageToken(pageToken string) *EnterprisesDevicesOperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesDevicesOperationsListCall) Fields(s ...googleapi.Field) *EnterprisesDevicesOperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesDevicesOperationsListCall) IfNoneMatch(entityTag string) *EnterprisesDevicesOperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesDevicesOperationsListCall) Context(ctx context.Context) *EnterprisesDevicesOperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesDevicesOperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesDevicesOperationsListCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "androidmanagement.enterprises.devices.operations.list" call.
// Exactly one of *ListOperationsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListOperationsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesDevicesOperationsListCall) Do(opts ...googleapi.CallOption) (*ListOperationsResponse, error) {
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
	//   "description": "Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns UNIMPLEMENTED.NOTE: the name binding allows API services to override the binding to use different resource name schemes, such as users/*/operations. To override the binding, API services can add a binding such as \"/v1/{name=users/*}/operations\" to their service configuration. For backwards compatibility, the default name includes the operations collection id, however overriding users must ensure the name binding is the parent resource, without the operations collection id.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/devices/{devicesId}/operations",
	//   "httpMethod": "GET",
	//   "id": "androidmanagement.enterprises.devices.operations.list",
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
	//       "pattern": "^enterprises/[^/]+/devices/[^/]+/operations$",
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
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *EnterprisesDevicesOperationsListCall) Pages(ctx context.Context, f func(*ListOperationsResponse) error) error {
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

// method id "androidmanagement.enterprises.enrollmentTokens.create":

type EnterprisesEnrollmentTokensCreateCall struct {
	s               *Service
	parent          string
	enrollmenttoken *EnrollmentToken
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// Create: Creates an enrollment token for a given enterprise.
func (r *EnterprisesEnrollmentTokensService) Create(parent string, enrollmenttoken *EnrollmentToken) *EnterprisesEnrollmentTokensCreateCall {
	c := &EnterprisesEnrollmentTokensCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.enrollmenttoken = enrollmenttoken
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesEnrollmentTokensCreateCall) Fields(s ...googleapi.Field) *EnterprisesEnrollmentTokensCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesEnrollmentTokensCreateCall) Context(ctx context.Context) *EnterprisesEnrollmentTokensCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesEnrollmentTokensCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesEnrollmentTokensCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.enrollmenttoken)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/enrollmentTokens")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidmanagement.enterprises.enrollmentTokens.create" call.
// Exactly one of *EnrollmentToken or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *EnrollmentToken.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesEnrollmentTokensCreateCall) Do(opts ...googleapi.CallOption) (*EnrollmentToken, error) {
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
	ret := &EnrollmentToken{
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
	//   "description": "Creates an enrollment token for a given enterprise.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/enrollmentTokens",
	//   "httpMethod": "POST",
	//   "id": "androidmanagement.enterprises.enrollmentTokens.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "The name of the enterprise in the form enterprises/{enterpriseId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/enrollmentTokens",
	//   "request": {
	//     "$ref": "EnrollmentToken"
	//   },
	//   "response": {
	//     "$ref": "EnrollmentToken"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.enrollmentTokens.delete":

type EnterprisesEnrollmentTokensDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes an enrollment token, which prevents future use of the
// token.
func (r *EnterprisesEnrollmentTokensService) Delete(name string) *EnterprisesEnrollmentTokensDeleteCall {
	c := &EnterprisesEnrollmentTokensDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesEnrollmentTokensDeleteCall) Fields(s ...googleapi.Field) *EnterprisesEnrollmentTokensDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesEnrollmentTokensDeleteCall) Context(ctx context.Context) *EnterprisesEnrollmentTokensDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesEnrollmentTokensDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesEnrollmentTokensDeleteCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "androidmanagement.enterprises.enrollmentTokens.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EnterprisesEnrollmentTokensDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes an enrollment token, which prevents future use of the token.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/enrollmentTokens/{enrollmentTokensId}",
	//   "httpMethod": "DELETE",
	//   "id": "androidmanagement.enterprises.enrollmentTokens.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the enrollment token in the form enterprises/{enterpriseId}/enrollmentTokens/{enrollmentTokenId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/enrollmentTokens/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.policies.delete":

type EnterprisesPoliciesDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a policy. This operation is only permitted if no
// devices are currently referencing the policy.
func (r *EnterprisesPoliciesService) Delete(name string) *EnterprisesPoliciesDeleteCall {
	c := &EnterprisesPoliciesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesPoliciesDeleteCall) Fields(s ...googleapi.Field) *EnterprisesPoliciesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesPoliciesDeleteCall) Context(ctx context.Context) *EnterprisesPoliciesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesPoliciesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesPoliciesDeleteCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "androidmanagement.enterprises.policies.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EnterprisesPoliciesDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes a policy. This operation is only permitted if no devices are currently referencing the policy.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/policies/{policiesId}",
	//   "httpMethod": "DELETE",
	//   "id": "androidmanagement.enterprises.policies.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the policy in the form enterprises/{enterpriseId}/policies/{policyId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/policies/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.policies.get":

type EnterprisesPoliciesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets a policy.
func (r *EnterprisesPoliciesService) Get(name string) *EnterprisesPoliciesGetCall {
	c := &EnterprisesPoliciesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesPoliciesGetCall) Fields(s ...googleapi.Field) *EnterprisesPoliciesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesPoliciesGetCall) IfNoneMatch(entityTag string) *EnterprisesPoliciesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesPoliciesGetCall) Context(ctx context.Context) *EnterprisesPoliciesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesPoliciesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesPoliciesGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "androidmanagement.enterprises.policies.get" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EnterprisesPoliciesGetCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Gets a policy.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/policies/{policiesId}",
	//   "httpMethod": "GET",
	//   "id": "androidmanagement.enterprises.policies.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the policy in the form enterprises/{enterpriseId}/policies/{policyId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/policies/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.policies.list":

type EnterprisesPoliciesListCall struct {
	s            *Service
	parent       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists policies for a given enterprise.
func (r *EnterprisesPoliciesService) List(parent string) *EnterprisesPoliciesListCall {
	c := &EnterprisesPoliciesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	return c
}

// PageSize sets the optional parameter "pageSize": The requested page
// size. The actual page size may be fixed to a min or max value.
func (c *EnterprisesPoliciesListCall) PageSize(pageSize int64) *EnterprisesPoliciesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return.
func (c *EnterprisesPoliciesListCall) PageToken(pageToken string) *EnterprisesPoliciesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesPoliciesListCall) Fields(s ...googleapi.Field) *EnterprisesPoliciesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *EnterprisesPoliciesListCall) IfNoneMatch(entityTag string) *EnterprisesPoliciesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesPoliciesListCall) Context(ctx context.Context) *EnterprisesPoliciesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesPoliciesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesPoliciesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/policies")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidmanagement.enterprises.policies.list" call.
// Exactly one of *ListPoliciesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListPoliciesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *EnterprisesPoliciesListCall) Do(opts ...googleapi.CallOption) (*ListPoliciesResponse, error) {
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
	ret := &ListPoliciesResponse{
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
	//   "description": "Lists policies for a given enterprise.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/policies",
	//   "httpMethod": "GET",
	//   "id": "androidmanagement.enterprises.policies.list",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "The requested page size. The actual page size may be fixed to a min or max value.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "parent": {
	//       "description": "The name of the enterprise in the form enterprises/{enterpriseId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/policies",
	//   "response": {
	//     "$ref": "ListPoliciesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *EnterprisesPoliciesListCall) Pages(ctx context.Context, f func(*ListPoliciesResponse) error) error {
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

// method id "androidmanagement.enterprises.policies.patch":

type EnterprisesPoliciesPatchCall struct {
	s          *Service
	name       string
	policy     *Policy
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates or creates a policy.
func (r *EnterprisesPoliciesService) Patch(name string, policy *Policy) *EnterprisesPoliciesPatchCall {
	c := &EnterprisesPoliciesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.policy = policy
	return c
}

// UpdateMask sets the optional parameter "updateMask": The field mask
// indicating the fields to update. If not set, all modifiable fields
// will be modified.
func (c *EnterprisesPoliciesPatchCall) UpdateMask(updateMask string) *EnterprisesPoliciesPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesPoliciesPatchCall) Fields(s ...googleapi.Field) *EnterprisesPoliciesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesPoliciesPatchCall) Context(ctx context.Context) *EnterprisesPoliciesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesPoliciesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesPoliciesPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.policy)
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
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidmanagement.enterprises.policies.patch" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *EnterprisesPoliciesPatchCall) Do(opts ...googleapi.CallOption) (*Policy, error) {
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
	//   "description": "Updates or creates a policy.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/policies/{policiesId}",
	//   "httpMethod": "PATCH",
	//   "id": "androidmanagement.enterprises.policies.patch",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the policy in the form enterprises/{enterpriseId}/policies/{policyId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+/policies/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "The field mask indicating the fields to update. If not set, all modifiable fields will be modified.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "request": {
	//     "$ref": "Policy"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.enterprises.webTokens.create":

type EnterprisesWebTokensCreateCall struct {
	s          *Service
	parent     string
	webtoken   *WebToken
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates a web token to access an embeddable managed Google
// Play web UI for a given enterprise.
func (r *EnterprisesWebTokensService) Create(parent string, webtoken *WebToken) *EnterprisesWebTokensCreateCall {
	c := &EnterprisesWebTokensCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.webtoken = webtoken
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *EnterprisesWebTokensCreateCall) Fields(s ...googleapi.Field) *EnterprisesWebTokensCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *EnterprisesWebTokensCreateCall) Context(ctx context.Context) *EnterprisesWebTokensCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *EnterprisesWebTokensCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *EnterprisesWebTokensCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.webtoken)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/webTokens")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidmanagement.enterprises.webTokens.create" call.
// Exactly one of *WebToken or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *WebToken.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *EnterprisesWebTokensCreateCall) Do(opts ...googleapi.CallOption) (*WebToken, error) {
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
	ret := &WebToken{
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
	//   "description": "Creates a web token to access an embeddable managed Google Play web UI for a given enterprise.",
	//   "flatPath": "v1/enterprises/{enterprisesId}/webTokens",
	//   "httpMethod": "POST",
	//   "id": "androidmanagement.enterprises.webTokens.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "The name of the enterprise in the form enterprises/{enterpriseId}",
	//       "location": "path",
	//       "pattern": "^enterprises/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/webTokens",
	//   "request": {
	//     "$ref": "WebToken"
	//   },
	//   "response": {
	//     "$ref": "WebToken"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}

// method id "androidmanagement.signupUrls.create":

type SignupUrlsCreateCall struct {
	s          *Service
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Creates an enterprise signup URL.
func (r *SignupUrlsService) Create() *SignupUrlsCreateCall {
	c := &SignupUrlsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// CallbackUrl sets the optional parameter "callbackUrl": The callback
// URL to which the admin will be redirected after successfully creating
// an enterprise. Before redirecting there the system will add a query
// parameter to this URL named enterpriseToken which will contain an
// opaque token to be used for the create enterprise request. The URL
// will be parsed then reformatted in order to add the enterpriseToken
// parameter, so there may be some minor formatting changes.
func (c *SignupUrlsCreateCall) CallbackUrl(callbackUrl string) *SignupUrlsCreateCall {
	c.urlParams_.Set("callbackUrl", callbackUrl)
	return c
}

// ProjectId sets the optional parameter "projectId": The id of the
// Google Cloud Platform project which will own the enterprise.
func (c *SignupUrlsCreateCall) ProjectId(projectId string) *SignupUrlsCreateCall {
	c.urlParams_.Set("projectId", projectId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SignupUrlsCreateCall) Fields(s ...googleapi.Field) *SignupUrlsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SignupUrlsCreateCall) Context(ctx context.Context) *SignupUrlsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SignupUrlsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SignupUrlsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/signupUrls")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androidmanagement.signupUrls.create" call.
// Exactly one of *SignupUrl or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *SignupUrl.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SignupUrlsCreateCall) Do(opts ...googleapi.CallOption) (*SignupUrl, error) {
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
	ret := &SignupUrl{
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
	//   "description": "Creates an enterprise signup URL.",
	//   "flatPath": "v1/signupUrls",
	//   "httpMethod": "POST",
	//   "id": "androidmanagement.signupUrls.create",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "callbackUrl": {
	//       "description": "The callback URL to which the admin will be redirected after successfully creating an enterprise. Before redirecting there the system will add a query parameter to this URL named enterpriseToken which will contain an opaque token to be used for the create enterprise request. The URL will be parsed then reformatted in order to add the enterpriseToken parameter, so there may be some minor formatting changes.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The id of the Google Cloud Platform project which will own the enterprise.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/signupUrls",
	//   "response": {
	//     "$ref": "SignupUrl"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/androidmanagement"
	//   ]
	// }

}
