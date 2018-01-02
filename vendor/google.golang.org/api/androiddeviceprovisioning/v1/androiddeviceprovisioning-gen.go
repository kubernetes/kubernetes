// Package androiddeviceprovisioning provides access to the Android Device Provisioning Partner API.
//
// See https://developers.google.com/zero-touch/
//
// Usage example:
//
//   import "google.golang.org/api/androiddeviceprovisioning/v1"
//   ...
//   androiddeviceprovisioningService, err := androiddeviceprovisioning.New(oauthHttpClient)
package androiddeviceprovisioning // import "google.golang.org/api/androiddeviceprovisioning/v1"

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

const apiId = "androiddeviceprovisioning:v1"
const apiName = "androiddeviceprovisioning"
const apiVersion = "v1"
const basePath = "https://androiddeviceprovisioning.googleapis.com/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Operations = NewOperationsService(s)
	s.Partners = NewPartnersService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Operations *OperationsService

	Partners *PartnersService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewOperationsService(s *Service) *OperationsService {
	rs := &OperationsService{s: s}
	return rs
}

type OperationsService struct {
	s *Service
}

func NewPartnersService(s *Service) *PartnersService {
	rs := &PartnersService{s: s}
	rs.Customers = NewPartnersCustomersService(s)
	rs.Devices = NewPartnersDevicesService(s)
	return rs
}

type PartnersService struct {
	s *Service

	Customers *PartnersCustomersService

	Devices *PartnersDevicesService
}

func NewPartnersCustomersService(s *Service) *PartnersCustomersService {
	rs := &PartnersCustomersService{s: s}
	return rs
}

type PartnersCustomersService struct {
	s *Service
}

func NewPartnersDevicesService(s *Service) *PartnersDevicesService {
	rs := &PartnersDevicesService{s: s}
	return rs
}

type PartnersDevicesService struct {
	s *Service
}

// ClaimDeviceRequest: Request message to claim a device on behalf of a
// customer.
type ClaimDeviceRequest struct {
	// CustomerId: The customer to claim for.
	CustomerId int64 `json:"customerId,omitempty,string"`

	// DeviceIdentifier: The device identifier of the device to claim.
	DeviceIdentifier *DeviceIdentifier `json:"deviceIdentifier,omitempty"`

	// SectionType: The section to claim.
	//
	// Possible values:
	//   "SECTION_TYPE_UNSPECIFIED" - Unspecified section type.
	//   "SECTION_TYPE_ZERO_TOUCH" - Zero touch section type.
	SectionType string `json:"sectionType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CustomerId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CustomerId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ClaimDeviceRequest) MarshalJSON() ([]byte, error) {
	type noMethod ClaimDeviceRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClaimDeviceResponse: Response message containing device id of the
// claim.
type ClaimDeviceResponse struct {
	// DeviceId: The device ID of the claimed device.
	DeviceId int64 `json:"deviceId,omitempty,string"`

	// DeviceName: The resource name of the device in the
	// format
	// `partners/[PARTNER_ID]/devices/[DEVICE_ID]`.
	DeviceName string `json:"deviceName,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *ClaimDeviceResponse) MarshalJSON() ([]byte, error) {
	type noMethod ClaimDeviceResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClaimDevicesRequest: Request to claim devices asynchronously in
// batch.
type ClaimDevicesRequest struct {
	// Claims: List of claims.
	Claims []*PartnerClaim `json:"claims,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Claims") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Claims") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ClaimDevicesRequest) MarshalJSON() ([]byte, error) {
	type noMethod ClaimDevicesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Company: Company
type Company struct {
	// AdminEmails: Admin emails.
	// Admins are able to operate on the portal.
	// This field is a write-only field at creation time.
	AdminEmails []string `json:"adminEmails,omitempty"`

	// CompanyId: Company ID.
	CompanyId int64 `json:"companyId,omitempty,string"`

	// CompanyName: Company name.
	CompanyName string `json:"companyName,omitempty"`

	// Name: The API resource name of the company in the
	// format
	// `partners/[PARTNER_ID]/customers/[CUSTOMER_ID]`.
	Name string `json:"name,omitempty"`

	// OwnerEmails: Owner emails.
	// Owners are able to operate on the portal, and modify admins or
	// other
	// owners. This field is a write-only field at creation time.
	OwnerEmails []string `json:"ownerEmails,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AdminEmails") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AdminEmails") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Company) MarshalJSON() ([]byte, error) {
	type noMethod Company
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateCustomerRequest: Request message to create a customer.
type CreateCustomerRequest struct {
	// Customer: The customer to create.
	Customer *Company `json:"customer,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Customer") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Customer") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateCustomerRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateCustomerRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Device: An Android device.
type Device struct {
	// Claims: Claims.
	Claims []*DeviceClaim `json:"claims,omitempty"`

	// Configuration: The resource name of the configuration.
	// Only set for customers.
	Configuration string `json:"configuration,omitempty"`

	// DeviceId: Device ID.
	DeviceId int64 `json:"deviceId,omitempty,string"`

	// DeviceIdentifier: Device identifier.
	DeviceIdentifier *DeviceIdentifier `json:"deviceIdentifier,omitempty"`

	// DeviceMetadata: Device metadata.
	DeviceMetadata *DeviceMetadata `json:"deviceMetadata,omitempty"`

	// Name: Resource name in `partners/[PARTNER_ID]/devices/[DEVICE_ID]`.
	Name string `json:"name,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Claims") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Claims") to include in API
	// requests with the JSON null value. By default, fields with empty
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

// DeviceClaim: Information about a device claimed for a partner.
type DeviceClaim struct {
	// OwnerCompanyId: Owner ID.
	OwnerCompanyId int64 `json:"ownerCompanyId,omitempty,string"`

	// SectionType: Section type of the device claim.
	//
	// Possible values:
	//   "SECTION_TYPE_UNSPECIFIED" - Unspecified section type.
	//   "SECTION_TYPE_ZERO_TOUCH" - Zero touch section type.
	SectionType string `json:"sectionType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OwnerCompanyId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OwnerCompanyId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DeviceClaim) MarshalJSON() ([]byte, error) {
	type noMethod DeviceClaim
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeviceIdentifier: Identifies a unique device.
type DeviceIdentifier struct {
	// Imei: IMEI number.
	Imei string `json:"imei,omitempty"`

	// Manufacturer: Manufacturer name to match
	// `android.os.Build.MANUFACTURER` (required).
	// Allowed values listed in
	// [manufacturer names](/zero-touch/resources/manufacturer-names).
	Manufacturer string `json:"manufacturer,omitempty"`

	// Meid: MEID number.
	Meid string `json:"meid,omitempty"`

	// SerialNumber: Serial number (optional).
	SerialNumber string `json:"serialNumber,omitempty"`

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

func (s *DeviceIdentifier) MarshalJSON() ([]byte, error) {
	type noMethod DeviceIdentifier
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeviceMetadata: A key-value pair of the device metadata.
type DeviceMetadata struct {
	// Entries: Metadata entries
	Entries map[string]string `json:"entries,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Entries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Entries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeviceMetadata) MarshalJSON() ([]byte, error) {
	type noMethod DeviceMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DevicesLongRunningOperationMetadata: Long running operation metadata.
type DevicesLongRunningOperationMetadata struct {
	// DevicesCount: Number of devices parsed in your requests.
	DevicesCount int64 `json:"devicesCount,omitempty"`

	// ProcessingStatus: The overall processing status.
	//
	// Possible values:
	//   "BATCH_PROCESS_STATUS_UNSPECIFIED" - Invalid code. Shouldn't be
	// used.
	//   "BATCH_PROCESS_PENDING" - Pending.
	//   "BATCH_PROCESS_IN_PROGRESS" - In progress.
	//   "BATCH_PROCESS_PROCESSED" - Processed.
	// This doesn't mean all items were processed sucessfully, you
	// should
	// check the `response` field for the result of every item.
	ProcessingStatus string `json:"processingStatus,omitempty"`

	// Progress: Processing progress from 0 to 100.
	Progress int64 `json:"progress,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DevicesCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DevicesCount") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DevicesLongRunningOperationMetadata) MarshalJSON() ([]byte, error) {
	type noMethod DevicesLongRunningOperationMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DevicesLongRunningOperationResponse: Long running operation response.
type DevicesLongRunningOperationResponse struct {
	// PerDeviceStatus: Processing status for each device.
	// One `PerDeviceStatus` per device. The order is the same as in your
	// requests.
	PerDeviceStatus []*OperationPerDevice `json:"perDeviceStatus,omitempty"`

	// SuccessCount: Number of succeesfully processed ones.
	SuccessCount int64 `json:"successCount,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PerDeviceStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PerDeviceStatus") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DevicesLongRunningOperationResponse) MarshalJSON() ([]byte, error) {
	type noMethod DevicesLongRunningOperationResponse
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

// FindDevicesByDeviceIdentifierRequest: Request to find devices.
type FindDevicesByDeviceIdentifierRequest struct {
	// DeviceIdentifier: The device identifier to search.
	DeviceIdentifier *DeviceIdentifier `json:"deviceIdentifier,omitempty"`

	// Limit: Number of devices to show.
	Limit int64 `json:"limit,omitempty,string"`

	// PageToken: Page token.
	PageToken string `json:"pageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DeviceIdentifier") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DeviceIdentifier") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *FindDevicesByDeviceIdentifierRequest) MarshalJSON() ([]byte, error) {
	type noMethod FindDevicesByDeviceIdentifierRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FindDevicesByDeviceIdentifierResponse: Response containing found
// devices.
type FindDevicesByDeviceIdentifierResponse struct {
	// Devices: Found devices.
	Devices []*Device `json:"devices,omitempty"`

	// NextPageToken: Page token of the next page.
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

func (s *FindDevicesByDeviceIdentifierResponse) MarshalJSON() ([]byte, error) {
	type noMethod FindDevicesByDeviceIdentifierResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FindDevicesByOwnerRequest: Request to find devices by customers.
type FindDevicesByOwnerRequest struct {
	// CustomerId: List of customer IDs to search for.
	CustomerId googleapi.Int64s `json:"customerId,omitempty"`

	// Limit: The number of devices to show in the result.
	Limit int64 `json:"limit,omitempty,string"`

	// PageToken: Page token.
	PageToken string `json:"pageToken,omitempty"`

	// SectionType: The section type.
	//
	// Possible values:
	//   "SECTION_TYPE_UNSPECIFIED" - Unspecified section type.
	//   "SECTION_TYPE_ZERO_TOUCH" - Zero touch section type.
	SectionType string `json:"sectionType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CustomerId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CustomerId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FindDevicesByOwnerRequest) MarshalJSON() ([]byte, error) {
	type noMethod FindDevicesByOwnerRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FindDevicesByOwnerResponse: Response containing found devices.
type FindDevicesByOwnerResponse struct {
	// Devices: Devices found.
	Devices []*Device `json:"devices,omitempty"`

	// NextPageToken: Page token of the next page.
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

func (s *FindDevicesByOwnerResponse) MarshalJSON() ([]byte, error) {
	type noMethod FindDevicesByOwnerResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListCustomersResponse: Response message of all customers related to
// this partner.
type ListCustomersResponse struct {
	// Customers: List of customers related to this partner.
	Customers []*Company `json:"customers,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Customers") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Customers") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListCustomersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListCustomersResponse
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

	// Error: This field will always be not set if the operation is created
	// by `claimAsync`, `unclaimAsync`, or `updateMetadataAsync`. In this
	// case, error information for each device is set in
	// `response.perDeviceStatus.result.status`.
	Error *Status `json:"error,omitempty"`

	// Metadata: This field will contain a
	// `DevicesLongRunningOperationMetadata` object if the operation is
	// created by `claimAsync`, `unclaimAsync`, or `updateMetadataAsync`.
	Metadata googleapi.RawMessage `json:"metadata,omitempty"`

	// Name: The server-assigned name, which is only unique within the same
	// service that
	// originally returns it. If you use the default HTTP mapping,
	// the
	// `name` should have the format of `operations/some/unique/name`.
	Name string `json:"name,omitempty"`

	// Response: This field will contain a
	// `DevicesLongRunningOperationResponse` object if the operation is
	// created by `claimAsync`, `unclaimAsync`, or `updateMetadataAsync`.
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

// OperationPerDevice: Operation the server received for every device.
type OperationPerDevice struct {
	// Claim: Request to claim a device.
	Claim *PartnerClaim `json:"claim,omitempty"`

	// Result: Processing result for every device.
	Result *PerDeviceStatusInBatch `json:"result,omitempty"`

	// Unclaim: Request to unclaim a device.
	Unclaim *PartnerUnclaim `json:"unclaim,omitempty"`

	// UpdateMetadata: Request to set metadata for a device.
	UpdateMetadata *UpdateMetadataArguments `json:"updateMetadata,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Claim") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Claim") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationPerDevice) MarshalJSON() ([]byte, error) {
	type noMethod OperationPerDevice
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PartnerClaim: Identifies one claim request.
type PartnerClaim struct {
	// CustomerId: Customer ID to claim for.
	CustomerId int64 `json:"customerId,omitempty,string"`

	// DeviceIdentifier: Device identifier of the device.
	DeviceIdentifier *DeviceIdentifier `json:"deviceIdentifier,omitempty"`

	// DeviceMetadata: Metadata to set at claim.
	DeviceMetadata *DeviceMetadata `json:"deviceMetadata,omitempty"`

	// SectionType: Section type to claim.
	//
	// Possible values:
	//   "SECTION_TYPE_UNSPECIFIED" - Unspecified section type.
	//   "SECTION_TYPE_ZERO_TOUCH" - Zero touch section type.
	SectionType string `json:"sectionType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CustomerId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CustomerId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PartnerClaim) MarshalJSON() ([]byte, error) {
	type noMethod PartnerClaim
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PartnerUnclaim: Identifies one unclaim request.
type PartnerUnclaim struct {
	// DeviceId: Device ID of the device.
	DeviceId int64 `json:"deviceId,omitempty,string"`

	// DeviceIdentifier: Device identifier of the device.
	DeviceIdentifier *DeviceIdentifier `json:"deviceIdentifier,omitempty"`

	// SectionType: Section type to unclaim.
	//
	// Possible values:
	//   "SECTION_TYPE_UNSPECIFIED" - Unspecified section type.
	//   "SECTION_TYPE_ZERO_TOUCH" - Zero touch section type.
	SectionType string `json:"sectionType,omitempty"`

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

func (s *PartnerUnclaim) MarshalJSON() ([]byte, error) {
	type noMethod PartnerUnclaim
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PerDeviceStatusInBatch: Stores the processing result for each device.
type PerDeviceStatusInBatch struct {
	// DeviceId: Device ID of the device if process succeeds.
	DeviceId int64 `json:"deviceId,omitempty,string"`

	// ErrorIdentifier: Error identifier.
	ErrorIdentifier string `json:"errorIdentifier,omitempty"`

	// ErrorMessage: Error message.
	ErrorMessage string `json:"errorMessage,omitempty"`

	// Status: Process result.
	//
	// Possible values:
	//   "SINGLE_DEVICE_STATUS_UNSPECIFIED" - Invalid code. Shouldn't be
	// used.
	//   "SINGLE_DEVICE_STATUS_UNKNOWN_ERROR" - Unknown error.
	// We don't expect this error to occur here.
	//   "SINGLE_DEVICE_STATUS_OTHER_ERROR" - Other error.
	// We know/expect this error, but there's no defined error code for
	// the
	// error.
	//   "SINGLE_DEVICE_STATUS_SUCCESS" - Success.
	//   "SINGLE_DEVICE_STATUS_PERMISSION_DENIED" - Permission denied.
	//   "SINGLE_DEVICE_STATUS_INVALID_DEVICE_IDENTIFIER" - Invalid device
	// identifier.
	//   "SINGLE_DEVICE_STATUS_INVALID_SECTION_TYPE" - Invalid section type.
	//   "SINGLE_DEVICE_STATUS_SECTION_NOT_YOURS" - This section is claimed
	// by another company.
	Status string `json:"status,omitempty"`

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

func (s *PerDeviceStatusInBatch) MarshalJSON() ([]byte, error) {
	type noMethod PerDeviceStatusInBatch
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

// UnclaimDeviceRequest: Request message to unclaim a device.
type UnclaimDeviceRequest struct {
	// DeviceId: The device ID returned by `ClaimDevice`.
	DeviceId int64 `json:"deviceId,omitempty,string"`

	// DeviceIdentifier: The device identifier you used when you claimed
	// this device.
	DeviceIdentifier *DeviceIdentifier `json:"deviceIdentifier,omitempty"`

	// SectionType: The section type to unclaim for.
	//
	// Possible values:
	//   "SECTION_TYPE_UNSPECIFIED" - Unspecified section type.
	//   "SECTION_TYPE_ZERO_TOUCH" - Zero touch section type.
	SectionType string `json:"sectionType,omitempty"`

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

func (s *UnclaimDeviceRequest) MarshalJSON() ([]byte, error) {
	type noMethod UnclaimDeviceRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UnclaimDevicesRequest: Request to unclaim devices asynchronously in
// batch.
type UnclaimDevicesRequest struct {
	// Unclaims: List of devices to unclaim.
	Unclaims []*PartnerUnclaim `json:"unclaims,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Unclaims") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Unclaims") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UnclaimDevicesRequest) MarshalJSON() ([]byte, error) {
	type noMethod UnclaimDevicesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateDeviceMetadataInBatchRequest: Request to update device metadata
// in batch.
type UpdateDeviceMetadataInBatchRequest struct {
	// Updates: List of metadata updates.
	Updates []*UpdateMetadataArguments `json:"updates,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Updates") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Updates") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateDeviceMetadataInBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateDeviceMetadataInBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateDeviceMetadataRequest: Request to set metadata for a device.
type UpdateDeviceMetadataRequest struct {
	// DeviceMetadata: The metdata to set.
	DeviceMetadata *DeviceMetadata `json:"deviceMetadata,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DeviceMetadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DeviceMetadata") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *UpdateDeviceMetadataRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateDeviceMetadataRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateMetadataArguments: Identifies metdata updates to one device.
type UpdateMetadataArguments struct {
	// DeviceId: Device ID of the device.
	DeviceId int64 `json:"deviceId,omitempty,string"`

	// DeviceIdentifier: Device identifier.
	DeviceIdentifier *DeviceIdentifier `json:"deviceIdentifier,omitempty"`

	// DeviceMetadata: The metadata to update.
	DeviceMetadata *DeviceMetadata `json:"deviceMetadata,omitempty"`

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

func (s *UpdateMetadataArguments) MarshalJSON() ([]byte, error) {
	type noMethod UpdateMetadataArguments
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "androiddeviceprovisioning.operations.get":

type OperationsGetCall struct {
	s            *Service
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

// Do executes the "androiddeviceprovisioning.operations.get" call.
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
	//   "id": "androiddeviceprovisioning.operations.get",
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
	//   }
	// }

}

// method id "androiddeviceprovisioning.partners.customers.create":

type PartnersCustomersCreateCall struct {
	s                     *Service
	parent                string
	createcustomerrequest *CreateCustomerRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Create: A customer for zero-touch enrollment will be created.
// After a Customer is created, their admins and owners will be able to
// manage
// devices on partner.android.com/zerotouch or via their API.
func (r *PartnersCustomersService) Create(parent string, createcustomerrequest *CreateCustomerRequest) *PartnersCustomersCreateCall {
	c := &PartnersCustomersCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.parent = parent
	c.createcustomerrequest = createcustomerrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PartnersCustomersCreateCall) Fields(s ...googleapi.Field) *PartnersCustomersCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PartnersCustomersCreateCall) Context(ctx context.Context) *PartnersCustomersCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PartnersCustomersCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PartnersCustomersCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createcustomerrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+parent}/customers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"parent": c.parent,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androiddeviceprovisioning.partners.customers.create" call.
// Exactly one of *Company or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Company.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *PartnersCustomersCreateCall) Do(opts ...googleapi.CallOption) (*Company, error) {
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
	ret := &Company{
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
	//   "description": "A customer for zero-touch enrollment will be created.\nAfter a Customer is created, their admins and owners will be able to manage\ndevices on partner.android.com/zerotouch or via their API.",
	//   "flatPath": "v1/partners/{partnersId}/customers",
	//   "httpMethod": "POST",
	//   "id": "androiddeviceprovisioning.partners.customers.create",
	//   "parameterOrder": [
	//     "parent"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "The parent resource in format `partners/[PARTNER_ID]'.",
	//       "location": "path",
	//       "pattern": "^partners/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+parent}/customers",
	//   "request": {
	//     "$ref": "CreateCustomerRequest"
	//   },
	//   "response": {
	//     "$ref": "Company"
	//   }
	// }

}

// method id "androiddeviceprovisioning.partners.customers.list":

type PartnersCustomersListCall struct {
	s            *Service
	partnerId    int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List the customers that are enrolled to the reseller identified
// by the
// `partnerId` argument. This list includes customers that the
// reseller
// created and customers that enrolled themselves using the portal.
func (r *PartnersCustomersService) List(partnerId int64) *PartnersCustomersListCall {
	c := &PartnersCustomersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.partnerId = partnerId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PartnersCustomersListCall) Fields(s ...googleapi.Field) *PartnersCustomersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PartnersCustomersListCall) IfNoneMatch(entityTag string) *PartnersCustomersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PartnersCustomersListCall) Context(ctx context.Context) *PartnersCustomersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PartnersCustomersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PartnersCustomersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/partners/{+partnerId}/customers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"partnerId": strconv.FormatInt(c.partnerId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androiddeviceprovisioning.partners.customers.list" call.
// Exactly one of *ListCustomersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListCustomersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PartnersCustomersListCall) Do(opts ...googleapi.CallOption) (*ListCustomersResponse, error) {
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
	ret := &ListCustomersResponse{
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
	//   "description": "List the customers that are enrolled to the reseller identified by the\n`partnerId` argument. This list includes customers that the reseller\ncreated and customers that enrolled themselves using the portal.",
	//   "flatPath": "v1/partners/{partnersId}/customers",
	//   "httpMethod": "GET",
	//   "id": "androiddeviceprovisioning.partners.customers.list",
	//   "parameterOrder": [
	//     "partnerId"
	//   ],
	//   "parameters": {
	//     "partnerId": {
	//       "description": "The ID of the partner.",
	//       "format": "int64",
	//       "location": "path",
	//       "pattern": "^[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/partners/{+partnerId}/customers",
	//   "response": {
	//     "$ref": "ListCustomersResponse"
	//   }
	// }

}

// method id "androiddeviceprovisioning.partners.devices.claim":

type PartnersDevicesClaimCall struct {
	s                  *Service
	partnerId          int64
	claimdevicerequest *ClaimDeviceRequest
	urlParams_         gensupport.URLParams
	ctx_               context.Context
	header_            http.Header
}

// Claim: Claim the device identified by device identifier.
func (r *PartnersDevicesService) Claim(partnerId int64, claimdevicerequest *ClaimDeviceRequest) *PartnersDevicesClaimCall {
	c := &PartnersDevicesClaimCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.partnerId = partnerId
	c.claimdevicerequest = claimdevicerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PartnersDevicesClaimCall) Fields(s ...googleapi.Field) *PartnersDevicesClaimCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PartnersDevicesClaimCall) Context(ctx context.Context) *PartnersDevicesClaimCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PartnersDevicesClaimCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PartnersDevicesClaimCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.claimdevicerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/partners/{+partnerId}/devices:claim")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"partnerId": strconv.FormatInt(c.partnerId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androiddeviceprovisioning.partners.devices.claim" call.
// Exactly one of *ClaimDeviceResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ClaimDeviceResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PartnersDevicesClaimCall) Do(opts ...googleapi.CallOption) (*ClaimDeviceResponse, error) {
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
	ret := &ClaimDeviceResponse{
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
	//   "description": "Claim the device identified by device identifier.",
	//   "flatPath": "v1/partners/{partnersId}/devices:claim",
	//   "httpMethod": "POST",
	//   "id": "androiddeviceprovisioning.partners.devices.claim",
	//   "parameterOrder": [
	//     "partnerId"
	//   ],
	//   "parameters": {
	//     "partnerId": {
	//       "description": "ID of the partner.",
	//       "format": "int64",
	//       "location": "path",
	//       "pattern": "^[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/partners/{+partnerId}/devices:claim",
	//   "request": {
	//     "$ref": "ClaimDeviceRequest"
	//   },
	//   "response": {
	//     "$ref": "ClaimDeviceResponse"
	//   }
	// }

}

// method id "androiddeviceprovisioning.partners.devices.claimAsync":

type PartnersDevicesClaimAsyncCall struct {
	s                   *Service
	partnerId           int64
	claimdevicesrequest *ClaimDevicesRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
	header_             http.Header
}

// ClaimAsync: Claim devices asynchronously.
func (r *PartnersDevicesService) ClaimAsync(partnerId int64, claimdevicesrequest *ClaimDevicesRequest) *PartnersDevicesClaimAsyncCall {
	c := &PartnersDevicesClaimAsyncCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.partnerId = partnerId
	c.claimdevicesrequest = claimdevicesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PartnersDevicesClaimAsyncCall) Fields(s ...googleapi.Field) *PartnersDevicesClaimAsyncCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PartnersDevicesClaimAsyncCall) Context(ctx context.Context) *PartnersDevicesClaimAsyncCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PartnersDevicesClaimAsyncCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PartnersDevicesClaimAsyncCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.claimdevicesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/partners/{+partnerId}/devices:claimAsync")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"partnerId": strconv.FormatInt(c.partnerId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androiddeviceprovisioning.partners.devices.claimAsync" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PartnersDevicesClaimAsyncCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Claim devices asynchronously.",
	//   "flatPath": "v1/partners/{partnersId}/devices:claimAsync",
	//   "httpMethod": "POST",
	//   "id": "androiddeviceprovisioning.partners.devices.claimAsync",
	//   "parameterOrder": [
	//     "partnerId"
	//   ],
	//   "parameters": {
	//     "partnerId": {
	//       "description": "Partner ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "pattern": "^[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/partners/{+partnerId}/devices:claimAsync",
	//   "request": {
	//     "$ref": "ClaimDevicesRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   }
	// }

}

// method id "androiddeviceprovisioning.partners.devices.findByIdentifier":

type PartnersDevicesFindByIdentifierCall struct {
	s                                    *Service
	partnerId                            int64
	finddevicesbydeviceidentifierrequest *FindDevicesByDeviceIdentifierRequest
	urlParams_                           gensupport.URLParams
	ctx_                                 context.Context
	header_                              http.Header
}

// FindByIdentifier: Find devices by device identifier.
func (r *PartnersDevicesService) FindByIdentifier(partnerId int64, finddevicesbydeviceidentifierrequest *FindDevicesByDeviceIdentifierRequest) *PartnersDevicesFindByIdentifierCall {
	c := &PartnersDevicesFindByIdentifierCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.partnerId = partnerId
	c.finddevicesbydeviceidentifierrequest = finddevicesbydeviceidentifierrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PartnersDevicesFindByIdentifierCall) Fields(s ...googleapi.Field) *PartnersDevicesFindByIdentifierCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PartnersDevicesFindByIdentifierCall) Context(ctx context.Context) *PartnersDevicesFindByIdentifierCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PartnersDevicesFindByIdentifierCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PartnersDevicesFindByIdentifierCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.finddevicesbydeviceidentifierrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/partners/{+partnerId}/devices:findByIdentifier")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"partnerId": strconv.FormatInt(c.partnerId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androiddeviceprovisioning.partners.devices.findByIdentifier" call.
// Exactly one of *FindDevicesByDeviceIdentifierResponse or error will
// be non-nil. Any non-2xx status code is an error. Response headers are
// in either
// *FindDevicesByDeviceIdentifierResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PartnersDevicesFindByIdentifierCall) Do(opts ...googleapi.CallOption) (*FindDevicesByDeviceIdentifierResponse, error) {
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
	ret := &FindDevicesByDeviceIdentifierResponse{
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
	//   "description": "Find devices by device identifier.",
	//   "flatPath": "v1/partners/{partnersId}/devices:findByIdentifier",
	//   "httpMethod": "POST",
	//   "id": "androiddeviceprovisioning.partners.devices.findByIdentifier",
	//   "parameterOrder": [
	//     "partnerId"
	//   ],
	//   "parameters": {
	//     "partnerId": {
	//       "description": "ID of the partner.",
	//       "format": "int64",
	//       "location": "path",
	//       "pattern": "^[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/partners/{+partnerId}/devices:findByIdentifier",
	//   "request": {
	//     "$ref": "FindDevicesByDeviceIdentifierRequest"
	//   },
	//   "response": {
	//     "$ref": "FindDevicesByDeviceIdentifierResponse"
	//   }
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *PartnersDevicesFindByIdentifierCall) Pages(ctx context.Context, f func(*FindDevicesByDeviceIdentifierResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.finddevicesbydeviceidentifierrequest.PageToken = pt }(c.finddevicesbydeviceidentifierrequest.PageToken) // reset paging to original point
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
		c.finddevicesbydeviceidentifierrequest.PageToken = x.NextPageToken
	}
}

// method id "androiddeviceprovisioning.partners.devices.findByOwner":

type PartnersDevicesFindByOwnerCall struct {
	s                         *Service
	partnerId                 int64
	finddevicesbyownerrequest *FindDevicesByOwnerRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// FindByOwner: Find devices by ownership.
func (r *PartnersDevicesService) FindByOwner(partnerId int64, finddevicesbyownerrequest *FindDevicesByOwnerRequest) *PartnersDevicesFindByOwnerCall {
	c := &PartnersDevicesFindByOwnerCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.partnerId = partnerId
	c.finddevicesbyownerrequest = finddevicesbyownerrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PartnersDevicesFindByOwnerCall) Fields(s ...googleapi.Field) *PartnersDevicesFindByOwnerCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PartnersDevicesFindByOwnerCall) Context(ctx context.Context) *PartnersDevicesFindByOwnerCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PartnersDevicesFindByOwnerCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PartnersDevicesFindByOwnerCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.finddevicesbyownerrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/partners/{+partnerId}/devices:findByOwner")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"partnerId": strconv.FormatInt(c.partnerId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androiddeviceprovisioning.partners.devices.findByOwner" call.
// Exactly one of *FindDevicesByOwnerResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *FindDevicesByOwnerResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PartnersDevicesFindByOwnerCall) Do(opts ...googleapi.CallOption) (*FindDevicesByOwnerResponse, error) {
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
	ret := &FindDevicesByOwnerResponse{
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
	//   "description": "Find devices by ownership.",
	//   "flatPath": "v1/partners/{partnersId}/devices:findByOwner",
	//   "httpMethod": "POST",
	//   "id": "androiddeviceprovisioning.partners.devices.findByOwner",
	//   "parameterOrder": [
	//     "partnerId"
	//   ],
	//   "parameters": {
	//     "partnerId": {
	//       "description": "ID of the partner.",
	//       "format": "int64",
	//       "location": "path",
	//       "pattern": "^[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/partners/{+partnerId}/devices:findByOwner",
	//   "request": {
	//     "$ref": "FindDevicesByOwnerRequest"
	//   },
	//   "response": {
	//     "$ref": "FindDevicesByOwnerResponse"
	//   }
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *PartnersDevicesFindByOwnerCall) Pages(ctx context.Context, f func(*FindDevicesByOwnerResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.finddevicesbyownerrequest.PageToken = pt }(c.finddevicesbyownerrequest.PageToken) // reset paging to original point
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
		c.finddevicesbyownerrequest.PageToken = x.NextPageToken
	}
}

// method id "androiddeviceprovisioning.partners.devices.get":

type PartnersDevicesGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Get a device.
func (r *PartnersDevicesService) Get(name string) *PartnersDevicesGetCall {
	c := &PartnersDevicesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PartnersDevicesGetCall) Fields(s ...googleapi.Field) *PartnersDevicesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PartnersDevicesGetCall) IfNoneMatch(entityTag string) *PartnersDevicesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PartnersDevicesGetCall) Context(ctx context.Context) *PartnersDevicesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PartnersDevicesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PartnersDevicesGetCall) doRequest(alt string) (*http.Response, error) {
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

// Do executes the "androiddeviceprovisioning.partners.devices.get" call.
// Exactly one of *Device or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Device.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *PartnersDevicesGetCall) Do(opts ...googleapi.CallOption) (*Device, error) {
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
	//   "description": "Get a device.",
	//   "flatPath": "v1/partners/{partnersId}/devices/{devicesId}",
	//   "httpMethod": "GET",
	//   "id": "androiddeviceprovisioning.partners.devices.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "Resource name in `partners/[PARTNER_ID]/devices/[DEVICE_ID]`.",
	//       "location": "path",
	//       "pattern": "^partners/[^/]+/devices/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+name}",
	//   "response": {
	//     "$ref": "Device"
	//   }
	// }

}

// method id "androiddeviceprovisioning.partners.devices.metadata":

type PartnersDevicesMetadataCall struct {
	s                           *Service
	metadataOwnerId             int64
	deviceId                    int64
	updatedevicemetadatarequest *UpdateDeviceMetadataRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
	header_                     http.Header
}

// Metadata: Update the metadata.
func (r *PartnersDevicesService) Metadata(metadataOwnerId int64, deviceId int64, updatedevicemetadatarequest *UpdateDeviceMetadataRequest) *PartnersDevicesMetadataCall {
	c := &PartnersDevicesMetadataCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.metadataOwnerId = metadataOwnerId
	c.deviceId = deviceId
	c.updatedevicemetadatarequest = updatedevicemetadatarequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PartnersDevicesMetadataCall) Fields(s ...googleapi.Field) *PartnersDevicesMetadataCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PartnersDevicesMetadataCall) Context(ctx context.Context) *PartnersDevicesMetadataCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PartnersDevicesMetadataCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PartnersDevicesMetadataCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.updatedevicemetadatarequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/partners/{+metadataOwnerId}/devices/{+deviceId}/metadata")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"metadataOwnerId": strconv.FormatInt(c.metadataOwnerId, 10),
		"deviceId":        strconv.FormatInt(c.deviceId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androiddeviceprovisioning.partners.devices.metadata" call.
// Exactly one of *DeviceMetadata or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *DeviceMetadata.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PartnersDevicesMetadataCall) Do(opts ...googleapi.CallOption) (*DeviceMetadata, error) {
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
	ret := &DeviceMetadata{
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
	//   "description": "Update the metadata.",
	//   "flatPath": "v1/partners/{partnersId}/devices/{devicesId}/metadata",
	//   "httpMethod": "POST",
	//   "id": "androiddeviceprovisioning.partners.devices.metadata",
	//   "parameterOrder": [
	//     "metadataOwnerId",
	//     "deviceId"
	//   ],
	//   "parameters": {
	//     "deviceId": {
	//       "description": "ID of the partner.",
	//       "format": "int64",
	//       "location": "path",
	//       "pattern": "^[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "metadataOwnerId": {
	//       "description": "The owner of the newly set metadata. Set this to the partner ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "pattern": "^[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/partners/{+metadataOwnerId}/devices/{+deviceId}/metadata",
	//   "request": {
	//     "$ref": "UpdateDeviceMetadataRequest"
	//   },
	//   "response": {
	//     "$ref": "DeviceMetadata"
	//   }
	// }

}

// method id "androiddeviceprovisioning.partners.devices.unclaim":

type PartnersDevicesUnclaimCall struct {
	s                    *Service
	partnerId            int64
	unclaimdevicerequest *UnclaimDeviceRequest
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
	header_              http.Header
}

// Unclaim: Unclaim the device identified by the `device_id` or the
// `deviceIdentifier`.
func (r *PartnersDevicesService) Unclaim(partnerId int64, unclaimdevicerequest *UnclaimDeviceRequest) *PartnersDevicesUnclaimCall {
	c := &PartnersDevicesUnclaimCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.partnerId = partnerId
	c.unclaimdevicerequest = unclaimdevicerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PartnersDevicesUnclaimCall) Fields(s ...googleapi.Field) *PartnersDevicesUnclaimCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PartnersDevicesUnclaimCall) Context(ctx context.Context) *PartnersDevicesUnclaimCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PartnersDevicesUnclaimCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PartnersDevicesUnclaimCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.unclaimdevicerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/partners/{+partnerId}/devices:unclaim")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"partnerId": strconv.FormatInt(c.partnerId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androiddeviceprovisioning.partners.devices.unclaim" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *PartnersDevicesUnclaimCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Unclaim the device identified by the `device_id` or the `deviceIdentifier`.",
	//   "flatPath": "v1/partners/{partnersId}/devices:unclaim",
	//   "httpMethod": "POST",
	//   "id": "androiddeviceprovisioning.partners.devices.unclaim",
	//   "parameterOrder": [
	//     "partnerId"
	//   ],
	//   "parameters": {
	//     "partnerId": {
	//       "description": "ID of the partner.",
	//       "format": "int64",
	//       "location": "path",
	//       "pattern": "^[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/partners/{+partnerId}/devices:unclaim",
	//   "request": {
	//     "$ref": "UnclaimDeviceRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   }
	// }

}

// method id "androiddeviceprovisioning.partners.devices.unclaimAsync":

type PartnersDevicesUnclaimAsyncCall struct {
	s                     *Service
	partnerId             int64
	unclaimdevicesrequest *UnclaimDevicesRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// UnclaimAsync: Unclaim devices asynchronously.
func (r *PartnersDevicesService) UnclaimAsync(partnerId int64, unclaimdevicesrequest *UnclaimDevicesRequest) *PartnersDevicesUnclaimAsyncCall {
	c := &PartnersDevicesUnclaimAsyncCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.partnerId = partnerId
	c.unclaimdevicesrequest = unclaimdevicesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PartnersDevicesUnclaimAsyncCall) Fields(s ...googleapi.Field) *PartnersDevicesUnclaimAsyncCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PartnersDevicesUnclaimAsyncCall) Context(ctx context.Context) *PartnersDevicesUnclaimAsyncCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PartnersDevicesUnclaimAsyncCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PartnersDevicesUnclaimAsyncCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.unclaimdevicesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/partners/{+partnerId}/devices:unclaimAsync")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"partnerId": strconv.FormatInt(c.partnerId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androiddeviceprovisioning.partners.devices.unclaimAsync" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PartnersDevicesUnclaimAsyncCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Unclaim devices asynchronously.",
	//   "flatPath": "v1/partners/{partnersId}/devices:unclaimAsync",
	//   "httpMethod": "POST",
	//   "id": "androiddeviceprovisioning.partners.devices.unclaimAsync",
	//   "parameterOrder": [
	//     "partnerId"
	//   ],
	//   "parameters": {
	//     "partnerId": {
	//       "description": "Partner ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "pattern": "^[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/partners/{+partnerId}/devices:unclaimAsync",
	//   "request": {
	//     "$ref": "UnclaimDevicesRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   }
	// }

}

// method id "androiddeviceprovisioning.partners.devices.updateMetadataAsync":

type PartnersDevicesUpdateMetadataAsyncCall struct {
	s                                  *Service
	partnerId                          int64
	updatedevicemetadatainbatchrequest *UpdateDeviceMetadataInBatchRequest
	urlParams_                         gensupport.URLParams
	ctx_                               context.Context
	header_                            http.Header
}

// UpdateMetadataAsync: Set metadata in batch asynchronously.
func (r *PartnersDevicesService) UpdateMetadataAsync(partnerId int64, updatedevicemetadatainbatchrequest *UpdateDeviceMetadataInBatchRequest) *PartnersDevicesUpdateMetadataAsyncCall {
	c := &PartnersDevicesUpdateMetadataAsyncCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.partnerId = partnerId
	c.updatedevicemetadatainbatchrequest = updatedevicemetadatainbatchrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PartnersDevicesUpdateMetadataAsyncCall) Fields(s ...googleapi.Field) *PartnersDevicesUpdateMetadataAsyncCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PartnersDevicesUpdateMetadataAsyncCall) Context(ctx context.Context) *PartnersDevicesUpdateMetadataAsyncCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PartnersDevicesUpdateMetadataAsyncCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PartnersDevicesUpdateMetadataAsyncCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.updatedevicemetadatainbatchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/partners/{+partnerId}/devices:updateMetadataAsync")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"partnerId": strconv.FormatInt(c.partnerId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "androiddeviceprovisioning.partners.devices.updateMetadataAsync" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PartnersDevicesUpdateMetadataAsyncCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Set metadata in batch asynchronously.",
	//   "flatPath": "v1/partners/{partnersId}/devices:updateMetadataAsync",
	//   "httpMethod": "POST",
	//   "id": "androiddeviceprovisioning.partners.devices.updateMetadataAsync",
	//   "parameterOrder": [
	//     "partnerId"
	//   ],
	//   "parameters": {
	//     "partnerId": {
	//       "description": "Partner ID.",
	//       "format": "int64",
	//       "location": "path",
	//       "pattern": "^[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/partners/{+partnerId}/devices:updateMetadataAsync",
	//   "request": {
	//     "$ref": "UpdateDeviceMetadataInBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   }
	// }

}
