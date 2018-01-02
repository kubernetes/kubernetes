// Package people provides access to the Google People API.
//
// See https://developers.google.com/people/
//
// Usage example:
//
//   import "google.golang.org/api/people/v1"
//   ...
//   peopleService, err := people.New(oauthHttpClient)
package people // import "google.golang.org/api/people/v1"

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

const apiId = "people:v1"
const apiName = "people"
const apiVersion = "v1"
const basePath = "https://people.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// Manage your contacts
	ContactsScope = "https://www.googleapis.com/auth/contacts"

	// View your contacts
	ContactsReadonlyScope = "https://www.googleapis.com/auth/contacts.readonly"

	// Know the list of people in your circles, your age range, and language
	PlusLoginScope = "https://www.googleapis.com/auth/plus.login"

	// View your street addresses
	UserAddressesReadScope = "https://www.googleapis.com/auth/user.addresses.read"

	// View your complete date of birth
	UserBirthdayReadScope = "https://www.googleapis.com/auth/user.birthday.read"

	// View your email addresses
	UserEmailsReadScope = "https://www.googleapis.com/auth/user.emails.read"

	// View your phone numbers
	UserPhonenumbersReadScope = "https://www.googleapis.com/auth/user.phonenumbers.read"

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
	s.ContactGroups = NewContactGroupsService(s)
	s.People = NewPeopleService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	ContactGroups *ContactGroupsService

	People *PeopleService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewContactGroupsService(s *Service) *ContactGroupsService {
	rs := &ContactGroupsService{s: s}
	rs.Members = NewContactGroupsMembersService(s)
	return rs
}

type ContactGroupsService struct {
	s *Service

	Members *ContactGroupsMembersService
}

func NewContactGroupsMembersService(s *Service) *ContactGroupsMembersService {
	rs := &ContactGroupsMembersService{s: s}
	return rs
}

type ContactGroupsMembersService struct {
	s *Service
}

func NewPeopleService(s *Service) *PeopleService {
	rs := &PeopleService{s: s}
	rs.Connections = NewPeopleConnectionsService(s)
	return rs
}

type PeopleService struct {
	s *Service

	Connections *PeopleConnectionsService
}

func NewPeopleConnectionsService(s *Service) *PeopleConnectionsService {
	rs := &PeopleConnectionsService{s: s}
	return rs
}

type PeopleConnectionsService struct {
	s *Service
}

// Address: A person's physical address. May be a P.O. box or street
// address. All fields
// are optional.
type Address struct {
	// City: The city of the address.
	City string `json:"city,omitempty"`

	// Country: The country of the address.
	Country string `json:"country,omitempty"`

	// CountryCode: The [ISO 3166-1
	// alpha-2](http://www.iso.org/iso/country_codes.htm) country
	// code of the address.
	CountryCode string `json:"countryCode,omitempty"`

	// ExtendedAddress: The extended address of the address; for example,
	// the apartment number.
	ExtendedAddress string `json:"extendedAddress,omitempty"`

	// FormattedType: The read-only type of the address translated and
	// formatted in the viewer's
	// account locale or the `Accept-Language` HTTP header locale.
	FormattedType string `json:"formattedType,omitempty"`

	// FormattedValue: The unstructured value of the address. If this is not
	// set by the user it
	// will be automatically constructed from structured values.
	FormattedValue string `json:"formattedValue,omitempty"`

	// Metadata: Metadata about the address.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// PoBox: The P.O. box of the address.
	PoBox string `json:"poBox,omitempty"`

	// PostalCode: The postal code of the address.
	PostalCode string `json:"postalCode,omitempty"`

	// Region: The region of the address; for example, the state or
	// province.
	Region string `json:"region,omitempty"`

	// StreetAddress: The street address.
	StreetAddress string `json:"streetAddress,omitempty"`

	// Type: The type of the address. The type can be custom or
	// predefined.
	// Possible values include, but are not limited to, the following:
	//
	// * `home`
	// * `work`
	// * `other`
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "City") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "City") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Address) MarshalJSON() ([]byte, error) {
	type noMethod Address
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AgeRangeType: A person's age range.
type AgeRangeType struct {
	// AgeRange: The age range.
	//
	// Possible values:
	//   "AGE_RANGE_UNSPECIFIED" - Unspecified.
	//   "LESS_THAN_EIGHTEEN" - Younger than eighteen.
	//   "EIGHTEEN_TO_TWENTY" - Between eighteen and twenty.
	//   "TWENTY_ONE_OR_OLDER" - Twenty-one and older.
	AgeRange string `json:"ageRange,omitempty"`

	// Metadata: Metadata about the age range.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AgeRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AgeRange") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AgeRangeType) MarshalJSON() ([]byte, error) {
	type noMethod AgeRangeType
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchGetContactGroupsResponse: The response to a batch get contact
// groups request.
type BatchGetContactGroupsResponse struct {
	// Responses: The list of responses for each requested contact group
	// resource.
	Responses []*ContactGroupResponse `json:"responses,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Responses") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Responses") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchGetContactGroupsResponse) MarshalJSON() ([]byte, error) {
	type noMethod BatchGetContactGroupsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Biography: A person's short biography.
type Biography struct {
	// ContentType: The content type of the biography.
	//
	// Possible values:
	//   "CONTENT_TYPE_UNSPECIFIED" - Unspecified.
	//   "TEXT_PLAIN" - Plain text.
	//   "TEXT_HTML" - HTML text.
	ContentType string `json:"contentType,omitempty"`

	// Metadata: Metadata about the biography.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The short biography.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ContentType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ContentType") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Biography) MarshalJSON() ([]byte, error) {
	type noMethod Biography
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Birthday: A person's birthday. At least one of the `date` and `text`
// fields are
// specified. The `date` and `text` fields typically represent the
// same
// date, but are not guaranteed to.
type Birthday struct {
	// Date: The date of the birthday.
	Date *Date `json:"date,omitempty"`

	// Metadata: Metadata about the birthday.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Text: A free-form string representing the user's birthday.
	Text string `json:"text,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Date") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Date") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Birthday) MarshalJSON() ([]byte, error) {
	type noMethod Birthday
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BraggingRights: A person's bragging rights.
type BraggingRights struct {
	// Metadata: Metadata about the bragging rights.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The bragging rights; for example, `climbed mount everest`.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Metadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Metadata") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BraggingRights) MarshalJSON() ([]byte, error) {
	type noMethod BraggingRights
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ContactGroup: A contact group.
type ContactGroup struct {
	// Etag: The [HTTP entity tag](https://en.wikipedia.org/wiki/HTTP_ETag)
	// of the
	// resource. Used for web cache validation.
	Etag string `json:"etag,omitempty"`

	// FormattedName: The read-only name translated and formatted in the
	// viewer's account locale
	// or the `Accept-Language` HTTP header locale for system groups
	// names.
	// Group names set by the owner are the same as name.
	FormattedName string `json:"formattedName,omitempty"`

	// GroupType: The read-only contact group type.
	//
	// Possible values:
	//   "GROUP_TYPE_UNSPECIFIED" - Unspecified.
	//   "USER_CONTACT_GROUP" - User defined contact group.
	//   "SYSTEM_CONTACT_GROUP" - System defined contact group.
	GroupType string `json:"groupType,omitempty"`

	// MemberCount: The total number of contacts in the group irrespective
	// of max members in
	// specified in the request.
	MemberCount int64 `json:"memberCount,omitempty"`

	// MemberResourceNames: The list of contact person resource names that
	// are members of the contact
	// group. The field is not populated for LIST requests and can only be
	// updated
	// through
	// the
	// [ModifyContactGroupMembers](/people/api/rest/v1/contactgroups/memb
	// ers/modify).
	MemberResourceNames []string `json:"memberResourceNames,omitempty"`

	// Metadata: Metadata about the contact group.
	Metadata *ContactGroupMetadata `json:"metadata,omitempty"`

	// Name: The contact group name set by the group owner or a system
	// provided name
	// for system groups.
	Name string `json:"name,omitempty"`

	// ResourceName: The resource name for the contact group, assigned by
	// the server. An ASCII
	// string, in the form of `contactGroups/`<var>contact_group_id</var>.
	ResourceName string `json:"resourceName,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Etag") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Etag") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ContactGroup) MarshalJSON() ([]byte, error) {
	type noMethod ContactGroup
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ContactGroupMembership: A Google contact group membership.
type ContactGroupMembership struct {
	// ContactGroupId: The contact group ID for the contact group
	// membership. The contact group
	// ID can be custom or predefined. Possible values include, but are
	// not
	// limited to, the following:
	//
	// *  `myContacts`
	// *  `starred`
	// *  A numerical ID for user-created groups.
	ContactGroupId string `json:"contactGroupId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ContactGroupId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ContactGroupId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ContactGroupMembership) MarshalJSON() ([]byte, error) {
	type noMethod ContactGroupMembership
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ContactGroupMetadata: The read-only metadata about a contact group.
type ContactGroupMetadata struct {
	// Deleted: True if the contact group resource has been deleted.
	// Populated only
	// for
	// [`ListContactGroups`](/people/api/rest/v1/contactgroups/list)
	// requests
	// that include a sync token.
	Deleted bool `json:"deleted,omitempty"`

	// UpdateTime: The time the group was last updated.
	UpdateTime string `json:"updateTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Deleted") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Deleted") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ContactGroupMetadata) MarshalJSON() ([]byte, error) {
	type noMethod ContactGroupMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ContactGroupResponse: The response for a specific contact group.
type ContactGroupResponse struct {
	// ContactGroup: The contact group.
	ContactGroup *ContactGroup `json:"contactGroup,omitempty"`

	// RequestedResourceName: The original requested resource name.
	RequestedResourceName string `json:"requestedResourceName,omitempty"`

	// Status: The status of the response.
	Status *Status `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ContactGroup") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ContactGroup") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ContactGroupResponse) MarshalJSON() ([]byte, error) {
	type noMethod ContactGroupResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CoverPhoto: A person's read-only cover photo. A large image shown on
// the person's
// profile page that represents who they are or what they care about.
type CoverPhoto struct {
	// Default: True if the cover photo is the default cover photo;
	// false if the cover photo is a user-provided cover photo.
	Default bool `json:"default,omitempty"`

	// Metadata: Metadata about the cover photo.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Url: The URL of the cover photo.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Default") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Default") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CoverPhoto) MarshalJSON() ([]byte, error) {
	type noMethod CoverPhoto
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CreateContactGroupRequest: A request to create a new contact group.
type CreateContactGroupRequest struct {
	// ContactGroup: The contact group to create.
	ContactGroup *ContactGroup `json:"contactGroup,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ContactGroup") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ContactGroup") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CreateContactGroupRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateContactGroupRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Date: Represents a whole calendar date, for example a date of birth.
// The time
// of day and time zone are either specified elsewhere or are
// not
// significant. The date is relative to the
// [Proleptic Gregorian
// Calendar](https://en.wikipedia.org/wiki/Proleptic_Gregorian_calendar).
//
// The day may be 0 to represent a year and month where the day is
// not
// significant. The year may be 0 to represent a month and day
// independent
// of year; for example, anniversary date.
type Date struct {
	// Day: Day of month. Must be from 1 to 31 and valid for the year and
	// month, or 0
	// if specifying a year/month where the day is not significant.
	Day int64 `json:"day,omitempty"`

	// Month: Month of year. Must be from 1 to 12.
	Month int64 `json:"month,omitempty"`

	// Year: Year of date. Must be from 1 to 9999, or 0 if specifying a date
	// without
	// a year.
	Year int64 `json:"year,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Day") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Day") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Date) MarshalJSON() ([]byte, error) {
	type noMethod Date
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DomainMembership: A Google Apps Domain membership.
type DomainMembership struct {
	// InViewerDomain: True if the person is in the viewer's Google Apps
	// domain.
	InViewerDomain bool `json:"inViewerDomain,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InViewerDomain") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InViewerDomain") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DomainMembership) MarshalJSON() ([]byte, error) {
	type noMethod DomainMembership
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// EmailAddress: A person's email address.
type EmailAddress struct {
	// DisplayName: The display name of the email.
	DisplayName string `json:"displayName,omitempty"`

	// FormattedType: The read-only type of the email address translated and
	// formatted in the
	// viewer's account locale or the `Accept-Language` HTTP header locale.
	FormattedType string `json:"formattedType,omitempty"`

	// Metadata: Metadata about the email address.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Type: The type of the email address. The type can be custom or
	// predefined.
	// Possible values include, but are not limited to, the following:
	//
	// * `home`
	// * `work`
	// * `other`
	Type string `json:"type,omitempty"`

	// Value: The email address.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DisplayName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EmailAddress) MarshalJSON() ([]byte, error) {
	type noMethod EmailAddress
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

// Event: An event related to the person.
type Event struct {
	// Date: The date of the event.
	Date *Date `json:"date,omitempty"`

	// FormattedType: The read-only type of the event translated and
	// formatted in the
	// viewer's account locale or the `Accept-Language` HTTP header locale.
	FormattedType string `json:"formattedType,omitempty"`

	// Metadata: Metadata about the event.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Type: The type of the event. The type can be custom or
	// predefined.
	// Possible values include, but are not limited to, the following:
	//
	// * `anniversary`
	// * `other`
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Date") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Date") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Event) MarshalJSON() ([]byte, error) {
	type noMethod Event
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FieldMetadata: Metadata about a field.
type FieldMetadata struct {
	// Primary: True if the field is the primary field; false if the field
	// is a secondary
	// field.
	Primary bool `json:"primary,omitempty"`

	// Source: The source of the field.
	Source *Source `json:"source,omitempty"`

	// Verified: True if the field is verified; false if the field is
	// unverified. A
	// verified field is typically a name, email address, phone number,
	// or
	// website that has been confirmed to be owned by the person.
	Verified bool `json:"verified,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Primary") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Primary") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FieldMetadata) MarshalJSON() ([]byte, error) {
	type noMethod FieldMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Gender: A person's gender.
type Gender struct {
	// FormattedValue: The read-only value of the gender translated and
	// formatted in the viewer's
	// account locale or the `Accept-Language` HTTP header locale.
	FormattedValue string `json:"formattedValue,omitempty"`

	// Metadata: Metadata about the gender.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The gender for the person. The gender can be custom or
	// predefined.
	// Possible values include, but are not limited to, the
	// following:
	//
	// * `male`
	// * `female`
	// * `other`
	// * `unknown`
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FormattedValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FormattedValue") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Gender) MarshalJSON() ([]byte, error) {
	type noMethod Gender
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type GetPeopleResponse struct {
	// Responses: The response for each requested resource name.
	Responses []*PersonResponse `json:"responses,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Responses") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Responses") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GetPeopleResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetPeopleResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ImClient: A person's instant messaging client.
type ImClient struct {
	// FormattedProtocol: The read-only protocol of the IM client formatted
	// in the viewer's account
	// locale or the `Accept-Language` HTTP header locale.
	FormattedProtocol string `json:"formattedProtocol,omitempty"`

	// FormattedType: The read-only type of the IM client translated and
	// formatted in the
	// viewer's account locale or the `Accept-Language` HTTP header locale.
	FormattedType string `json:"formattedType,omitempty"`

	// Metadata: Metadata about the IM client.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Protocol: The protocol of the IM client. The protocol can be custom
	// or predefined.
	// Possible values include, but are not limited to, the following:
	//
	// * `aim`
	// * `msn`
	// * `yahoo`
	// * `skype`
	// * `qq`
	// * `googleTalk`
	// * `icq`
	// * `jabber`
	// * `netMeeting`
	Protocol string `json:"protocol,omitempty"`

	// Type: The type of the IM client. The type can be custom or
	// predefined.
	// Possible values include, but are not limited to, the following:
	//
	// * `home`
	// * `work`
	// * `other`
	Type string `json:"type,omitempty"`

	// Username: The user name used in the IM client.
	Username string `json:"username,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FormattedProtocol")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FormattedProtocol") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ImClient) MarshalJSON() ([]byte, error) {
	type noMethod ImClient
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Interest: One of the person's interests.
type Interest struct {
	// Metadata: Metadata about the interest.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The interest; for example, `stargazing`.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Metadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Metadata") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Interest) MarshalJSON() ([]byte, error) {
	type noMethod Interest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type ListConnectionsResponse struct {
	// Connections: The list of people that the requestor is connected to.
	Connections []*Person `json:"connections,omitempty"`

	// NextPageToken: The token that can be used to retrieve the next page
	// of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// NextSyncToken: The token that can be used to retrieve changes since
	// the last request.
	NextSyncToken string `json:"nextSyncToken,omitempty"`

	// TotalItems: The total number of items in the list without pagination.
	TotalItems int64 `json:"totalItems,omitempty"`

	// TotalPeople: **DEPRECATED** (Please use totalItems)
	// The total number of people in the list without pagination.
	TotalPeople int64 `json:"totalPeople,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Connections") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Connections") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListConnectionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListConnectionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListContactGroupsResponse: The response to a list contact groups
// request.
type ListContactGroupsResponse struct {
	// ContactGroups: The list of contact groups. Members of the contact
	// groups are not
	// populated.
	ContactGroups []*ContactGroup `json:"contactGroups,omitempty"`

	// NextPageToken: The token that can be used to retrieve the next page
	// of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// NextSyncToken: The token that can be used to retrieve changes since
	// the last request.
	NextSyncToken string `json:"nextSyncToken,omitempty"`

	// TotalItems: The total number of items in the list without pagination.
	TotalItems int64 `json:"totalItems,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ContactGroups") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ContactGroups") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListContactGroupsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListContactGroupsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Locale: A person's locale preference.
type Locale struct {
	// Metadata: Metadata about the locale.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The well-formed [IETF BCP
	// 47](https://tools.ietf.org/html/bcp47)
	// language tag representing the locale.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Metadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Metadata") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Locale) MarshalJSON() ([]byte, error) {
	type noMethod Locale
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Membership: A person's read-only membership in a group.
type Membership struct {
	// ContactGroupMembership: The contact group membership.
	ContactGroupMembership *ContactGroupMembership `json:"contactGroupMembership,omitempty"`

	// DomainMembership: The domain membership.
	DomainMembership *DomainMembership `json:"domainMembership,omitempty"`

	// Metadata: Metadata about the membership.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ContactGroupMembership") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ContactGroupMembership")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Membership) MarshalJSON() ([]byte, error) {
	type noMethod Membership
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ModifyContactGroupMembersRequest: A request to modify an existing
// contact group's members.
type ModifyContactGroupMembersRequest struct {
	// ResourceNamesToAdd: The resource names of the contact people to add
	// in the form of in the form
	// `people/`<var>person_id</var>.
	ResourceNamesToAdd []string `json:"resourceNamesToAdd,omitempty"`

	// ResourceNamesToRemove: The resource names of the contact people to
	// remove in the form of in the
	// form of `people/`<var>person_id</var>.
	ResourceNamesToRemove []string `json:"resourceNamesToRemove,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ResourceNamesToAdd")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ResourceNamesToAdd") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ModifyContactGroupMembersRequest) MarshalJSON() ([]byte, error) {
	type noMethod ModifyContactGroupMembersRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ModifyContactGroupMembersResponse: The response to a modify contact
// group members request.
type ModifyContactGroupMembersResponse struct {
	// NotFoundResourceNames: The contact people resource names that were
	// not found.
	NotFoundResourceNames []string `json:"notFoundResourceNames,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "NotFoundResourceNames") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NotFoundResourceNames") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ModifyContactGroupMembersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ModifyContactGroupMembersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Name: A person's name. If the name is a mononym, the family name is
// empty.
type Name struct {
	// DisplayName: The read-only display name formatted according to the
	// locale specified by
	// the viewer's account or the `Accept-Language` HTTP header.
	DisplayName string `json:"displayName,omitempty"`

	// DisplayNameLastFirst: The read-only display name with the last name
	// first formatted according to
	// the locale specified by the viewer's account or the
	// `Accept-Language` HTTP header.
	DisplayNameLastFirst string `json:"displayNameLastFirst,omitempty"`

	// FamilyName: The family name.
	FamilyName string `json:"familyName,omitempty"`

	// GivenName: The given name.
	GivenName string `json:"givenName,omitempty"`

	// HonorificPrefix: The honorific prefixes, such as `Mrs.` or `Dr.`
	HonorificPrefix string `json:"honorificPrefix,omitempty"`

	// HonorificSuffix: The honorific suffixes, such as `Jr.`
	HonorificSuffix string `json:"honorificSuffix,omitempty"`

	// Metadata: Metadata about the name.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// MiddleName: The middle name(s).
	MiddleName string `json:"middleName,omitempty"`

	// PhoneticFamilyName: The family name spelled as it sounds.
	PhoneticFamilyName string `json:"phoneticFamilyName,omitempty"`

	// PhoneticFullName: The full name spelled as it sounds.
	PhoneticFullName string `json:"phoneticFullName,omitempty"`

	// PhoneticGivenName: The given name spelled as it sounds.
	PhoneticGivenName string `json:"phoneticGivenName,omitempty"`

	// PhoneticHonorificPrefix: The honorific prefixes spelled as they
	// sound.
	PhoneticHonorificPrefix string `json:"phoneticHonorificPrefix,omitempty"`

	// PhoneticHonorificSuffix: The honorific suffixes spelled as they
	// sound.
	PhoneticHonorificSuffix string `json:"phoneticHonorificSuffix,omitempty"`

	// PhoneticMiddleName: The middle name(s) spelled as they sound.
	PhoneticMiddleName string `json:"phoneticMiddleName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DisplayName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Name) MarshalJSON() ([]byte, error) {
	type noMethod Name
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Nickname: A person's nickname.
type Nickname struct {
	// Metadata: Metadata about the nickname.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Type: The type of the nickname.
	//
	// Possible values:
	//   "DEFAULT" - Generic nickname.
	//   "MAIDEN_NAME" - Maiden name or birth family name. Used when the
	// person's family name has
	// changed as a result of marriage.
	//   "INITIALS" - Initials.
	//   "GPLUS" - Google+ profile nickname.
	//   "OTHER_NAME" - A professional affiliation or other name; for
	// example, `Dr. Smith.`
	Type string `json:"type,omitempty"`

	// Value: The nickname.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Metadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Metadata") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Nickname) MarshalJSON() ([]byte, error) {
	type noMethod Nickname
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Occupation: A person's occupation.
type Occupation struct {
	// Metadata: Metadata about the occupation.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The occupation; for example, `carpenter`.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Metadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Metadata") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Occupation) MarshalJSON() ([]byte, error) {
	type noMethod Occupation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Organization: A person's past or current organization. Overlapping
// date ranges are
// permitted.
type Organization struct {
	// Current: True if the organization is the person's current
	// organization;
	// false if the organization is a past organization.
	Current bool `json:"current,omitempty"`

	// Department: The person's department at the organization.
	Department string `json:"department,omitempty"`

	// Domain: The domain name associated with the organization; for
	// example, `google.com`.
	Domain string `json:"domain,omitempty"`

	// EndDate: The end date when the person left the organization.
	EndDate *Date `json:"endDate,omitempty"`

	// FormattedType: The read-only type of the organization translated and
	// formatted in the
	// viewer's account locale or the `Accept-Language` HTTP header locale.
	FormattedType string `json:"formattedType,omitempty"`

	// JobDescription: The person's job description at the organization.
	JobDescription string `json:"jobDescription,omitempty"`

	// Location: The location of the organization office the person works
	// at.
	Location string `json:"location,omitempty"`

	// Metadata: Metadata about the organization.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Name: The name of the organization.
	Name string `json:"name,omitempty"`

	// PhoneticName: The phonetic name of the organization.
	PhoneticName string `json:"phoneticName,omitempty"`

	// StartDate: The start date when the person joined the organization.
	StartDate *Date `json:"startDate,omitempty"`

	// Symbol: The symbol associated with the organization; for example, a
	// stock ticker
	// symbol, abbreviation, or acronym.
	Symbol string `json:"symbol,omitempty"`

	// Title: The person's job title at the organization.
	Title string `json:"title,omitempty"`

	// Type: The type of the organization. The type can be custom or
	// predefined.
	// Possible values include, but are not limited to, the following:
	//
	// * `work`
	// * `school`
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Current") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Current") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Organization) MarshalJSON() ([]byte, error) {
	type noMethod Organization
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Person: Information about a person merged from various data sources
// such as the
// authenticated user's contacts and profile data.
//
// Most fields can have multiple items. The items in a field have no
// guaranteed
// order, but each non-empty field is guaranteed to have exactly one
// field with
// `metadata.primary` set to true.
type Person struct {
	// Addresses: The person's street addresses.
	Addresses []*Address `json:"addresses,omitempty"`

	// AgeRange: **DEPRECATED** (Please use `person.ageRanges`
	// instead)**
	//
	// The person's read-only age range.
	//
	// Possible values:
	//   "AGE_RANGE_UNSPECIFIED" - Unspecified.
	//   "LESS_THAN_EIGHTEEN" - Younger than eighteen.
	//   "EIGHTEEN_TO_TWENTY" - Between eighteen and twenty.
	//   "TWENTY_ONE_OR_OLDER" - Twenty-one and older.
	AgeRange string `json:"ageRange,omitempty"`

	// AgeRanges: The person's read-only age ranges.
	AgeRanges []*AgeRangeType `json:"ageRanges,omitempty"`

	// Biographies: The person's biographies.
	Biographies []*Biography `json:"biographies,omitempty"`

	// Birthdays: The person's birthdays.
	Birthdays []*Birthday `json:"birthdays,omitempty"`

	// BraggingRights: The person's bragging rights.
	BraggingRights []*BraggingRights `json:"braggingRights,omitempty"`

	// CoverPhotos: The person's read-only cover photos.
	CoverPhotos []*CoverPhoto `json:"coverPhotos,omitempty"`

	// EmailAddresses: The person's email addresses.
	EmailAddresses []*EmailAddress `json:"emailAddresses,omitempty"`

	// Etag: The [HTTP entity tag](https://en.wikipedia.org/wiki/HTTP_ETag)
	// of the
	// resource. Used for web cache validation.
	Etag string `json:"etag,omitempty"`

	// Events: The person's events.
	Events []*Event `json:"events,omitempty"`

	// Genders: The person's genders.
	Genders []*Gender `json:"genders,omitempty"`

	// ImClients: The person's instant messaging clients.
	ImClients []*ImClient `json:"imClients,omitempty"`

	// Interests: The person's interests.
	Interests []*Interest `json:"interests,omitempty"`

	// Locales: The person's locale preferences.
	Locales []*Locale `json:"locales,omitempty"`

	// Memberships: The person's read-only group memberships.
	Memberships []*Membership `json:"memberships,omitempty"`

	// Metadata: Read-only metadata about the person.
	Metadata *PersonMetadata `json:"metadata,omitempty"`

	// Names: The person's names.
	Names []*Name `json:"names,omitempty"`

	// Nicknames: The person's nicknames.
	Nicknames []*Nickname `json:"nicknames,omitempty"`

	// Occupations: The person's occupations.
	Occupations []*Occupation `json:"occupations,omitempty"`

	// Organizations: The person's past or current organizations.
	Organizations []*Organization `json:"organizations,omitempty"`

	// PhoneNumbers: The person's phone numbers.
	PhoneNumbers []*PhoneNumber `json:"phoneNumbers,omitempty"`

	// Photos: The person's read-only photos.
	Photos []*Photo `json:"photos,omitempty"`

	// Relations: The person's relations.
	Relations []*Relation `json:"relations,omitempty"`

	// RelationshipInterests: The person's read-only relationship interests.
	RelationshipInterests []*RelationshipInterest `json:"relationshipInterests,omitempty"`

	// RelationshipStatuses: The person's read-only relationship statuses.
	RelationshipStatuses []*RelationshipStatus `json:"relationshipStatuses,omitempty"`

	// Residences: The person's residences.
	Residences []*Residence `json:"residences,omitempty"`

	// ResourceName: The resource name for the person, assigned by the
	// server. An ASCII string
	// with a max length of 27 characters, in the form
	// of
	// `people/`<var>person_id</var>.
	ResourceName string `json:"resourceName,omitempty"`

	// Skills: The person's skills.
	Skills []*Skill `json:"skills,omitempty"`

	// Taglines: The person's read-only taglines.
	Taglines []*Tagline `json:"taglines,omitempty"`

	// Urls: The person's associated URLs.
	Urls []*Url `json:"urls,omitempty"`

	// UserDefined: The person's user defined data.
	UserDefined []*UserDefined `json:"userDefined,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Addresses") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Addresses") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Person) MarshalJSON() ([]byte, error) {
	type noMethod Person
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PersonMetadata: The read-only metadata about a person.
type PersonMetadata struct {
	// Deleted: True if the person resource has been deleted. Populated only
	// for
	// [`connections.list`](/people/api/rest/v1/people.connections/list)
	// requests
	// that include a sync token.
	Deleted bool `json:"deleted,omitempty"`

	// LinkedPeopleResourceNames: Resource names of people linked to this
	// resource.
	LinkedPeopleResourceNames []string `json:"linkedPeopleResourceNames,omitempty"`

	// ObjectType: **DEPRECATED** (Please
	// use
	// `person.metadata.sources.profileMetadata.objectType` instead)
	//
	// The type of the person object.
	//
	// Possible values:
	//   "OBJECT_TYPE_UNSPECIFIED" - Unspecified.
	//   "PERSON" - Person.
	//   "PAGE" - [Google+ Page.](http://www.google.com/+/brands/)
	ObjectType string `json:"objectType,omitempty"`

	// PreviousResourceNames: Any former resource names this person has had.
	// Populated only
	// for
	// [`connections.list`](/people/api/rest/v1/people.connections/list)
	// requests
	// that include a sync token.
	//
	// The resource name may change when adding or removing fields that link
	// a
	// contact and profile such as a verified email, verified phone number,
	// or
	// profile URL.
	PreviousResourceNames []string `json:"previousResourceNames,omitempty"`

	// Sources: The sources of data for the person.
	Sources []*Source `json:"sources,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Deleted") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Deleted") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PersonMetadata) MarshalJSON() ([]byte, error) {
	type noMethod PersonMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PersonResponse: The response for a single person
type PersonResponse struct {
	// HttpStatusCode: **DEPRECATED** (Please use status instead)
	//
	// [HTTP 1.1 status
	// code]
	// (http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html).
	HttpStatusCode int64 `json:"httpStatusCode,omitempty"`

	// Person: The person.
	Person *Person `json:"person,omitempty"`

	// RequestedResourceName: The original requested resource name. May be
	// different than the resource
	// name on the returned person.
	//
	// The resource name can change when adding or removing fields that link
	// a
	// contact and profile such as a verified email, verified phone number,
	// or a
	// profile URL.
	RequestedResourceName string `json:"requestedResourceName,omitempty"`

	// Status: The status of the response.
	Status *Status `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "HttpStatusCode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "HttpStatusCode") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *PersonResponse) MarshalJSON() ([]byte, error) {
	type noMethod PersonResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PhoneNumber: A person's phone number.
type PhoneNumber struct {
	// CanonicalForm: The read-only canonicalized [ITU-T
	// E.164](https://law.resource.org/pub/us/cfr/ibr/004/itu-t.E.164.1.2008.
	// pdf)
	// form of the phone number.
	CanonicalForm string `json:"canonicalForm,omitempty"`

	// FormattedType: The read-only type of the phone number translated and
	// formatted in the
	// viewer's account locale or the `Accept-Language` HTTP header locale.
	FormattedType string `json:"formattedType,omitempty"`

	// Metadata: Metadata about the phone number.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Type: The type of the phone number. The type can be custom or
	// predefined.
	// Possible values include, but are not limited to, the following:
	//
	// * `home`
	// * `work`
	// * `mobile`
	// * `homeFax`
	// * `workFax`
	// * `otherFax`
	// * `pager`
	// * `workMobile`
	// * `workPager`
	// * `main`
	// * `googleVoice`
	// * `other`
	Type string `json:"type,omitempty"`

	// Value: The phone number.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CanonicalForm") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CanonicalForm") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PhoneNumber) MarshalJSON() ([]byte, error) {
	type noMethod PhoneNumber
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Photo: A person's read-only photo. A picture shown next to the
// person's name to
// help others recognize the person.
type Photo struct {
	// Default: True if the photo is a default photo;
	// false if the photo is a user-provided photo.
	Default bool `json:"default,omitempty"`

	// Metadata: Metadata about the photo.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Url: The URL of the photo. You can change the desired size by
	// appending a query
	// parameter `sz=`<var>size</var> at the end of the url.
	// Example:
	// `https://lh3.googleusercontent.com/-T_wVWLlmg7w/AAAAAAAAAAI/A
	// AAAAAAABa8/00gzXvDBYqw/s100/photo.jpg?sz=50`
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Default") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Default") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Photo) MarshalJSON() ([]byte, error) {
	type noMethod Photo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProfileMetadata: The read-only metadata about a profile.
type ProfileMetadata struct {
	// ObjectType: The profile object type.
	//
	// Possible values:
	//   "OBJECT_TYPE_UNSPECIFIED" - Unspecified.
	//   "PERSON" - Person.
	//   "PAGE" - [Google+ Page.](http://www.google.com/+/brands/)
	ObjectType string `json:"objectType,omitempty"`

	// UserTypes: The user types.
	//
	// Possible values:
	//   "USER_TYPE_UNKNOWN" - The user type is not known.
	//   "GOOGLE_USER" - The user is a Google user.
	//   "GPLUS_USER" - The user is a Google+ user.
	//   "GOOGLE_APPS_USER" - The user is a Google Apps for Work user.
	UserTypes []string `json:"userTypes,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ObjectType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ObjectType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ProfileMetadata) MarshalJSON() ([]byte, error) {
	type noMethod ProfileMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Relation: A person's relation to another person.
type Relation struct {
	// FormattedType: The type of the relation translated and formatted in
	// the viewer's account
	// locale or the locale specified in the Accept-Language HTTP header.
	FormattedType string `json:"formattedType,omitempty"`

	// Metadata: Metadata about the relation.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Person: The name of the other person this relation refers to.
	Person string `json:"person,omitempty"`

	// Type: The person's relation to the other person. The type can be
	// custom or predefined.
	// Possible values include, but are not limited to, the following
	// values:
	//
	// * `spouse`
	// * `child`
	// * `mother`
	// * `father`
	// * `parent`
	// * `brother`
	// * `sister`
	// * `friend`
	// * `relative`
	// * `domesticPartner`
	// * `manager`
	// * `assistant`
	// * `referredBy`
	// * `partner`
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FormattedType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FormattedType") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Relation) MarshalJSON() ([]byte, error) {
	type noMethod Relation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RelationshipInterest: A person's read-only relationship interest .
type RelationshipInterest struct {
	// FormattedValue: The value of the relationship interest translated and
	// formatted in the
	// viewer's account locale or the locale specified in the
	// Accept-Language
	// HTTP header.
	FormattedValue string `json:"formattedValue,omitempty"`

	// Metadata: Metadata about the relationship interest.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The kind of relationship the person is looking for. The value
	// can be custom
	// or predefined. Possible values include, but are not limited to,
	// the
	// following values:
	//
	// * `friend`
	// * `date`
	// * `relationship`
	// * `networking`
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FormattedValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FormattedValue") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *RelationshipInterest) MarshalJSON() ([]byte, error) {
	type noMethod RelationshipInterest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RelationshipStatus: A person's read-only relationship status.
type RelationshipStatus struct {
	// FormattedValue: The read-only value of the relationship status
	// translated and formatted in
	// the viewer's account locale or the `Accept-Language` HTTP header
	// locale.
	FormattedValue string `json:"formattedValue,omitempty"`

	// Metadata: Metadata about the relationship status.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The relationship status. The value can be custom or
	// predefined.
	// Possible values include, but are not limited to, the following:
	//
	// * `single`
	// * `inARelationship`
	// * `engaged`
	// * `married`
	// * `itsComplicated`
	// * `openRelationship`
	// * `widowed`
	// * `inDomesticPartnership`
	// * `inCivilUnion`
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FormattedValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FormattedValue") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *RelationshipStatus) MarshalJSON() ([]byte, error) {
	type noMethod RelationshipStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Residence: A person's past or current residence.
type Residence struct {
	// Current: True if the residence is the person's current
	// residence;
	// false if the residence is a past residence.
	Current bool `json:"current,omitempty"`

	// Metadata: Metadata about the residence.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The address of the residence.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Current") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Current") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Residence) MarshalJSON() ([]byte, error) {
	type noMethod Residence
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Skill: A skill that the person has.
type Skill struct {
	// Metadata: Metadata about the skill.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The skill; for example, `underwater basket weaving`.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Metadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Metadata") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Skill) MarshalJSON() ([]byte, error) {
	type noMethod Skill
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Source: The source of a field.
type Source struct {
	// Etag: **Only populated in `person.metadata.sources`.**
	//
	// The [HTTP entity tag](https://en.wikipedia.org/wiki/HTTP_ETag) of
	// the
	// source. Used for web cache validation.
	Etag string `json:"etag,omitempty"`

	// Id: The unique identifier within the source type generated by the
	// server.
	Id string `json:"id,omitempty"`

	// ProfileMetadata: **Only populated in
	// `person.metadata.sources`.**
	//
	// Metadata about a source of type PROFILE.
	ProfileMetadata *ProfileMetadata `json:"profileMetadata,omitempty"`

	// Type: The source type.
	//
	// Possible values:
	//   "SOURCE_TYPE_UNSPECIFIED" - Unspecified.
	//   "ACCOUNT" - [Google Account](https://accounts.google.com).
	//   "PROFILE" - [Google profile](https://profiles.google.com). You can
	// view the
	// profile at https://profiles.google.com/<var>id</var>
	// where
	// <var>id</var> is the source id.
	//   "DOMAIN_PROFILE" - [Google Apps domain
	// profile](https://admin.google.com).
	//   "CONTACT" - [Google contact](https://contacts.google.com). You can
	// view the
	// contact at https://contact.google.com/<var>id</var> where
	// <var>id</var>
	// is the source id.
	Type string `json:"type,omitempty"`

	// UpdateTime: **Only populated in `person.metadata.sources`.**
	//
	// Last update timestamp of this source.
	UpdateTime string `json:"updateTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Etag") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Etag") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Source) MarshalJSON() ([]byte, error) {
	type noMethod Source
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

// Tagline: A read-only brief one-line description of the person.
type Tagline struct {
	// Metadata: Metadata about the tagline.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The tagline.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Metadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Metadata") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Tagline) MarshalJSON() ([]byte, error) {
	type noMethod Tagline
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateContactGroupRequest: A request to update an existing contact
// group. Only the name can be updated.
type UpdateContactGroupRequest struct {
	// ContactGroup: The contact group to update.
	ContactGroup *ContactGroup `json:"contactGroup,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ContactGroup") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ContactGroup") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateContactGroupRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateContactGroupRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Url: A person's associated URLs.
type Url struct {
	// FormattedType: The read-only type of the URL translated and formatted
	// in the viewer's
	// account locale or the `Accept-Language` HTTP header locale.
	FormattedType string `json:"formattedType,omitempty"`

	// Metadata: Metadata about the URL.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Type: The type of the URL. The type can be custom or
	// predefined.
	// Possible values include, but are not limited to, the following:
	//
	// * `home`
	// * `work`
	// * `blog`
	// * `profile`
	// * `homePage`
	// * `ftp`
	// * `reservations`
	// * `appInstallPage`: website for a Google+ application.
	// * `other`
	Type string `json:"type,omitempty"`

	// Value: The URL.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FormattedType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FormattedType") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Url) MarshalJSON() ([]byte, error) {
	type noMethod Url
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UserDefined: Arbitrary user data that is populated by the end users.
type UserDefined struct {
	// Key: The end user specified key of the user defined data.
	Key string `json:"key,omitempty"`

	// Metadata: Metadata about the user defined data.
	Metadata *FieldMetadata `json:"metadata,omitempty"`

	// Value: The end user specified value of the user defined data.
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

func (s *UserDefined) MarshalJSON() ([]byte, error) {
	type noMethod UserDefined
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "people.contactGroups.batchGet":

type ContactGroupsBatchGetCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// BatchGet: Get a list of contact groups owned by the authenticated
// user by specifying
// a list of contact group resource names.
func (r *ContactGroupsService) BatchGet() *ContactGroupsBatchGetCall {
	c := &ContactGroupsBatchGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// MaxMembers sets the optional parameter "maxMembers": Specifies the
// maximum number of members to return for each group.
func (c *ContactGroupsBatchGetCall) MaxMembers(maxMembers int64) *ContactGroupsBatchGetCall {
	c.urlParams_.Set("maxMembers", fmt.Sprint(maxMembers))
	return c
}

// ResourceNames sets the optional parameter "resourceNames": The
// resource names of the contact groups to get.
func (c *ContactGroupsBatchGetCall) ResourceNames(resourceNames ...string) *ContactGroupsBatchGetCall {
	c.urlParams_.SetMulti("resourceNames", append([]string{}, resourceNames...))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ContactGroupsBatchGetCall) Fields(s ...googleapi.Field) *ContactGroupsBatchGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ContactGroupsBatchGetCall) IfNoneMatch(entityTag string) *ContactGroupsBatchGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ContactGroupsBatchGetCall) Context(ctx context.Context) *ContactGroupsBatchGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ContactGroupsBatchGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ContactGroupsBatchGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/contactGroups:batchGet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.contactGroups.batchGet" call.
// Exactly one of *BatchGetContactGroupsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *BatchGetContactGroupsResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ContactGroupsBatchGetCall) Do(opts ...googleapi.CallOption) (*BatchGetContactGroupsResponse, error) {
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
	ret := &BatchGetContactGroupsResponse{
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
	//   "description": "Get a list of contact groups owned by the authenticated user by specifying\na list of contact group resource names.",
	//   "flatPath": "v1/contactGroups:batchGet",
	//   "httpMethod": "GET",
	//   "id": "people.contactGroups.batchGet",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "maxMembers": {
	//       "description": "Specifies the maximum number of members to return for each group.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "resourceNames": {
	//       "description": "The resource names of the contact groups to get.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/contactGroups:batchGet",
	//   "response": {
	//     "$ref": "BatchGetContactGroupsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts",
	//     "https://www.googleapis.com/auth/contacts.readonly"
	//   ]
	// }

}

// method id "people.contactGroups.create":

type ContactGroupsCreateCall struct {
	s                         *Service
	createcontactgrouprequest *CreateContactGroupRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// Create: Create a new contact group owned by the authenticated user.
func (r *ContactGroupsService) Create(createcontactgrouprequest *CreateContactGroupRequest) *ContactGroupsCreateCall {
	c := &ContactGroupsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.createcontactgrouprequest = createcontactgrouprequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ContactGroupsCreateCall) Fields(s ...googleapi.Field) *ContactGroupsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ContactGroupsCreateCall) Context(ctx context.Context) *ContactGroupsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ContactGroupsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ContactGroupsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createcontactgrouprequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/contactGroups")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.contactGroups.create" call.
// Exactly one of *ContactGroup or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ContactGroup.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ContactGroupsCreateCall) Do(opts ...googleapi.CallOption) (*ContactGroup, error) {
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
	ret := &ContactGroup{
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
	//   "description": "Create a new contact group owned by the authenticated user.",
	//   "flatPath": "v1/contactGroups",
	//   "httpMethod": "POST",
	//   "id": "people.contactGroups.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/contactGroups",
	//   "request": {
	//     "$ref": "CreateContactGroupRequest"
	//   },
	//   "response": {
	//     "$ref": "ContactGroup"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts"
	//   ]
	// }

}

// method id "people.contactGroups.delete":

type ContactGroupsDeleteCall struct {
	s            *Service
	resourceName string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// Delete: Delete an existing contact group owned by the authenticated
// user by
// specifying a contact group resource name.
func (r *ContactGroupsService) Delete(resourceName string) *ContactGroupsDeleteCall {
	c := &ContactGroupsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resourceName = resourceName
	return c
}

// DeleteContacts sets the optional parameter "deleteContacts": Set to
// true to also delete the contacts in the specified group.
func (c *ContactGroupsDeleteCall) DeleteContacts(deleteContacts bool) *ContactGroupsDeleteCall {
	c.urlParams_.Set("deleteContacts", fmt.Sprint(deleteContacts))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ContactGroupsDeleteCall) Fields(s ...googleapi.Field) *ContactGroupsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ContactGroupsDeleteCall) Context(ctx context.Context) *ContactGroupsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ContactGroupsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ContactGroupsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resourceName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resourceName": c.resourceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.contactGroups.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ContactGroupsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Delete an existing contact group owned by the authenticated user by\nspecifying a contact group resource name.",
	//   "flatPath": "v1/contactGroups/{contactGroupsId}",
	//   "httpMethod": "DELETE",
	//   "id": "people.contactGroups.delete",
	//   "parameterOrder": [
	//     "resourceName"
	//   ],
	//   "parameters": {
	//     "deleteContacts": {
	//       "description": "Set to true to also delete the contacts in the specified group.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "resourceName": {
	//       "description": "The resource name of the contact group to delete.",
	//       "location": "path",
	//       "pattern": "^contactGroups/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resourceName}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts"
	//   ]
	// }

}

// method id "people.contactGroups.get":

type ContactGroupsGetCall struct {
	s            *Service
	resourceName string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Get a specific contact group owned by the authenticated user by
// specifying
// a contact group resource name.
func (r *ContactGroupsService) Get(resourceName string) *ContactGroupsGetCall {
	c := &ContactGroupsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resourceName = resourceName
	return c
}

// MaxMembers sets the optional parameter "maxMembers": Specifies the
// maximum number of members to return.
func (c *ContactGroupsGetCall) MaxMembers(maxMembers int64) *ContactGroupsGetCall {
	c.urlParams_.Set("maxMembers", fmt.Sprint(maxMembers))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ContactGroupsGetCall) Fields(s ...googleapi.Field) *ContactGroupsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ContactGroupsGetCall) IfNoneMatch(entityTag string) *ContactGroupsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ContactGroupsGetCall) Context(ctx context.Context) *ContactGroupsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ContactGroupsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ContactGroupsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resourceName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resourceName": c.resourceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.contactGroups.get" call.
// Exactly one of *ContactGroup or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ContactGroup.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ContactGroupsGetCall) Do(opts ...googleapi.CallOption) (*ContactGroup, error) {
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
	ret := &ContactGroup{
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
	//   "description": "Get a specific contact group owned by the authenticated user by specifying\na contact group resource name.",
	//   "flatPath": "v1/contactGroups/{contactGroupsId}",
	//   "httpMethod": "GET",
	//   "id": "people.contactGroups.get",
	//   "parameterOrder": [
	//     "resourceName"
	//   ],
	//   "parameters": {
	//     "maxMembers": {
	//       "description": "Specifies the maximum number of members to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "resourceName": {
	//       "description": "The resource name of the contact group to get.",
	//       "location": "path",
	//       "pattern": "^contactGroups/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resourceName}",
	//   "response": {
	//     "$ref": "ContactGroup"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts",
	//     "https://www.googleapis.com/auth/contacts.readonly"
	//   ]
	// }

}

// method id "people.contactGroups.list":

type ContactGroupsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List all contact groups owned by the authenticated user.
// Members of the
// contact groups are not populated.
func (r *ContactGroupsService) List() *ContactGroupsListCall {
	c := &ContactGroupsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of resources to return.
func (c *ContactGroupsListCall) PageSize(pageSize int64) *ContactGroupsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The
// next_page_token value returned from a previous call
// to
// [ListContactGroups](/people/api/rest/v1/contactgroups/list).
// Reques
// ts the next page of resources.
func (c *ContactGroupsListCall) PageToken(pageToken string) *ContactGroupsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// SyncToken sets the optional parameter "syncToken": A sync token,
// returned by a previous call to `contactgroups.list`.
// Only resources changed since the sync token was created will be
// returned.
func (c *ContactGroupsListCall) SyncToken(syncToken string) *ContactGroupsListCall {
	c.urlParams_.Set("syncToken", syncToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ContactGroupsListCall) Fields(s ...googleapi.Field) *ContactGroupsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ContactGroupsListCall) IfNoneMatch(entityTag string) *ContactGroupsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ContactGroupsListCall) Context(ctx context.Context) *ContactGroupsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ContactGroupsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ContactGroupsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/contactGroups")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.contactGroups.list" call.
// Exactly one of *ListContactGroupsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListContactGroupsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ContactGroupsListCall) Do(opts ...googleapi.CallOption) (*ListContactGroupsResponse, error) {
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
	ret := &ListContactGroupsResponse{
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
	//   "description": "List all contact groups owned by the authenticated user. Members of the\ncontact groups are not populated.",
	//   "flatPath": "v1/contactGroups",
	//   "httpMethod": "GET",
	//   "id": "people.contactGroups.list",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "The maximum number of resources to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The next_page_token value returned from a previous call to\n[ListContactGroups](/people/api/rest/v1/contactgroups/list).\nRequests the next page of resources.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "syncToken": {
	//       "description": "A sync token, returned by a previous call to `contactgroups.list`.\nOnly resources changed since the sync token was created will be returned.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/contactGroups",
	//   "response": {
	//     "$ref": "ListContactGroupsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts",
	//     "https://www.googleapis.com/auth/contacts.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ContactGroupsListCall) Pages(ctx context.Context, f func(*ListContactGroupsResponse) error) error {
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

// method id "people.contactGroups.update":

type ContactGroupsUpdateCall struct {
	s                         *Service
	resourceName              string
	updatecontactgrouprequest *UpdateContactGroupRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
	header_                   http.Header
}

// Update: Update the name of an existing contact group owned by the
// authenticated
// user.
func (r *ContactGroupsService) Update(resourceName string, updatecontactgrouprequest *UpdateContactGroupRequest) *ContactGroupsUpdateCall {
	c := &ContactGroupsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resourceName = resourceName
	c.updatecontactgrouprequest = updatecontactgrouprequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ContactGroupsUpdateCall) Fields(s ...googleapi.Field) *ContactGroupsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ContactGroupsUpdateCall) Context(ctx context.Context) *ContactGroupsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ContactGroupsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ContactGroupsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.updatecontactgrouprequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resourceName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resourceName": c.resourceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.contactGroups.update" call.
// Exactly one of *ContactGroup or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ContactGroup.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ContactGroupsUpdateCall) Do(opts ...googleapi.CallOption) (*ContactGroup, error) {
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
	ret := &ContactGroup{
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
	//   "description": "Update the name of an existing contact group owned by the authenticated\nuser.",
	//   "flatPath": "v1/contactGroups/{contactGroupsId}",
	//   "httpMethod": "PUT",
	//   "id": "people.contactGroups.update",
	//   "parameterOrder": [
	//     "resourceName"
	//   ],
	//   "parameters": {
	//     "resourceName": {
	//       "description": "The resource name for the contact group, assigned by the server. An ASCII\nstring, in the form of `contactGroups/`\u003cvar\u003econtact_group_id\u003c/var\u003e.",
	//       "location": "path",
	//       "pattern": "^contactGroups/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resourceName}",
	//   "request": {
	//     "$ref": "UpdateContactGroupRequest"
	//   },
	//   "response": {
	//     "$ref": "ContactGroup"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts"
	//   ]
	// }

}

// method id "people.contactGroups.members.modify":

type ContactGroupsMembersModifyCall struct {
	s                                *Service
	resourceName                     string
	modifycontactgroupmembersrequest *ModifyContactGroupMembersRequest
	urlParams_                       gensupport.URLParams
	ctx_                             context.Context
	header_                          http.Header
}

// Modify: Modify the members of a contact group owned by the
// authenticated user.
func (r *ContactGroupsMembersService) Modify(resourceName string, modifycontactgroupmembersrequest *ModifyContactGroupMembersRequest) *ContactGroupsMembersModifyCall {
	c := &ContactGroupsMembersModifyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resourceName = resourceName
	c.modifycontactgroupmembersrequest = modifycontactgroupmembersrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ContactGroupsMembersModifyCall) Fields(s ...googleapi.Field) *ContactGroupsMembersModifyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ContactGroupsMembersModifyCall) Context(ctx context.Context) *ContactGroupsMembersModifyCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ContactGroupsMembersModifyCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ContactGroupsMembersModifyCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.modifycontactgroupmembersrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resourceName}/members:modify")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resourceName": c.resourceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.contactGroups.members.modify" call.
// Exactly one of *ModifyContactGroupMembersResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ModifyContactGroupMembersResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ContactGroupsMembersModifyCall) Do(opts ...googleapi.CallOption) (*ModifyContactGroupMembersResponse, error) {
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
	ret := &ModifyContactGroupMembersResponse{
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
	//   "description": "Modify the members of a contact group owned by the authenticated user.",
	//   "flatPath": "v1/contactGroups/{contactGroupsId}/members:modify",
	//   "httpMethod": "POST",
	//   "id": "people.contactGroups.members.modify",
	//   "parameterOrder": [
	//     "resourceName"
	//   ],
	//   "parameters": {
	//     "resourceName": {
	//       "description": "The resource name of the contact group to modify.",
	//       "location": "path",
	//       "pattern": "^contactGroups/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resourceName}/members:modify",
	//   "request": {
	//     "$ref": "ModifyContactGroupMembersRequest"
	//   },
	//   "response": {
	//     "$ref": "ModifyContactGroupMembersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts"
	//   ]
	// }

}

// method id "people.people.createContact":

type PeopleCreateContactCall struct {
	s          *Service
	person     *Person
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// CreateContact: Create a new contact and return the person resource
// for that contact.
func (r *PeopleService) CreateContact(person *Person) *PeopleCreateContactCall {
	c := &PeopleCreateContactCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.person = person
	return c
}

// Parent sets the optional parameter "parent": The resource name of the
// owning person resource.
func (c *PeopleCreateContactCall) Parent(parent string) *PeopleCreateContactCall {
	c.urlParams_.Set("parent", parent)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PeopleCreateContactCall) Fields(s ...googleapi.Field) *PeopleCreateContactCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PeopleCreateContactCall) Context(ctx context.Context) *PeopleCreateContactCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PeopleCreateContactCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PeopleCreateContactCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.person)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/people:createContact")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.people.createContact" call.
// Exactly one of *Person or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Person.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *PeopleCreateContactCall) Do(opts ...googleapi.CallOption) (*Person, error) {
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
	ret := &Person{
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
	//   "description": "Create a new contact and return the person resource for that contact.",
	//   "flatPath": "v1/people:createContact",
	//   "httpMethod": "POST",
	//   "id": "people.people.createContact",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "parent": {
	//       "description": "The resource name of the owning person resource.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/people:createContact",
	//   "request": {
	//     "$ref": "Person"
	//   },
	//   "response": {
	//     "$ref": "Person"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts"
	//   ]
	// }

}

// method id "people.people.deleteContact":

type PeopleDeleteContactCall struct {
	s            *Service
	resourceName string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// DeleteContact: Delete a contact person. Any non-contact data will not
// be deleted.
func (r *PeopleService) DeleteContact(resourceName string) *PeopleDeleteContactCall {
	c := &PeopleDeleteContactCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resourceName = resourceName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PeopleDeleteContactCall) Fields(s ...googleapi.Field) *PeopleDeleteContactCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PeopleDeleteContactCall) Context(ctx context.Context) *PeopleDeleteContactCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PeopleDeleteContactCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PeopleDeleteContactCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resourceName}:deleteContact")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resourceName": c.resourceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.people.deleteContact" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *PeopleDeleteContactCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Delete a contact person. Any non-contact data will not be deleted.",
	//   "flatPath": "v1/people/{peopleId}:deleteContact",
	//   "httpMethod": "DELETE",
	//   "id": "people.people.deleteContact",
	//   "parameterOrder": [
	//     "resourceName"
	//   ],
	//   "parameters": {
	//     "resourceName": {
	//       "description": "The resource name of the contact to delete.",
	//       "location": "path",
	//       "pattern": "^people/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resourceName}:deleteContact",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts"
	//   ]
	// }

}

// method id "people.people.get":

type PeopleGetCall struct {
	s            *Service
	resourceName string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Provides information about a person by specifying a resource
// name. Use
// `people/me` to indicate the authenticated user.
// <br>
// The request throws a 400 error if 'personFields' is not specified.
func (r *PeopleService) Get(resourceName string) *PeopleGetCall {
	c := &PeopleGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resourceName = resourceName
	return c
}

// PersonFields sets the optional parameter "personFields":
// **Required.** A field mask to restrict which fields on the person
// are
// returned. Valid values are:
//
// * addresses
// * ageRanges
// * biographies
// * birthdays
// * braggingRights
// * coverPhotos
// * emailAddresses
// * events
// * genders
// * imClients
// * interests
// * locales
// * memberships
// * metadata
// * names
// * nicknames
// * occupations
// * organizations
// * phoneNumbers
// * photos
// * relations
// * relationshipInterests
// * relationshipStatuses
// * residences
// * skills
// * taglines
// * urls
func (c *PeopleGetCall) PersonFields(personFields string) *PeopleGetCall {
	c.urlParams_.Set("personFields", personFields)
	return c
}

// RequestMaskIncludeField sets the optional parameter
// "requestMask.includeField": **Required.** Comma-separated list of
// person fields to be included in the
// response. Each path should start with `person.`: for
// example,
// `person.names` or `person.photos`.
func (c *PeopleGetCall) RequestMaskIncludeField(requestMaskIncludeField string) *PeopleGetCall {
	c.urlParams_.Set("requestMask.includeField", requestMaskIncludeField)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PeopleGetCall) Fields(s ...googleapi.Field) *PeopleGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PeopleGetCall) IfNoneMatch(entityTag string) *PeopleGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PeopleGetCall) Context(ctx context.Context) *PeopleGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PeopleGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PeopleGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resourceName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resourceName": c.resourceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.people.get" call.
// Exactly one of *Person or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Person.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *PeopleGetCall) Do(opts ...googleapi.CallOption) (*Person, error) {
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
	ret := &Person{
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
	//   "description": "Provides information about a person by specifying a resource name. Use\n`people/me` to indicate the authenticated user.\n\u003cbr\u003e\nThe request throws a 400 error if 'personFields' is not specified.",
	//   "flatPath": "v1/people/{peopleId}",
	//   "httpMethod": "GET",
	//   "id": "people.people.get",
	//   "parameterOrder": [
	//     "resourceName"
	//   ],
	//   "parameters": {
	//     "personFields": {
	//       "description": "**Required.** A field mask to restrict which fields on the person are\nreturned. Valid values are:\n\n* addresses\n* ageRanges\n* biographies\n* birthdays\n* braggingRights\n* coverPhotos\n* emailAddresses\n* events\n* genders\n* imClients\n* interests\n* locales\n* memberships\n* metadata\n* names\n* nicknames\n* occupations\n* organizations\n* phoneNumbers\n* photos\n* relations\n* relationshipInterests\n* relationshipStatuses\n* residences\n* skills\n* taglines\n* urls",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMask.includeField": {
	//       "description": "**Required.** Comma-separated list of person fields to be included in the\nresponse. Each path should start with `person.`: for example,\n`person.names` or `person.photos`.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "resourceName": {
	//       "description": "The resource name of the person to provide information about.\n\n- To get information about the authenticated user, specify `people/me`.\n- To get information about a google account, specify\n `people/`\u003cvar\u003eaccount_id\u003c/var\u003e.\n- To get information about a contact, specify the resource name that\n  identifies the contact as returned by\n[`people.connections.list`](/people/api/rest/v1/people.connections/list).",
	//       "location": "path",
	//       "pattern": "^people/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resourceName}",
	//   "response": {
	//     "$ref": "Person"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts",
	//     "https://www.googleapis.com/auth/contacts.readonly",
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/user.addresses.read",
	//     "https://www.googleapis.com/auth/user.birthday.read",
	//     "https://www.googleapis.com/auth/user.emails.read",
	//     "https://www.googleapis.com/auth/user.phonenumbers.read",
	//     "https://www.googleapis.com/auth/userinfo.email",
	//     "https://www.googleapis.com/auth/userinfo.profile"
	//   ]
	// }

}

// method id "people.people.getBatchGet":

type PeopleGetBatchGetCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// GetBatchGet: Provides information about a list of specific people by
// specifying a list
// of requested resource names. Use `people/me` to indicate the
// authenticated
// user.
// <br>
// The request throws a 400 error if 'personFields' is not specified.
func (r *PeopleService) GetBatchGet() *PeopleGetBatchGetCall {
	c := &PeopleGetBatchGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PersonFields sets the optional parameter "personFields":
// **Required.** A field mask to restrict which fields on each person
// are
// returned. Valid values are:
//
// * addresses
// * ageRanges
// * biographies
// * birthdays
// * braggingRights
// * coverPhotos
// * emailAddresses
// * events
// * genders
// * imClients
// * interests
// * locales
// * memberships
// * metadata
// * names
// * nicknames
// * occupations
// * organizations
// * phoneNumbers
// * photos
// * relations
// * relationshipInterests
// * relationshipStatuses
// * residences
// * skills
// * taglines
// * urls
func (c *PeopleGetBatchGetCall) PersonFields(personFields string) *PeopleGetBatchGetCall {
	c.urlParams_.Set("personFields", personFields)
	return c
}

// RequestMaskIncludeField sets the optional parameter
// "requestMask.includeField": **Required.** Comma-separated list of
// person fields to be included in the
// response. Each path should start with `person.`: for
// example,
// `person.names` or `person.photos`.
func (c *PeopleGetBatchGetCall) RequestMaskIncludeField(requestMaskIncludeField string) *PeopleGetBatchGetCall {
	c.urlParams_.Set("requestMask.includeField", requestMaskIncludeField)
	return c
}

// ResourceNames sets the optional parameter "resourceNames": The
// resource names of the people to provide information about.
//
// - To get information about the authenticated user, specify
// `people/me`.
// - To get information about a google account, specify
//   `people/`<var>account_id</var>.
// - To get information about a contact, specify the resource name that
//   identifies the contact as returned
// by
// [`people.connections.list`](/people/api/rest/v1/people.connections/
// list).
//
// You can include up to 50 resource names in one request.
func (c *PeopleGetBatchGetCall) ResourceNames(resourceNames ...string) *PeopleGetBatchGetCall {
	c.urlParams_.SetMulti("resourceNames", append([]string{}, resourceNames...))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PeopleGetBatchGetCall) Fields(s ...googleapi.Field) *PeopleGetBatchGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PeopleGetBatchGetCall) IfNoneMatch(entityTag string) *PeopleGetBatchGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PeopleGetBatchGetCall) Context(ctx context.Context) *PeopleGetBatchGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PeopleGetBatchGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PeopleGetBatchGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/people:batchGet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.people.getBatchGet" call.
// Exactly one of *GetPeopleResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GetPeopleResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PeopleGetBatchGetCall) Do(opts ...googleapi.CallOption) (*GetPeopleResponse, error) {
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
	ret := &GetPeopleResponse{
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
	//   "description": "Provides information about a list of specific people by specifying a list\nof requested resource names. Use `people/me` to indicate the authenticated\nuser.\n\u003cbr\u003e\nThe request throws a 400 error if 'personFields' is not specified.",
	//   "flatPath": "v1/people:batchGet",
	//   "httpMethod": "GET",
	//   "id": "people.people.getBatchGet",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "personFields": {
	//       "description": "**Required.** A field mask to restrict which fields on each person are\nreturned. Valid values are:\n\n* addresses\n* ageRanges\n* biographies\n* birthdays\n* braggingRights\n* coverPhotos\n* emailAddresses\n* events\n* genders\n* imClients\n* interests\n* locales\n* memberships\n* metadata\n* names\n* nicknames\n* occupations\n* organizations\n* phoneNumbers\n* photos\n* relations\n* relationshipInterests\n* relationshipStatuses\n* residences\n* skills\n* taglines\n* urls",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMask.includeField": {
	//       "description": "**Required.** Comma-separated list of person fields to be included in the\nresponse. Each path should start with `person.`: for example,\n`person.names` or `person.photos`.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "resourceNames": {
	//       "description": "The resource names of the people to provide information about.\n\n- To get information about the authenticated user, specify `people/me`.\n- To get information about a google account, specify\n  `people/`\u003cvar\u003eaccount_id\u003c/var\u003e.\n- To get information about a contact, specify the resource name that\n  identifies the contact as returned by\n[`people.connections.list`](/people/api/rest/v1/people.connections/list).\n\nYou can include up to 50 resource names in one request.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/people:batchGet",
	//   "response": {
	//     "$ref": "GetPeopleResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts",
	//     "https://www.googleapis.com/auth/contacts.readonly",
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/user.addresses.read",
	//     "https://www.googleapis.com/auth/user.birthday.read",
	//     "https://www.googleapis.com/auth/user.emails.read",
	//     "https://www.googleapis.com/auth/user.phonenumbers.read",
	//     "https://www.googleapis.com/auth/userinfo.email",
	//     "https://www.googleapis.com/auth/userinfo.profile"
	//   ]
	// }

}

// method id "people.people.updateContact":

type PeopleUpdateContactCall struct {
	s            *Service
	resourceName string
	person       *Person
	urlParams_   gensupport.URLParams
	ctx_         context.Context
	header_      http.Header
}

// UpdateContact: Update contact data for an existing contact person.
// Any non-contact data
// will not be modified.
//
// The request throws a 400 error if `updatePersonFields` is not
// specified.
// <br>
// The request throws a 400 error if `person.metadata.sources` is
// not
// specified for the contact to be updated.
// <br>
// The request throws a 412 error if `person.metadata.sources.etag`
// is
// different than the contact's etag, which indicates the contact has
// changed
// since its data was read. Clients should get the latest person and
// re-apply
// their updates to the latest person.
func (r *PeopleService) UpdateContact(resourceName string, person *Person) *PeopleUpdateContactCall {
	c := &PeopleUpdateContactCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resourceName = resourceName
	c.person = person
	return c
}

// UpdatePersonFields sets the optional parameter "updatePersonFields":
// **Required.** A field mask to restrict which fields on the person
// are
// updated. Valid values are:
//
// * addresses
// * biographies
// * birthdays
// * braggingRights
// * emailAddresses
// * events
// * genders
// * imClients
// * interests
// * locales
// * names
// * nicknames
// * occupations
// * organizations
// * phoneNumbers
// * relations
// * residences
// * skills
// * urls
func (c *PeopleUpdateContactCall) UpdatePersonFields(updatePersonFields string) *PeopleUpdateContactCall {
	c.urlParams_.Set("updatePersonFields", updatePersonFields)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PeopleUpdateContactCall) Fields(s ...googleapi.Field) *PeopleUpdateContactCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PeopleUpdateContactCall) Context(ctx context.Context) *PeopleUpdateContactCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PeopleUpdateContactCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PeopleUpdateContactCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.person)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resourceName}:updateContact")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resourceName": c.resourceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.people.updateContact" call.
// Exactly one of *Person or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Person.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *PeopleUpdateContactCall) Do(opts ...googleapi.CallOption) (*Person, error) {
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
	ret := &Person{
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
	//   "description": "Update contact data for an existing contact person. Any non-contact data\nwill not be modified.\n\nThe request throws a 400 error if `updatePersonFields` is not specified.\n\u003cbr\u003e\nThe request throws a 400 error if `person.metadata.sources` is not\nspecified for the contact to be updated.\n\u003cbr\u003e\nThe request throws a 412 error if `person.metadata.sources.etag` is\ndifferent than the contact's etag, which indicates the contact has changed\nsince its data was read. Clients should get the latest person and re-apply\ntheir updates to the latest person.",
	//   "flatPath": "v1/people/{peopleId}:updateContact",
	//   "httpMethod": "PATCH",
	//   "id": "people.people.updateContact",
	//   "parameterOrder": [
	//     "resourceName"
	//   ],
	//   "parameters": {
	//     "resourceName": {
	//       "description": "The resource name for the person, assigned by the server. An ASCII string\nwith a max length of 27 characters, in the form of\n`people/`\u003cvar\u003eperson_id\u003c/var\u003e.",
	//       "location": "path",
	//       "pattern": "^people/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updatePersonFields": {
	//       "description": "**Required.** A field mask to restrict which fields on the person are\nupdated. Valid values are:\n\n* addresses\n* biographies\n* birthdays\n* braggingRights\n* emailAddresses\n* events\n* genders\n* imClients\n* interests\n* locales\n* names\n* nicknames\n* occupations\n* organizations\n* phoneNumbers\n* relations\n* residences\n* skills\n* urls",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resourceName}:updateContact",
	//   "request": {
	//     "$ref": "Person"
	//   },
	//   "response": {
	//     "$ref": "Person"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts"
	//   ]
	// }

}

// method id "people.people.connections.list":

type PeopleConnectionsListCall struct {
	s            *Service
	resourceName string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Provides a list of the authenticated user's contacts merged
// with any
// connected profiles.
// <br>
// The request throws a 400 error if 'personFields' is not specified.
func (r *PeopleConnectionsService) List(resourceName string) *PeopleConnectionsListCall {
	c := &PeopleConnectionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resourceName = resourceName
	return c
}

// PageSize sets the optional parameter "pageSize": The number of
// connections to include in the response. Valid values are
// between 1 and 2000, inclusive. Defaults to 100.
func (c *PeopleConnectionsListCall) PageSize(pageSize int64) *PeopleConnectionsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The token of the
// page to be returned.
func (c *PeopleConnectionsListCall) PageToken(pageToken string) *PeopleConnectionsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// PersonFields sets the optional parameter "personFields":
// **Required.** A field mask to restrict which fields on each person
// are
// returned. Valid values are:
//
// * addresses
// * ageRanges
// * biographies
// * birthdays
// * braggingRights
// * coverPhotos
// * emailAddresses
// * events
// * genders
// * imClients
// * interests
// * locales
// * memberships
// * metadata
// * names
// * nicknames
// * occupations
// * organizations
// * phoneNumbers
// * photos
// * relations
// * relationshipInterests
// * relationshipStatuses
// * residences
// * skills
// * taglines
// * urls
func (c *PeopleConnectionsListCall) PersonFields(personFields string) *PeopleConnectionsListCall {
	c.urlParams_.Set("personFields", personFields)
	return c
}

// RequestMaskIncludeField sets the optional parameter
// "requestMask.includeField": **Required.** Comma-separated list of
// person fields to be included in the
// response. Each path should start with `person.`: for
// example,
// `person.names` or `person.photos`.
func (c *PeopleConnectionsListCall) RequestMaskIncludeField(requestMaskIncludeField string) *PeopleConnectionsListCall {
	c.urlParams_.Set("requestMask.includeField", requestMaskIncludeField)
	return c
}

// RequestSyncToken sets the optional parameter "requestSyncToken":
// Whether the response should include a sync token, which can be used
// to get
// all changes since the last request.
func (c *PeopleConnectionsListCall) RequestSyncToken(requestSyncToken bool) *PeopleConnectionsListCall {
	c.urlParams_.Set("requestSyncToken", fmt.Sprint(requestSyncToken))
	return c
}

// SortOrder sets the optional parameter "sortOrder": The order in which
// the connections should be sorted. Defaults
// to
// `LAST_MODIFIED_ASCENDING`.
//
// Possible values:
//   "LAST_MODIFIED_ASCENDING"
//   "FIRST_NAME_ASCENDING"
//   "LAST_NAME_ASCENDING"
func (c *PeopleConnectionsListCall) SortOrder(sortOrder string) *PeopleConnectionsListCall {
	c.urlParams_.Set("sortOrder", sortOrder)
	return c
}

// SyncToken sets the optional parameter "syncToken": A sync token,
// returned by a previous call to `people.connections.list`.
// Only resources changed since the sync token was created will be
// returned.
func (c *PeopleConnectionsListCall) SyncToken(syncToken string) *PeopleConnectionsListCall {
	c.urlParams_.Set("syncToken", syncToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PeopleConnectionsListCall) Fields(s ...googleapi.Field) *PeopleConnectionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PeopleConnectionsListCall) IfNoneMatch(entityTag string) *PeopleConnectionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PeopleConnectionsListCall) Context(ctx context.Context) *PeopleConnectionsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PeopleConnectionsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PeopleConnectionsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/{+resourceName}/connections")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"resourceName": c.resourceName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "people.people.connections.list" call.
// Exactly one of *ListConnectionsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListConnectionsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PeopleConnectionsListCall) Do(opts ...googleapi.CallOption) (*ListConnectionsResponse, error) {
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
	ret := &ListConnectionsResponse{
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
	//   "description": "Provides a list of the authenticated user's contacts merged with any\nconnected profiles.\n\u003cbr\u003e\nThe request throws a 400 error if 'personFields' is not specified.",
	//   "flatPath": "v1/people/{peopleId}/connections",
	//   "httpMethod": "GET",
	//   "id": "people.people.connections.list",
	//   "parameterOrder": [
	//     "resourceName"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "The number of connections to include in the response. Valid values are\nbetween 1 and 2000, inclusive. Defaults to 100.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The token of the page to be returned.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "personFields": {
	//       "description": "**Required.** A field mask to restrict which fields on each person are\nreturned. Valid values are:\n\n* addresses\n* ageRanges\n* biographies\n* birthdays\n* braggingRights\n* coverPhotos\n* emailAddresses\n* events\n* genders\n* imClients\n* interests\n* locales\n* memberships\n* metadata\n* names\n* nicknames\n* occupations\n* organizations\n* phoneNumbers\n* photos\n* relations\n* relationshipInterests\n* relationshipStatuses\n* residences\n* skills\n* taglines\n* urls",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMask.includeField": {
	//       "description": "**Required.** Comma-separated list of person fields to be included in the\nresponse. Each path should start with `person.`: for example,\n`person.names` or `person.photos`.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestSyncToken": {
	//       "description": "Whether the response should include a sync token, which can be used to get\nall changes since the last request.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "resourceName": {
	//       "description": "The resource name to return connections for. Only `people/me` is valid.",
	//       "location": "path",
	//       "pattern": "^people/[^/]+$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sortOrder": {
	//       "description": "The order in which the connections should be sorted. Defaults to\n`LAST_MODIFIED_ASCENDING`.",
	//       "enum": [
	//         "LAST_MODIFIED_ASCENDING",
	//         "FIRST_NAME_ASCENDING",
	//         "LAST_NAME_ASCENDING"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "syncToken": {
	//       "description": "A sync token, returned by a previous call to `people.connections.list`.\nOnly resources changed since the sync token was created will be returned.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/{+resourceName}/connections",
	//   "response": {
	//     "$ref": "ListConnectionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/contacts",
	//     "https://www.googleapis.com/auth/contacts.readonly"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *PeopleConnectionsListCall) Pages(ctx context.Context, f func(*ListConnectionsResponse) error) error {
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
