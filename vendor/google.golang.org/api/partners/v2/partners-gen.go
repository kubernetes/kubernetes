// Package partners provides access to the Google Partners API.
//
// See https://developers.google.com/partners/
//
// Usage example:
//
//   import "google.golang.org/api/partners/v2"
//   ...
//   partnersService, err := partners.New(oauthHttpClient)
package partners // import "google.golang.org/api/partners/v2"

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

const apiId = "partners:v2"
const apiName = "partners"
const apiVersion = "v2"
const basePath = "https://partners.googleapis.com/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.ClientMessages = NewClientMessagesService(s)
	s.Companies = NewCompaniesService(s)
	s.UserEvents = NewUserEventsService(s)
	s.UserStates = NewUserStatesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	ClientMessages *ClientMessagesService

	Companies *CompaniesService

	UserEvents *UserEventsService

	UserStates *UserStatesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewClientMessagesService(s *Service) *ClientMessagesService {
	rs := &ClientMessagesService{s: s}
	return rs
}

type ClientMessagesService struct {
	s *Service
}

func NewCompaniesService(s *Service) *CompaniesService {
	rs := &CompaniesService{s: s}
	rs.Leads = NewCompaniesLeadsService(s)
	return rs
}

type CompaniesService struct {
	s *Service

	Leads *CompaniesLeadsService
}

func NewCompaniesLeadsService(s *Service) *CompaniesLeadsService {
	rs := &CompaniesLeadsService{s: s}
	return rs
}

type CompaniesLeadsService struct {
	s *Service
}

func NewUserEventsService(s *Service) *UserEventsService {
	rs := &UserEventsService{s: s}
	return rs
}

type UserEventsService struct {
	s *Service
}

func NewUserStatesService(s *Service) *UserStatesService {
	rs := &UserStatesService{s: s}
	return rs
}

type UserStatesService struct {
	s *Service
}

// CertificationExamStatus: Status for a Google Partners certification
// exam.
type CertificationExamStatus struct {
	// NumberUsersPass: The number of people who have passed the
	// certification exam.
	NumberUsersPass int64 `json:"numberUsersPass,omitempty"`

	// Type: The type of certification exam.
	//
	// Possible values:
	//   "CERTIFICATION_EXAM_TYPE_UNSPECIFIED"
	//   "CET_ADWORDS_ADVANCED_SEARCH"
	//   "CET_ADWORDS_ADVANCED_DISPLAY"
	//   "CET_VIDEO_ADS"
	//   "CET_ANALYTICS"
	//   "CET_DOUBLECLICK"
	//   "CET_SHOPPING"
	//   "CET_MOBILE"
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NumberUsersPass") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CertificationExamStatus) MarshalJSON() ([]byte, error) {
	type noMethod CertificationExamStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CertificationStatus: Google Partners certification status.
type CertificationStatus struct {
	// ExamStatuses: List of certification exam statuses.
	ExamStatuses []*CertificationExamStatus `json:"examStatuses,omitempty"`

	// IsCertified: Whether certification is passing.
	IsCertified bool `json:"isCertified,omitempty"`

	// Type: The type of the certification.
	//
	// Possible values:
	//   "CERTIFICATION_TYPE_UNSPECIFIED"
	//   "CT_ADWORDS"
	//   "CT_YOUTUBE"
	//   "CT_VIDEOADS"
	//   "CT_ANALYTICS"
	//   "CT_DOUBLECLICK"
	//   "CT_SHOPPING"
	//   "CT_MOBILE"
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExamStatuses") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CertificationStatus) MarshalJSON() ([]byte, error) {
	type noMethod CertificationStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Company: A company resource in the Google Partners API. Once
// certified, it qualifies for being searched by advertisers.
type Company struct {
	// CertificationStatuses: The list of Google Partners certification
	// statuses for the company.
	CertificationStatuses []*CertificationStatus `json:"certificationStatuses,omitempty"`

	// ConvertedMinMonthlyBudget: The minimum monthly budget that the
	// company accepts for partner business, converted to the requested
	// currency code.
	ConvertedMinMonthlyBudget *Money `json:"convertedMinMonthlyBudget,omitempty"`

	// Id: The ID of the company.
	Id string `json:"id,omitempty"`

	// Industries: Industries the company can help with.
	//
	// Possible values:
	//   "INDUSTRY_UNSPECIFIED" - Unchosen.
	//   "I_AUTOMOTIVE" - The automotive industry.
	//   "I_BUSINESS_TO_BUSINESS" - The business-to-business industry.
	//   "I_CONSUMER_PACKAGED_GOODS" - The consumer packaged goods industry.
	//   "I_EDUCATION" - The education industry.
	//   "I_FINANCE" - The finance industry.
	//   "I_HEALTHCARE" - The healthcare industry.
	//   "I_MEDIA_AND_ENTERTAINMENT" - The media and entertainment industry.
	//   "I_RETAIL" - The retail industry.
	//   "I_TECHNOLOGY" - The technology industry.
	//   "I_TRAVEL" - The travel industry.
	Industries []string `json:"industries,omitempty"`

	// LocalizedInfos: The list of localized info for the company.
	LocalizedInfos []*LocalizedCompanyInfo `json:"localizedInfos,omitempty"`

	// Locations: The list of company locations.
	Locations []*Location `json:"locations,omitempty"`

	// Name: The name of the company.
	Name string `json:"name,omitempty"`

	// OriginalMinMonthlyBudget: The unconverted minimum monthly budget that
	// the company accepts for partner business.
	OriginalMinMonthlyBudget *Money `json:"originalMinMonthlyBudget,omitempty"`

	// PublicProfile: Basic information from the company's public profile.
	PublicProfile *PublicProfile `json:"publicProfile,omitempty"`

	// Ranks: Information related to the ranking of the company within the
	// list of companies.
	Ranks []*Rank `json:"ranks,omitempty"`

	// Services: Services the company can help with.
	//
	// Possible values:
	//   "SERVICE_UNSPECIFIED" - Unchosen.
	//   "S_ADVANCED_ADWORDS_SUPPORT" - Help with advanced AdWords support.
	//   "S_ADVERTISING_ON_GOOGLE" - Help with advertising on Google.
	//   "S_AN_ENHANCED_WEBSITE" - Help with an enhanced website.
	//   "S_AN_ONLINE_MARKETING_PLAN" - Help with an online marketing plan.
	//   "S_MOBILE_AND_VIDEO_ADS" - Help with mobile and video ads.
	Services []string `json:"services,omitempty"`

	// WebsiteUrl: URL of the company's website.
	WebsiteUrl string `json:"websiteUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "CertificationStatuses") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Company) MarshalJSON() ([]byte, error) {
	type noMethod Company
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CreateLeadRequest: Request message for CreateLead.
type CreateLeadRequest struct {
	// Lead: The lead resource. The `LeadType` must not be
	// `LEAD_TYPE_UNSPECIFIED` and either `email` or `phone_number` must be
	// provided.
	Lead *Lead `json:"lead,omitempty"`

	// RecaptchaChallenge: reCaptcha challenge info.
	RecaptchaChallenge *RecaptchaChallenge `json:"recaptchaChallenge,omitempty"`

	// RequestMetadata: Current request metadata.
	RequestMetadata *RequestMetadata `json:"requestMetadata,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Lead") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreateLeadRequest) MarshalJSON() ([]byte, error) {
	type noMethod CreateLeadRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CreateLeadResponse: Response message for CreateLead. Debug
// information about this request.
type CreateLeadResponse struct {
	// Lead: Lead that was created depending on the outcome of reCaptcha
	// validation.
	Lead *Lead `json:"lead,omitempty"`

	// RecaptchaStatus: The outcome of reCaptcha validation.
	//
	// Possible values:
	//   "RECAPTCHA_STATUS_UNSPECIFIED"
	//   "RS_NOT_NEEDED"
	//   "RS_PASSED"
	//   "RS_FAILED"
	RecaptchaStatus string `json:"recaptchaStatus,omitempty"`

	// ResponseMetadata: Current response metadata.
	ResponseMetadata *ResponseMetadata `json:"responseMetadata,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Lead") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CreateLeadResponse) MarshalJSON() ([]byte, error) {
	type noMethod CreateLeadResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DebugInfo: Debug information about this request.
type DebugInfo struct {
	// ServerInfo: Info about the server that serviced this request.
	ServerInfo string `json:"serverInfo,omitempty"`

	// ServerTraceInfo: Server-side debug stack trace.
	ServerTraceInfo string `json:"serverTraceInfo,omitempty"`

	// ServiceUrl: URL of the service that handled this request.
	ServiceUrl string `json:"serviceUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ServerInfo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DebugInfo) MarshalJSON() ([]byte, error) {
	type noMethod DebugInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// EventData: Key value data pair for an event.
type EventData struct {
	// Key: Data type.
	//
	// Possible values:
	//   "EVENT_DATA_TYPE_UNSPECIFIED"
	//   "ACTION"
	//   "AGENCY_ID"
	//   "AGENCY_NAME"
	//   "AGENCY_PHONE_NUMBER"
	//   "AGENCY_WEBSITE"
	//   "BUDGET"
	//   "CENTER_POINT"
	//   "CERTIFICATION"
	//   "COMMENT"
	//   "COUNTRY"
	//   "CURRENCY"
	//   "CURRENTLY_VIEWED_AGENCY_ID"
	//   "DISTANCE"
	//   "DISTANCE_TYPE"
	//   "EXAM"
	//   "HISTORY_TOKEN"
	//   "IDENTIFIER"
	//   "INDUSTRY"
	//   "INSIGHT_TAG"
	//   "LANGUAGE"
	//   "LOCATION"
	//   "MARKETING_OPT_IN"
	//   "QUERY"
	//   "SEARCH_START_INDEX"
	//   "SERVICE"
	//   "SHOW_VOW"
	//   "SOLUTION"
	//   "TRAFFIC_SOURCE_ID"
	//   "TRAFFIC_SUB_ID"
	//   "VIEW_PORT"
	//   "WEBSITE"
	//   "DETAILS"
	//   "EXPERIMENT_ID"
	//   "GPS_MOTIVATION"
	//   "URL"
	//   "ELEMENT_FOCUS"
	//   "PROGRESS"
	Key string `json:"key,omitempty"`

	// Values: Data values.
	Values []string `json:"values,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Key") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *EventData) MarshalJSON() ([]byte, error) {
	type noMethod EventData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// GetCompanyResponse: Response message for GetCompany.
type GetCompanyResponse struct {
	// Company: The company.
	Company *Company `json:"company,omitempty"`

	// ResponseMetadata: Current response metadata.
	ResponseMetadata *ResponseMetadata `json:"responseMetadata,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Company") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GetCompanyResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetCompanyResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LatLng: An object representing a latitude/longitude pair. This is
// expressed as a pair of doubles representing degrees latitude and
// degrees longitude. Unless specified otherwise, this must conform to
// the WGS84 standard. Values must be within normalized ranges. Example
// of normalization code in Python: def NormalizeLongitude(longitude):
// """Wrapsdecimal degrees longitude to [-180.0, 180.0].""" q, r =
// divmod(longitude, 360.0) if r > 180.0 or (r == 180.0 and q <= -1.0):
// return r - 360.0 return r def NormalizeLatLng(latitude, longitude):
// """Wraps decimal degrees latitude and longitude to [-180.0, 180.0]
// and [-90.0, 90.0], respectively.""" r = latitude % 360.0 if r =
// 270.0: return r - 360, NormalizeLongitude(longitude) else: return 180
// - r, NormalizeLongitude(longitude + 180.0) assert 180.0 ==
// NormalizeLongitude(180.0) assert -180.0 == NormalizeLongitude(-180.0)
// assert -179.0 == NormalizeLongitude(181.0) assert (0.0, 0.0) ==
// NormalizeLatLng(360.0, 0.0) assert (0.0, 0.0) ==
// NormalizeLatLng(-360.0, 0.0) assert (85.0, 180.0) ==
// NormalizeLatLng(95.0, 0.0) assert (-85.0, -170.0) ==
// NormalizeLatLng(-95.0, 10.0) assert (90.0, 10.0) ==
// NormalizeLatLng(90.0, 10.0) assert (-90.0, -10.0) ==
// NormalizeLatLng(-90.0, -10.0) assert (0.0, -170.0) ==
// NormalizeLatLng(-180.0, 10.0) assert (0.0, -170.0) ==
// NormalizeLatLng(180.0, 10.0) assert (-90.0, 10.0) ==
// NormalizeLatLng(270.0, 10.0) assert (90.0, 10.0) ==
// NormalizeLatLng(-270.0, 10.0)
type LatLng struct {
	// Latitude: The latitude in degrees. It must be in the range [-90.0,
	// +90.0].
	Latitude float64 `json:"latitude,omitempty"`

	// Longitude: The longitude in degrees. It must be in the range [-180.0,
	// +180.0].
	Longitude float64 `json:"longitude,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Latitude") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LatLng) MarshalJSON() ([]byte, error) {
	type noMethod LatLng
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Lead: A lead resource that represents an advertiser contact for a
// `Company`. These are usually generated via Google Partner Search (the
// advertiser portal).
type Lead struct {
	// Comments: Comments lead source gave.
	Comments string `json:"comments,omitempty"`

	// Email: Email address of lead source.
	Email string `json:"email,omitempty"`

	// FamilyName: Last name of lead source.
	FamilyName string `json:"familyName,omitempty"`

	// GivenName: First name of lead source.
	GivenName string `json:"givenName,omitempty"`

	// GpsMotivations: List of reasons for using Google Partner Search and
	// creating a lead.
	//
	// Possible values:
	//   "GPS_MOTIVATION_UNSPECIFIED" - Unchosen.
	//   "GPSM_HELP_WITH_ADVERTISING" - Advertiser needs help with their
	// advertising.
	//   "GPSM_HELP_WITH_WEBSITE" - Advertiser needs help with their
	// website.
	//   "GPSM_NO_WEBSITE" - Advertiser does not have a website.
	GpsMotivations []string `json:"gpsMotivations,omitempty"`

	// Id: ID of the lead.
	Id string `json:"id,omitempty"`

	// MinMonthlyBudget: The minimum monthly budget lead source is willing
	// to spend.
	MinMonthlyBudget *Money `json:"minMonthlyBudget,omitempty"`

	// PhoneNumber: Phone number of lead source.
	PhoneNumber string `json:"phoneNumber,omitempty"`

	// Type: Type of lead.
	//
	// Possible values:
	//   "LEAD_TYPE_UNSPECIFIED"
	//   "LT_GPS"
	Type string `json:"type,omitempty"`

	// WebsiteUrl: Website URL of lead source.
	WebsiteUrl string `json:"websiteUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Comments") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Lead) MarshalJSON() ([]byte, error) {
	type noMethod Lead
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListCompaniesResponse: Response message for ListCompanies.
type ListCompaniesResponse struct {
	// Companies: The list of companies.
	Companies []*Company `json:"companies,omitempty"`

	// NextPageToken: A token to retrieve next page of results. Pass this
	// value in the `ListCompaniesRequest.page_token` field in the
	// subsequent call to ListCompanies to retrieve the next page of
	// results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ResponseMetadata: Current response metadata.
	ResponseMetadata *ResponseMetadata `json:"responseMetadata,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Companies") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListCompaniesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListCompaniesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListUserStatesResponse: Response message for ListUserStates.
type ListUserStatesResponse struct {
	// ResponseMetadata: Current response metadata.
	ResponseMetadata *ResponseMetadata `json:"responseMetadata,omitempty"`

	// UserStates: User's states.
	//
	// Possible values:
	//   "USER_STATE_UNSPECIFIED" - Unchosen.
	//   "US_REQUIRES_RECAPTCHA_FOR_GPS_CONTACT" - User must pass reCaptcha
	// to contact a Partner via Google Partner Search.
	UserStates []string `json:"userStates,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ResponseMetadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListUserStatesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListUserStatesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LocalizedCompanyInfo: The localized company information.
type LocalizedCompanyInfo struct {
	// CountryCodes: List of country codes for the localized company info.
	CountryCodes []string `json:"countryCodes,omitempty"`

	// DisplayName: Localized display name.
	DisplayName string `json:"displayName,omitempty"`

	// LanguageCode: Language code of the localized company info, as defined
	// by BCP 47 (IETF BCP 47, "Tags for Identifying Languages").
	LanguageCode string `json:"languageCode,omitempty"`

	// Overview: Localized brief description that the company uses to
	// advertise themselves.
	Overview string `json:"overview,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CountryCodes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LocalizedCompanyInfo) MarshalJSON() ([]byte, error) {
	type noMethod LocalizedCompanyInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Location: A location with address and geographic coordinates.
type Location struct {
	// Address: The complete address of the location.
	Address string `json:"address,omitempty"`

	// LatLng: The latitude and longitude of the location, in degrees.
	LatLng *LatLng `json:"latLng,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Address") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Location) MarshalJSON() ([]byte, error) {
	type noMethod Location
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LogMessageRequest: Request message for LogClientMessage.
type LogMessageRequest struct {
	// ClientInfo: Map of client info, such as URL, browser navigator,
	// browser platform, etc.
	ClientInfo map[string]string `json:"clientInfo,omitempty"`

	// Details: Details about the client message.
	Details string `json:"details,omitempty"`

	// Level: Message level of client message.
	//
	// Possible values:
	//   "MESSAGE_LEVEL_UNSPECIFIED"
	//   "ML_FINE"
	//   "ML_INFO"
	//   "ML_WARNING"
	//   "ML_SEVERE"
	Level string `json:"level,omitempty"`

	// RequestMetadata: Current request metadata.
	RequestMetadata *RequestMetadata `json:"requestMetadata,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClientInfo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LogMessageRequest) MarshalJSON() ([]byte, error) {
	type noMethod LogMessageRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LogMessageResponse: Response message for LogClientMessage.
type LogMessageResponse struct {
	// ResponseMetadata: Current response metadata.
	ResponseMetadata *ResponseMetadata `json:"responseMetadata,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ResponseMetadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LogMessageResponse) MarshalJSON() ([]byte, error) {
	type noMethod LogMessageResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LogUserEventRequest: Request message for LogUserEvent.
type LogUserEventRequest struct {
	// EventAction: The action that occurred.
	//
	// Possible values:
	//   "EVENT_ACTION_UNSPECIFIED"
	//   "SMB_CLICKED_FIND_A_PARTNER_BUTTON_BOTTOM"
	//   "SMB_CLICKED_FIND_A_PARTNER_BUTTON_TOP"
	//   "AGENCY_CLICKED_JOIN_NOW_BUTTON_BOTTOM"
	//   "AGENCY_CLICKED_JOIN_NOW_BUTTON_TOP"
	//   "SMB_CANCELED_PARTNER_CONTACT_FORM"
	//   "SMB_CLICKED_CONTACT_A_PARTNER"
	//   "SMB_COMPLETED_PARTNER_CONTACT_FORM"
	//   "SMB_ENTERED_EMAIL_IN_CONTACT_PARTNER_FORM"
	//   "SMB_ENTERED_NAME_IN_CONTACT_PARTNER_FORM"
	//   "SMB_ENTERED_PHONE_IN_CONTACT_PARTNER_FORM"
	//   "SMB_FAILED_RECAPTCHA_IN_CONTACT_PARTNER_FORM"
	//   "PARTNER_VIEWED_BY_SMB"
	//   "SMB_CANCELED_PARTNER_CONTACT_FORM_ON_GPS"
	//   "SMB_CHANGED_A_SEARCH_PARAMETER_TOP"
	//   "SMB_CLICKED_CONTACT_A_PARTNER_ON_GPS"
	//   "SMB_CLICKED_SHOW_MORE_PARTNERS_BUTTON_BOTTOM"
	//   "SMB_COMPLETED_PARTNER_CONTACT_FORM_ON_GPS"
	//   "SMB_NO_PARTNERS_AVAILABLE_WITH_SEARCH_CRITERIA"
	//   "SMB_PERFORMED_SEARCH_ON_GPS"
	//   "SMB_VIEWED_A_PARTNER_ON_GPS"
	//   "SMB_CANCELED_PARTNER_CONTACT_FORM_ON_PROFILE_PAGE"
	//   "SMB_CLICKED_CONTACT_A_PARTNER_ON_PROFILE_PAGE"
	//   "SMB_CLICKED_PARTNER_WEBSITE"
	//   "SMB_COMPLETED_PARTNER_CONTACT_FORM_ON_PROFILE_PAGE"
	//   "SMB_VIEWED_A_PARTNER_PROFILE"
	//   "AGENCY_CLICKED_ACCEPT_TOS_BUTTON"
	//   "AGENCY_CHANGED_TOS_COUNTRY"
	//   "AGENCY_ADDED_ADDRESS_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_ADDED_PHONE_NUMBER_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_CHANGED_PRIMARY_ACCOUNT_ASSOCIATION"
	//   "AGENCY_CHANGED_PRIMARY_COUNTRY_ASSOCIATION"
	//   "AGENCY_CLICKED_AFFILIATE_BUTTON_IN_MY_PROFILE_IN_PORTAL"
	//   "AGENCY_CLICKED_GIVE_EDIT_ACCESS_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_CLICKED_LOG_OUT_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_CLICKED_MY_PROFILE_LEFT_NAV_IN_PORTAL"
	//   "AGENCY_CLICKED_SAVE_AND_CONTINUE_AT_BOT_OF_COMPLETE_PROFILE"
	//   "AGENCY_CLICKED_UNAFFILIATE_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_FILLED_OUT_COMP_AFFILIATION_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_SUCCESSFULLY_CONNECTED_WITH_COMPANY_IN_MY_PROFILE"
	//   "AGENCY_CLICKED_CREATE_MCC_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_DIDNT_HAVE_AN_MCC_ASSOCIATED_ON_COMPLETE_PROFILE"
	//   "AGENCY_HAD_AN_MCC_ASSOCIATED_ON_COMPLETE_PROFILE"
	//   "AGENCY_ADDED_JOB_FUNCTION_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_LOOKED_AT_JOB_FUNCTION_DROP_DOWN"
	//   "AGENCY_SELECTED_ACCOUNT_MANAGER_AS_JOB_FUNCTION"
	//   "AGENCY_SELECTED_ACCOUNT_PLANNER_AS_JOB_FUNCTION"
	//   "AGENCY_SELECTED_ANALYTICS_AS_JOB_FUNCTION"
	//   "AGENCY_SELECTED_CREATIVE_AS_JOB_FUNCTION"
	//   "AGENCY_SELECTED_MEDIA_BUYER_AS_JOB_FUNCTION"
	//   "AGENCY_SELECTED_MEDIA_PLANNER_AS_JOB_FUNCTION"
	//   "AGENCY_SELECTED_OTHER_AS_JOB_FUNCTION"
	//   "AGENCY_SELECTED_PRODUCTION_AS_JOB_FUNCTION"
	//   "AGENCY_SELECTED_SEO_AS_JOB_FUNCTION"
	//   "AGENCY_SELECTED_SALES_REP_AS_JOB_FUNCTION"
	//   "AGENCY_SELECTED_SEARCH_SPECIALIST_AS_JOB_FUNCTION"
	//   "AGENCY_ADDED_CHANNELS_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_LOOKED_AT_ADD_CHANNEL_DROP_DOWN"
	//   "AGENCY_SELECTED_CROSS_CHANNEL_FROM_ADD_CHANNEL"
	//   "AGENCY_SELECTED_DISPLAY_FROM_ADD_CHANNEL"
	//   "AGENCY_SELECTED_MOBILE_FROM_ADD_CHANNEL"
	//   "AGENCY_SELECTED_SEARCH_FROM_ADD_CHANNEL"
	//   "AGENCY_SELECTED_SOCIAL_FROM_ADD_CHANNEL"
	//   "AGENCY_SELECTED_TOOLS_FROM_ADD_CHANNEL"
	//   "AGENCY_SELECTED_YOUTUBE_FROM_ADD_CHANNEL"
	//   "AGENCY_ADDED_INDUSTRIES_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_CHANGED_ADD_INDUSTRIES_DROP_DOWN"
	//   "AGENCY_ADDED_MARKETS_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_CHANGED_ADD_MARKETS_DROP_DOWN"
	//   "AGENCY_CHECKED_RECIEVE_MAIL_PROMOTIONS_MYPROFILE"
	//   "AGENCY_CHECKED_RECIEVE_MAIL_PROMOTIONS_SIGNUP"
	//   "AGENCY_SELECTED_OPT_IN_BETA_TESTS_AND_MKT_RESEARCH"
	//   "AGENCY_SELECTED_OPT_IN_BETA_TESTS_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_SELECTED_OPT_IN_NEWS_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_SELECTED_OPT_IN_NEWS_INVITATIONS_AND_PROMOS"
	//   "AGENCY_SELECTED_OPT_IN_PERFORMANCE_SUG_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_SELECTED_OPT_IN_PERFORMANCE_SUGGESTIONS"
	//   "AGENCY_SELECTED_OPT_IN_SELECT_ALL_EMAIL_NOTIFICATIONS"
	//   "AGENCY_SELECTED_SELECT_ALL_OPT_INS_IN_MY_PROFILE_PORTAL"
	//   "AGENCY_CLICKED_BACK_BUTTON_ON_CONNECT_WITH_COMPANY"
	//   "AGENCY_CLICKED_CONTINUE_TO_OVERVIEW_ON_CONNECT_WITH_COMPANY"
	//   "AGECNY_CLICKED_CREATE_MCC_CONNECT_WITH_COMPANY_NOT_FOUND"
	//   "AGECNY_CLICKED_GIVE_EDIT_ACCESS_CONNECT_WITH_COMPANY_NOT_FOUND"
	//   "AGECNY_CLICKED_LOG_OUT_CONNECT_WITH_COMPANY_NOT_FOUND"
	//   "AGENCY_CLICKED_SKIP_FOR_NOW_ON_CONNECT_WITH_COMPANY_PAGE"
	//   "AGENCY_CLOSED_CONNECTED_TO_COMPANY_X_BUTTON_WRONG_COMPANY"
	//   "AGENCY_COMPLETED_FIELD_CONNECT_WITH_COMPANY"
	//   "AGECNY_FOUND_COMPANY_TO_CONNECT_WITH"
	//   "AGENCY_SUCCESSFULLY_CREATED_COMPANY"
	//   "AGENCY_ADDED_NEW_COMPANY_LOCATION"
	//   "AGENCY_CLICKED_COMMUNITY_JOIN_NOW_LINK_IN_PORTAL_NOTIFICATIONS"
	//   "AGENCY_CLICKED_CONNECT_TO_COMPANY_LINK_IN_PORTAL_NOTIFICATIONS"
	//   "AGENCY_CLICKED_GET_CERTIFIED_LINK_IN_PORTAL_NOTIFICATIONS"
	//
	// "AGENCY_CLICKED_GET_VIDEO_ADS_CERTIFIED_LINK_IN_PORTAL_NOTIFICATIONS"
	//   "AGENCY_CLICKED_LINK_TO_MCC_LINK_IN_PORTAL_NOTIFICATIONS"
	//   "AGENCY_CLICKED_INSIGHT_CONTENT_IN_PORTAL"
	//   "AGENCY_CLICKED_INSIGHTS_VIEW_NOW_PITCH_DECKS_IN_PORTAL"
	//   "AGENCY_CLICKED_INSIGHTS_LEFT_NAV_IN_PORTAL"
	//   "AGENCY_CLICKED_INSIGHTS_UPLOAD_CONTENT"
	//   "AGENCY_CLICKED_INSIGHTS_VIEWED_DEPRECATED"
	//   "AGENCY_CLICKED_COMMUNITY_LEFT_NAV_IN_PORTAL"
	//   "AGENCY_CLICKED_JOIN_COMMUNITY_BUTTON_COMMUNITY_PORTAL"
	//   "AGENCY_CLICKED_CERTIFICATIONS_LEFT_NAV_IN_PORTAL"
	//   "AGENCY_CLICKED_CERTIFICATIONS_PRODUCT_LEFT_NAV_IN_PORTAL"
	//   "AGENCY_CLICKED_PARTNER_STATUS_LEFT_NAV_IN_PORTAL"
	//   "AGENCY_CLICKED_PARTNER_STATUS_PRODUCT_LEFT_NAV_IN_PORTAL"
	//   "AGENCY_CLICKED_OFFERS_LEFT_NAV_IN_PORTAL"
	//   "AGENCY_CLICKED_SEND_BUTTON_ON_OFFERS_PAGE"
	//   "AGENCY_CLICKED_EXAM_DETAILS_ON_CERT_ADWORDS_PAGE"
	//   "AGENCY_CLICKED_SEE_EXAMS_CERTIFICATION_MAIN_PAGE"
	//   "AGENCY_CLICKED_TAKE_EXAM_ON_CERT_EXAM_PAGE"
	//   "AGENCY_OPENED_LAST_ADMIN_DIALOG"
	//   "AGENCY_OPENED_DIALOG_WITH_NO_USERS"
	//   "AGENCY_PROMOTED_USER_TO_ADMIN"
	//   "AGENCY_UNAFFILIATED"
	//   "AGENCY_CHANGED_ROLES"
	//   "SMB_CLICKED_COMPANY_NAME_LINK_TO_PROFILE"
	//   "SMB_VIEWED_ADWORDS_CERTIFICATE"
	//   "SMB_CLICKED_ADWORDS_CERTIFICATE_HELP_ICON"
	//   "SMB_VIEWED_ANALYTICS_CERTIFICATE"
	//   "SMB_VIEWED_DOUBLECLICK_CERTIFICATE"
	//   "SMB_VIEWED_VIDEO_ADS_CERTIFICATE"
	//   "SMB_VIEWED_SHOPPING_CERTIFICATE"
	//   "SMB_CLICKED_VIDEO_ADS_CERTIFICATE_HELP_ICON"
	//   "CLICKED_HELP_AT_BOTTOM"
	//   "CLICKED_HELP_AT_TOP"
	//   "CLIENT_ERROR"
	//   "AGENCY_CLICKED_LEFT_NAV_STORIES"
	//   "CLICKED"
	//   "SMB_VIEWED_MOBILE_CERTIFICATE"
	//   "AGENCY_FAILED_COMPANY_VERIFICATION"
	//   "VISITED_LANDING"
	//   "VISITED_GPS"
	//   "VISITED_AGENCY_PORTAL"
	//   "CANCELLED_INDIVIDUAL_SIGN_UP"
	//   "CANCELLED_COMPANY_SIGN_UP"
	//   "AGENCY_CLICKED_SIGN_IN_BUTTON_TOP"
	//   "AGENCY_CLICKED_SAVE_AND_CONTINUE_AT_BOT_OF_INCOMPLETE_PROFILE"
	//   "AGENCY_UNSELECTED_OPT_IN_NEWS_INVITATIONS_AND_PROMOS"
	//   "AGENCY_UNSELECTED_OPT_IN_BETA_TESTS_AND_MKT_RESEARCH"
	//   "AGENCY_UNSELECTED_OPT_IN_PERFORMANCE_SUGGESTIONS"
	//   "AGENCY_SELECTED_OPT_OUT_UNSELECT_ALL_EMAIL_NOTIFICATIONS"
	//   "AGENCY_LINKED_INDIVIDUAL_MCC"
	//   "AGENCY_SUGGESTED_TO_USER"
	//   "AGENCY_IGNORED_SUGGESTED_AGENCIES_AND_SEARCHED"
	//   "AGENCY_PICKED_SUGGESTED_AGENCY"
	//   "AGENCY_SEARCHED_FOR_AGENCIES"
	//   "AGENCY_PICKED_SEARCHED_AGENCY"
	//   "AGENCY_DISMISSED_AFFILIATION_WIDGET"
	//   "AGENCY_CLICKED_INSIGHTS_DOWNLOAD_CONTENT"
	//   "AGENCY_PROGRESS_INSIGHTS_VIEW_CONTENT"
	//   "AGENCY_CLICKED_CANCEL_ACCEPT_TOS_BUTTON"
	//   "SMB_ENTERED_WEBSITE_IN_CONTACT_PARTNER_FORM"
	EventAction string `json:"eventAction,omitempty"`

	// EventCategory: The category the action belongs to.
	//
	// Possible values:
	//   "EVENT_CATEGORY_UNSPECIFIED"
	//   "GOOGLE_PARTNER_SEARCH"
	//   "GOOGLE_PARTNER_SIGNUP_FLOW"
	//   "GOOGLE_PARTNER_PORTAL"
	//   "GOOGLE_PARTNER_PORTAL_MY_PROFILE"
	//   "GOOGLE_PARTNER_PORTAL_CERTIFICATIONS"
	//   "GOOGLE_PARTNER_PORTAL_COMMUNITY"
	//   "GOOGLE_PARTNER_PORTAL_INSIGHTS"
	//   "GOOGLE_PARTNER_PORTAL_CLIENTS"
	//   "GOOGLE_PARTNER_PUBLIC_USER_PROFILE"
	//   "GOOGLE_PARTNER_PANEL"
	//   "GOOGLE_PARTNER_PORTAL_LAST_ADMIN_DIALOG"
	//   "GOOGLE_PARTNER_CLIENT"
	//   "GOOGLE_PARTNER_PORTAL_COMPANY_PROFILE"
	//   "EXTERNAL_LINKS"
	//   "GOOGLE_PARTNER_LANDING"
	EventCategory string `json:"eventCategory,omitempty"`

	// EventDatas: List of event data for the event.
	EventDatas []*EventData `json:"eventDatas,omitempty"`

	// EventScope: The scope of the event.
	//
	// Possible values:
	//   "EVENT_SCOPE_UNSPECIFIED"
	//   "VISITOR"
	//   "SESSION"
	//   "PAGE"
	EventScope string `json:"eventScope,omitempty"`

	// Lead: Advertiser lead information.
	Lead *Lead `json:"lead,omitempty"`

	// RequestMetadata: Current request metadata.
	RequestMetadata *RequestMetadata `json:"requestMetadata,omitempty"`

	// Url: The URL where the event occurred.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EventAction") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LogUserEventRequest) MarshalJSON() ([]byte, error) {
	type noMethod LogUserEventRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LogUserEventResponse: Response message for LogUserEvent.
type LogUserEventResponse struct {
	// ResponseMetadata: Current response metadata.
	ResponseMetadata *ResponseMetadata `json:"responseMetadata,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ResponseMetadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LogUserEventResponse) MarshalJSON() ([]byte, error) {
	type noMethod LogUserEventResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Money: Represents an amount of money with its currency type.
type Money struct {
	// CurrencyCode: The 3-letter currency code defined in ISO 4217.
	CurrencyCode string `json:"currencyCode,omitempty"`

	// Nanos: Number of nano (10^-9) units of the amount. The value must be
	// between -999,999,999 and +999,999,999 inclusive. If `units` is
	// positive, `nanos` must be positive or zero. If `units` is zero,
	// `nanos` can be positive, zero, or negative. If `units` is negative,
	// `nanos` must be negative or zero. For example $-1.75 is represented
	// as `units`=-1 and `nanos`=-750,000,000.
	Nanos int64 `json:"nanos,omitempty"`

	// Units: The whole units of the amount. For example if `currencyCode`
	// is "USD", then 1 unit is one US dollar.
	Units int64 `json:"units,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "CurrencyCode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Money) MarshalJSON() ([]byte, error) {
	type noMethod Money
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PublicProfile: Basic information from a public profile.
type PublicProfile struct {
	// DisplayImageUrl: The URL to the main display image of the public
	// profile.
	DisplayImageUrl string `json:"displayImageUrl,omitempty"`

	// DisplayName: The display name of the public profile.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The ID which can be used to retrieve more details about the
	// public profile.
	Id string `json:"id,omitempty"`

	// Url: The URL of the public profile.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayImageUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PublicProfile) MarshalJSON() ([]byte, error) {
	type noMethod PublicProfile
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Rank: Information related to ranking of results.
type Rank struct {
	// Type: The type of rank.
	//
	// Possible values:
	//   "RANK_TYPE_UNSPECIFIED"
	//   "RT_FINAL_SCORE"
	Type string `json:"type,omitempty"`

	// Value: The numerical value of the rank.
	Value float64 `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Type") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Rank) MarshalJSON() ([]byte, error) {
	type noMethod Rank
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// RecaptchaChallenge: reCaptcha challenge info.
type RecaptchaChallenge struct {
	// Id: The ID of the reCaptcha challenge.
	Id string `json:"id,omitempty"`

	// Response: The response to the reCaptcha challenge.
	Response string `json:"response,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *RecaptchaChallenge) MarshalJSON() ([]byte, error) {
	type noMethod RecaptchaChallenge
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// RequestMetadata: Common data that is in each API request.
type RequestMetadata struct {
	// ExperimentIds: Experiment IDs the current request belongs to.
	ExperimentIds []string `json:"experimentIds,omitempty"`

	// Locale: Locale to use for the current request.
	Locale string `json:"locale,omitempty"`

	// PartnersSessionId: Google Partners session ID.
	PartnersSessionId string `json:"partnersSessionId,omitempty"`

	// TrafficSource: Source of traffic for the current request.
	TrafficSource *TrafficSource `json:"trafficSource,omitempty"`

	// UserOverrides: Values to use instead of the user's respective
	// defaults for the current request. These are only honored by
	// whitelisted products.
	UserOverrides *UserOverrides `json:"userOverrides,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExperimentIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *RequestMetadata) MarshalJSON() ([]byte, error) {
	type noMethod RequestMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ResponseMetadata: Common data that is in each API response.
type ResponseMetadata struct {
	// DebugInfo: Debug information about this request.
	DebugInfo *DebugInfo `json:"debugInfo,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DebugInfo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ResponseMetadata) MarshalJSON() ([]byte, error) {
	type noMethod ResponseMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TrafficSource: Source of traffic for the current request.
type TrafficSource struct {
	// TrafficSourceId: Identifier to indicate where the traffic comes from.
	// An identifier has multiple letters created by a team which redirected
	// the traffic to us.
	TrafficSourceId string `json:"trafficSourceId,omitempty"`

	// TrafficSubId: Second level identifier to indicate where the traffic
	// comes from. An identifier has multiple letters created by a team
	// which redirected the traffic to us.
	TrafficSubId string `json:"trafficSubId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "TrafficSourceId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TrafficSource) MarshalJSON() ([]byte, error) {
	type noMethod TrafficSource
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// UserOverrides: Values to use instead of the user's respective
// defaults. These are only honored by whitelisted products.
type UserOverrides struct {
	// IpAddress: IP address to use instead of the user's geo-located IP
	// address.
	IpAddress string `json:"ipAddress,omitempty"`

	// UserId: Logged-in user ID to impersonate instead of the user's ID.
	UserId string `json:"userId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IpAddress") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *UserOverrides) MarshalJSON() ([]byte, error) {
	type noMethod UserOverrides
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "partners.clientMessages.log":

type ClientMessagesLogCall struct {
	s                 *Service
	logmessagerequest *LogMessageRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
}

// Log: Logs a generic message from the client, such as `Failed to
// render component`, `Profile page is running slow`, `More than 500
// users have accessed this result.`, etc.
func (r *ClientMessagesService) Log(logmessagerequest *LogMessageRequest) *ClientMessagesLogCall {
	c := &ClientMessagesLogCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.logmessagerequest = logmessagerequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ClientMessagesLogCall) QuotaUser(quotaUser string) *ClientMessagesLogCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ClientMessagesLogCall) Fields(s ...googleapi.Field) *ClientMessagesLogCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ClientMessagesLogCall) Context(ctx context.Context) *ClientMessagesLogCall {
	c.ctx_ = ctx
	return c
}

func (c *ClientMessagesLogCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logmessagerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2/clientMessages:log")
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

// Do executes the "partners.clientMessages.log" call.
// Exactly one of *LogMessageResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *LogMessageResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ClientMessagesLogCall) Do() (*LogMessageResponse, error) {
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
	ret := &LogMessageResponse{
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
	//   "description": "Logs a generic message from the client, such as `Failed to render component`, `Profile page is running slow`, `More than 500 users have accessed this result.`, etc.",
	//   "httpMethod": "POST",
	//   "id": "partners.clientMessages.log",
	//   "path": "v2/clientMessages:log",
	//   "request": {
	//     "$ref": "LogMessageRequest"
	//   },
	//   "response": {
	//     "$ref": "LogMessageResponse"
	//   }
	// }

}

// method id "partners.companies.get":

type CompaniesGetCall struct {
	s            *Service
	companyId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets a company.
func (r *CompaniesService) Get(companyId string) *CompaniesGetCall {
	c := &CompaniesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.companyId = companyId
	return c
}

// Address sets the optional parameter "address": The address to use for
// sorting the company's addresses by proximity. If not given, the
// geo-located address of the request is used. Used when order_by is
// set.
func (c *CompaniesGetCall) Address(address string) *CompaniesGetCall {
	c.urlParams_.Set("address", address)
	return c
}

// CurrencyCode sets the optional parameter "currencyCode": If the
// company's budget is in a different currency code than this one, then
// the converted budget is converted to this currency code.
func (c *CompaniesGetCall) CurrencyCode(currencyCode string) *CompaniesGetCall {
	c.urlParams_.Set("currencyCode", currencyCode)
	return c
}

// OrderBy sets the optional parameter "orderBy": How to order addresses
// within the returned company. Currently, only `address` and `address
// desc` is supported which will sorted by closest to farthest in
// distance from given address and farthest to closest distance from
// given address respectively.
func (c *CompaniesGetCall) OrderBy(orderBy string) *CompaniesGetCall {
	c.urlParams_.Set("orderBy", orderBy)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CompaniesGetCall) QuotaUser(quotaUser string) *CompaniesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// RequestMetadataExperimentIds sets the optional parameter
// "requestMetadata.experimentIds": Experiment IDs the current request
// belongs to.
func (c *CompaniesGetCall) RequestMetadataExperimentIds(requestMetadataExperimentIds ...string) *CompaniesGetCall {
	c.urlParams_.SetMulti("requestMetadata.experimentIds", append([]string{}, requestMetadataExperimentIds...))
	return c
}

// RequestMetadataLocale sets the optional parameter
// "requestMetadata.locale": Locale to use for the current request.
func (c *CompaniesGetCall) RequestMetadataLocale(requestMetadataLocale string) *CompaniesGetCall {
	c.urlParams_.Set("requestMetadata.locale", requestMetadataLocale)
	return c
}

// RequestMetadataPartnersSessionId sets the optional parameter
// "requestMetadata.partnersSessionId": Google Partners session ID.
func (c *CompaniesGetCall) RequestMetadataPartnersSessionId(requestMetadataPartnersSessionId string) *CompaniesGetCall {
	c.urlParams_.Set("requestMetadata.partnersSessionId", requestMetadataPartnersSessionId)
	return c
}

// RequestMetadataTrafficSourceTrafficSourceId sets the optional
// parameter "requestMetadata.trafficSource.trafficSourceId": Identifier
// to indicate where the traffic comes from. An identifier has multiple
// letters created by a team which redirected the traffic to us.
func (c *CompaniesGetCall) RequestMetadataTrafficSourceTrafficSourceId(requestMetadataTrafficSourceTrafficSourceId string) *CompaniesGetCall {
	c.urlParams_.Set("requestMetadata.trafficSource.trafficSourceId", requestMetadataTrafficSourceTrafficSourceId)
	return c
}

// RequestMetadataTrafficSourceTrafficSubId sets the optional parameter
// "requestMetadata.trafficSource.trafficSubId": Second level identifier
// to indicate where the traffic comes from. An identifier has multiple
// letters created by a team which redirected the traffic to us.
func (c *CompaniesGetCall) RequestMetadataTrafficSourceTrafficSubId(requestMetadataTrafficSourceTrafficSubId string) *CompaniesGetCall {
	c.urlParams_.Set("requestMetadata.trafficSource.trafficSubId", requestMetadataTrafficSourceTrafficSubId)
	return c
}

// RequestMetadataUserOverridesIpAddress sets the optional parameter
// "requestMetadata.userOverrides.ipAddress": IP address to use instead
// of the user's geo-located IP address.
func (c *CompaniesGetCall) RequestMetadataUserOverridesIpAddress(requestMetadataUserOverridesIpAddress string) *CompaniesGetCall {
	c.urlParams_.Set("requestMetadata.userOverrides.ipAddress", requestMetadataUserOverridesIpAddress)
	return c
}

// RequestMetadataUserOverridesUserId sets the optional parameter
// "requestMetadata.userOverrides.userId": Logged-in user ID to
// impersonate instead of the user's ID.
func (c *CompaniesGetCall) RequestMetadataUserOverridesUserId(requestMetadataUserOverridesUserId string) *CompaniesGetCall {
	c.urlParams_.Set("requestMetadata.userOverrides.userId", requestMetadataUserOverridesUserId)
	return c
}

// View sets the optional parameter "view": The view of `Company`
// resource to be returned. This must not be `COMPANY_VIEW_UNSPECIFIED`.
//
// Possible values:
//   "COMPANY_VIEW_UNSPECIFIED"
//   "CV_GOOGLE_PARTNER_SEARCH"
func (c *CompaniesGetCall) View(view string) *CompaniesGetCall {
	c.urlParams_.Set("view", view)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CompaniesGetCall) Fields(s ...googleapi.Field) *CompaniesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CompaniesGetCall) IfNoneMatch(entityTag string) *CompaniesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CompaniesGetCall) Context(ctx context.Context) *CompaniesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *CompaniesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2/companies/{companyId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"companyId": c.companyId,
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

// Do executes the "partners.companies.get" call.
// Exactly one of *GetCompanyResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GetCompanyResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CompaniesGetCall) Do() (*GetCompanyResponse, error) {
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
	ret := &GetCompanyResponse{
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
	//   "description": "Gets a company.",
	//   "httpMethod": "GET",
	//   "id": "partners.companies.get",
	//   "parameterOrder": [
	//     "companyId"
	//   ],
	//   "parameters": {
	//     "address": {
	//       "description": "The address to use for sorting the company's addresses by proximity. If not given, the geo-located address of the request is used. Used when order_by is set.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "companyId": {
	//       "description": "The ID of the company to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "currencyCode": {
	//       "description": "If the company's budget is in a different currency code than this one, then the converted budget is converted to this currency code.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "orderBy": {
	//       "description": "How to order addresses within the returned company. Currently, only `address` and `address desc` is supported which will sorted by closest to farthest in distance from given address and farthest to closest distance from given address respectively.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.experimentIds": {
	//       "description": "Experiment IDs the current request belongs to.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "requestMetadata.locale": {
	//       "description": "Locale to use for the current request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.partnersSessionId": {
	//       "description": "Google Partners session ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.trafficSource.trafficSourceId": {
	//       "description": "Identifier to indicate where the traffic comes from. An identifier has multiple letters created by a team which redirected the traffic to us.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.trafficSource.trafficSubId": {
	//       "description": "Second level identifier to indicate where the traffic comes from. An identifier has multiple letters created by a team which redirected the traffic to us.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.userOverrides.ipAddress": {
	//       "description": "IP address to use instead of the user's geo-located IP address.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.userOverrides.userId": {
	//       "description": "Logged-in user ID to impersonate instead of the user's ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "The view of `Company` resource to be returned. This must not be `COMPANY_VIEW_UNSPECIFIED`.",
	//       "enum": [
	//         "COMPANY_VIEW_UNSPECIFIED",
	//         "CV_GOOGLE_PARTNER_SEARCH"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2/companies/{companyId}",
	//   "response": {
	//     "$ref": "GetCompanyResponse"
	//   }
	// }

}

// method id "partners.companies.list":

type CompaniesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists companies.
func (r *CompaniesService) List() *CompaniesListCall {
	c := &CompaniesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Address sets the optional parameter "address": The address to use
// when searching for companies. If not given, the geo-located address
// of the request is used.
func (c *CompaniesListCall) Address(address string) *CompaniesListCall {
	c.urlParams_.Set("address", address)
	return c
}

// CompanyName sets the optional parameter "companyName": Company name
// to search for.
func (c *CompaniesListCall) CompanyName(companyName string) *CompaniesListCall {
	c.urlParams_.Set("companyName", companyName)
	return c
}

// GpsMotivations sets the optional parameter "gpsMotivations": List of
// reasons for using Google Partner Search to get companies.
//
// Possible values:
//   "GPS_MOTIVATION_UNSPECIFIED"
//   "GPSM_HELP_WITH_ADVERTISING"
//   "GPSM_HELP_WITH_WEBSITE"
//   "GPSM_NO_WEBSITE"
func (c *CompaniesListCall) GpsMotivations(gpsMotivations ...string) *CompaniesListCall {
	c.urlParams_.SetMulti("gpsMotivations", append([]string{}, gpsMotivations...))
	return c
}

// Industries sets the optional parameter "industries": List of
// industries the company can help with.
//
// Possible values:
//   "INDUSTRY_UNSPECIFIED"
//   "I_AUTOMOTIVE"
//   "I_BUSINESS_TO_BUSINESS"
//   "I_CONSUMER_PACKAGED_GOODS"
//   "I_EDUCATION"
//   "I_FINANCE"
//   "I_HEALTHCARE"
//   "I_MEDIA_AND_ENTERTAINMENT"
//   "I_RETAIL"
//   "I_TECHNOLOGY"
//   "I_TRAVEL"
func (c *CompaniesListCall) Industries(industries ...string) *CompaniesListCall {
	c.urlParams_.SetMulti("industries", append([]string{}, industries...))
	return c
}

// LanguageCodes sets the optional parameter "languageCodes": List of
// language codes that company can support. Only primary language
// subtags are accepted as defined by BCP 47 (IETF BCP 47, "Tags for
// Identifying Languages").
func (c *CompaniesListCall) LanguageCodes(languageCodes ...string) *CompaniesListCall {
	c.urlParams_.SetMulti("languageCodes", append([]string{}, languageCodes...))
	return c
}

// MaxMonthlyBudgetCurrencyCode sets the optional parameter
// "maxMonthlyBudget.currencyCode": The 3-letter currency code defined
// in ISO 4217.
func (c *CompaniesListCall) MaxMonthlyBudgetCurrencyCode(maxMonthlyBudgetCurrencyCode string) *CompaniesListCall {
	c.urlParams_.Set("maxMonthlyBudget.currencyCode", maxMonthlyBudgetCurrencyCode)
	return c
}

// MaxMonthlyBudgetNanos sets the optional parameter
// "maxMonthlyBudget.nanos": Number of nano (10^-9) units of the amount.
// The value must be between -999,999,999 and +999,999,999 inclusive. If
// `units` is positive, `nanos` must be positive or zero. If `units` is
// zero, `nanos` can be positive, zero, or negative. If `units` is
// negative, `nanos` must be negative or zero. For example $-1.75 is
// represented as `units`=-1 and `nanos`=-750,000,000.
func (c *CompaniesListCall) MaxMonthlyBudgetNanos(maxMonthlyBudgetNanos int64) *CompaniesListCall {
	c.urlParams_.Set("maxMonthlyBudget.nanos", fmt.Sprint(maxMonthlyBudgetNanos))
	return c
}

// MaxMonthlyBudgetUnits sets the optional parameter
// "maxMonthlyBudget.units": The whole units of the amount. For example
// if `currencyCode` is "USD", then 1 unit is one US dollar.
func (c *CompaniesListCall) MaxMonthlyBudgetUnits(maxMonthlyBudgetUnits int64) *CompaniesListCall {
	c.urlParams_.Set("maxMonthlyBudget.units", fmt.Sprint(maxMonthlyBudgetUnits))
	return c
}

// MinMonthlyBudgetCurrencyCode sets the optional parameter
// "minMonthlyBudget.currencyCode": The 3-letter currency code defined
// in ISO 4217.
func (c *CompaniesListCall) MinMonthlyBudgetCurrencyCode(minMonthlyBudgetCurrencyCode string) *CompaniesListCall {
	c.urlParams_.Set("minMonthlyBudget.currencyCode", minMonthlyBudgetCurrencyCode)
	return c
}

// MinMonthlyBudgetNanos sets the optional parameter
// "minMonthlyBudget.nanos": Number of nano (10^-9) units of the amount.
// The value must be between -999,999,999 and +999,999,999 inclusive. If
// `units` is positive, `nanos` must be positive or zero. If `units` is
// zero, `nanos` can be positive, zero, or negative. If `units` is
// negative, `nanos` must be negative or zero. For example $-1.75 is
// represented as `units`=-1 and `nanos`=-750,000,000.
func (c *CompaniesListCall) MinMonthlyBudgetNanos(minMonthlyBudgetNanos int64) *CompaniesListCall {
	c.urlParams_.Set("minMonthlyBudget.nanos", fmt.Sprint(minMonthlyBudgetNanos))
	return c
}

// MinMonthlyBudgetUnits sets the optional parameter
// "minMonthlyBudget.units": The whole units of the amount. For example
// if `currencyCode` is "USD", then 1 unit is one US dollar.
func (c *CompaniesListCall) MinMonthlyBudgetUnits(minMonthlyBudgetUnits int64) *CompaniesListCall {
	c.urlParams_.Set("minMonthlyBudget.units", fmt.Sprint(minMonthlyBudgetUnits))
	return c
}

// OrderBy sets the optional parameter "orderBy": How to order addresses
// within the returned companies. Currently, only `address` and `address
// desc` is supported which will sorted by closest to farthest in
// distance from given address and farthest to closest distance from
// given address respectively.
func (c *CompaniesListCall) OrderBy(orderBy string) *CompaniesListCall {
	c.urlParams_.Set("orderBy", orderBy)
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// Server may return fewer companies than requested. If unspecified,
// server picks an appropriate default.
func (c *CompaniesListCall) PageSize(pageSize int64) *CompaniesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results that the server returns. Typically,
// this is the value of `ListCompaniesResponse.next_page_token` returned
// from the previous call to ListCompanies.
func (c *CompaniesListCall) PageToken(pageToken string) *CompaniesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CompaniesListCall) QuotaUser(quotaUser string) *CompaniesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// RequestMetadataExperimentIds sets the optional parameter
// "requestMetadata.experimentIds": Experiment IDs the current request
// belongs to.
func (c *CompaniesListCall) RequestMetadataExperimentIds(requestMetadataExperimentIds ...string) *CompaniesListCall {
	c.urlParams_.SetMulti("requestMetadata.experimentIds", append([]string{}, requestMetadataExperimentIds...))
	return c
}

// RequestMetadataLocale sets the optional parameter
// "requestMetadata.locale": Locale to use for the current request.
func (c *CompaniesListCall) RequestMetadataLocale(requestMetadataLocale string) *CompaniesListCall {
	c.urlParams_.Set("requestMetadata.locale", requestMetadataLocale)
	return c
}

// RequestMetadataPartnersSessionId sets the optional parameter
// "requestMetadata.partnersSessionId": Google Partners session ID.
func (c *CompaniesListCall) RequestMetadataPartnersSessionId(requestMetadataPartnersSessionId string) *CompaniesListCall {
	c.urlParams_.Set("requestMetadata.partnersSessionId", requestMetadataPartnersSessionId)
	return c
}

// RequestMetadataTrafficSourceTrafficSourceId sets the optional
// parameter "requestMetadata.trafficSource.trafficSourceId": Identifier
// to indicate where the traffic comes from. An identifier has multiple
// letters created by a team which redirected the traffic to us.
func (c *CompaniesListCall) RequestMetadataTrafficSourceTrafficSourceId(requestMetadataTrafficSourceTrafficSourceId string) *CompaniesListCall {
	c.urlParams_.Set("requestMetadata.trafficSource.trafficSourceId", requestMetadataTrafficSourceTrafficSourceId)
	return c
}

// RequestMetadataTrafficSourceTrafficSubId sets the optional parameter
// "requestMetadata.trafficSource.trafficSubId": Second level identifier
// to indicate where the traffic comes from. An identifier has multiple
// letters created by a team which redirected the traffic to us.
func (c *CompaniesListCall) RequestMetadataTrafficSourceTrafficSubId(requestMetadataTrafficSourceTrafficSubId string) *CompaniesListCall {
	c.urlParams_.Set("requestMetadata.trafficSource.trafficSubId", requestMetadataTrafficSourceTrafficSubId)
	return c
}

// RequestMetadataUserOverridesIpAddress sets the optional parameter
// "requestMetadata.userOverrides.ipAddress": IP address to use instead
// of the user's geo-located IP address.
func (c *CompaniesListCall) RequestMetadataUserOverridesIpAddress(requestMetadataUserOverridesIpAddress string) *CompaniesListCall {
	c.urlParams_.Set("requestMetadata.userOverrides.ipAddress", requestMetadataUserOverridesIpAddress)
	return c
}

// RequestMetadataUserOverridesUserId sets the optional parameter
// "requestMetadata.userOverrides.userId": Logged-in user ID to
// impersonate instead of the user's ID.
func (c *CompaniesListCall) RequestMetadataUserOverridesUserId(requestMetadataUserOverridesUserId string) *CompaniesListCall {
	c.urlParams_.Set("requestMetadata.userOverrides.userId", requestMetadataUserOverridesUserId)
	return c
}

// Services sets the optional parameter "services": List of services the
// company can help with.
//
// Possible values:
//   "SERVICE_UNSPECIFIED"
//   "S_ADVANCED_ADWORDS_SUPPORT"
//   "S_ADVERTISING_ON_GOOGLE"
//   "S_AN_ENHANCED_WEBSITE"
//   "S_AN_ONLINE_MARKETING_PLAN"
//   "S_MOBILE_AND_VIDEO_ADS"
func (c *CompaniesListCall) Services(services ...string) *CompaniesListCall {
	c.urlParams_.SetMulti("services", append([]string{}, services...))
	return c
}

// View sets the optional parameter "view": The view of the `Company`
// resource to be returned. This must not be `COMPANY_VIEW_UNSPECIFIED`.
//
// Possible values:
//   "COMPANY_VIEW_UNSPECIFIED"
//   "CV_GOOGLE_PARTNER_SEARCH"
func (c *CompaniesListCall) View(view string) *CompaniesListCall {
	c.urlParams_.Set("view", view)
	return c
}

// WebsiteUrl sets the optional parameter "websiteUrl": Website URL that
// will help to find a better matched company. .
func (c *CompaniesListCall) WebsiteUrl(websiteUrl string) *CompaniesListCall {
	c.urlParams_.Set("websiteUrl", websiteUrl)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CompaniesListCall) Fields(s ...googleapi.Field) *CompaniesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CompaniesListCall) IfNoneMatch(entityTag string) *CompaniesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CompaniesListCall) Context(ctx context.Context) *CompaniesListCall {
	c.ctx_ = ctx
	return c
}

func (c *CompaniesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2/companies")
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

// Do executes the "partners.companies.list" call.
// Exactly one of *ListCompaniesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListCompaniesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CompaniesListCall) Do() (*ListCompaniesResponse, error) {
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
	ret := &ListCompaniesResponse{
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
	//   "description": "Lists companies.",
	//   "httpMethod": "GET",
	//   "id": "partners.companies.list",
	//   "parameters": {
	//     "address": {
	//       "description": "The address to use when searching for companies. If not given, the geo-located address of the request is used.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "companyName": {
	//       "description": "Company name to search for.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "gpsMotivations": {
	//       "description": "List of reasons for using Google Partner Search to get companies.",
	//       "enum": [
	//         "GPS_MOTIVATION_UNSPECIFIED",
	//         "GPSM_HELP_WITH_ADVERTISING",
	//         "GPSM_HELP_WITH_WEBSITE",
	//         "GPSM_NO_WEBSITE"
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "industries": {
	//       "description": "List of industries the company can help with.",
	//       "enum": [
	//         "INDUSTRY_UNSPECIFIED",
	//         "I_AUTOMOTIVE",
	//         "I_BUSINESS_TO_BUSINESS",
	//         "I_CONSUMER_PACKAGED_GOODS",
	//         "I_EDUCATION",
	//         "I_FINANCE",
	//         "I_HEALTHCARE",
	//         "I_MEDIA_AND_ENTERTAINMENT",
	//         "I_RETAIL",
	//         "I_TECHNOLOGY",
	//         "I_TRAVEL"
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "languageCodes": {
	//       "description": "List of language codes that company can support. Only primary language subtags are accepted as defined by BCP 47 (IETF BCP 47, \"Tags for Identifying Languages\").",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "maxMonthlyBudget.currencyCode": {
	//       "description": "The 3-letter currency code defined in ISO 4217.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxMonthlyBudget.nanos": {
	//       "description": "Number of nano (10^-9) units of the amount. The value must be between -999,999,999 and +999,999,999 inclusive. If `units` is positive, `nanos` must be positive or zero. If `units` is zero, `nanos` can be positive, zero, or negative. If `units` is negative, `nanos` must be negative or zero. For example $-1.75 is represented as `units`=-1 and `nanos`=-750,000,000.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "maxMonthlyBudget.units": {
	//       "description": "The whole units of the amount. For example if `currencyCode` is `\"USD\"`, then 1 unit is one US dollar.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "minMonthlyBudget.currencyCode": {
	//       "description": "The 3-letter currency code defined in ISO 4217.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "minMonthlyBudget.nanos": {
	//       "description": "Number of nano (10^-9) units of the amount. The value must be between -999,999,999 and +999,999,999 inclusive. If `units` is positive, `nanos` must be positive or zero. If `units` is zero, `nanos` can be positive, zero, or negative. If `units` is negative, `nanos` must be negative or zero. For example $-1.75 is represented as `units`=-1 and `nanos`=-750,000,000.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "minMonthlyBudget.units": {
	//       "description": "The whole units of the amount. For example if `currencyCode` is `\"USD\"`, then 1 unit is one US dollar.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "orderBy": {
	//       "description": "How to order addresses within the returned companies. Currently, only `address` and `address desc` is supported which will sorted by closest to farthest in distance from given address and farthest to closest distance from given address respectively.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. Server may return fewer companies than requested. If unspecified, server picks an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results that the server returns. Typically, this is the value of `ListCompaniesResponse.next_page_token` returned from the previous call to ListCompanies.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.experimentIds": {
	//       "description": "Experiment IDs the current request belongs to.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "requestMetadata.locale": {
	//       "description": "Locale to use for the current request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.partnersSessionId": {
	//       "description": "Google Partners session ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.trafficSource.trafficSourceId": {
	//       "description": "Identifier to indicate where the traffic comes from. An identifier has multiple letters created by a team which redirected the traffic to us.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.trafficSource.trafficSubId": {
	//       "description": "Second level identifier to indicate where the traffic comes from. An identifier has multiple letters created by a team which redirected the traffic to us.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.userOverrides.ipAddress": {
	//       "description": "IP address to use instead of the user's geo-located IP address.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.userOverrides.userId": {
	//       "description": "Logged-in user ID to impersonate instead of the user's ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "services": {
	//       "description": "List of services the company can help with.",
	//       "enum": [
	//         "SERVICE_UNSPECIFIED",
	//         "S_ADVANCED_ADWORDS_SUPPORT",
	//         "S_ADVERTISING_ON_GOOGLE",
	//         "S_AN_ENHANCED_WEBSITE",
	//         "S_AN_ONLINE_MARKETING_PLAN",
	//         "S_MOBILE_AND_VIDEO_ADS"
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "The view of the `Company` resource to be returned. This must not be `COMPANY_VIEW_UNSPECIFIED`.",
	//       "enum": [
	//         "COMPANY_VIEW_UNSPECIFIED",
	//         "CV_GOOGLE_PARTNER_SEARCH"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "websiteUrl": {
	//       "description": "Website URL that will help to find a better matched company. .",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2/companies",
	//   "response": {
	//     "$ref": "ListCompaniesResponse"
	//   }
	// }

}

// method id "partners.companies.leads.create":

type CompaniesLeadsCreateCall struct {
	s                 *Service
	companyId         string
	createleadrequest *CreateLeadRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
}

// Create: Creates an advertiser lead for the given company ID.
func (r *CompaniesLeadsService) Create(companyId string, createleadrequest *CreateLeadRequest) *CompaniesLeadsCreateCall {
	c := &CompaniesLeadsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.companyId = companyId
	c.createleadrequest = createleadrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CompaniesLeadsCreateCall) QuotaUser(quotaUser string) *CompaniesLeadsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CompaniesLeadsCreateCall) Fields(s ...googleapi.Field) *CompaniesLeadsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CompaniesLeadsCreateCall) Context(ctx context.Context) *CompaniesLeadsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *CompaniesLeadsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.createleadrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2/companies/{companyId}/leads")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"companyId": c.companyId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "partners.companies.leads.create" call.
// Exactly one of *CreateLeadResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *CreateLeadResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CompaniesLeadsCreateCall) Do() (*CreateLeadResponse, error) {
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
	ret := &CreateLeadResponse{
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
	//   "description": "Creates an advertiser lead for the given company ID.",
	//   "httpMethod": "POST",
	//   "id": "partners.companies.leads.create",
	//   "parameterOrder": [
	//     "companyId"
	//   ],
	//   "parameters": {
	//     "companyId": {
	//       "description": "The ID of the company to contact.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2/companies/{companyId}/leads",
	//   "request": {
	//     "$ref": "CreateLeadRequest"
	//   },
	//   "response": {
	//     "$ref": "CreateLeadResponse"
	//   }
	// }

}

// method id "partners.userEvents.log":

type UserEventsLogCall struct {
	s                   *Service
	logusereventrequest *LogUserEventRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
}

// Log: Logs a user event.
func (r *UserEventsService) Log(logusereventrequest *LogUserEventRequest) *UserEventsLogCall {
	c := &UserEventsLogCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.logusereventrequest = logusereventrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *UserEventsLogCall) QuotaUser(quotaUser string) *UserEventsLogCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UserEventsLogCall) Fields(s ...googleapi.Field) *UserEventsLogCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UserEventsLogCall) Context(ctx context.Context) *UserEventsLogCall {
	c.ctx_ = ctx
	return c
}

func (c *UserEventsLogCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.logusereventrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2/userEvents:log")
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

// Do executes the "partners.userEvents.log" call.
// Exactly one of *LogUserEventResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *LogUserEventResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UserEventsLogCall) Do() (*LogUserEventResponse, error) {
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
	ret := &LogUserEventResponse{
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
	//   "description": "Logs a user event.",
	//   "httpMethod": "POST",
	//   "id": "partners.userEvents.log",
	//   "path": "v2/userEvents:log",
	//   "request": {
	//     "$ref": "LogUserEventRequest"
	//   },
	//   "response": {
	//     "$ref": "LogUserEventResponse"
	//   }
	// }

}

// method id "partners.userStates.list":

type UserStatesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists states for current user.
func (r *UserStatesService) List() *UserStatesListCall {
	c := &UserStatesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *UserStatesListCall) QuotaUser(quotaUser string) *UserStatesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// RequestMetadataExperimentIds sets the optional parameter
// "requestMetadata.experimentIds": Experiment IDs the current request
// belongs to.
func (c *UserStatesListCall) RequestMetadataExperimentIds(requestMetadataExperimentIds ...string) *UserStatesListCall {
	c.urlParams_.SetMulti("requestMetadata.experimentIds", append([]string{}, requestMetadataExperimentIds...))
	return c
}

// RequestMetadataLocale sets the optional parameter
// "requestMetadata.locale": Locale to use for the current request.
func (c *UserStatesListCall) RequestMetadataLocale(requestMetadataLocale string) *UserStatesListCall {
	c.urlParams_.Set("requestMetadata.locale", requestMetadataLocale)
	return c
}

// RequestMetadataPartnersSessionId sets the optional parameter
// "requestMetadata.partnersSessionId": Google Partners session ID.
func (c *UserStatesListCall) RequestMetadataPartnersSessionId(requestMetadataPartnersSessionId string) *UserStatesListCall {
	c.urlParams_.Set("requestMetadata.partnersSessionId", requestMetadataPartnersSessionId)
	return c
}

// RequestMetadataTrafficSourceTrafficSourceId sets the optional
// parameter "requestMetadata.trafficSource.trafficSourceId": Identifier
// to indicate where the traffic comes from. An identifier has multiple
// letters created by a team which redirected the traffic to us.
func (c *UserStatesListCall) RequestMetadataTrafficSourceTrafficSourceId(requestMetadataTrafficSourceTrafficSourceId string) *UserStatesListCall {
	c.urlParams_.Set("requestMetadata.trafficSource.trafficSourceId", requestMetadataTrafficSourceTrafficSourceId)
	return c
}

// RequestMetadataTrafficSourceTrafficSubId sets the optional parameter
// "requestMetadata.trafficSource.trafficSubId": Second level identifier
// to indicate where the traffic comes from. An identifier has multiple
// letters created by a team which redirected the traffic to us.
func (c *UserStatesListCall) RequestMetadataTrafficSourceTrafficSubId(requestMetadataTrafficSourceTrafficSubId string) *UserStatesListCall {
	c.urlParams_.Set("requestMetadata.trafficSource.trafficSubId", requestMetadataTrafficSourceTrafficSubId)
	return c
}

// RequestMetadataUserOverridesIpAddress sets the optional parameter
// "requestMetadata.userOverrides.ipAddress": IP address to use instead
// of the user's geo-located IP address.
func (c *UserStatesListCall) RequestMetadataUserOverridesIpAddress(requestMetadataUserOverridesIpAddress string) *UserStatesListCall {
	c.urlParams_.Set("requestMetadata.userOverrides.ipAddress", requestMetadataUserOverridesIpAddress)
	return c
}

// RequestMetadataUserOverridesUserId sets the optional parameter
// "requestMetadata.userOverrides.userId": Logged-in user ID to
// impersonate instead of the user's ID.
func (c *UserStatesListCall) RequestMetadataUserOverridesUserId(requestMetadataUserOverridesUserId string) *UserStatesListCall {
	c.urlParams_.Set("requestMetadata.userOverrides.userId", requestMetadataUserOverridesUserId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UserStatesListCall) Fields(s ...googleapi.Field) *UserStatesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UserStatesListCall) IfNoneMatch(entityTag string) *UserStatesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UserStatesListCall) Context(ctx context.Context) *UserStatesListCall {
	c.ctx_ = ctx
	return c
}

func (c *UserStatesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v2/userStates")
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

// Do executes the "partners.userStates.list" call.
// Exactly one of *ListUserStatesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListUserStatesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UserStatesListCall) Do() (*ListUserStatesResponse, error) {
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
	ret := &ListUserStatesResponse{
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
	//   "description": "Lists states for current user.",
	//   "httpMethod": "GET",
	//   "id": "partners.userStates.list",
	//   "parameters": {
	//     "requestMetadata.experimentIds": {
	//       "description": "Experiment IDs the current request belongs to.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "requestMetadata.locale": {
	//       "description": "Locale to use for the current request.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.partnersSessionId": {
	//       "description": "Google Partners session ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.trafficSource.trafficSourceId": {
	//       "description": "Identifier to indicate where the traffic comes from. An identifier has multiple letters created by a team which redirected the traffic to us.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.trafficSource.trafficSubId": {
	//       "description": "Second level identifier to indicate where the traffic comes from. An identifier has multiple letters created by a team which redirected the traffic to us.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.userOverrides.ipAddress": {
	//       "description": "IP address to use instead of the user's geo-located IP address.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "requestMetadata.userOverrides.userId": {
	//       "description": "Logged-in user ID to impersonate instead of the user's ID.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v2/userStates",
	//   "response": {
	//     "$ref": "ListUserStatesResponse"
	//   }
	// }

}
