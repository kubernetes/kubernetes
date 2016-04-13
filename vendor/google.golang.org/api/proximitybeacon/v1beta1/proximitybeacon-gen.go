// Package proximitybeacon provides access to the Google Proximity Beacon API.
//
// See https://developers.google.com/beacons/proximity/
//
// Usage example:
//
//   import "google.golang.org/api/proximitybeacon/v1beta1"
//   ...
//   proximitybeaconService, err := proximitybeacon.New(oauthHttpClient)
package proximitybeacon // import "google.golang.org/api/proximitybeacon/v1beta1"

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

const apiId = "proximitybeacon:v1beta1"
const apiName = "proximitybeacon"
const apiVersion = "v1beta1"
const basePath = "https://proximitybeacon.googleapis.com/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Beaconinfo = NewBeaconinfoService(s)
	s.Beacons = NewBeaconsService(s)
	s.Namespaces = NewNamespacesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Beaconinfo *BeaconinfoService

	Beacons *BeaconsService

	Namespaces *NamespacesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewBeaconinfoService(s *Service) *BeaconinfoService {
	rs := &BeaconinfoService{s: s}
	return rs
}

type BeaconinfoService struct {
	s *Service
}

func NewBeaconsService(s *Service) *BeaconsService {
	rs := &BeaconsService{s: s}
	rs.Attachments = NewBeaconsAttachmentsService(s)
	rs.Diagnostics = NewBeaconsDiagnosticsService(s)
	return rs
}

type BeaconsService struct {
	s *Service

	Attachments *BeaconsAttachmentsService

	Diagnostics *BeaconsDiagnosticsService
}

func NewBeaconsAttachmentsService(s *Service) *BeaconsAttachmentsService {
	rs := &BeaconsAttachmentsService{s: s}
	return rs
}

type BeaconsAttachmentsService struct {
	s *Service
}

func NewBeaconsDiagnosticsService(s *Service) *BeaconsDiagnosticsService {
	rs := &BeaconsDiagnosticsService{s: s}
	return rs
}

type BeaconsDiagnosticsService struct {
	s *Service
}

func NewNamespacesService(s *Service) *NamespacesService {
	rs := &NamespacesService{s: s}
	return rs
}

type NamespacesService struct {
	s *Service
}

// AdvertisedId: Defines a unique identifier of a beacon as broadcast by
// the device.
type AdvertisedId struct {
	// Id: The actual beacon identifier, as broadcast by the beacon
	// hardware. Must be
	// [base64](http://tools.ietf.org/html/rfc4648#section-4) encoded in
	// HTTP requests, and will be so encoded (with padding) in responses.
	// The base64 encoding should be of the binary byte-stream and not any
	// textual (such as hex) representation thereof. Required.
	Id string `json:"id,omitempty"`

	// Type: Specifies the identifier type. Required.
	//
	// Possible values:
	//   "TYPE_UNSPECIFIED"
	//   "EDDYSTONE"
	//   "IBEACON"
	//   "ALTBEACON"
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AdvertisedId) MarshalJSON() ([]byte, error) {
	type noMethod AdvertisedId
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// AttachmentInfo: A subset of attachment information served via the
// `beaconinfo.getforobserved` method, used when your users encounter
// your beacons.
type AttachmentInfo struct {
	// Data: An opaque data container for client-provided data.
	Data string `json:"data,omitempty"`

	// NamespacedType: Specifies what kind of attachment this is. Tells a
	// client how to interpret the `data` field. Format is namespace/type,
	// for example scrupulous-wombat-12345/welcome-message
	NamespacedType string `json:"namespacedType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Data") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AttachmentInfo) MarshalJSON() ([]byte, error) {
	type noMethod AttachmentInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Beacon: Details of a beacon device.
type Beacon struct {
	// AdvertisedId: The identifier of a beacon as advertised by it. This
	// field must be populated when registering. It may be empty when
	// updating a beacon record because it is ignored in updates.
	AdvertisedId *AdvertisedId `json:"advertisedId,omitempty"`

	// BeaconName: Resource name of this beacon. A beacon name has the
	// format "beacons/N!beaconId" where the beaconId is the base16 ID
	// broadcast by the beacon and N is a code for the beacon's type.
	// Possible values are `3` for Eddystone, `1` for iBeacon, or `5` for
	// AltBeacon. This field must be left empty when registering. After
	// reading a beacon, clients can use the name for future operations.
	BeaconName string `json:"beaconName,omitempty"`

	// Description: Free text used to identify and describe the beacon.
	// Maximum length 140 characters. Optional.
	Description string `json:"description,omitempty"`

	// ExpectedStability: Expected location stability. This is set when the
	// beacon is registered or updated, not automatically detected in any
	// way. Optional.
	//
	// Possible values:
	//   "STABILITY_UNSPECIFIED"
	//   "STABLE"
	//   "PORTABLE"
	//   "MOBILE"
	//   "ROVING"
	ExpectedStability string `json:"expectedStability,omitempty"`

	// IndoorLevel: The indoor level information for this beacon, if known.
	// As returned by the Google Maps API. Optional.
	IndoorLevel *IndoorLevel `json:"indoorLevel,omitempty"`

	// LatLng: The location of the beacon, expressed as a latitude and
	// longitude pair. This location is given when the beacon is registered
	// or updated. It does not necessarily indicate the actual current
	// location of the beacon. Optional.
	LatLng *LatLng `json:"latLng,omitempty"`

	// PlaceId: The [Google Places API](/places/place-id) Place ID of the
	// place where the beacon is deployed. This is given when the beacon is
	// registered or updated, not automatically detected in any way.
	// Optional.
	PlaceId string `json:"placeId,omitempty"`

	// Properties: Properties of the beacon device, for example battery type
	// or firmware version. Optional.
	Properties map[string]string `json:"properties,omitempty"`

	// Status: Current status of the beacon. Required.
	//
	// Possible values:
	//   "STATUS_UNSPECIFIED"
	//   "ACTIVE"
	//   "DECOMMISSIONED"
	//   "INACTIVE"
	Status string `json:"status,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AdvertisedId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Beacon) MarshalJSON() ([]byte, error) {
	type noMethod Beacon
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// BeaconAttachment: Project-specific data associated with a beacon.
type BeaconAttachment struct {
	// AttachmentName: Resource name of this attachment. Attachment names
	// have the format: beacons/beacon_id/attachments/attachment_id. Leave
	// this empty on creation.
	AttachmentName string `json:"attachmentName,omitempty"`

	// Data: An opaque data container for client-provided data. Must be
	// [base64](http://tools.ietf.org/html/rfc4648#section-4) encoded in
	// HTTP requests, and will be so encoded (with padding) in responses.
	// Required.
	Data string `json:"data,omitempty"`

	// NamespacedType: Specifies what kind of attachment this is. Tells a
	// client how to interpret the `data` field. Format is namespace/type.
	// Namespace provides type separation between clients. Type describes
	// the type of `data`, for use by the client when parsing the `data`
	// field. Required.
	NamespacedType string `json:"namespacedType,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AttachmentName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BeaconAttachment) MarshalJSON() ([]byte, error) {
	type noMethod BeaconAttachment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// BeaconInfo: A subset of beacon information served via the
// `beaconinfo.getforobserved` method, which you call when users of your
// app encounter your beacons.
type BeaconInfo struct {
	// AdvertisedId: The ID advertised by the beacon.
	AdvertisedId *AdvertisedId `json:"advertisedId,omitempty"`

	// Attachments: Attachments matching the type(s) requested. May be empty
	// if no attachment types were requested, or if none matched.
	Attachments []*AttachmentInfo `json:"attachments,omitempty"`

	// BeaconName: The name under which the beacon is registered.
	BeaconName string `json:"beaconName,omitempty"`

	// Description: Free text used to identify or describe the beacon in a
	// registered establishment. For example: "entrance", "room 101", etc.
	// May be empty.
	Description string `json:"description,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AdvertisedId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BeaconInfo) MarshalJSON() ([]byte, error) {
	type noMethod BeaconInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Date: Represents a whole calendar date, e.g. date of birth. The time
// of day and time zone are either specified elsewhere or are not
// significant. The date is relative to the Proleptic Gregorian
// Calendar. The day may be 0 to represent a year and month where the
// day is not significant, e.g. credit card expiration date. The year
// may be 0 to represent a month and day independent of year, e.g.
// anniversary date. Related types are [google.type.TimeOfDay][] and
// `google.protobuf.Timestamp`.
type Date struct {
	// Day: Day of month. Must be from 1 to 31 and valid for the year and
	// month, or 0 if specifying a year/month where the day is not
	// sigificant.
	Day int64 `json:"day,omitempty"`

	// Month: Month of year of date. Must be from 1 to 12.
	Month int64 `json:"month,omitempty"`

	// Year: Year of date. Must be from 1 to 9,999, or 0 if specifying a
	// date without a year.
	Year int64 `json:"year,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Day") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Date) MarshalJSON() ([]byte, error) {
	type noMethod Date
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DeleteAttachmentsResponse: Response for a request to delete
// attachments.
type DeleteAttachmentsResponse struct {
	// NumDeleted: The number of attachments that were deleted.
	NumDeleted int64 `json:"numDeleted,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NumDeleted") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DeleteAttachmentsResponse) MarshalJSON() ([]byte, error) {
	type noMethod DeleteAttachmentsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Diagnostics: Diagnostics for a single beacon.
type Diagnostics struct {
	// Alerts: An unordered list of Alerts that the beacon has.
	//
	// Possible values:
	//   "ALERT_UNSPECIFIED" - Invalid value. Should never appear.
	//   "WRONG_LOCATION" - The beacon has been reported in a location
	// different than its registered location. This may indicate that the
	// beacon has been moved. This signal is not 100% accurate, but
	// indicates that further investigation is worth while.
	//   "LOW_BATTERY" - The battery level for the beacon is low enough
	// that, given the beacon's current use, its battery will run out with
	// in the next 60 days. This indicates that the battery should be
	// replaced soon.
	Alerts []string `json:"alerts,omitempty"`

	// BeaconName: Resource name of the beacon.
	BeaconName string `json:"beaconName,omitempty"`

	// EstimatedLowBatteryDate: The date when the battery is expected to be
	// low. If the value is missing then there is no estimate for when the
	// battery will be low. This value is only an estimate, not an exact
	// date.
	EstimatedLowBatteryDate *Date `json:"estimatedLowBatteryDate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Alerts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Diagnostics) MarshalJSON() ([]byte, error) {
	type noMethod Diagnostics
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Empty: A generic empty message that you can re-use to avoid defining
// duplicated empty messages in your APIs. A typical example is to use
// it as the request or the response type of an API method. For
// instance: service Foo { rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty); } The JSON representation for `Empty` is
// empty JSON object `{}`.
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// GetInfoForObservedBeaconsRequest: Request for beacon and attachment
// information about beacons that a mobile client has encountered "in
// the wild".
type GetInfoForObservedBeaconsRequest struct {
	// NamespacedTypes: Specifies what kind of attachments to include in the
	// response. When given, the response will include only attachments of
	// the given types. When empty, no attachments will be returned. Must be
	// in the format namespace/type. Accepts `*` to specify all types in all
	// namespaces. Optional.
	NamespacedTypes []string `json:"namespacedTypes,omitempty"`

	// Observations: The beacons that the client has encountered. At least
	// one must be given.
	Observations []*Observation `json:"observations,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NamespacedTypes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GetInfoForObservedBeaconsRequest) MarshalJSON() ([]byte, error) {
	type noMethod GetInfoForObservedBeaconsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// GetInfoForObservedBeaconsResponse: Information about the requested
// beacons, optionally including attachment data.
type GetInfoForObservedBeaconsResponse struct {
	// Beacons: Public information about beacons. May be empty if the
	// request matched no beacons.
	Beacons []*BeaconInfo `json:"beacons,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Beacons") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GetInfoForObservedBeaconsResponse) MarshalJSON() ([]byte, error) {
	type noMethod GetInfoForObservedBeaconsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// IndoorLevel: Indoor level, a human-readable string as returned by
// Google Maps APIs, useful to indicate which floor of a building a
// beacon is located on.
type IndoorLevel struct {
	// Name: The name of this level.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Name") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IndoorLevel) MarshalJSON() ([]byte, error) {
	type noMethod IndoorLevel
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

// ListBeaconAttachmentsResponse: Response to ListBeaconAttachments that
// contains the requested attachments.
type ListBeaconAttachmentsResponse struct {
	// Attachments: The attachments that corresponded to the request params.
	Attachments []*BeaconAttachment `json:"attachments,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Attachments") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListBeaconAttachmentsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListBeaconAttachmentsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListBeaconsResponse: Response that contains list beacon results and
// pagination help.
type ListBeaconsResponse struct {
	// Beacons: The beacons that matched the search criteria.
	Beacons []*Beacon `json:"beacons,omitempty"`

	// NextPageToken: An opaque pagination token that the client may provide
	// in their next request to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// TotalCount: Estimate of the total number of beacons matched by the
	// query. Higher values may be less accurate.
	TotalCount int64 `json:"totalCount,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Beacons") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListBeaconsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListBeaconsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListDiagnosticsResponse: Response that contains the requested
// diagnostics.
type ListDiagnosticsResponse struct {
	// Diagnostics: The diagnostics matching the given request.
	Diagnostics []*Diagnostics `json:"diagnostics,omitempty"`

	// NextPageToken: Token that can be used for pagination. Returned only
	// if the request matches more beacons than can be returned in this
	// response.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Diagnostics") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListDiagnosticsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListDiagnosticsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListNamespacesResponse: Response to ListNamespacesRequest that
// contains all the project's namespaces.
type ListNamespacesResponse struct {
	// Namespaces: The attachments that corresponded to the request params.
	Namespaces []*Namespace `json:"namespaces,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Namespaces") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListNamespacesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListNamespacesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Namespace: An attachment namespace defines read and write access for
// all the attachments created under it. Each namespace is globally
// unique, and owned by one project which is the only project that can
// create attachments under it.
type Namespace struct {
	// NamespaceName: Resource name of this namespace. Namespaces names have
	// the format: namespaces/namespace.
	NamespaceName string `json:"namespaceName,omitempty"`

	// ServingVisibility: Specifies what clients may receive attachments
	// under this namespace via `beaconinfo.getforobserved`.
	//
	// Possible values:
	//   "VISIBILITY_UNSPECIFIED"
	//   "UNLISTED"
	//   "PUBLIC"
	ServingVisibility string `json:"servingVisibility,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NamespaceName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Namespace) MarshalJSON() ([]byte, error) {
	type noMethod Namespace
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Observation: Represents one beacon observed once.
type Observation struct {
	// AdvertisedId: The ID advertised by the beacon the client has
	// encountered. Required.
	AdvertisedId *AdvertisedId `json:"advertisedId,omitempty"`

	// Telemetry: The array of telemetry bytes received from the beacon. The
	// server is responsible for parsing it. This field may frequently be
	// empty, as with a beacon that transmits telemetry only occasionally.
	Telemetry string `json:"telemetry,omitempty"`

	// TimestampMs: Time when the beacon was observed. Being sourced from a
	// mobile device, this time may be suspect.
	TimestampMs string `json:"timestampMs,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AdvertisedId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Observation) MarshalJSON() ([]byte, error) {
	type noMethod Observation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "proximitybeacon.beaconinfo.getforobserved":

type BeaconinfoGetforobservedCall struct {
	s                                *Service
	getinfoforobservedbeaconsrequest *GetInfoForObservedBeaconsRequest
	urlParams_                       gensupport.URLParams
	ctx_                             context.Context
}

// Getforobserved: Given one or more beacon observations, returns any
// beacon information and attachments accessible to your application.
func (r *BeaconinfoService) Getforobserved(getinfoforobservedbeaconsrequest *GetInfoForObservedBeaconsRequest) *BeaconinfoGetforobservedCall {
	c := &BeaconinfoGetforobservedCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.getinfoforobservedbeaconsrequest = getinfoforobservedbeaconsrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconinfoGetforobservedCall) QuotaUser(quotaUser string) *BeaconinfoGetforobservedCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconinfoGetforobservedCall) Fields(s ...googleapi.Field) *BeaconinfoGetforobservedCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconinfoGetforobservedCall) Context(ctx context.Context) *BeaconinfoGetforobservedCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconinfoGetforobservedCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.getinfoforobservedbeaconsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/beaconinfo:getforobserved")
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

// Do executes the "proximitybeacon.beaconinfo.getforobserved" call.
// Exactly one of *GetInfoForObservedBeaconsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *GetInfoForObservedBeaconsResponse.ServerResponse.Header or
// (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *BeaconinfoGetforobservedCall) Do() (*GetInfoForObservedBeaconsResponse, error) {
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
	ret := &GetInfoForObservedBeaconsResponse{
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
	//   "description": "Given one or more beacon observations, returns any beacon information and attachments accessible to your application.",
	//   "httpMethod": "POST",
	//   "id": "proximitybeacon.beaconinfo.getforobserved",
	//   "path": "v1beta1/beaconinfo:getforobserved",
	//   "request": {
	//     "$ref": "GetInfoForObservedBeaconsRequest"
	//   },
	//   "response": {
	//     "$ref": "GetInfoForObservedBeaconsResponse"
	//   }
	// }

}

// method id "proximitybeacon.beacons.activate":

type BeaconsActivateCall struct {
	s          *Service
	beaconName string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Activate: (Re)activates a beacon. A beacon that is active will return
// information and attachment data when queried via
// `beaconinfo.getforobserved`. Calling this method on an already active
// beacon will do nothing (but will return a successful response code).
func (r *BeaconsService) Activate(beaconName string) *BeaconsActivateCall {
	c := &BeaconsActivateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.beaconName = beaconName
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsActivateCall) QuotaUser(quotaUser string) *BeaconsActivateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsActivateCall) Fields(s ...googleapi.Field) *BeaconsActivateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsActivateCall) Context(ctx context.Context) *BeaconsActivateCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsActivateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/{+beaconName}:activate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"beaconName": c.beaconName,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "proximitybeacon.beacons.activate" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *BeaconsActivateCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "(Re)activates a beacon. A beacon that is active will return information and attachment data when queried via `beaconinfo.getforobserved`. Calling this method on an already active beacon will do nothing (but will return a successful response code).",
	//   "httpMethod": "POST",
	//   "id": "proximitybeacon.beacons.activate",
	//   "parameterOrder": [
	//     "beaconName"
	//   ],
	//   "parameters": {
	//     "beaconName": {
	//       "description": "The beacon to activate. Required.",
	//       "location": "path",
	//       "pattern": "^beacons/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/{+beaconName}:activate",
	//   "response": {
	//     "$ref": "Empty"
	//   }
	// }

}

// method id "proximitybeacon.beacons.deactivate":

type BeaconsDeactivateCall struct {
	s          *Service
	beaconName string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Deactivate: Deactivates a beacon. Once deactivated, the API will not
// return information nor attachment data for the beacon when queried
// via `beaconinfo.getforobserved`. Calling this method on an already
// inactive beacon will do nothing (but will return a successful
// response code).
func (r *BeaconsService) Deactivate(beaconName string) *BeaconsDeactivateCall {
	c := &BeaconsDeactivateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.beaconName = beaconName
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsDeactivateCall) QuotaUser(quotaUser string) *BeaconsDeactivateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsDeactivateCall) Fields(s ...googleapi.Field) *BeaconsDeactivateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsDeactivateCall) Context(ctx context.Context) *BeaconsDeactivateCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsDeactivateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/{+beaconName}:deactivate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"beaconName": c.beaconName,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "proximitybeacon.beacons.deactivate" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *BeaconsDeactivateCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deactivates a beacon. Once deactivated, the API will not return information nor attachment data for the beacon when queried via `beaconinfo.getforobserved`. Calling this method on an already inactive beacon will do nothing (but will return a successful response code).",
	//   "httpMethod": "POST",
	//   "id": "proximitybeacon.beacons.deactivate",
	//   "parameterOrder": [
	//     "beaconName"
	//   ],
	//   "parameters": {
	//     "beaconName": {
	//       "description": "The beacon name of this beacon.",
	//       "location": "path",
	//       "pattern": "^beacons/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/{+beaconName}:deactivate",
	//   "response": {
	//     "$ref": "Empty"
	//   }
	// }

}

// method id "proximitybeacon.beacons.decommission":

type BeaconsDecommissionCall struct {
	s          *Service
	beaconName string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Decommission: Decommissions the specified beacon in the service. This
// beacon will no longer be returned from `beaconinfo.getforobserved`.
// This operation is permanent -- you will not be able to re-register a
// beacon with this ID again.
func (r *BeaconsService) Decommission(beaconName string) *BeaconsDecommissionCall {
	c := &BeaconsDecommissionCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.beaconName = beaconName
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsDecommissionCall) QuotaUser(quotaUser string) *BeaconsDecommissionCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsDecommissionCall) Fields(s ...googleapi.Field) *BeaconsDecommissionCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsDecommissionCall) Context(ctx context.Context) *BeaconsDecommissionCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsDecommissionCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/{+beaconName}:decommission")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"beaconName": c.beaconName,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "proximitybeacon.beacons.decommission" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *BeaconsDecommissionCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Decommissions the specified beacon in the service. This beacon will no longer be returned from `beaconinfo.getforobserved`. This operation is permanent -- you will not be able to re-register a beacon with this ID again.",
	//   "httpMethod": "POST",
	//   "id": "proximitybeacon.beacons.decommission",
	//   "parameterOrder": [
	//     "beaconName"
	//   ],
	//   "parameters": {
	//     "beaconName": {
	//       "description": "Beacon that should be decommissioned. Required.",
	//       "location": "path",
	//       "pattern": "^beacons/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/{+beaconName}:decommission",
	//   "response": {
	//     "$ref": "Empty"
	//   }
	// }

}

// method id "proximitybeacon.beacons.get":

type BeaconsGetCall struct {
	s            *Service
	beaconName   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Returns detailed information about the specified beacon.
func (r *BeaconsService) Get(beaconName string) *BeaconsGetCall {
	c := &BeaconsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.beaconName = beaconName
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsGetCall) QuotaUser(quotaUser string) *BeaconsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsGetCall) Fields(s ...googleapi.Field) *BeaconsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BeaconsGetCall) IfNoneMatch(entityTag string) *BeaconsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsGetCall) Context(ctx context.Context) *BeaconsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/{+beaconName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"beaconName": c.beaconName,
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

// Do executes the "proximitybeacon.beacons.get" call.
// Exactly one of *Beacon or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Beacon.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *BeaconsGetCall) Do() (*Beacon, error) {
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
	ret := &Beacon{
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
	//   "description": "Returns detailed information about the specified beacon.",
	//   "httpMethod": "GET",
	//   "id": "proximitybeacon.beacons.get",
	//   "parameterOrder": [
	//     "beaconName"
	//   ],
	//   "parameters": {
	//     "beaconName": {
	//       "description": "Beacon that is requested.",
	//       "location": "path",
	//       "pattern": "^beacons/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/{+beaconName}",
	//   "response": {
	//     "$ref": "Beacon"
	//   }
	// }

}

// method id "proximitybeacon.beacons.list":

type BeaconsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Searches the beacon registry for beacons that match the given
// search criteria. Only those beacons that the client has permission to
// list will be returned.
func (r *BeaconsService) List() *BeaconsListCall {
	c := &BeaconsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of records to return for this request, up to a server-defined upper
// limit.
func (c *BeaconsListCall) PageSize(pageSize int64) *BeaconsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A pagination token
// obtained from a previous request to list beacons.
func (c *BeaconsListCall) PageToken(pageToken string) *BeaconsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Q sets the optional parameter "q": Filter query string that supports
// the following field filters: * `description:"" For example:
// `description:"Room 3" Returns beacons whose description matches
// tokens in the string "Room 3" (not necessarily that exact string).
// The string must be double-quoted. * `status:` For example:
// `status:active` Returns beacons whose status matches the given value.
// Values must be one of the Beacon.Status enum values (case
// insensitive). Accepts multiple filters which will be combined with OR
// logic. * `stability:` For example: `stability:mobile` Returns beacons
// whose expected stability matches the given value. Values must be one
// of the Beacon.Stability enum values (case insensitive). Accepts
// multiple filters which will be combined with OR logic. *
// `place_id:"" For example: `place_id:"ChIJVSZzVR8FdkgRXGmmm6SslKw="
// Returns beacons explicitly registered at the given place, expressed
// as a Place ID obtained from [Google Places API](/places/place-id).
// Does not match places inside the given place. Does not consider the
// beacon's actual location (which may be different from its registered
// place). Accepts multiple filters that will be combined with OR logic.
// The place ID must be double-quoted. * `registration_time[|=]` For
// example: `registration_time>=1433116800` Returns beacons whose
// registration time matches the given filter. Supports the operators: ,
// =. Timestamp must be expressed as an integer number of seconds since
// midnight January 1, 1970 UTC. Accepts at most two filters that will
// be combined with AND logic, to support "between" semantics. If more
// than two are supplied, the latter ones are ignored. * `lat: lng:
// radius:` For example: `lat:51.1232343 lng:-1.093852 radius:1000`
// Returns beacons whose registered location is within the given circle.
// When any of these fields are given, all are required. Latitude and
// longitude must be decimal degrees between -90.0 and 90.0 and between
// -180.0 and 180.0 respectively. Radius must be an integer number of
// meters less than 1,000,000 (1000 km). * `property:"=" For example:
// `property:"battery-type=CR2032" Returns beacons which have a
// property of the given name and value. Supports multiple filters which
// will be combined with OR logic. The entire name=value string must be
// double-quoted as one string. * `attachment_type:"" For example:
// `attachment_type:"my-namespace/my-type" Returns beacons having at
// least one attachment of the given namespaced type. Supports "any
// within this namespace" via the partial wildcard syntax:
// "my-namespace/*". Supports multiple filters which will be combined
// with OR logic. The string must be double-quoted. Multiple filters on
// the same field are combined with OR logic (except registration_time
// which is combined with AND logic). Multiple filters on different
// fields are combined with AND logic. Filters should be separated by
// spaces. As with any HTTP query string parameter, the whole filter
// expression must be URL-encoded. Example REST request: `GET
// /v1beta1/beacons?q=status:active%20lat:51.123%20lng:-1.095%20radius:10
// 00`
func (c *BeaconsListCall) Q(q string) *BeaconsListCall {
	c.urlParams_.Set("q", q)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsListCall) QuotaUser(quotaUser string) *BeaconsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsListCall) Fields(s ...googleapi.Field) *BeaconsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BeaconsListCall) IfNoneMatch(entityTag string) *BeaconsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsListCall) Context(ctx context.Context) *BeaconsListCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/beacons")
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

// Do executes the "proximitybeacon.beacons.list" call.
// Exactly one of *ListBeaconsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListBeaconsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *BeaconsListCall) Do() (*ListBeaconsResponse, error) {
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
	ret := &ListBeaconsResponse{
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
	//   "description": "Searches the beacon registry for beacons that match the given search criteria. Only those beacons that the client has permission to list will be returned.",
	//   "httpMethod": "GET",
	//   "id": "proximitybeacon.beacons.list",
	//   "parameters": {
	//     "pageSize": {
	//       "description": "The maximum number of records to return for this request, up to a server-defined upper limit.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A pagination token obtained from a previous request to list beacons.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "q": {
	//       "description": "Filter query string that supports the following field filters: * `description:\"\"` For example: `description:\"Room 3\"` Returns beacons whose description matches tokens in the string \"Room 3\" (not necessarily that exact string). The string must be double-quoted. * `status:` For example: `status:active` Returns beacons whose status matches the given value. Values must be one of the Beacon.Status enum values (case insensitive). Accepts multiple filters which will be combined with OR logic. * `stability:` For example: `stability:mobile` Returns beacons whose expected stability matches the given value. Values must be one of the Beacon.Stability enum values (case insensitive). Accepts multiple filters which will be combined with OR logic. * `place_id:\"\"` For example: `place_id:\"ChIJVSZzVR8FdkgRXGmmm6SslKw=\"` Returns beacons explicitly registered at the given place, expressed as a Place ID obtained from [Google Places API](/places/place-id). Does not match places inside the given place. Does not consider the beacon's actual location (which may be different from its registered place). Accepts multiple filters that will be combined with OR logic. The place ID must be double-quoted. * `registration_time[|=]` For example: `registration_time\u003e=1433116800` Returns beacons whose registration time matches the given filter. Supports the operators: , =. Timestamp must be expressed as an integer number of seconds since midnight January 1, 1970 UTC. Accepts at most two filters that will be combined with AND logic, to support \"between\" semantics. If more than two are supplied, the latter ones are ignored. * `lat: lng: radius:` For example: `lat:51.1232343 lng:-1.093852 radius:1000` Returns beacons whose registered location is within the given circle. When any of these fields are given, all are required. Latitude and longitude must be decimal degrees between -90.0 and 90.0 and between -180.0 and 180.0 respectively. Radius must be an integer number of meters less than 1,000,000 (1000 km). * `property:\"=\"` For example: `property:\"battery-type=CR2032\"` Returns beacons which have a property of the given name and value. Supports multiple filters which will be combined with OR logic. The entire name=value string must be double-quoted as one string. * `attachment_type:\"\"` For example: `attachment_type:\"my-namespace/my-type\"` Returns beacons having at least one attachment of the given namespaced type. Supports \"any within this namespace\" via the partial wildcard syntax: \"my-namespace/*\". Supports multiple filters which will be combined with OR logic. The string must be double-quoted. Multiple filters on the same field are combined with OR logic (except registration_time which is combined with AND logic). Multiple filters on different fields are combined with AND logic. Filters should be separated by spaces. As with any HTTP query string parameter, the whole filter expression must be URL-encoded. Example REST request: `GET /v1beta1/beacons?q=status:active%20lat:51.123%20lng:-1.095%20radius:1000`",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/beacons",
	//   "response": {
	//     "$ref": "ListBeaconsResponse"
	//   }
	// }

}

// method id "proximitybeacon.beacons.register":

type BeaconsRegisterCall struct {
	s          *Service
	beacon     *Beacon
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Register: Registers a previously unregistered beacon given its
// `advertisedId`. These IDs are unique within the system. An ID can be
// registered only once.
func (r *BeaconsService) Register(beacon *Beacon) *BeaconsRegisterCall {
	c := &BeaconsRegisterCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.beacon = beacon
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsRegisterCall) QuotaUser(quotaUser string) *BeaconsRegisterCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsRegisterCall) Fields(s ...googleapi.Field) *BeaconsRegisterCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsRegisterCall) Context(ctx context.Context) *BeaconsRegisterCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsRegisterCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.beacon)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/beacons:register")
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

// Do executes the "proximitybeacon.beacons.register" call.
// Exactly one of *Beacon or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Beacon.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *BeaconsRegisterCall) Do() (*Beacon, error) {
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
	ret := &Beacon{
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
	//   "description": "Registers a previously unregistered beacon given its `advertisedId`. These IDs are unique within the system. An ID can be registered only once.",
	//   "httpMethod": "POST",
	//   "id": "proximitybeacon.beacons.register",
	//   "path": "v1beta1/beacons:register",
	//   "request": {
	//     "$ref": "Beacon"
	//   },
	//   "response": {
	//     "$ref": "Beacon"
	//   }
	// }

}

// method id "proximitybeacon.beacons.update":

type BeaconsUpdateCall struct {
	s          *Service
	beaconName string
	beacon     *Beacon
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates the information about the specified beacon. **Any
// field that you do not populate in the submitted beacon will be
// permanently erased**, so you should follow the "read, modify, write"
// pattern to avoid inadvertently destroying data. Changes to the beacon
// status via this method will be silently ignored. To update beacon
// status, use the separate methods on this API for (de)activation and
// decommissioning.
func (r *BeaconsService) Update(beaconName string, beacon *Beacon) *BeaconsUpdateCall {
	c := &BeaconsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.beaconName = beaconName
	c.beacon = beacon
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsUpdateCall) QuotaUser(quotaUser string) *BeaconsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsUpdateCall) Fields(s ...googleapi.Field) *BeaconsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsUpdateCall) Context(ctx context.Context) *BeaconsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.beacon)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/{+beaconName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"beaconName": c.beaconName,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "proximitybeacon.beacons.update" call.
// Exactly one of *Beacon or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Beacon.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *BeaconsUpdateCall) Do() (*Beacon, error) {
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
	ret := &Beacon{
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
	//   "description": "Updates the information about the specified beacon. **Any field that you do not populate in the submitted beacon will be permanently erased**, so you should follow the \"read, modify, write\" pattern to avoid inadvertently destroying data. Changes to the beacon status via this method will be silently ignored. To update beacon status, use the separate methods on this API for (de)activation and decommissioning.",
	//   "httpMethod": "PUT",
	//   "id": "proximitybeacon.beacons.update",
	//   "parameterOrder": [
	//     "beaconName"
	//   ],
	//   "parameters": {
	//     "beaconName": {
	//       "description": "Resource name of this beacon. A beacon name has the format \"beacons/N!beaconId\" where the beaconId is the base16 ID broadcast by the beacon and N is a code for the beacon's type. Possible values are `3` for Eddystone, `1` for iBeacon, or `5` for AltBeacon. This field must be left empty when registering. After reading a beacon, clients can use the name for future operations.",
	//       "location": "path",
	//       "pattern": "^beacons/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/{+beaconName}",
	//   "request": {
	//     "$ref": "Beacon"
	//   },
	//   "response": {
	//     "$ref": "Beacon"
	//   }
	// }

}

// method id "proximitybeacon.beacons.attachments.batchDelete":

type BeaconsAttachmentsBatchDeleteCall struct {
	s          *Service
	beaconName string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// BatchDelete: Deletes multiple attachments on a given beacon. This
// operation is permanent and cannot be undone. You can optionally
// specify `namespacedType` to choose which attachments should be
// deleted. If you do not specify `namespacedType`, all your attachments
// on the given beacon will be deleted. You also may explicitly specify
// `*/*` to delete all.
func (r *BeaconsAttachmentsService) BatchDelete(beaconName string) *BeaconsAttachmentsBatchDeleteCall {
	c := &BeaconsAttachmentsBatchDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.beaconName = beaconName
	return c
}

// NamespacedType sets the optional parameter "namespacedType":
// Specifies the namespace and type of attachments to delete in
// `namespace/type` format. Accepts `*/*` to specify "all types in all
// namespaces".
func (c *BeaconsAttachmentsBatchDeleteCall) NamespacedType(namespacedType string) *BeaconsAttachmentsBatchDeleteCall {
	c.urlParams_.Set("namespacedType", namespacedType)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsAttachmentsBatchDeleteCall) QuotaUser(quotaUser string) *BeaconsAttachmentsBatchDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsAttachmentsBatchDeleteCall) Fields(s ...googleapi.Field) *BeaconsAttachmentsBatchDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsAttachmentsBatchDeleteCall) Context(ctx context.Context) *BeaconsAttachmentsBatchDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsAttachmentsBatchDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/{+beaconName}/attachments:batchDelete")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"beaconName": c.beaconName,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "proximitybeacon.beacons.attachments.batchDelete" call.
// Exactly one of *DeleteAttachmentsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *DeleteAttachmentsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *BeaconsAttachmentsBatchDeleteCall) Do() (*DeleteAttachmentsResponse, error) {
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
	ret := &DeleteAttachmentsResponse{
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
	//   "description": "Deletes multiple attachments on a given beacon. This operation is permanent and cannot be undone. You can optionally specify `namespacedType` to choose which attachments should be deleted. If you do not specify `namespacedType`, all your attachments on the given beacon will be deleted. You also may explicitly specify `*/*` to delete all.",
	//   "httpMethod": "POST",
	//   "id": "proximitybeacon.beacons.attachments.batchDelete",
	//   "parameterOrder": [
	//     "beaconName"
	//   ],
	//   "parameters": {
	//     "beaconName": {
	//       "description": "The beacon whose attachments are to be deleted. Required.",
	//       "location": "path",
	//       "pattern": "^beacons/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "namespacedType": {
	//       "description": "Specifies the namespace and type of attachments to delete in `namespace/type` format. Accepts `*/*` to specify \"all types in all namespaces\". Optional.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/{+beaconName}/attachments:batchDelete",
	//   "response": {
	//     "$ref": "DeleteAttachmentsResponse"
	//   }
	// }

}

// method id "proximitybeacon.beacons.attachments.create":

type BeaconsAttachmentsCreateCall struct {
	s                *Service
	beaconName       string
	beaconattachment *BeaconAttachment
	urlParams_       gensupport.URLParams
	ctx_             context.Context
}

// Create: Associates the given data with the specified beacon.
// Attachment data must contain two parts:
// - A namespaced type.
// - The actual attachment data itself.  The namespaced type consists of
// two parts, the namespace and the type. The namespace must be one of
// the values returned by the `namespaces` endpoint, while the type can
// be a string of any characters except for the forward slash (`/`) up
// to 100 characters in length. Attachment data can be up to 1024 bytes
// long.
func (r *BeaconsAttachmentsService) Create(beaconName string, beaconattachment *BeaconAttachment) *BeaconsAttachmentsCreateCall {
	c := &BeaconsAttachmentsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.beaconName = beaconName
	c.beaconattachment = beaconattachment
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsAttachmentsCreateCall) QuotaUser(quotaUser string) *BeaconsAttachmentsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsAttachmentsCreateCall) Fields(s ...googleapi.Field) *BeaconsAttachmentsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsAttachmentsCreateCall) Context(ctx context.Context) *BeaconsAttachmentsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsAttachmentsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.beaconattachment)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/{+beaconName}/attachments")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"beaconName": c.beaconName,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "proximitybeacon.beacons.attachments.create" call.
// Exactly one of *BeaconAttachment or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *BeaconAttachment.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *BeaconsAttachmentsCreateCall) Do() (*BeaconAttachment, error) {
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
	ret := &BeaconAttachment{
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
	//   "description": "Associates the given data with the specified beacon. Attachment data must contain two parts:  \n- A namespaced type. \n- The actual attachment data itself.  The namespaced type consists of two parts, the namespace and the type. The namespace must be one of the values returned by the `namespaces` endpoint, while the type can be a string of any characters except for the forward slash (`/`) up to 100 characters in length. Attachment data can be up to 1024 bytes long.",
	//   "httpMethod": "POST",
	//   "id": "proximitybeacon.beacons.attachments.create",
	//   "parameterOrder": [
	//     "beaconName"
	//   ],
	//   "parameters": {
	//     "beaconName": {
	//       "description": "The beacon on which the attachment should be created. Required.",
	//       "location": "path",
	//       "pattern": "^beacons/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/{+beaconName}/attachments",
	//   "request": {
	//     "$ref": "BeaconAttachment"
	//   },
	//   "response": {
	//     "$ref": "BeaconAttachment"
	//   }
	// }

}

// method id "proximitybeacon.beacons.attachments.delete":

type BeaconsAttachmentsDeleteCall struct {
	s              *Service
	attachmentName string
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Delete: Deletes the specified attachment for the given beacon. Each
// attachment has a unique attachment name (`attachmentName`) which is
// returned when you fetch the attachment data via this API. You specify
// this with the delete request to control which attachment is removed.
// This operation cannot be undone.
func (r *BeaconsAttachmentsService) Delete(attachmentName string) *BeaconsAttachmentsDeleteCall {
	c := &BeaconsAttachmentsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.attachmentName = attachmentName
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsAttachmentsDeleteCall) QuotaUser(quotaUser string) *BeaconsAttachmentsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsAttachmentsDeleteCall) Fields(s ...googleapi.Field) *BeaconsAttachmentsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsAttachmentsDeleteCall) Context(ctx context.Context) *BeaconsAttachmentsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsAttachmentsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/{+attachmentName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"attachmentName": c.attachmentName,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "proximitybeacon.beacons.attachments.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *BeaconsAttachmentsDeleteCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the specified attachment for the given beacon. Each attachment has a unique attachment name (`attachmentName`) which is returned when you fetch the attachment data via this API. You specify this with the delete request to control which attachment is removed. This operation cannot be undone.",
	//   "httpMethod": "DELETE",
	//   "id": "proximitybeacon.beacons.attachments.delete",
	//   "parameterOrder": [
	//     "attachmentName"
	//   ],
	//   "parameters": {
	//     "attachmentName": {
	//       "description": "The attachment name (`attachmentName`) of the attachment to remove. For example: `beacons/3!893737abc9/attachments/c5e937-af0-494-959-ec49d12738` Required.",
	//       "location": "path",
	//       "pattern": "^beacons/[^/]*/attachments/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/{+attachmentName}",
	//   "response": {
	//     "$ref": "Empty"
	//   }
	// }

}

// method id "proximitybeacon.beacons.attachments.list":

type BeaconsAttachmentsListCall struct {
	s            *Service
	beaconName   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Returns the attachments for the specified beacon that match the
// specified namespaced-type pattern. To control which namespaced types
// are returned, you add the `namespacedType` query parameter to the
// request. You must either use `*/*`, to return all attachments, or the
// namespace must be one of the ones returned from the `namespaces`
// endpoint.
func (r *BeaconsAttachmentsService) List(beaconName string) *BeaconsAttachmentsListCall {
	c := &BeaconsAttachmentsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.beaconName = beaconName
	return c
}

// NamespacedType sets the optional parameter "namespacedType":
// Specifies the namespace and type of attachment to include in response
// in namespace/type format. Accepts `*/*` to specify "all types in all
// namespaces".
func (c *BeaconsAttachmentsListCall) NamespacedType(namespacedType string) *BeaconsAttachmentsListCall {
	c.urlParams_.Set("namespacedType", namespacedType)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsAttachmentsListCall) QuotaUser(quotaUser string) *BeaconsAttachmentsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsAttachmentsListCall) Fields(s ...googleapi.Field) *BeaconsAttachmentsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BeaconsAttachmentsListCall) IfNoneMatch(entityTag string) *BeaconsAttachmentsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsAttachmentsListCall) Context(ctx context.Context) *BeaconsAttachmentsListCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsAttachmentsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/{+beaconName}/attachments")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"beaconName": c.beaconName,
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

// Do executes the "proximitybeacon.beacons.attachments.list" call.
// Exactly one of *ListBeaconAttachmentsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListBeaconAttachmentsResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *BeaconsAttachmentsListCall) Do() (*ListBeaconAttachmentsResponse, error) {
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
	ret := &ListBeaconAttachmentsResponse{
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
	//   "description": "Returns the attachments for the specified beacon that match the specified namespaced-type pattern. To control which namespaced types are returned, you add the `namespacedType` query parameter to the request. You must either use `*/*`, to return all attachments, or the namespace must be one of the ones returned from the `namespaces` endpoint.",
	//   "httpMethod": "GET",
	//   "id": "proximitybeacon.beacons.attachments.list",
	//   "parameterOrder": [
	//     "beaconName"
	//   ],
	//   "parameters": {
	//     "beaconName": {
	//       "description": "The beacon whose attachments are to be fetched. Required.",
	//       "location": "path",
	//       "pattern": "^beacons/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "namespacedType": {
	//       "description": "Specifies the namespace and type of attachment to include in response in namespace/type format. Accepts `*/*` to specify \"all types in all namespaces\".",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/{+beaconName}/attachments",
	//   "response": {
	//     "$ref": "ListBeaconAttachmentsResponse"
	//   }
	// }

}

// method id "proximitybeacon.beacons.diagnostics.list":

type BeaconsDiagnosticsListCall struct {
	s            *Service
	beaconName   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: List the diagnostics for a single beacon. You can also list
// diagnostics for all the beacons owned by your Google Developers
// Console project by using the beacon name `beacons/-`.
func (r *BeaconsDiagnosticsService) List(beaconName string) *BeaconsDiagnosticsListCall {
	c := &BeaconsDiagnosticsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.beaconName = beaconName
	return c
}

// AlertFilter sets the optional parameter "alertFilter": Requests only
// beacons that have the given alert. For example, to find beacons that
// have low batteries use `alert_filter=LOW_BATTERY`.
//
// Possible values:
//   "ALERT_UNSPECIFIED"
//   "WRONG_LOCATION"
//   "LOW_BATTERY"
func (c *BeaconsDiagnosticsListCall) AlertFilter(alertFilter string) *BeaconsDiagnosticsListCall {
	c.urlParams_.Set("alertFilter", alertFilter)
	return c
}

// PageSize sets the optional parameter "pageSize": Specifies the
// maximum number of results to return. Defaults to 10. Maximum 1000.
func (c *BeaconsDiagnosticsListCall) PageSize(pageSize int64) *BeaconsDiagnosticsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Requests results
// that occur after the `page_token`, obtained from the response to a
// previous request.
func (c *BeaconsDiagnosticsListCall) PageToken(pageToken string) *BeaconsDiagnosticsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *BeaconsDiagnosticsListCall) QuotaUser(quotaUser string) *BeaconsDiagnosticsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BeaconsDiagnosticsListCall) Fields(s ...googleapi.Field) *BeaconsDiagnosticsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BeaconsDiagnosticsListCall) IfNoneMatch(entityTag string) *BeaconsDiagnosticsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BeaconsDiagnosticsListCall) Context(ctx context.Context) *BeaconsDiagnosticsListCall {
	c.ctx_ = ctx
	return c
}

func (c *BeaconsDiagnosticsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/{+beaconName}/diagnostics")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"beaconName": c.beaconName,
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

// Do executes the "proximitybeacon.beacons.diagnostics.list" call.
// Exactly one of *ListDiagnosticsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListDiagnosticsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *BeaconsDiagnosticsListCall) Do() (*ListDiagnosticsResponse, error) {
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
	ret := &ListDiagnosticsResponse{
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
	//   "description": "List the diagnostics for a single beacon. You can also list diagnostics for all the beacons owned by your Google Developers Console project by using the beacon name `beacons/-`.",
	//   "httpMethod": "GET",
	//   "id": "proximitybeacon.beacons.diagnostics.list",
	//   "parameterOrder": [
	//     "beaconName"
	//   ],
	//   "parameters": {
	//     "alertFilter": {
	//       "description": "Requests only beacons that have the given alert. For example, to find beacons that have low batteries use `alert_filter=LOW_BATTERY`.",
	//       "enum": [
	//         "ALERT_UNSPECIFIED",
	//         "WRONG_LOCATION",
	//         "LOW_BATTERY"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "beaconName": {
	//       "description": "Beacon that the diagnostics are for.",
	//       "location": "path",
	//       "pattern": "^beacons/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Specifies the maximum number of results to return. Defaults to 10. Maximum 1000. Optional.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Requests results that occur after the `page_token`, obtained from the response to a previous request. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1/{+beaconName}/diagnostics",
	//   "response": {
	//     "$ref": "ListDiagnosticsResponse"
	//   }
	// }

}

// method id "proximitybeacon.namespaces.list":

type NamespacesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists all attachment namespaces owned by your Google Developers
// Console project. Attachment data associated with a beacon must
// include a namespaced type, and the namespace must be owned by your
// project.
func (r *NamespacesService) List() *NamespacesListCall {
	c := &NamespacesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *NamespacesListCall) QuotaUser(quotaUser string) *NamespacesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *NamespacesListCall) Fields(s ...googleapi.Field) *NamespacesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *NamespacesListCall) IfNoneMatch(entityTag string) *NamespacesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *NamespacesListCall) Context(ctx context.Context) *NamespacesListCall {
	c.ctx_ = ctx
	return c
}

func (c *NamespacesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/namespaces")
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

// Do executes the "proximitybeacon.namespaces.list" call.
// Exactly one of *ListNamespacesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListNamespacesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *NamespacesListCall) Do() (*ListNamespacesResponse, error) {
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
	ret := &ListNamespacesResponse{
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
	//   "description": "Lists all attachment namespaces owned by your Google Developers Console project. Attachment data associated with a beacon must include a namespaced type, and the namespace must be owned by your project.",
	//   "httpMethod": "GET",
	//   "id": "proximitybeacon.namespaces.list",
	//   "path": "v1beta1/namespaces",
	//   "response": {
	//     "$ref": "ListNamespacesResponse"
	//   }
	// }

}
