// Package spectrum provides access to the Google Spectrum Database API.
//
// See http://developers.google.com/spectrum
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/spectrum/v1explorer"
//   ...
//   spectrumService, err := spectrum.New(oauthHttpClient)
package spectrum

import (
	"bytes"
	"code.google.com/p/google-api-go-client/googleapi"
	"encoding/json"
	"errors"
	"fmt"
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
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace

const apiId = "spectrum:v1explorer"
const apiName = "spectrum"
const apiVersion = "v1explorer"
const basePath = "https://www.googleapis.com/spectrum/v1explorer/paws/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Paws = NewPawsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Paws *PawsService
}

func NewPawsService(s *Service) *PawsService {
	rs := &PawsService{s: s}
	return rs
}

type PawsService struct {
	s *Service
}

type AntennaCharacteristics struct {
	// Height: The antenna height in meters. Whether the antenna height is
	// required depends on the device type and the regulatory domain. Note
	// that the height may be negative.
	Height float64 `json:"height,omitempty"`

	// HeightType: If the height is required, then the height type (AGL for
	// above ground level or AMSL for above mean sea level) is also
	// required. The default is AGL.
	HeightType string `json:"heightType,omitempty"`

	// HeightUncertainty: The height uncertainty in meters. Whether this is
	// required depends on the regulatory domain.
	HeightUncertainty float64 `json:"heightUncertainty,omitempty"`
}

type DatabaseSpec struct {
	// Name: The display name for a database.
	Name string `json:"name,omitempty"`

	// Uri: The corresponding URI of the database.
	Uri string `json:"uri,omitempty"`
}

type DbUpdateSpec struct {
	// Databases: A required list of one or more databases. A device should
	// update its preconfigured list of databases to replace (only) the
	// database that provided the response with the specified entries.
	Databases []*DatabaseSpec `json:"databases,omitempty"`
}

type DeviceCapabilities struct {
	// FrequencyRanges: An optional list of frequency ranges supported by
	// the device. Each element must contain start and stop frequencies in
	// which the device can operate. Channel identifiers are optional. When
	// specified, the database should not return available spectrum that
	// falls outside these ranges or channel IDs.
	FrequencyRanges []*FrequencyRange `json:"frequencyRanges,omitempty"`
}

type DeviceDescriptor struct {
	// EtsiEnDeviceCategory: Specifies the ETSI white space device category.
	// Valid values are the strings master and slave. This field is
	// case-insensitive. Consult the ETSI documentation for details about
	// the device types.
	EtsiEnDeviceCategory string `json:"etsiEnDeviceCategory,omitempty"`

	// EtsiEnDeviceEmissionsClass: Specifies the ETSI white space device
	// emissions class. The values are represented by numeric strings, such
	// as 1, 2, etc. Consult the ETSI documentation for details about the
	// device types.
	EtsiEnDeviceEmissionsClass string `json:"etsiEnDeviceEmissionsClass,omitempty"`

	// EtsiEnDeviceType: Specifies the ETSI white space device type. Valid
	// values are single-letter strings, such as A, B, etc. Consult the ETSI
	// documentation for details about the device types.
	EtsiEnDeviceType string `json:"etsiEnDeviceType,omitempty"`

	// EtsiEnTechnologyId: Specifies the ETSI white space device technology
	// identifier. The string value must not exceed 64 characters in length.
	// Consult the ETSI documentation for details about the device types.
	EtsiEnTechnologyId string `json:"etsiEnTechnologyId,omitempty"`

	// FccId: Specifies the device's FCC certification identifier. The value
	// is an identifier string whose length should not exceed 32 characters.
	// Note that, in practice, a valid FCC ID may be limited to 19
	// characters.
	FccId string `json:"fccId,omitempty"`

	// FccTvbdDeviceType: Specifies the TV Band White Space device type, as
	// defined by the FCC. Valid values are FIXED, MODE_1, MODE_2.
	FccTvbdDeviceType string `json:"fccTvbdDeviceType,omitempty"`

	// ManufacturerId: The manufacturer's ID may be required by the
	// regulatory domain. This should represent the name of the device
	// manufacturer, should be consistent across all devices from the same
	// manufacturer, and should be distinct from that of other
	// manufacturers. The string value must not exceed 64 characters in
	// length.
	ManufacturerId string `json:"manufacturerId,omitempty"`

	// ModelId: The device's model ID may be required by the regulatory
	// domain. The string value must not exceed 64 characters in length.
	ModelId string `json:"modelId,omitempty"`

	// RulesetIds: The list of identifiers for rulesets supported by the
	// device. A database may require that the device provide this list
	// before servicing the device requests. If the database does not
	// support any of the rulesets specified in the list, the database may
	// refuse to service the device requests. If present, the list must
	// contain at least one entry.
	//
	// For information about the valid
	// requests, see section 9.2 of the PAWS specification. Currently,
	// FccTvBandWhiteSpace-2010 is the only supported ruleset.
	RulesetIds []string `json:"rulesetIds,omitempty"`

	// SerialNumber: The manufacturer's device serial number; required by
	// the applicable regulatory domain. The length of the value must not
	// exceed 64 characters.
	SerialNumber string `json:"serialNumber,omitempty"`
}

type DeviceOwner struct {
	// Operator: The vCard contact information for the device operator is
	// optional, but may be required by specific regulatory domains.
	Operator *Vcard `json:"operator,omitempty"`

	// Owner: The vCard contact information for the individual or business
	// that owns the device is required.
	Owner *Vcard `json:"owner,omitempty"`
}

type DeviceValidity struct {
	// DeviceDesc: The descriptor of the device for which the validity check
	// was requested. It will always be present.
	DeviceDesc *DeviceDescriptor `json:"deviceDesc,omitempty"`

	// IsValid: The validity status: true if the device is valid for
	// operation, false otherwise. It will always be present.
	IsValid bool `json:"isValid,omitempty"`

	// Reason: If the device identifier is not valid, the database may
	// include a reason. The reason may be in any language. The length of
	// the value should not exceed 128 characters.
	Reason string `json:"reason,omitempty"`
}

type EventTime struct {
	// StartTime: The inclusive start of the event. It will be present.
	StartTime string `json:"startTime,omitempty"`

	// StopTime: The exclusive end of the event. It will be present.
	StopTime string `json:"stopTime,omitempty"`
}

type FrequencyRange struct {
	// ChannelId: The database may include a channel identifier, when
	// applicable. When it is included, the device should treat it as
	// informative. The length of the identifier should not exceed 16
	// characters.
	ChannelId string `json:"channelId,omitempty"`

	// MaxPowerDBm: The maximum total power level (EIRP)—computed over the
	// corresponding operating bandwidth—that is permitted within the
	// frequency range. Depending on the context in which the
	// frequency-range element appears, this value may be required. For
	// example, it is required in the available-spectrum response,
	// available-spectrum-batch response, and spectrum-use notification
	// message, but it should not be present (it is not applicable) when the
	// frequency range appears inside a device-capabilities message.
	MaxPowerDBm float64 `json:"maxPowerDBm,omitempty"`

	// StartHz: The required inclusive start of the frequency range (in
	// Hertz).
	StartHz float64 `json:"startHz,omitempty"`

	// StopHz: The required exclusive end of the frequency range (in Hertz).
	StopHz float64 `json:"stopHz,omitempty"`
}

type GeoLocation struct {
	// Confidence: The location confidence level, as an integer percentage,
	// may be required, depending on the regulatory domain. When the
	// parameter is optional and not provided, its value is assumed to be
	// 95. Valid values range from 0 to 99, since, in practice, 100-percent
	// confidence is not achievable. The confidence value is meaningful only
	// when geolocation refers to a point with uncertainty.
	Confidence int64 `json:"confidence,omitempty"`

	// Point: If present, indicates that the geolocation represents a point.
	// Paradoxically, a point is parameterized using an ellipse, where the
	// center represents the location of the point and the distances along
	// the major and minor axes represent the uncertainty. The uncertainty
	// values may be required, depending on the regulatory domain.
	Point *GeoLocationEllipse `json:"point,omitempty"`

	// Region: If present, indicates that the geolocation represents a
	// region. Database support for regions is optional.
	Region *GeoLocationPolygon `json:"region,omitempty"`
}

type GeoLocationEllipse struct {
	// Center: A required geo-spatial point representing the center of the
	// ellipse.
	Center *GeoLocationPoint `json:"center,omitempty"`

	// Orientation: A floating-point number that expresses the orientation
	// of the ellipse, representing the rotation, in degrees, of the
	// semi-major axis from North towards the East. For example, when the
	// uncertainty is greatest along the North-South direction, orientation
	// is 0 degrees; conversely, if the uncertainty is greatest along the
	// East-West direction, orientation is 90 degrees. When orientation is
	// not present, the orientation is assumed to be 0.
	Orientation float64 `json:"orientation,omitempty"`

	// SemiMajorAxis: A floating-point number that expresses the location
	// uncertainty along the major axis of the ellipse. May be required by
	// the regulatory domain. When the uncertainty is optional, the default
	// value is 0.
	SemiMajorAxis float64 `json:"semiMajorAxis,omitempty"`

	// SemiMinorAxis: A floating-point number that expresses the location
	// uncertainty along the minor axis of the ellipse. May be required by
	// the regulatory domain. When the uncertainty is optional, the default
	// value is 0.
	SemiMinorAxis float64 `json:"semiMinorAxis,omitempty"`
}

type GeoLocationPoint struct {
	// Latitude: A required floating-point number that expresses the
	// latitude in degrees using the WGS84 datum. For details on this
	// encoding, see the National Imagery and Mapping Agency's Technical
	// Report TR8350.2.
	Latitude float64 `json:"latitude,omitempty"`

	// Longitude: A required floating-point number that expresses the
	// longitude in degrees using the WGS84 datum. For details on this
	// encoding, see the National Imagery and Mapping Agency's Technical
	// Report TR8350.2.
	Longitude float64 `json:"longitude,omitempty"`
}

type GeoLocationPolygon struct {
	// Exterior: When the geolocation describes a region, the exterior field
	// refers to a list of latitude/longitude points that represent the
	// vertices of a polygon. The first and last points must be the same.
	// Thus, a minimum of four points is required. The following polygon
	// restrictions from RFC5491 apply:
	// - A connecting line shall not
	// cross another connecting line of the same polygon.
	// - The vertices
	// must be defined in a counterclockwise order.
	// - The edges of a
	// polygon are defined by the shortest path between two points in space
	// (not a geodesic curve). Consequently, the length between two adjacent
	// vertices should be restricted to a maximum of 130 km.
	// - All vertices
	// are assumed to be at the same altitude.
	// - Polygon shapes should be
	// restricted to a maximum of 15 vertices (16 points that include the
	// repeated vertex).
	Exterior []*GeoLocationPoint `json:"exterior,omitempty"`
}

type GeoSpectrumSchedule struct {
	// Location: The geolocation identifies the location at which the
	// spectrum schedule applies. It will always be present.
	Location *GeoLocation `json:"location,omitempty"`

	// SpectrumSchedules: A list of available spectrum profiles and
	// associated times. It will always be present, and at least one
	// schedule must be included (though it may be empty if there is no
	// available spectrum). More than one schedule may be included to
	// represent future changes to the available spectrum.
	SpectrumSchedules []*SpectrumSchedule `json:"spectrumSchedules,omitempty"`
}

type PawsGetSpectrumBatchRequest struct {
	// Antenna: Depending on device type and regulatory domain, antenna
	// characteristics may be required.
	Antenna *AntennaCharacteristics `json:"antenna,omitempty"`

	// Capabilities: The master device may include its device capabilities
	// to limit the available-spectrum batch response to the spectrum that
	// is compatible with its capabilities. The database should not return
	// spectrum that is incompatible with the specified capabilities.
	Capabilities *DeviceCapabilities `json:"capabilities,omitempty"`

	// DeviceDesc: When the available spectrum request is made on behalf of
	// a specific device (a master or slave device), device descriptor
	// information for the device on whose behalf the request is made is
	// required (in such cases, the requestType parameter must be empty).
	// When a requestType value is specified, device descriptor information
	// may be optional or required according to the rules of the applicable
	// regulatory domain.
	DeviceDesc *DeviceDescriptor `json:"deviceDesc,omitempty"`

	// Locations: A geolocation list is required. This allows a device to
	// specify its current location plus additional anticipated locations
	// when allowed by the regulatory domain. At least one location must be
	// included. Geolocation must be given as the location of the radiation
	// center of the device's antenna. If a location specifies a region,
	// rather than a point, the database may return an UNIMPLEMENTED error
	// if it does not support query by region.
	//
	// There is no upper limit on
	// the number of locations included in a available spectrum batch
	// request, but the database may restrict the number of locations it
	// supports by returning a response with fewer locations than specified
	// in the batch request. Note that geolocations must be those of the
	// master device (a device with geolocation capability that makes an
	// available spectrum batch request), whether the master device is
	// making the request on its own behalf or on behalf of a slave device
	// (one without geolocation capability).
	Locations []*GeoLocation `json:"locations,omitempty"`

	// MasterDeviceDesc: When an available spectrum batch request is made by
	// the master device (a device with geolocation capability) on behalf of
	// a slave device (a device without geolocation capability), the rules
	// of the applicable regulatory domain may require the master device to
	// provide its own device descriptor information (in addition to device
	// descriptor information for the slave device in a separate parameter).
	MasterDeviceDesc *DeviceDescriptor `json:"masterDeviceDesc,omitempty"`

	// Owner: Depending on device type and regulatory domain, device owner
	// information may be included in an available spectrum batch request.
	// This allows the device to register and get spectrum-availability
	// information in a single request.
	Owner *DeviceOwner `json:"owner,omitempty"`

	// RequestType: The request type parameter is an optional parameter that
	// can be used to modify an available spectrum batch request, but its
	// use depends on applicable regulatory rules. For example, It may be
	// used to request generic slave device parameters without having to
	// specify the device descriptor for a specific device. When the
	// requestType parameter is missing, the request is for a specific
	// device (master or slave), and the device descriptor parameter for the
	// device on whose behalf the batch request is made is required.
	RequestType string `json:"requestType,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type PawsGetSpectrumBatchResponse struct {
	// DatabaseChange: A database may include the databaseChange parameter
	// to notify a device of a change to its database URI, providing one or
	// more alternate database URIs. The device should use this information
	// to update its list of pre-configured databases by (only) replacing
	// its entry for the responding database with the list of alternate
	// URIs.
	DatabaseChange *DbUpdateSpec `json:"databaseChange,omitempty"`

	// DeviceDesc: The database must return in its available spectrum
	// response the device descriptor information it received in the master
	// device's available spectrum batch request.
	DeviceDesc *DeviceDescriptor `json:"deviceDesc,omitempty"`

	// GeoSpectrumSchedules: The available spectrum batch response must
	// contain a geo-spectrum schedule list, The list may be empty if
	// spectrum is not available. The database may return more than one
	// geo-spectrum schedule to represent future changes to the available
	// spectrum. How far in advance a schedule may be provided depends upon
	// the applicable regulatory domain. The database may return available
	// spectrum for fewer geolocations than requested. The device must not
	// make assumptions about the order of the entries in the list, and must
	// use the geolocation value in each geo-spectrum schedule entry to
	// match available spectrum to a location.
	GeoSpectrumSchedules []*GeoSpectrumSchedule `json:"geoSpectrumSchedules,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "spectrum#pawsGetSpectrumBatchResponse".
	Kind string `json:"kind,omitempty"`

	// MaxContiguousBwHz: The database may return a constraint on the
	// allowed maximum contiguous bandwidth (in Hertz). A regulatory domain
	// may require the database to return this parameter. When this
	// parameter is present in the response, the device must apply this
	// constraint to its spectrum-selection logic to ensure that no single
	// block of spectrum has bandwidth that exceeds this value.
	MaxContiguousBwHz float64 `json:"maxContiguousBwHz,omitempty"`

	// MaxTotalBwHz: The database may return a constraint on the allowed
	// maximum total bandwidth (in Hertz), which does not need to be
	// contiguous. A regulatory domain may require the database to return
	// this parameter. When this parameter is present in the available
	// spectrum batch response, the device must apply this constraint to its
	// spectrum-selection logic to ensure that total bandwidth does not
	// exceed this value.
	MaxTotalBwHz float64 `json:"maxTotalBwHz,omitempty"`

	// NeedsSpectrumReport: For regulatory domains that require a
	// spectrum-usage report from devices, the database must return true for
	// this parameter if the geo-spectrum schedules list is not empty;
	// otherwise, the database should either return false or omit this
	// parameter. If this parameter is present and its value is true, the
	// device must send a spectrum use notify message to the database;
	// otherwise, the device should not send the notification.
	NeedsSpectrumReport bool `json:"needsSpectrumReport,omitempty"`

	// RulesetInfo: The database should return ruleset information, which
	// identifies the applicable regulatory authority and ruleset for the
	// available spectrum batch response. If included, the device must use
	// the corresponding ruleset to interpret the response. Values provided
	// in the returned ruleset information, such as maxLocationChange, take
	// precedence over any conflicting values provided in the ruleset
	// information returned in a prior initialization response sent by the
	// database to the device.
	RulesetInfo *RulesetInfo `json:"rulesetInfo,omitempty"`

	// Timestamp: The database includes a timestamp of the form,
	// YYYY-MM-DDThh:mm:ssZ (Internet timestamp format per RFC3339), in its
	// available spectrum batch response. The timestamp should be used by
	// the device as a reference for the start and stop times specified in
	// the response spectrum schedules.
	Timestamp string `json:"timestamp,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type PawsGetSpectrumRequest struct {
	// Antenna: Depending on device type and regulatory domain, the
	// characteristics of the antenna may be required.
	Antenna *AntennaCharacteristics `json:"antenna,omitempty"`

	// Capabilities: The master device may include its device capabilities
	// to limit the available-spectrum response to the spectrum that is
	// compatible with its capabilities. The database should not return
	// spectrum that is incompatible with the specified capabilities.
	Capabilities *DeviceCapabilities `json:"capabilities,omitempty"`

	// DeviceDesc: When the available spectrum request is made on behalf of
	// a specific device (a master or slave device), device descriptor
	// information for that device is required (in such cases, the
	// requestType parameter must be empty). When a requestType value is
	// specified, device descriptor information may be optional or required
	// according to the rules of the applicable regulatory domain.
	DeviceDesc *DeviceDescriptor `json:"deviceDesc,omitempty"`

	// Location: The geolocation of the master device (a device with
	// geolocation capability that makes an available spectrum request) is
	// required whether the master device is making the request on its own
	// behalf or on behalf of a slave device (one without geolocation
	// capability). The location must be the location of the radiation
	// center of the master device's antenna. To support mobile devices, a
	// regulatory domain may allow the anticipated position of the master
	// device to be given instead. If the location specifies a region,
	// rather than a point, the database may return an UNIMPLEMENTED error
	// code if it does not support query by region.
	Location *GeoLocation `json:"location,omitempty"`

	// MasterDeviceDesc: When an available spectrum request is made by the
	// master device (a device with geolocation capability) on behalf of a
	// slave device (a device without geolocation capability), the rules of
	// the applicable regulatory domain may require the master device to
	// provide its own device descriptor information (in addition to device
	// descriptor information for the slave device, which is provided in a
	// separate parameter).
	MasterDeviceDesc *DeviceDescriptor `json:"masterDeviceDesc,omitempty"`

	// Owner: Depending on device type and regulatory domain, device owner
	// information may be included in an available spectrum request. This
	// allows the device to register and get spectrum-availability
	// information in a single request.
	Owner *DeviceOwner `json:"owner,omitempty"`

	// RequestType: The request type parameter is an optional parameter that
	// can be used to modify an available spectrum request, but its use
	// depends on applicable regulatory rules. It may be used, for example,
	// to request generic slave device parameters without having to specify
	// the device descriptor for a specific device. When the requestType
	// parameter is missing, the request is for a specific device (master or
	// slave), and the deviceDesc parameter for the device on whose behalf
	// the request is made is required.
	RequestType string `json:"requestType,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type PawsGetSpectrumResponse struct {
	// DatabaseChange: A database may include the databaseChange parameter
	// to notify a device of a change to its database URI, providing one or
	// more alternate database URIs. The device should use this information
	// to update its list of pre-configured databases by (only) replacing
	// its entry for the responding database with the list of alternate
	// URIs.
	DatabaseChange *DbUpdateSpec `json:"databaseChange,omitempty"`

	// DeviceDesc: The database must return, in its available spectrum
	// response, the device descriptor information it received in the master
	// device's available spectrum request.
	DeviceDesc *DeviceDescriptor `json:"deviceDesc,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "spectrum#pawsGetSpectrumResponse".
	Kind string `json:"kind,omitempty"`

	// MaxContiguousBwHz: The database may return a constraint on the
	// allowed maximum contiguous bandwidth (in Hertz). A regulatory domain
	// may require the database to return this parameter. When this
	// parameter is present in the response, the device must apply this
	// constraint to its spectrum-selection logic to ensure that no single
	// block of spectrum has bandwidth that exceeds this value.
	MaxContiguousBwHz float64 `json:"maxContiguousBwHz,omitempty"`

	// MaxTotalBwHz: The database may return a constraint on the allowed
	// maximum total bandwidth (in Hertz), which need not be contiguous. A
	// regulatory domain may require the database to return this parameter.
	// When this parameter is present in the available spectrum response,
	// the device must apply this constraint to its spectrum-selection logic
	// to ensure that total bandwidth does not exceed this value.
	MaxTotalBwHz float64 `json:"maxTotalBwHz,omitempty"`

	// NeedsSpectrumReport: For regulatory domains that require a
	// spectrum-usage report from devices, the database must return true for
	// this parameter if the spectrum schedule list is not empty; otherwise,
	// the database will either return false or omit this parameter. If this
	// parameter is present and its value is true, the device must send a
	// spectrum use notify message to the database; otherwise, the device
	// must not send the notification.
	NeedsSpectrumReport bool `json:"needsSpectrumReport,omitempty"`

	// RulesetInfo: The database should return ruleset information, which
	// identifies the applicable regulatory authority and ruleset for the
	// available spectrum response. If included, the device must use the
	// corresponding ruleset to interpret the response. Values provided in
	// the returned ruleset information, such as maxLocationChange, take
	// precedence over any conflicting values provided in the ruleset
	// information returned in a prior initialization response sent by the
	// database to the device.
	RulesetInfo *RulesetInfo `json:"rulesetInfo,omitempty"`

	// SpectrumSchedules: The available spectrum response must contain a
	// spectrum schedule list. The list may be empty if spectrum is not
	// available. The database may return more than one spectrum schedule to
	// represent future changes to the available spectrum. How far in
	// advance a schedule may be provided depends on the applicable
	// regulatory domain.
	SpectrumSchedules []*SpectrumSchedule `json:"spectrumSchedules,omitempty"`

	// Timestamp: The database includes a timestamp of the form
	// YYYY-MM-DDThh:mm:ssZ (Internet timestamp format per RFC3339) in its
	// available spectrum response. The timestamp should be used by the
	// device as a reference for the start and stop times specified in the
	// response spectrum schedules.
	Timestamp string `json:"timestamp,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type PawsInitRequest struct {
	// DeviceDesc: The DeviceDescriptor parameter is required. If the
	// database does not support the device or any of the rulesets specified
	// in the device descriptor, it must return an UNSUPPORTED error code in
	// the error response.
	DeviceDesc *DeviceDescriptor `json:"deviceDesc,omitempty"`

	// Location: A device's geolocation is required.
	Location *GeoLocation `json:"location,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type PawsInitResponse struct {
	// DatabaseChange: A database may include the databaseChange parameter
	// to notify a device of a change to its database URI, providing one or
	// more alternate database URIs. The device should use this information
	// to update its list of pre-configured databases by (only) replacing
	// its entry for the responding database with the list of alternate
	// URIs.
	DatabaseChange *DbUpdateSpec `json:"databaseChange,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "spectrum#pawsInitResponse".
	Kind string `json:"kind,omitempty"`

	// RulesetInfo: The rulesetInfo parameter must be included in the
	// response. This parameter specifies the regulatory domain and
	// parameters applicable to that domain. The database must include the
	// authority field, which defines the regulatory domain for the location
	// specified in the INIT_REQ message.
	RulesetInfo *RulesetInfo `json:"rulesetInfo,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type PawsNotifySpectrumUseRequest struct {
	// DeviceDesc: Device descriptor information is required in the
	// spectrum-use notification message.
	DeviceDesc *DeviceDescriptor `json:"deviceDesc,omitempty"`

	// Location: The geolocation of the master device (the device that is
	// sending the spectrum-use notification) to the database is required in
	// the spectrum-use notification message.
	Location *GeoLocation `json:"location,omitempty"`

	// Spectra: A spectrum list is required in the spectrum-use
	// notification. The list specifies the spectrum that the device expects
	// to use, which includes frequency ranges and maximum power levels. The
	// list may be empty if the device decides not to use any of spectrum.
	// For consistency, the psdBandwidthHz value should match that from one
	// of the spectrum elements in the corresponding available spectrum
	// response previously sent to the device by the database. Note that
	// maximum power levels in the spectrum element must be expressed as
	// power spectral density over the specified psdBandwidthHz value. The
	// actual bandwidth to be used (as computed from the start and stop
	// frequencies) may be different from the psdBandwidthHz value. As an
	// example, when regulatory rules express maximum power spectral density
	// in terms of maximum power over any 100 kHz band, then the
	// psdBandwidthHz value should be set to 100 kHz, even though the actual
	// bandwidth used can be 20 kHz.
	Spectra []*SpectrumMessage `json:"spectra,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type PawsNotifySpectrumUseResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "spectrum#pawsNotifySpectrumUseResponse".
	Kind string `json:"kind,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type PawsRegisterRequest struct {
	// Antenna: Antenna characteristics, including its height and height
	// type.
	Antenna *AntennaCharacteristics `json:"antenna,omitempty"`

	// DeviceDesc: A DeviceDescriptor is required.
	DeviceDesc *DeviceDescriptor `json:"deviceDesc,omitempty"`

	// DeviceOwner: Device owner information is required.
	DeviceOwner *DeviceOwner `json:"deviceOwner,omitempty"`

	// Location: A device's geolocation is required.
	Location *GeoLocation `json:"location,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type PawsRegisterResponse struct {
	// DatabaseChange: A database may include the databaseChange parameter
	// to notify a device of a change to its database URI, providing one or
	// more alternate database URIs. The device should use this information
	// to update its list of pre-configured databases by (only) replacing
	// its entry for the responding database with the list of alternate
	// URIs.
	DatabaseChange *DbUpdateSpec `json:"databaseChange,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "spectrum#pawsRegisterResponse".
	Kind string `json:"kind,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type PawsVerifyDeviceRequest struct {
	// DeviceDescs: A list of device descriptors, which specifies the slave
	// devices to be validated, is required.
	DeviceDescs []*DeviceDescriptor `json:"deviceDescs,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type PawsVerifyDeviceResponse struct {
	// DatabaseChange: A database may include the databaseChange parameter
	// to notify a device of a change to its database URI, providing one or
	// more alternate database URIs. The device should use this information
	// to update its list of pre-configured databases by (only) replacing
	// its entry for the responding database with the list of alternate
	// URIs.
	DatabaseChange *DbUpdateSpec `json:"databaseChange,omitempty"`

	// DeviceValidities: A device validities list is required in the device
	// validation response to report whether each slave device listed in a
	// previous device validation request is valid. The number of entries
	// must match the number of device descriptors listed in the previous
	// device validation request.
	DeviceValidities []*DeviceValidity `json:"deviceValidities,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "spectrum#pawsVerifyDeviceResponse".
	Kind string `json:"kind,omitempty"`

	// Type: The message type (e.g., INIT_REQ, AVAIL_SPECTRUM_REQ,
	// ...).
	//
	// Required field.
	Type string `json:"type,omitempty"`

	// Version: The PAWS version. Must be exactly 1.0.
	//
	// Required field.
	Version string `json:"version,omitempty"`
}

type RulesetInfo struct {
	// Authority: The regulatory domain to which the ruleset belongs is
	// required. It must be a 2-letter country code. The device should use
	// this to determine additional device behavior required by the
	// associated regulatory domain.
	Authority string `json:"authority,omitempty"`

	// MaxLocationChange: The maximum location change in meters is required
	// in the initialization response, but optional otherwise. When the
	// device changes location by more than this specified distance, it must
	// contact the database to get the available spectrum for the new
	// location. If the device is using spectrum that is no longer
	// available, it must immediately cease use of the spectrum under rules
	// for database-managed spectrum. If this value is provided within the
	// context of an available-spectrum response, it takes precedence over
	// the value within the initialization response.
	MaxLocationChange float64 `json:"maxLocationChange,omitempty"`

	// MaxPollingSecs: The maximum duration, in seconds, between requests
	// for available spectrum. It is required in the initialization
	// response, but optional otherwise. The device must contact the
	// database to get available spectrum no less frequently than this
	// duration. If the new spectrum information indicates that the device
	// is using spectrum that is no longer available, it must immediately
	// cease use of those frequencies under rules for database-managed
	// spectrum. If this value is provided within the context of an
	// available-spectrum response, it takes precedence over the value
	// within the initialization response.
	MaxPollingSecs int64 `json:"maxPollingSecs,omitempty"`

	// RulesetIds: The identifiers of the rulesets supported for the
	// device's location. The database should include at least one
	// applicable ruleset in the initialization response. The device may use
	// the ruleset identifiers to determine parameters to include in
	// subsequent requests. Within the context of the available-spectrum
	// responses, the database should include the identifier of the ruleset
	// that it used to determine the available-spectrum response. If
	// included, the device must use the specified ruleset to interpret the
	// response. If the device does not support the indicated ruleset, it
	// must not operate in the spectrum governed by the ruleset.
	RulesetIds []string `json:"rulesetIds,omitempty"`
}

type SpectrumMessage struct {
	// Bandwidth: The bandwidth (in Hertz) for which permissible power
	// levels are specified. For example, FCC regulation would require only
	// one spectrum specification at 6MHz bandwidth, but Ofcom regulation
	// would require two specifications, at 0.1MHz and 8MHz. This parameter
	// may be empty if there is no available spectrum. It will be present
	// otherwise.
	Bandwidth float64 `json:"bandwidth,omitempty"`

	// FrequencyRanges: The list of frequency ranges and permissible power
	// levels. The list may be empty if there is no available spectrum,
	// otherwise it will be present.
	FrequencyRanges []*FrequencyRange `json:"frequencyRanges,omitempty"`
}

type SpectrumSchedule struct {
	// EventTime: The event time expresses when the spectrum profile is
	// valid. It will always be present.
	EventTime *EventTime `json:"eventTime,omitempty"`

	// Spectra: A list of spectrum messages representing the usable profile.
	// It will always be present, but may be empty when there is no
	// available spectrum.
	Spectra []*SpectrumMessage `json:"spectra,omitempty"`
}

type Vcard struct {
	// Adr: The street address of the entity.
	Adr *VcardAddress `json:"adr,omitempty"`

	// Email: An email address that can be used to reach the contact.
	Email *VcardTypedText `json:"email,omitempty"`

	// Fn: The full name of the contact person. For example: John A. Smith.
	Fn string `json:"fn,omitempty"`

	// Org: The organization associated with the registering entity.
	Org *VcardTypedText `json:"org,omitempty"`

	// Tel: A telephone number that can be used to call the contact.
	Tel *VcardTelephone `json:"tel,omitempty"`
}

type VcardAddress struct {
	// Code: The postal code associated with the address. For example:
	// 94423.
	Code string `json:"code,omitempty"`

	// Country: The country name. For example: US.
	Country string `json:"country,omitempty"`

	// Locality: The city or local equivalent portion of the address. For
	// example: San Jose.
	Locality string `json:"locality,omitempty"`

	// Pobox: An optional post office box number.
	Pobox string `json:"pobox,omitempty"`

	// Region: The state or local equivalent portion of the address. For
	// example: CA.
	Region string `json:"region,omitempty"`

	// Street: The street number and name. For example: 123 Any St.
	Street string `json:"street,omitempty"`
}

type VcardTelephone struct {
	// Uri: A nested telephone URI of the form: tel:+1-123-456-7890.
	Uri string `json:"uri,omitempty"`
}

type VcardTypedText struct {
	// Text: The text string associated with this item. For example, for an
	// org field: ACME, inc. For an email field: smith@example.com.
	Text string `json:"text,omitempty"`
}

// method id "spectrum.paws.getSpectrum":

type PawsGetSpectrumCall struct {
	s                      *Service
	pawsgetspectrumrequest *PawsGetSpectrumRequest
	opt_                   map[string]interface{}
}

// GetSpectrum: Requests information about the available spectrum for a
// device at a location. Requests from a fixed-mode device must include
// owner information so the device can be registered with the database.
func (r *PawsService) GetSpectrum(pawsgetspectrumrequest *PawsGetSpectrumRequest) *PawsGetSpectrumCall {
	c := &PawsGetSpectrumCall{s: r.s, opt_: make(map[string]interface{})}
	c.pawsgetspectrumrequest = pawsgetspectrumrequest
	return c
}

func (c *PawsGetSpectrumCall) Do() (*PawsGetSpectrumResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pawsgetspectrumrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "getSpectrum")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PawsGetSpectrumResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Requests information about the available spectrum for a device at a location. Requests from a fixed-mode device must include owner information so the device can be registered with the database.",
	//   "httpMethod": "POST",
	//   "id": "spectrum.paws.getSpectrum",
	//   "path": "getSpectrum",
	//   "request": {
	//     "$ref": "PawsGetSpectrumRequest"
	//   },
	//   "response": {
	//     "$ref": "PawsGetSpectrumResponse"
	//   }
	// }

}

// method id "spectrum.paws.getSpectrumBatch":

type PawsGetSpectrumBatchCall struct {
	s                           *Service
	pawsgetspectrumbatchrequest *PawsGetSpectrumBatchRequest
	opt_                        map[string]interface{}
}

// GetSpectrumBatch: The Google Spectrum Database does not support batch
// requests, so this method always yields an UNIMPLEMENTED error.
func (r *PawsService) GetSpectrumBatch(pawsgetspectrumbatchrequest *PawsGetSpectrumBatchRequest) *PawsGetSpectrumBatchCall {
	c := &PawsGetSpectrumBatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.pawsgetspectrumbatchrequest = pawsgetspectrumbatchrequest
	return c
}

func (c *PawsGetSpectrumBatchCall) Do() (*PawsGetSpectrumBatchResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pawsgetspectrumbatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "getSpectrumBatch")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PawsGetSpectrumBatchResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "The Google Spectrum Database does not support batch requests, so this method always yields an UNIMPLEMENTED error.",
	//   "httpMethod": "POST",
	//   "id": "spectrum.paws.getSpectrumBatch",
	//   "path": "getSpectrumBatch",
	//   "request": {
	//     "$ref": "PawsGetSpectrumBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "PawsGetSpectrumBatchResponse"
	//   }
	// }

}

// method id "spectrum.paws.init":

type PawsInitCall struct {
	s               *Service
	pawsinitrequest *PawsInitRequest
	opt_            map[string]interface{}
}

// Init: Initializes the connection between a white space device and the
// database.
func (r *PawsService) Init(pawsinitrequest *PawsInitRequest) *PawsInitCall {
	c := &PawsInitCall{s: r.s, opt_: make(map[string]interface{})}
	c.pawsinitrequest = pawsinitrequest
	return c
}

func (c *PawsInitCall) Do() (*PawsInitResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pawsinitrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "init")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PawsInitResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Initializes the connection between a white space device and the database.",
	//   "httpMethod": "POST",
	//   "id": "spectrum.paws.init",
	//   "path": "init",
	//   "request": {
	//     "$ref": "PawsInitRequest"
	//   },
	//   "response": {
	//     "$ref": "PawsInitResponse"
	//   }
	// }

}

// method id "spectrum.paws.notifySpectrumUse":

type PawsNotifySpectrumUseCall struct {
	s                            *Service
	pawsnotifyspectrumuserequest *PawsNotifySpectrumUseRequest
	opt_                         map[string]interface{}
}

// NotifySpectrumUse: Notifies the database that the device has selected
// certain frequency ranges for transmission. Only to be invoked when
// required by the regulator. The Google Spectrum Database does not
// operate in domains that require notification, so this always yields
// an UNIMPLEMENTED error.
func (r *PawsService) NotifySpectrumUse(pawsnotifyspectrumuserequest *PawsNotifySpectrumUseRequest) *PawsNotifySpectrumUseCall {
	c := &PawsNotifySpectrumUseCall{s: r.s, opt_: make(map[string]interface{})}
	c.pawsnotifyspectrumuserequest = pawsnotifyspectrumuserequest
	return c
}

func (c *PawsNotifySpectrumUseCall) Do() (*PawsNotifySpectrumUseResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pawsnotifyspectrumuserequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "notifySpectrumUse")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PawsNotifySpectrumUseResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Notifies the database that the device has selected certain frequency ranges for transmission. Only to be invoked when required by the regulator. The Google Spectrum Database does not operate in domains that require notification, so this always yields an UNIMPLEMENTED error.",
	//   "httpMethod": "POST",
	//   "id": "spectrum.paws.notifySpectrumUse",
	//   "path": "notifySpectrumUse",
	//   "request": {
	//     "$ref": "PawsNotifySpectrumUseRequest"
	//   },
	//   "response": {
	//     "$ref": "PawsNotifySpectrumUseResponse"
	//   }
	// }

}

// method id "spectrum.paws.register":

type PawsRegisterCall struct {
	s                   *Service
	pawsregisterrequest *PawsRegisterRequest
	opt_                map[string]interface{}
}

// Register: The Google Spectrum Database implements registration in the
// getSpectrum method. As such this always returns an UNIMPLEMENTED
// error.
func (r *PawsService) Register(pawsregisterrequest *PawsRegisterRequest) *PawsRegisterCall {
	c := &PawsRegisterCall{s: r.s, opt_: make(map[string]interface{})}
	c.pawsregisterrequest = pawsregisterrequest
	return c
}

func (c *PawsRegisterCall) Do() (*PawsRegisterResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pawsregisterrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "register")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PawsRegisterResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "The Google Spectrum Database implements registration in the getSpectrum method. As such this always returns an UNIMPLEMENTED error.",
	//   "httpMethod": "POST",
	//   "id": "spectrum.paws.register",
	//   "path": "register",
	//   "request": {
	//     "$ref": "PawsRegisterRequest"
	//   },
	//   "response": {
	//     "$ref": "PawsRegisterResponse"
	//   }
	// }

}

// method id "spectrum.paws.verifyDevice":

type PawsVerifyDeviceCall struct {
	s                       *Service
	pawsverifydevicerequest *PawsVerifyDeviceRequest
	opt_                    map[string]interface{}
}

// VerifyDevice: Validates a device for white space use in accordance
// with regulatory rules. The Google Spectrum Database does not support
// master/slave configurations, so this always yields an UNIMPLEMENTED
// error.
func (r *PawsService) VerifyDevice(pawsverifydevicerequest *PawsVerifyDeviceRequest) *PawsVerifyDeviceCall {
	c := &PawsVerifyDeviceCall{s: r.s, opt_: make(map[string]interface{})}
	c.pawsverifydevicerequest = pawsverifydevicerequest
	return c
}

func (c *PawsVerifyDeviceCall) Do() (*PawsVerifyDeviceResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pawsverifydevicerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "verifyDevice")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PawsVerifyDeviceResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Validates a device for white space use in accordance with regulatory rules. The Google Spectrum Database does not support master/slave configurations, so this always yields an UNIMPLEMENTED error.",
	//   "httpMethod": "POST",
	//   "id": "spectrum.paws.verifyDevice",
	//   "path": "verifyDevice",
	//   "request": {
	//     "$ref": "PawsVerifyDeviceRequest"
	//   },
	//   "response": {
	//     "$ref": "PawsVerifyDeviceResponse"
	//   }
	// }

}
