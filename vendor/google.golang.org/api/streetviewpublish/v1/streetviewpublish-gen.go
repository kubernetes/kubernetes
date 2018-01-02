// Package streetviewpublish provides access to the Street View Publish API.
//
// See https://developers.google.com/streetview/publish/
//
// Usage example:
//
//   import "google.golang.org/api/streetviewpublish/v1"
//   ...
//   streetviewpublishService, err := streetviewpublish.New(oauthHttpClient)
package streetviewpublish // import "google.golang.org/api/streetviewpublish/v1"

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

const apiId = "streetviewpublish:v1"
const apiName = "streetviewpublish"
const apiVersion = "v1"
const basePath = "https://streetviewpublish.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// Publish and manage your 360 photos on Google Street View
	StreetviewpublishScope = "https://www.googleapis.com/auth/streetviewpublish"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Photo = NewPhotoService(s)
	s.Photos = NewPhotosService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Photo *PhotoService

	Photos *PhotosService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewPhotoService(s *Service) *PhotoService {
	rs := &PhotoService{s: s}
	return rs
}

type PhotoService struct {
	s *Service
}

func NewPhotosService(s *Service) *PhotosService {
	rs := &PhotosService{s: s}
	return rs
}

type PhotosService struct {
	s *Service
}

// BatchDeletePhotosRequest: Request to delete multiple Photos.
type BatchDeletePhotosRequest struct {
	// PhotoIds: Required. IDs of the Photos. For HTTP
	// GET requests, the URL query parameter should
	// be
	// `photoIds=<id1>&photoIds=<id2>&...`.
	PhotoIds []string `json:"photoIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PhotoIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PhotoIds") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchDeletePhotosRequest) MarshalJSON() ([]byte, error) {
	type noMethod BatchDeletePhotosRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchDeletePhotosResponse: Response to batch delete of one or
// more
// Photos.
type BatchDeletePhotosResponse struct {
	// Status: The status for the operation to delete a single
	// Photo in the batch request.
	Status []*Status `json:"status,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Status") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Status") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchDeletePhotosResponse) MarshalJSON() ([]byte, error) {
	type noMethod BatchDeletePhotosResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchGetPhotosResponse: Response to batch get of Photos.
type BatchGetPhotosResponse struct {
	// Results: List of results for each individual
	// Photo requested, in the same order as
	// the requests in
	// BatchGetPhotos.
	Results []*PhotoResponse `json:"results,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Results") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Results") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchGetPhotosResponse) MarshalJSON() ([]byte, error) {
	type noMethod BatchGetPhotosResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchUpdatePhotosRequest: Request to update the metadata of
// photos.
// Updating the pixels of photos is not supported.
type BatchUpdatePhotosRequest struct {
	// UpdatePhotoRequests: Required. List of
	// UpdatePhotoRequests.
	UpdatePhotoRequests []*UpdatePhotoRequest `json:"updatePhotoRequests,omitempty"`

	// ForceSendFields is a list of field names (e.g. "UpdatePhotoRequests")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "UpdatePhotoRequests") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BatchUpdatePhotosRequest) MarshalJSON() ([]byte, error) {
	type noMethod BatchUpdatePhotosRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchUpdatePhotosResponse: Response to batch update of metadata of
// one or more
// Photos.
type BatchUpdatePhotosResponse struct {
	// Results: List of results for each individual
	// Photo updated, in the same order as
	// the request.
	Results []*PhotoResponse `json:"results,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Results") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Results") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchUpdatePhotosResponse) MarshalJSON() ([]byte, error) {
	type noMethod BatchUpdatePhotosResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Connection: A connection is the link from a source photo to a
// destination photo.
type Connection struct {
	// Target: Required. The destination of the connection from the
	// containing photo to
	// another photo.
	Target *PhotoId `json:"target,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Target") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Target") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Connection) MarshalJSON() ([]byte, error) {
	type noMethod Connection
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

// LatLng: An object representing a latitude/longitude pair. This is
// expressed as a pair
// of doubles representing degrees latitude and degrees longitude.
// Unless
// specified otherwise, this must conform to the
// <a
// href="http://www.unoosa.org/pdf/icg/2012/template/WGS_84.pdf">WGS84
// st
// andard</a>. Values must be within normalized ranges.
//
// Example of normalization code in Python:
//
//     def NormalizeLongitude(longitude):
//       """Wraps decimal degrees longitude to [-180.0, 180.0]."""
//       q, r = divmod(longitude, 360.0)
//       if r > 180.0 or (r == 180.0 and q <= -1.0):
//         return r - 360.0
//       return r
//
//     def NormalizeLatLng(latitude, longitude):
//       """Wraps decimal degrees latitude and longitude to
//       [-90.0, 90.0] and [-180.0, 180.0], respectively."""
//       r = latitude % 360.0
//       if r <= 90.0:
//         return r, NormalizeLongitude(longitude)
//       elif r >= 270.0:
//         return r - 360, NormalizeLongitude(longitude)
//       else:
//         return 180 - r, NormalizeLongitude(longitude + 180.0)
//
//     assert 180.0 == NormalizeLongitude(180.0)
//     assert -180.0 == NormalizeLongitude(-180.0)
//     assert -179.0 == NormalizeLongitude(181.0)
//     assert (0.0, 0.0) == NormalizeLatLng(360.0, 0.0)
//     assert (0.0, 0.0) == NormalizeLatLng(-360.0, 0.0)
//     assert (85.0, 180.0) == NormalizeLatLng(95.0, 0.0)
//     assert (-85.0, -170.0) == NormalizeLatLng(-95.0, 10.0)
//     assert (90.0, 10.0) == NormalizeLatLng(90.0, 10.0)
//     assert (-90.0, -10.0) == NormalizeLatLng(-90.0, -10.0)
//     assert (0.0, -170.0) == NormalizeLatLng(-180.0, 10.0)
//     assert (0.0, -170.0) == NormalizeLatLng(180.0, 10.0)
//     assert (-90.0, 10.0) == NormalizeLatLng(270.0, 10.0)
//     assert (90.0, 10.0) == NormalizeLatLng(-270.0, 10.0)
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

	// NullFields is a list of field names (e.g. "Latitude") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LatLng) MarshalJSON() ([]byte, error) {
	type noMethod LatLng
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *LatLng) UnmarshalJSON(data []byte) error {
	type noMethod LatLng
	var s1 struct {
		Latitude  gensupport.JSONFloat64 `json:"latitude"`
		Longitude gensupport.JSONFloat64 `json:"longitude"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Latitude = float64(s1.Latitude)
	s.Longitude = float64(s1.Longitude)
	return nil
}

// Level: Level information containing level number and its
// corresponding name.
type Level struct {
	// Name: Required. A name assigned to this Level, restricted to 3
	// characters.
	// Consider how the elevator buttons would be labeled for this level if
	// there
	// was an elevator.
	Name string `json:"name,omitempty"`

	// Number: Floor number, used for ordering. 0 indicates the ground
	// level, 1 indicates
	// the first level above ground level, -1 indicates the first level
	// under
	// ground level. Non-integer values are OK.
	Number float64 `json:"number,omitempty"`

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

func (s *Level) MarshalJSON() ([]byte, error) {
	type noMethod Level
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *Level) UnmarshalJSON(data []byte) error {
	type noMethod Level
	var s1 struct {
		Number gensupport.JSONFloat64 `json:"number"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Number = float64(s1.Number)
	return nil
}

// ListPhotosResponse: Response to list all photos that belong to a
// user.
type ListPhotosResponse struct {
	// NextPageToken: Token to retrieve the next page of results, or empty
	// if there are no more
	// results in the list.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Photos: List of photos. The maximum number of items returned is based
	// on the
	// pageSize field
	// in the request.
	Photos []*Photo `json:"photos,omitempty"`

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

func (s *ListPhotosResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListPhotosResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Photo: Photo is used to store 360 photos along with photo metadata.
type Photo struct {
	// CaptureTime: Absolute time when the photo was captured.
	// When the photo has no exif timestamp, this is used to set a timestamp
	// in
	// the photo metadata.
	CaptureTime string `json:"captureTime,omitempty"`

	// Connections: Connections to other photos. A connection represents the
	// link from this
	// photo to another photo.
	Connections []*Connection `json:"connections,omitempty"`

	// DownloadUrl: Output only. The download URL for the photo bytes. This
	// field is set only
	// when
	// GetPhotoRequest.view
	// is set to
	// PhotoView.INCLUDE_DOWNLOAD_URL.
	DownloadUrl string `json:"downloadUrl,omitempty"`

	// PhotoId: Required when updating photo. Output only when creating
	// photo.
	// Identifier for the photo, which is unique among all photos in
	// Google.
	PhotoId *PhotoId `json:"photoId,omitempty"`

	// Places: Places where this photo belongs.
	Places []*Place `json:"places,omitempty"`

	// Pose: Pose of the photo.
	Pose *Pose `json:"pose,omitempty"`

	// ShareLink: Output only. The share link for the photo.
	ShareLink string `json:"shareLink,omitempty"`

	// ThumbnailUrl: Output only. The thumbnail URL for showing a preview of
	// the given photo.
	ThumbnailUrl string `json:"thumbnailUrl,omitempty"`

	// UploadReference: Required when creating photo. Input only. The
	// resource URL where the photo
	// bytes are uploaded to.
	UploadReference *UploadRef `json:"uploadReference,omitempty"`

	// ViewCount: Output only. View count of the photo.
	ViewCount int64 `json:"viewCount,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CaptureTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CaptureTime") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Photo) MarshalJSON() ([]byte, error) {
	type noMethod Photo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PhotoId: Identifier for a Photo.
type PhotoId struct {
	// Id: Required. A unique identifier for a photo.
	Id string `json:"id,omitempty"`

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

func (s *PhotoId) MarshalJSON() ([]byte, error) {
	type noMethod PhotoId
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PhotoResponse: Response payload for a single
// Photo
// in batch operations including
// BatchGetPhotos
// and
// BatchUpdatePhotos.
type PhotoResponse struct {
	// Photo: The Photo resource, if the request
	// was successful.
	Photo *Photo `json:"photo,omitempty"`

	// Status: The status for the operation to get or update a single photo
	// in the batch
	// request.
	Status *Status `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Photo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Photo") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PhotoResponse) MarshalJSON() ([]byte, error) {
	type noMethod PhotoResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Place: Place metadata for an entity.
type Place struct {
	// PlaceId: Required. Place identifier, as described
	// in
	// https://developers.google.com/places/place-id.
	PlaceId string `json:"placeId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PlaceId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PlaceId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Place) MarshalJSON() ([]byte, error) {
	type noMethod Place
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Pose: Raw pose measurement for an entity.
type Pose struct {
	// Altitude: Altitude of the pose in meters above ground level (as
	// defined by WGS84).
	// NaN indicates an unmeasured quantity.
	Altitude float64 `json:"altitude,omitempty"`

	// Heading: Compass heading, measured at the center of the photo in
	// degrees clockwise
	// from North. Value must be >=0 and <360.
	// NaN indicates an unmeasured quantity.
	Heading float64 `json:"heading,omitempty"`

	// LatLngPair: Latitude and longitude pair of the pose, as explained
	// here:
	// https://cloud.google.com/datastore/docs/reference/rest/Shared.Ty
	// pes/LatLng
	// When creating a Photo, if the
	// latitude and longitude pair are not provided here, the geolocation
	// from the
	// exif header will be used. If the latitude and longitude pair is
	// not
	// provided and cannot be found in the exif header, the create photo
	// process
	// will fail.
	LatLngPair *LatLng `json:"latLngPair,omitempty"`

	// Level: Level (the floor in a building) used to configure vertical
	// navigation.
	Level *Level `json:"level,omitempty"`

	// Pitch: Pitch, measured at the center of the photo in degrees. Value
	// must be >=-90
	// and <= 90. A value of -90 means looking directly down, and a value of
	// 90
	// means looking directly up.
	// NaN indicates an unmeasured quantity.
	Pitch float64 `json:"pitch,omitempty"`

	// Roll: Roll, measured in degrees. Value must be >= 0 and <360. A value
	// of 0
	// means level with the horizon.
	// NaN indicates an unmeasured quantity.
	Roll float64 `json:"roll,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Altitude") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Altitude") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Pose) MarshalJSON() ([]byte, error) {
	type noMethod Pose
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *Pose) UnmarshalJSON(data []byte) error {
	type noMethod Pose
	var s1 struct {
		Altitude gensupport.JSONFloat64 `json:"altitude"`
		Heading  gensupport.JSONFloat64 `json:"heading"`
		Pitch    gensupport.JSONFloat64 `json:"pitch"`
		Roll     gensupport.JSONFloat64 `json:"roll"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Altitude = float64(s1.Altitude)
	s.Heading = float64(s1.Heading)
	s.Pitch = float64(s1.Pitch)
	s.Roll = float64(s1.Roll)
	return nil
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

// UpdatePhotoRequest: Request to update the metadata of a
// Photo. Updating the pixels of a photo
// is not supported.
type UpdatePhotoRequest struct {
	// Photo: Required. Photo object containing the
	// new metadata.
	Photo *Photo `json:"photo,omitempty"`

	// UpdateMask: Mask that identifies fields on the photo metadata to
	// update.
	// If not present, the old Photo metadata will be entirely replaced with
	// the
	// new Photo metadata in this request. The update fails if invalid
	// fields are
	// specified. Multiple fields can be specified in a comma-delimited
	// list.
	//
	// The following fields are valid:
	//
	// * `pose.heading`
	// * `pose.latLngPair`
	// * `pose.pitch`
	// * `pose.roll`
	// * `pose.level`
	// * `pose.altitude`
	// * `connections`
	// * `places`
	//
	//
	// <aside class="note"><b>Note:</b> Repeated fields in
	// updateMask
	// mean the entire set of repeated values will be replaced with the
	// new
	// contents. For example, if
	// updateMask
	// contains `connections` and `UpdatePhotoRequest.photo.connections` is
	// empty,
	// all connections will be removed.</aside>
	UpdateMask string `json:"updateMask,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Photo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Photo") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdatePhotoRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdatePhotoRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UploadRef: Upload reference for media files.
type UploadRef struct {
	// UploadUrl: Required. An upload reference should be unique for each
	// user. It follows
	// the
	// form:
	// "https://streetviewpublish.googleapis.com/media/user/{account_id
	// }/photo/{upload_reference}"
	UploadUrl string `json:"uploadUrl,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "UploadUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "UploadUrl") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UploadRef) MarshalJSON() ([]byte, error) {
	type noMethod UploadRef
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "streetviewpublish.photo.create":

type PhotoCreateCall struct {
	s          *Service
	photo      *Photo
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: After the client finishes uploading the photo with the
// returned
// UploadRef,
// CreatePhoto
// publishes the uploaded Photo to
// Street View on Google Maps.
//
// This method returns the following error codes:
//
// * google.rpc.Code.INVALID_ARGUMENT if the request is malformed.
// * google.rpc.Code.NOT_FOUND if the upload reference does not exist.
// * google.rpc.Code.RESOURCE_EXHAUSTED if the account has reached
// the
// storage limit.
func (r *PhotoService) Create(photo *Photo) *PhotoCreateCall {
	c := &PhotoCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.photo = photo
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PhotoCreateCall) Fields(s ...googleapi.Field) *PhotoCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PhotoCreateCall) Context(ctx context.Context) *PhotoCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PhotoCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PhotoCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.photo)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/photo")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "streetviewpublish.photo.create" call.
// Exactly one of *Photo or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Photo.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *PhotoCreateCall) Do(opts ...googleapi.CallOption) (*Photo, error) {
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
	ret := &Photo{
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
	//   "description": "After the client finishes uploading the photo with the returned\nUploadRef,\nCreatePhoto\npublishes the uploaded Photo to\nStreet View on Google Maps.\n\nThis method returns the following error codes:\n\n* google.rpc.Code.INVALID_ARGUMENT if the request is malformed.\n* google.rpc.Code.NOT_FOUND if the upload reference does not exist.\n* google.rpc.Code.RESOURCE_EXHAUSTED if the account has reached the\nstorage limit.",
	//   "flatPath": "v1/photo",
	//   "httpMethod": "POST",
	//   "id": "streetviewpublish.photo.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/photo",
	//   "request": {
	//     "$ref": "Photo"
	//   },
	//   "response": {
	//     "$ref": "Photo"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/streetviewpublish"
	//   ]
	// }

}

// method id "streetviewpublish.photo.delete":

type PhotoDeleteCall struct {
	s          *Service
	photoId    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a Photo and its metadata.
//
// This method returns the following error codes:
//
// * google.rpc.Code.PERMISSION_DENIED if the requesting user did
// not
// create the requested photo.
// * google.rpc.Code.NOT_FOUND if the photo ID does not exist.
func (r *PhotoService) Delete(photoId string) *PhotoDeleteCall {
	c := &PhotoDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.photoId = photoId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PhotoDeleteCall) Fields(s ...googleapi.Field) *PhotoDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PhotoDeleteCall) Context(ctx context.Context) *PhotoDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PhotoDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PhotoDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/photo/{photoId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"photoId": c.photoId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "streetviewpublish.photo.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *PhotoDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	//   "description": "Deletes a Photo and its metadata.\n\nThis method returns the following error codes:\n\n* google.rpc.Code.PERMISSION_DENIED if the requesting user did not\ncreate the requested photo.\n* google.rpc.Code.NOT_FOUND if the photo ID does not exist.",
	//   "flatPath": "v1/photo/{photoId}",
	//   "httpMethod": "DELETE",
	//   "id": "streetviewpublish.photo.delete",
	//   "parameterOrder": [
	//     "photoId"
	//   ],
	//   "parameters": {
	//     "photoId": {
	//       "description": "Required. ID of the Photo.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/photo/{photoId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/streetviewpublish"
	//   ]
	// }

}

// method id "streetviewpublish.photo.get":

type PhotoGetCall struct {
	s            *Service
	photoId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the metadata of the specified
// Photo.
//
// This method returns the following error codes:
//
// * google.rpc.Code.PERMISSION_DENIED if the requesting user did
// not
// create the requested Photo.
// * google.rpc.Code.NOT_FOUND if the requested
// Photo does not exist.
func (r *PhotoService) Get(photoId string) *PhotoGetCall {
	c := &PhotoGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.photoId = photoId
	return c
}

// View sets the optional parameter "view": Specifies if a download URL
// for the photo bytes should be returned in the
// Photo response.
//
// Possible values:
//   "BASIC"
//   "INCLUDE_DOWNLOAD_URL"
func (c *PhotoGetCall) View(view string) *PhotoGetCall {
	c.urlParams_.Set("view", view)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PhotoGetCall) Fields(s ...googleapi.Field) *PhotoGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PhotoGetCall) IfNoneMatch(entityTag string) *PhotoGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PhotoGetCall) Context(ctx context.Context) *PhotoGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PhotoGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PhotoGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/photo/{photoId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"photoId": c.photoId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "streetviewpublish.photo.get" call.
// Exactly one of *Photo or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Photo.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *PhotoGetCall) Do(opts ...googleapi.CallOption) (*Photo, error) {
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
	ret := &Photo{
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
	//   "description": "Gets the metadata of the specified\nPhoto.\n\nThis method returns the following error codes:\n\n* google.rpc.Code.PERMISSION_DENIED if the requesting user did not\ncreate the requested Photo.\n* google.rpc.Code.NOT_FOUND if the requested\nPhoto does not exist.",
	//   "flatPath": "v1/photo/{photoId}",
	//   "httpMethod": "GET",
	//   "id": "streetviewpublish.photo.get",
	//   "parameterOrder": [
	//     "photoId"
	//   ],
	//   "parameters": {
	//     "photoId": {
	//       "description": "Required. ID of the Photo.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Specifies if a download URL for the photo bytes should be returned in the\nPhoto response.",
	//       "enum": [
	//         "BASIC",
	//         "INCLUDE_DOWNLOAD_URL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/photo/{photoId}",
	//   "response": {
	//     "$ref": "Photo"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/streetviewpublish"
	//   ]
	// }

}

// method id "streetviewpublish.photo.startUpload":

type PhotoStartUploadCall struct {
	s          *Service
	empty      *Empty
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// StartUpload: Creates an upload session to start uploading photo
// bytes. The upload URL of
// the returned UploadRef is used to
// upload the bytes for the Photo.
//
// In addition to the photo requirements shown
// in
// https://support.google.com/maps/answer/7012050?hl=en&ref_topic=6275
// 604,
// the photo must also meet the following requirements:
//
// * Photo Sphere XMP metadata must be included in the photo medadata.
// See
// https://developers.google.com/streetview/spherical-metadata for
// the
// required fields.
// * The pixel size of the photo must meet the size requirements listed
// in
// https://support.google.com/maps/answer/7012050?hl=en&ref_topic=6275
// 604, and
// the photo must be a full 360 horizontally.
//
// After the upload is complete, the
// UploadRef is used with
// CreatePhoto
// to create the Photo object entry.
func (r *PhotoService) StartUpload(empty *Empty) *PhotoStartUploadCall {
	c := &PhotoStartUploadCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.empty = empty
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PhotoStartUploadCall) Fields(s ...googleapi.Field) *PhotoStartUploadCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PhotoStartUploadCall) Context(ctx context.Context) *PhotoStartUploadCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PhotoStartUploadCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PhotoStartUploadCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.empty)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/photo:startUpload")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "streetviewpublish.photo.startUpload" call.
// Exactly one of *UploadRef or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *UploadRef.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PhotoStartUploadCall) Do(opts ...googleapi.CallOption) (*UploadRef, error) {
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
	ret := &UploadRef{
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
	//   "description": "Creates an upload session to start uploading photo bytes. The upload URL of\nthe returned UploadRef is used to\nupload the bytes for the Photo.\n\nIn addition to the photo requirements shown in\nhttps://support.google.com/maps/answer/7012050?hl=en\u0026ref_topic=6275604,\nthe photo must also meet the following requirements:\n\n* Photo Sphere XMP metadata must be included in the photo medadata. See\nhttps://developers.google.com/streetview/spherical-metadata for the\nrequired fields.\n* The pixel size of the photo must meet the size requirements listed in\nhttps://support.google.com/maps/answer/7012050?hl=en\u0026ref_topic=6275604, and\nthe photo must be a full 360 horizontally.\n\nAfter the upload is complete, the\nUploadRef is used with\nCreatePhoto\nto create the Photo object entry.",
	//   "flatPath": "v1/photo:startUpload",
	//   "httpMethod": "POST",
	//   "id": "streetviewpublish.photo.startUpload",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/photo:startUpload",
	//   "request": {
	//     "$ref": "Empty"
	//   },
	//   "response": {
	//     "$ref": "UploadRef"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/streetviewpublish"
	//   ]
	// }

}

// method id "streetviewpublish.photo.update":

type PhotoUpdateCall struct {
	s          *Service
	id         string
	photo      *Photo
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates the metadata of a Photo, such
// as pose, place association, connections, etc. Changing the pixels of
// a
// photo is not supported.
//
// Only the fields specified in
// updateMask
// field are used. If `updateMask` is not present, the update applies to
// all
// fields.
//
// <aside class="note"><b>Note:</b> To
// update
// Pose.altitude,
// Pose.latLngPair has to be
// filled as well. Otherwise, the request will fail.</aside>
//
// This method returns the following error codes:
//
// * google.rpc.Code.PERMISSION_DENIED if the requesting user did
// not
// create the requested photo.
// * google.rpc.Code.INVALID_ARGUMENT if the request is malformed.
// * google.rpc.Code.NOT_FOUND if the requested photo does not exist.
func (r *PhotoService) Update(id string, photo *Photo) *PhotoUpdateCall {
	c := &PhotoUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.id = id
	c.photo = photo
	return c
}

// UpdateMask sets the optional parameter "updateMask": Mask that
// identifies fields on the photo metadata to update.
// If not present, the old Photo metadata will be entirely replaced with
// the
// new Photo metadata in this request. The update fails if invalid
// fields are
// specified. Multiple fields can be specified in a comma-delimited
// list.
//
// The following fields are valid:
//
// * `pose.heading`
// * `pose.latLngPair`
// * `pose.pitch`
// * `pose.roll`
// * `pose.level`
// * `pose.altitude`
// * `connections`
// * `places`
//
//
// <aside class="note"><b>Note:</b> Repeated fields in
// updateMask
// mean the entire set of repeated values will be replaced with the
// new
// contents. For example, if
// updateMask
// contains `connections` and `UpdatePhotoRequest.photo.connections` is
// empty,
// all connections will be removed.</aside>
func (c *PhotoUpdateCall) UpdateMask(updateMask string) *PhotoUpdateCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PhotoUpdateCall) Fields(s ...googleapi.Field) *PhotoUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PhotoUpdateCall) Context(ctx context.Context) *PhotoUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PhotoUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PhotoUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.photo)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/photo/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"id": c.id,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "streetviewpublish.photo.update" call.
// Exactly one of *Photo or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Photo.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *PhotoUpdateCall) Do(opts ...googleapi.CallOption) (*Photo, error) {
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
	ret := &Photo{
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
	//   "description": "Updates the metadata of a Photo, such\nas pose, place association, connections, etc. Changing the pixels of a\nphoto is not supported.\n\nOnly the fields specified in\nupdateMask\nfield are used. If `updateMask` is not present, the update applies to all\nfields.\n\n\u003caside class=\"note\"\u003e\u003cb\u003eNote:\u003c/b\u003e To update\nPose.altitude,\nPose.latLngPair has to be\nfilled as well. Otherwise, the request will fail.\u003c/aside\u003e\n\nThis method returns the following error codes:\n\n* google.rpc.Code.PERMISSION_DENIED if the requesting user did not\ncreate the requested photo.\n* google.rpc.Code.INVALID_ARGUMENT if the request is malformed.\n* google.rpc.Code.NOT_FOUND if the requested photo does not exist.",
	//   "flatPath": "v1/photo/{id}",
	//   "httpMethod": "PUT",
	//   "id": "streetviewpublish.photo.update",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "Required. A unique identifier for a photo.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "Mask that identifies fields on the photo metadata to update.\nIf not present, the old Photo metadata will be entirely replaced with the\nnew Photo metadata in this request. The update fails if invalid fields are\nspecified. Multiple fields can be specified in a comma-delimited list.\n\nThe following fields are valid:\n\n* `pose.heading`\n* `pose.latLngPair`\n* `pose.pitch`\n* `pose.roll`\n* `pose.level`\n* `pose.altitude`\n* `connections`\n* `places`\n\n\n\u003caside class=\"note\"\u003e\u003cb\u003eNote:\u003c/b\u003e Repeated fields in\nupdateMask\nmean the entire set of repeated values will be replaced with the new\ncontents. For example, if\nupdateMask\ncontains `connections` and `UpdatePhotoRequest.photo.connections` is empty,\nall connections will be removed.\u003c/aside\u003e",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/photo/{id}",
	//   "request": {
	//     "$ref": "Photo"
	//   },
	//   "response": {
	//     "$ref": "Photo"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/streetviewpublish"
	//   ]
	// }

}

// method id "streetviewpublish.photos.batchDelete":

type PhotosBatchDeleteCall struct {
	s                        *Service
	batchdeletephotosrequest *BatchDeletePhotosRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// BatchDelete: Deletes a list of Photos and their
// metadata.
//
// Note that if
// BatchDeletePhotos
// fails, either critical fields are missing or there was an
// authentication
// error. Even if
// BatchDeletePhotos
// succeeds, there may have been failures for single photos in the
// batch.
// These failures will be specified in
// each
// PhotoResponse.status
// in
// BatchDeletePhotosResponse.results.
// See
// De
// letePhoto
// for specific failures that can occur per photo.
func (r *PhotosService) BatchDelete(batchdeletephotosrequest *BatchDeletePhotosRequest) *PhotosBatchDeleteCall {
	c := &PhotosBatchDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.batchdeletephotosrequest = batchdeletephotosrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PhotosBatchDeleteCall) Fields(s ...googleapi.Field) *PhotosBatchDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PhotosBatchDeleteCall) Context(ctx context.Context) *PhotosBatchDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PhotosBatchDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PhotosBatchDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.batchdeletephotosrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/photos:batchDelete")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "streetviewpublish.photos.batchDelete" call.
// Exactly one of *BatchDeletePhotosResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *BatchDeletePhotosResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PhotosBatchDeleteCall) Do(opts ...googleapi.CallOption) (*BatchDeletePhotosResponse, error) {
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
	ret := &BatchDeletePhotosResponse{
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
	//   "description": "Deletes a list of Photos and their\nmetadata.\n\nNote that if\nBatchDeletePhotos\nfails, either critical fields are missing or there was an authentication\nerror. Even if\nBatchDeletePhotos\nsucceeds, there may have been failures for single photos in the batch.\nThese failures will be specified in each\nPhotoResponse.status\nin\nBatchDeletePhotosResponse.results.\nSee\nDeletePhoto\nfor specific failures that can occur per photo.",
	//   "flatPath": "v1/photos:batchDelete",
	//   "httpMethod": "POST",
	//   "id": "streetviewpublish.photos.batchDelete",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/photos:batchDelete",
	//   "request": {
	//     "$ref": "BatchDeletePhotosRequest"
	//   },
	//   "response": {
	//     "$ref": "BatchDeletePhotosResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/streetviewpublish"
	//   ]
	// }

}

// method id "streetviewpublish.photos.batchGet":

type PhotosBatchGetCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// BatchGet: Gets the metadata of the specified
// Photo batch.
//
// Note that if
// BatchGetPhotos
// fails, either critical fields are missing or there was an
// authentication
// error. Even if
// BatchGetPhotos
// succeeds, there may have been failures for single photos in the
// batch.
// These failures will be specified in
// each
// PhotoResponse.status
// in
// BatchGetPhotosResponse.results.
// See
// GetPh
// oto
// for specific failures that can occur per photo.
func (r *PhotosService) BatchGet() *PhotosBatchGetCall {
	c := &PhotosBatchGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PhotoIds sets the optional parameter "photoIds": Required. IDs of the
// Photos. For HTTP
// GET requests, the URL query parameter should
// be
// `photoIds=<id1>&photoIds=<id2>&...`.
func (c *PhotosBatchGetCall) PhotoIds(photoIds ...string) *PhotosBatchGetCall {
	c.urlParams_.SetMulti("photoIds", append([]string{}, photoIds...))
	return c
}

// View sets the optional parameter "view": Specifies if a download URL
// for the photo bytes should be returned in the
// Photo response.
//
// Possible values:
//   "BASIC"
//   "INCLUDE_DOWNLOAD_URL"
func (c *PhotosBatchGetCall) View(view string) *PhotosBatchGetCall {
	c.urlParams_.Set("view", view)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PhotosBatchGetCall) Fields(s ...googleapi.Field) *PhotosBatchGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PhotosBatchGetCall) IfNoneMatch(entityTag string) *PhotosBatchGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PhotosBatchGetCall) Context(ctx context.Context) *PhotosBatchGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PhotosBatchGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PhotosBatchGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/photos:batchGet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "streetviewpublish.photos.batchGet" call.
// Exactly one of *BatchGetPhotosResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *BatchGetPhotosResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PhotosBatchGetCall) Do(opts ...googleapi.CallOption) (*BatchGetPhotosResponse, error) {
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
	ret := &BatchGetPhotosResponse{
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
	//   "description": "Gets the metadata of the specified\nPhoto batch.\n\nNote that if\nBatchGetPhotos\nfails, either critical fields are missing or there was an authentication\nerror. Even if\nBatchGetPhotos\nsucceeds, there may have been failures for single photos in the batch.\nThese failures will be specified in each\nPhotoResponse.status\nin\nBatchGetPhotosResponse.results.\nSee\nGetPhoto\nfor specific failures that can occur per photo.",
	//   "flatPath": "v1/photos:batchGet",
	//   "httpMethod": "GET",
	//   "id": "streetviewpublish.photos.batchGet",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "photoIds": {
	//       "description": "Required. IDs of the Photos. For HTTP\nGET requests, the URL query parameter should be\n`photoIds=\u003cid1\u003e\u0026photoIds=\u003cid2\u003e\u0026...`.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Specifies if a download URL for the photo bytes should be returned in the\nPhoto response.",
	//       "enum": [
	//         "BASIC",
	//         "INCLUDE_DOWNLOAD_URL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/photos:batchGet",
	//   "response": {
	//     "$ref": "BatchGetPhotosResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/streetviewpublish"
	//   ]
	// }

}

// method id "streetviewpublish.photos.batchUpdate":

type PhotosBatchUpdateCall struct {
	s                        *Service
	batchupdatephotosrequest *BatchUpdatePhotosRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// BatchUpdate: Updates the metadata of Photos, such
// as pose, place association, connections, etc. Changing the pixels of
// photos
// is not supported.
//
// Note that if
// BatchUpdatePhotos
// fails, either critical fields are missing or there was an
// authentication
// error. Even if
// BatchUpdatePhotos
// succeeds, there may have been failures for single photos in the
// batch.
// These failures will be specified in
// each
// PhotoResponse.status
// in
// BatchUpdatePhotosResponse.results.
// See
// Up
// datePhoto
// for specific failures that can occur per photo.
//
// Only the fields specified in
// updateMask
// field are used. If `updateMask` is not present, the update applies to
// all
// fields.
//
// <aside class="note"><b>Note:</b> To
// update
// Pose.altitude,
// Pose.latLngPair has to be
// filled as well. Otherwise, the request will fail.</aside>
func (r *PhotosService) BatchUpdate(batchupdatephotosrequest *BatchUpdatePhotosRequest) *PhotosBatchUpdateCall {
	c := &PhotosBatchUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.batchupdatephotosrequest = batchupdatephotosrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PhotosBatchUpdateCall) Fields(s ...googleapi.Field) *PhotosBatchUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PhotosBatchUpdateCall) Context(ctx context.Context) *PhotosBatchUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PhotosBatchUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PhotosBatchUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.batchupdatephotosrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/photos:batchUpdate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "streetviewpublish.photos.batchUpdate" call.
// Exactly one of *BatchUpdatePhotosResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *BatchUpdatePhotosResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PhotosBatchUpdateCall) Do(opts ...googleapi.CallOption) (*BatchUpdatePhotosResponse, error) {
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
	ret := &BatchUpdatePhotosResponse{
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
	//   "description": "Updates the metadata of Photos, such\nas pose, place association, connections, etc. Changing the pixels of photos\nis not supported.\n\nNote that if\nBatchUpdatePhotos\nfails, either critical fields are missing or there was an authentication\nerror. Even if\nBatchUpdatePhotos\nsucceeds, there may have been failures for single photos in the batch.\nThese failures will be specified in each\nPhotoResponse.status\nin\nBatchUpdatePhotosResponse.results.\nSee\nUpdatePhoto\nfor specific failures that can occur per photo.\n\nOnly the fields specified in\nupdateMask\nfield are used. If `updateMask` is not present, the update applies to all\nfields.\n\n\u003caside class=\"note\"\u003e\u003cb\u003eNote:\u003c/b\u003e To update\nPose.altitude,\nPose.latLngPair has to be\nfilled as well. Otherwise, the request will fail.\u003c/aside\u003e",
	//   "flatPath": "v1/photos:batchUpdate",
	//   "httpMethod": "POST",
	//   "id": "streetviewpublish.photos.batchUpdate",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1/photos:batchUpdate",
	//   "request": {
	//     "$ref": "BatchUpdatePhotosRequest"
	//   },
	//   "response": {
	//     "$ref": "BatchUpdatePhotosResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/streetviewpublish"
	//   ]
	// }

}

// method id "streetviewpublish.photos.list":

type PhotosListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all the Photos that belong to
// the user.
func (r *PhotosService) List() *PhotosListCall {
	c := &PhotosListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Filter sets the optional parameter "filter": The filter expression.
// For example: `placeId=ChIJj61dQgK6j4AR4GeTYWZsKWw`.
func (c *PhotosListCall) Filter(filter string) *PhotosListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The maximum number
// of photos to return.
// `pageSize` must be non-negative. If `pageSize` is zero or is not
// provided,
// the default page size of 100 will be used.
// The number of photos returned in the response may be less than
// `pageSize`
// if the number of photos that belong to the user is less than
// `pageSize`.
func (c *PhotosListCall) PageSize(pageSize int64) *PhotosListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken":
// The
// nextPageToken
// value returned from a previous
// ListPhotos
// request, if any.
func (c *PhotosListCall) PageToken(pageToken string) *PhotosListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// View sets the optional parameter "view": Specifies if a download URL
// for the photos bytes should be returned in the
// Photos response.
//
// Possible values:
//   "BASIC"
//   "INCLUDE_DOWNLOAD_URL"
func (c *PhotosListCall) View(view string) *PhotosListCall {
	c.urlParams_.Set("view", view)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PhotosListCall) Fields(s ...googleapi.Field) *PhotosListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PhotosListCall) IfNoneMatch(entityTag string) *PhotosListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PhotosListCall) Context(ctx context.Context) *PhotosListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PhotosListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PhotosListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/photos")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "streetviewpublish.photos.list" call.
// Exactly one of *ListPhotosResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListPhotosResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *PhotosListCall) Do(opts ...googleapi.CallOption) (*ListPhotosResponse, error) {
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
	ret := &ListPhotosResponse{
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
	//   "description": "Lists all the Photos that belong to\nthe user.",
	//   "flatPath": "v1/photos",
	//   "httpMethod": "GET",
	//   "id": "streetviewpublish.photos.list",
	//   "parameterOrder": [],
	//   "parameters": {
	//     "filter": {
	//       "description": "The filter expression. For example: `placeId=ChIJj61dQgK6j4AR4GeTYWZsKWw`.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The maximum number of photos to return.\n`pageSize` must be non-negative. If `pageSize` is zero or is not provided,\nthe default page size of 100 will be used.\nThe number of photos returned in the response may be less than `pageSize`\nif the number of photos that belong to the user is less than `pageSize`.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The\nnextPageToken\nvalue returned from a previous\nListPhotos\nrequest, if any.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Specifies if a download URL for the photos bytes should be returned in the\nPhotos response.",
	//       "enum": [
	//         "BASIC",
	//         "INCLUDE_DOWNLOAD_URL"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/photos",
	//   "response": {
	//     "$ref": "ListPhotosResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/streetviewpublish"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *PhotosListCall) Pages(ctx context.Context, f func(*ListPhotosResponse) error) error {
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
