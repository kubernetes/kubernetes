// Package videointelligence provides access to the Google Cloud Video Intelligence API.
//
// See https://cloud.google.com/video-intelligence/docs/
//
// Usage example:
//
//   import "google.golang.org/api/videointelligence/v1beta1"
//   ...
//   videointelligenceService, err := videointelligence.New(oauthHttpClient)
package videointelligence // import "google.golang.org/api/videointelligence/v1beta1"

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

const apiId = "videointelligence:v1beta1"
const apiName = "videointelligence"
const apiVersion = "v1beta1"
const basePath = "https://videointelligence.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Videos = NewVideosService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Videos *VideosService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewVideosService(s *Service) *VideosService {
	rs := &VideosService{s: s}
	return rs
}

type VideosService struct {
	s *Service
}

// GoogleCloudVideointelligenceV1AnnotateVideoProgress: Video annotation
// progress. Included in the `metadata`
// field of the `Operation` returned by the `GetOperation`
// call of the `google::longrunning::Operations` service.
type GoogleCloudVideointelligenceV1AnnotateVideoProgress struct {
	// AnnotationProgress: Progress metadata for all videos specified in
	// `AnnotateVideoRequest`.
	AnnotationProgress []*GoogleCloudVideointelligenceV1VideoAnnotationProgress `json:"annotationProgress,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AnnotationProgress")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AnnotationProgress") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1AnnotateVideoProgress) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1AnnotateVideoProgress
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1AnnotateVideoResponse: Video annotation
// response. Included in the `response`
// field of the `Operation` returned by the `GetOperation`
// call of the `google::longrunning::Operations` service.
type GoogleCloudVideointelligenceV1AnnotateVideoResponse struct {
	// AnnotationResults: Annotation results for all videos specified in
	// `AnnotateVideoRequest`.
	AnnotationResults []*GoogleCloudVideointelligenceV1VideoAnnotationResults `json:"annotationResults,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AnnotationResults")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AnnotationResults") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1AnnotateVideoResponse) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1AnnotateVideoResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1LabelAnnotation: Label annotation.
type GoogleCloudVideointelligenceV1LabelAnnotation struct {
	// Description: Textual description, e.g. `Fixed-gear bicycle`.
	Description string `json:"description,omitempty"`

	// LanguageCode: Language code for `description` in BCP-47 format.
	LanguageCode string `json:"languageCode,omitempty"`

	// Locations: Where the label was detected and with what confidence.
	Locations []*GoogleCloudVideointelligenceV1LabelLocation `json:"locations,omitempty"`

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

func (s *GoogleCloudVideointelligenceV1LabelAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1LabelAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1LabelLocation: Label location.
type GoogleCloudVideointelligenceV1LabelLocation struct {
	// Confidence: Confidence that the label is accurate. Range: [0, 1].
	Confidence float64 `json:"confidence,omitempty"`

	// Level: Label level.
	//
	// Possible values:
	//   "LABEL_LEVEL_UNSPECIFIED" - Unspecified.
	//   "VIDEO_LEVEL" - Video-level. Corresponds to the whole video.
	//   "SEGMENT_LEVEL" - Segment-level. Corresponds to one of
	// `AnnotateSpec.segments`.
	//   "SHOT_LEVEL" - Shot-level. Corresponds to a single shot (i.e. a
	// series of frames
	// without a major camera position or background change).
	//   "FRAME_LEVEL" - Frame-level. Corresponds to a single video frame.
	Level string `json:"level,omitempty"`

	// Segment: Video segment. Unset for video-level labels.
	// Set to a frame timestamp for frame-level labels.
	// Otherwise, corresponds to one of `AnnotateSpec.segments`
	// (if specified) or to shot boundaries (if requested).
	Segment *GoogleCloudVideointelligenceV1VideoSegment `json:"segment,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Confidence") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Confidence") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1LabelLocation) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1LabelLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *GoogleCloudVideointelligenceV1LabelLocation) UnmarshalJSON(data []byte) error {
	type noMethod GoogleCloudVideointelligenceV1LabelLocation
	var s1 struct {
		Confidence gensupport.JSONFloat64 `json:"confidence"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Confidence = float64(s1.Confidence)
	return nil
}

// GoogleCloudVideointelligenceV1SafeSearchAnnotation: Safe search
// annotation (based on per-frame visual signals only).
// If no unsafe content has been detected in a frame, no annotations
// are present for that frame.
type GoogleCloudVideointelligenceV1SafeSearchAnnotation struct {
	// Adult: Likelihood of adult content.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - Very unlikely.
	//   "UNLIKELY" - Unlikely.
	//   "POSSIBLE" - Possible.
	//   "LIKELY" - Likely.
	//   "VERY_LIKELY" - Very likely.
	Adult string `json:"adult,omitempty"`

	// Time: Time-offset, relative to the beginning of the
	// video,
	// corresponding to the video frame for this annotation.
	Time string `json:"time,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Adult") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Adult") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1SafeSearchAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1SafeSearchAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1VideoAnnotationProgress: Annotation
// progress for a single video.
type GoogleCloudVideointelligenceV1VideoAnnotationProgress struct {
	// InputUri: Video file location in
	// [Google Cloud Storage](https://cloud.google.com/storage/).
	InputUri string `json:"inputUri,omitempty"`

	// ProgressPercent: Approximate percentage processed thus
	// far.
	// Guaranteed to be 100 when fully processed.
	ProgressPercent int64 `json:"progressPercent,omitempty"`

	// StartTime: Time when the request was received.
	StartTime string `json:"startTime,omitempty"`

	// UpdateTime: Time of the most recent update.
	UpdateTime string `json:"updateTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InputUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InputUri") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1VideoAnnotationProgress) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1VideoAnnotationProgress
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1VideoAnnotationResults: Annotation
// results for a single video.
type GoogleCloudVideointelligenceV1VideoAnnotationResults struct {
	// Error: If set, indicates an error. Note that for a single
	// `AnnotateVideoRequest`
	// some videos may succeed and some may fail.
	Error *GoogleRpcStatus `json:"error,omitempty"`

	// InputUri: Video file location in
	// [Google Cloud Storage](https://cloud.google.com/storage/).
	InputUri string `json:"inputUri,omitempty"`

	// LabelAnnotations: Label annotations. There is exactly one element for
	// each unique label.
	LabelAnnotations []*GoogleCloudVideointelligenceV1LabelAnnotation `json:"labelAnnotations,omitempty"`

	// SafeSearchAnnotations: Safe search annotations.
	SafeSearchAnnotations []*GoogleCloudVideointelligenceV1SafeSearchAnnotation `json:"safeSearchAnnotations,omitempty"`

	// ShotAnnotations: Shot annotations. Each shot is represented as a
	// video segment.
	ShotAnnotations []*GoogleCloudVideointelligenceV1VideoSegment `json:"shotAnnotations,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Error") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Error") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1VideoAnnotationResults) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1VideoAnnotationResults
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1VideoSegment: Video segment.
type GoogleCloudVideointelligenceV1VideoSegment struct {
	// EndTime: Time-offset, relative to the beginning of the
	// video,
	// corresponding to the end of the segment (inclusive).
	EndTime string `json:"endTime,omitempty"`

	// StartTime: Time-offset, relative to the beginning of the
	// video,
	// corresponding to the start of the segment (inclusive).
	StartTime string `json:"startTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1VideoSegment) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1VideoSegment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta1AnnotateVideoProgress: Video
// annotation progress. Included in the `metadata`
// field of the `Operation` returned by the `GetOperation`
// call of the `google::longrunning::Operations` service.
type GoogleCloudVideointelligenceV1beta1AnnotateVideoProgress struct {
	// AnnotationProgress: Progress metadata for all videos specified in
	// `AnnotateVideoRequest`.
	AnnotationProgress []*GoogleCloudVideointelligenceV1beta1VideoAnnotationProgress `json:"annotationProgress,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AnnotationProgress")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AnnotationProgress") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta1AnnotateVideoProgress) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta1AnnotateVideoProgress
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta1AnnotateVideoRequest: Video
// annotation request.
type GoogleCloudVideointelligenceV1beta1AnnotateVideoRequest struct {
	// Features: Requested video annotation features.
	//
	// Possible values:
	//   "FEATURE_UNSPECIFIED" - Unspecified.
	//   "LABEL_DETECTION" - Label detection. Detect objects, such as dog or
	// flower.
	//   "SHOT_CHANGE_DETECTION" - Shot change detection.
	//   "SAFE_SEARCH_DETECTION" - Safe search detection.
	Features []string `json:"features,omitempty"`

	// InputContent: The video data bytes. Encoding: base64. If unset, the
	// input video(s)
	// should be specified via `input_uri`. If set, `input_uri` should be
	// unset.
	InputContent string `json:"inputContent,omitempty"`

	// InputUri: Input video location. Currently, only
	// [Google Cloud Storage](https://cloud.google.com/storage/) URIs
	// are
	// supported, which must be specified in the following
	// format:
	// `gs://bucket-id/object-id` (other URI formats
	// return
	// google.rpc.Code.INVALID_ARGUMENT). For more information, see
	// [Request URIs](/storage/docs/reference-uris).
	// A video URI may include wildcards in `object-id`, and thus
	// identify
	// multiple videos. Supported wildcards: '*' to match 0 or more
	// characters;
	// '?' to match 1 character. If unset, the input video should be
	// embedded
	// in the request as `input_content`. If set, `input_content` should be
	// unset.
	InputUri string `json:"inputUri,omitempty"`

	// LocationId: Optional cloud region where annotation should take place.
	// Supported cloud
	// regions: `us-east1`, `us-west1`, `europe-west1`, `asia-east1`. If no
	// region
	// is specified, a region will be determined based on video file
	// location.
	LocationId string `json:"locationId,omitempty"`

	// OutputUri: Optional location where the output (in JSON format) should
	// be stored.
	// Currently, only [Google Cloud
	// Storage](https://cloud.google.com/storage/)
	// URIs are supported, which must be specified in the following
	// format:
	// `gs://bucket-id/object-id` (other URI formats
	// return
	// google.rpc.Code.INVALID_ARGUMENT). For more information, see
	// [Request URIs](/storage/docs/reference-uris).
	OutputUri string `json:"outputUri,omitempty"`

	// VideoContext: Additional video context and/or feature-specific
	// parameters.
	VideoContext *GoogleCloudVideointelligenceV1beta1VideoContext `json:"videoContext,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Features") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Features") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta1AnnotateVideoRequest) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta1AnnotateVideoRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta1AnnotateVideoResponse: Video
// annotation response. Included in the `response`
// field of the `Operation` returned by the `GetOperation`
// call of the `google::longrunning::Operations` service.
type GoogleCloudVideointelligenceV1beta1AnnotateVideoResponse struct {
	// AnnotationResults: Annotation results for all videos specified in
	// `AnnotateVideoRequest`.
	AnnotationResults []*GoogleCloudVideointelligenceV1beta1VideoAnnotationResults `json:"annotationResults,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AnnotationResults")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AnnotationResults") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta1AnnotateVideoResponse) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta1AnnotateVideoResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta1LabelAnnotation: Label annotation.
type GoogleCloudVideointelligenceV1beta1LabelAnnotation struct {
	// Description: Textual description, e.g. `Fixed-gear bicycle`.
	Description string `json:"description,omitempty"`

	// LanguageCode: Language code for `description` in BCP-47 format.
	LanguageCode string `json:"languageCode,omitempty"`

	// Locations: Where the label was detected and with what confidence.
	Locations []*GoogleCloudVideointelligenceV1beta1LabelLocation `json:"locations,omitempty"`

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

func (s *GoogleCloudVideointelligenceV1beta1LabelAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta1LabelAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta1LabelLocation: Label location.
type GoogleCloudVideointelligenceV1beta1LabelLocation struct {
	// Confidence: Confidence that the label is accurate. Range: [0, 1].
	Confidence float64 `json:"confidence,omitempty"`

	// Level: Label level.
	//
	// Possible values:
	//   "LABEL_LEVEL_UNSPECIFIED" - Unspecified.
	//   "VIDEO_LEVEL" - Video-level. Corresponds to the whole video.
	//   "SEGMENT_LEVEL" - Segment-level. Corresponds to one of
	// `AnnotateSpec.segments`.
	//   "SHOT_LEVEL" - Shot-level. Corresponds to a single shot (i.e. a
	// series of frames
	// without a major camera position or background change).
	//   "FRAME_LEVEL" - Frame-level. Corresponds to a single video frame.
	Level string `json:"level,omitempty"`

	// Segment: Video segment. Set to [-1, -1] for video-level labels.
	// Set to [timestamp, timestamp] for frame-level labels.
	// Otherwise, corresponds to one of `AnnotateSpec.segments`
	// (if specified) or to shot boundaries (if requested).
	Segment *GoogleCloudVideointelligenceV1beta1VideoSegment `json:"segment,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Confidence") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Confidence") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta1LabelLocation) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta1LabelLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *GoogleCloudVideointelligenceV1beta1LabelLocation) UnmarshalJSON(data []byte) error {
	type noMethod GoogleCloudVideointelligenceV1beta1LabelLocation
	var s1 struct {
		Confidence gensupport.JSONFloat64 `json:"confidence"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Confidence = float64(s1.Confidence)
	return nil
}

// GoogleCloudVideointelligenceV1beta1SafeSearchAnnotation: Safe search
// annotation (based on per-frame visual signals only).
// If no unsafe content has been detected in a frame, no annotations
// are present for that frame. If only some types of unsafe content
// have been detected in a frame, the likelihood is set to `UNKNOWN`
// for all other types of unsafe content.
type GoogleCloudVideointelligenceV1beta1SafeSearchAnnotation struct {
	// Adult: Likelihood of adult content.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - Very unlikely.
	//   "UNLIKELY" - Unlikely.
	//   "POSSIBLE" - Possible.
	//   "LIKELY" - Likely.
	//   "VERY_LIKELY" - Very likely.
	Adult string `json:"adult,omitempty"`

	// Medical: Likelihood of medical content.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - Very unlikely.
	//   "UNLIKELY" - Unlikely.
	//   "POSSIBLE" - Possible.
	//   "LIKELY" - Likely.
	//   "VERY_LIKELY" - Very likely.
	Medical string `json:"medical,omitempty"`

	// Racy: Likelihood of racy content.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - Very unlikely.
	//   "UNLIKELY" - Unlikely.
	//   "POSSIBLE" - Possible.
	//   "LIKELY" - Likely.
	//   "VERY_LIKELY" - Very likely.
	Racy string `json:"racy,omitempty"`

	// Spoof: Likelihood that an obvious modification was made to the
	// original
	// version to make it appear funny or offensive.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - Very unlikely.
	//   "UNLIKELY" - Unlikely.
	//   "POSSIBLE" - Possible.
	//   "LIKELY" - Likely.
	//   "VERY_LIKELY" - Very likely.
	Spoof string `json:"spoof,omitempty"`

	// TimeOffset: Video time offset in microseconds.
	TimeOffset int64 `json:"timeOffset,omitempty,string"`

	// Violent: Likelihood of violent content.
	//
	// Possible values:
	//   "UNKNOWN" - Unknown likelihood.
	//   "VERY_UNLIKELY" - Very unlikely.
	//   "UNLIKELY" - Unlikely.
	//   "POSSIBLE" - Possible.
	//   "LIKELY" - Likely.
	//   "VERY_LIKELY" - Very likely.
	Violent string `json:"violent,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Adult") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Adult") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta1SafeSearchAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta1SafeSearchAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta1VideoAnnotationProgress:
// Annotation progress for a single video.
type GoogleCloudVideointelligenceV1beta1VideoAnnotationProgress struct {
	// InputUri: Video file location in
	// [Google Cloud Storage](https://cloud.google.com/storage/).
	InputUri string `json:"inputUri,omitempty"`

	// ProgressPercent: Approximate percentage processed thus
	// far.
	// Guaranteed to be 100 when fully processed.
	ProgressPercent int64 `json:"progressPercent,omitempty"`

	// StartTime: Time when the request was received.
	StartTime string `json:"startTime,omitempty"`

	// UpdateTime: Time of the most recent update.
	UpdateTime string `json:"updateTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InputUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InputUri") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta1VideoAnnotationProgress) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta1VideoAnnotationProgress
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta1VideoAnnotationResults: Annotation
// results for a single video.
type GoogleCloudVideointelligenceV1beta1VideoAnnotationResults struct {
	// Error: If set, indicates an error. Note that for a single
	// `AnnotateVideoRequest`
	// some videos may succeed and some may fail.
	Error *GoogleRpcStatus `json:"error,omitempty"`

	// InputUri: Video file location in
	// [Google Cloud Storage](https://cloud.google.com/storage/).
	InputUri string `json:"inputUri,omitempty"`

	// LabelAnnotations: Label annotations. There is exactly one element for
	// each unique label.
	LabelAnnotations []*GoogleCloudVideointelligenceV1beta1LabelAnnotation `json:"labelAnnotations,omitempty"`

	// SafeSearchAnnotations: Safe search annotations.
	SafeSearchAnnotations []*GoogleCloudVideointelligenceV1beta1SafeSearchAnnotation `json:"safeSearchAnnotations,omitempty"`

	// ShotAnnotations: Shot annotations. Each shot is represented as a
	// video segment.
	ShotAnnotations []*GoogleCloudVideointelligenceV1beta1VideoSegment `json:"shotAnnotations,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Error") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Error") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta1VideoAnnotationResults) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta1VideoAnnotationResults
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta1VideoContext: Video context and/or
// feature-specific parameters.
type GoogleCloudVideointelligenceV1beta1VideoContext struct {
	// LabelDetectionMode: If label detection has been requested, what
	// labels should be detected
	// in addition to video-level labels or segment-level labels. If
	// unspecified,
	// defaults to `SHOT_MODE`.
	//
	// Possible values:
	//   "LABEL_DETECTION_MODE_UNSPECIFIED" - Unspecified.
	//   "SHOT_MODE" - Detect shot-level labels.
	//   "FRAME_MODE" - Detect frame-level labels.
	//   "SHOT_AND_FRAME_MODE" - Detect both shot-level and frame-level
	// labels.
	LabelDetectionMode string `json:"labelDetectionMode,omitempty"`

	// LabelDetectionModel: Model to use for label detection.
	// Supported values: "latest" and "stable" (the default).
	LabelDetectionModel string `json:"labelDetectionModel,omitempty"`

	// SafeSearchDetectionModel: Model to use for safe search
	// detection.
	// Supported values: "latest" and "stable" (the default).
	SafeSearchDetectionModel string `json:"safeSearchDetectionModel,omitempty"`

	// Segments: Video segments to annotate. The segments may overlap and
	// are not required
	// to be contiguous or span the whole video. If unspecified, each
	// video
	// is treated as a single segment.
	Segments []*GoogleCloudVideointelligenceV1beta1VideoSegment `json:"segments,omitempty"`

	// ShotChangeDetectionModel: Model to use for shot change
	// detection.
	// Supported values: "latest" and "stable" (the default).
	ShotChangeDetectionModel string `json:"shotChangeDetectionModel,omitempty"`

	// StationaryCamera: Whether the video has been shot from a stationary
	// (i.e. non-moving) camera.
	// When set to true, might improve detection accuracy for moving
	// objects.
	StationaryCamera bool `json:"stationaryCamera,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LabelDetectionMode")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LabelDetectionMode") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta1VideoContext) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta1VideoContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta1VideoSegment: Video segment.
type GoogleCloudVideointelligenceV1beta1VideoSegment struct {
	// EndTimeOffset: End offset in microseconds (inclusive). Unset means 0.
	EndTimeOffset int64 `json:"endTimeOffset,omitempty,string"`

	// StartTimeOffset: Start offset in microseconds (inclusive). Unset
	// means 0.
	StartTimeOffset int64 `json:"startTimeOffset,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "EndTimeOffset") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndTimeOffset") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta1VideoSegment) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta1VideoSegment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta2AnnotateVideoProgress: Video
// annotation progress. Included in the `metadata`
// field of the `Operation` returned by the `GetOperation`
// call of the `google::longrunning::Operations` service.
type GoogleCloudVideointelligenceV1beta2AnnotateVideoProgress struct {
	// AnnotationProgress: Progress metadata for all videos specified in
	// `AnnotateVideoRequest`.
	AnnotationProgress []*GoogleCloudVideointelligenceV1beta2VideoAnnotationProgress `json:"annotationProgress,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AnnotationProgress")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AnnotationProgress") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta2AnnotateVideoProgress) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta2AnnotateVideoProgress
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta2AnnotateVideoResponse: Video
// annotation response. Included in the `response`
// field of the `Operation` returned by the `GetOperation`
// call of the `google::longrunning::Operations` service.
type GoogleCloudVideointelligenceV1beta2AnnotateVideoResponse struct {
	// AnnotationResults: Annotation results for all videos specified in
	// `AnnotateVideoRequest`.
	AnnotationResults []*GoogleCloudVideointelligenceV1beta2VideoAnnotationResults `json:"annotationResults,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AnnotationResults")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AnnotationResults") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta2AnnotateVideoResponse) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta2AnnotateVideoResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta2Entity: Detected entity from video
// analysis.
type GoogleCloudVideointelligenceV1beta2Entity struct {
	// Description: Textual description, e.g. `Fixed-gear bicycle`.
	Description string `json:"description,omitempty"`

	// EntityId: Opaque entity ID. Some IDs may be available in
	// [Google Knowledge Graph
	// Search
	// API](https://developers.google.com/knowledge-graph/).
	EntityId string `json:"entityId,omitempty"`

	// LanguageCode: Language code for `description` in BCP-47 format.
	LanguageCode string `json:"languageCode,omitempty"`

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

func (s *GoogleCloudVideointelligenceV1beta2Entity) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta2Entity
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta2ExplicitContentAnnotation:
// Explicit content annotation (based on per-frame visual signals
// only).
// If no explicit content has been detected in a frame, no annotations
// are
// present for that frame.
type GoogleCloudVideointelligenceV1beta2ExplicitContentAnnotation struct {
	// Frames: All video frames where explicit content was detected.
	Frames []*GoogleCloudVideointelligenceV1beta2ExplicitContentFrame `json:"frames,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Frames") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Frames") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta2ExplicitContentAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta2ExplicitContentAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta2ExplicitContentFrame: Video frame
// level annotation results for explicit content.
type GoogleCloudVideointelligenceV1beta2ExplicitContentFrame struct {
	// PornographyLikelihood: Likelihood of the pornography content..
	//
	// Possible values:
	//   "LIKELIHOOD_UNSPECIFIED" - Unspecified likelihood.
	//   "VERY_UNLIKELY" - Very unlikely.
	//   "UNLIKELY" - Unlikely.
	//   "POSSIBLE" - Possible.
	//   "LIKELY" - Likely.
	//   "VERY_LIKELY" - Very likely.
	PornographyLikelihood string `json:"pornographyLikelihood,omitempty"`

	// TimeOffset: Time-offset, relative to the beginning of the video,
	// corresponding to the
	// video frame for this location.
	TimeOffset string `json:"timeOffset,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "PornographyLikelihood") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PornographyLikelihood") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta2ExplicitContentFrame) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta2ExplicitContentFrame
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta2LabelAnnotation: Label annotation.
type GoogleCloudVideointelligenceV1beta2LabelAnnotation struct {
	// CategoryEntities: Common categories for the detected entity.
	// E.g. when the label is `Terrier` the category is likely `dog`. And in
	// some
	// cases there might be more than one categories e.g. `Terrier` could
	// also be
	// a `pet`.
	CategoryEntities []*GoogleCloudVideointelligenceV1beta2Entity `json:"categoryEntities,omitempty"`

	// Entity: Detected entity.
	Entity *GoogleCloudVideointelligenceV1beta2Entity `json:"entity,omitempty"`

	// Frames: All video frames where a label was detected.
	Frames []*GoogleCloudVideointelligenceV1beta2LabelFrame `json:"frames,omitempty"`

	// Segments: All video segments where a label was detected.
	Segments []*GoogleCloudVideointelligenceV1beta2LabelSegment `json:"segments,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CategoryEntities") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CategoryEntities") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta2LabelAnnotation) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta2LabelAnnotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta2LabelFrame: Video frame level
// annotation results for label detection.
type GoogleCloudVideointelligenceV1beta2LabelFrame struct {
	// Confidence: Confidence that the label is accurate. Range: [0, 1].
	Confidence float64 `json:"confidence,omitempty"`

	// TimeOffset: Time-offset, relative to the beginning of the video,
	// corresponding to the
	// video frame for this location.
	TimeOffset string `json:"timeOffset,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Confidence") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Confidence") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta2LabelFrame) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta2LabelFrame
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *GoogleCloudVideointelligenceV1beta2LabelFrame) UnmarshalJSON(data []byte) error {
	type noMethod GoogleCloudVideointelligenceV1beta2LabelFrame
	var s1 struct {
		Confidence gensupport.JSONFloat64 `json:"confidence"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Confidence = float64(s1.Confidence)
	return nil
}

// GoogleCloudVideointelligenceV1beta2LabelSegment: Video segment level
// annotation results for label detection.
type GoogleCloudVideointelligenceV1beta2LabelSegment struct {
	// Confidence: Confidence that the label is accurate. Range: [0, 1].
	Confidence float64 `json:"confidence,omitempty"`

	// Segment: Video segment where a label was detected.
	Segment *GoogleCloudVideointelligenceV1beta2VideoSegment `json:"segment,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Confidence") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Confidence") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta2LabelSegment) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta2LabelSegment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *GoogleCloudVideointelligenceV1beta2LabelSegment) UnmarshalJSON(data []byte) error {
	type noMethod GoogleCloudVideointelligenceV1beta2LabelSegment
	var s1 struct {
		Confidence gensupport.JSONFloat64 `json:"confidence"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Confidence = float64(s1.Confidence)
	return nil
}

// GoogleCloudVideointelligenceV1beta2VideoAnnotationProgress:
// Annotation progress for a single video.
type GoogleCloudVideointelligenceV1beta2VideoAnnotationProgress struct {
	// InputUri: Video file location in
	// [Google Cloud Storage](https://cloud.google.com/storage/).
	InputUri string `json:"inputUri,omitempty"`

	// ProgressPercent: Approximate percentage processed thus
	// far.
	// Guaranteed to be 100 when fully processed.
	ProgressPercent int64 `json:"progressPercent,omitempty"`

	// StartTime: Time when the request was received.
	StartTime string `json:"startTime,omitempty"`

	// UpdateTime: Time of the most recent update.
	UpdateTime string `json:"updateTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InputUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InputUri") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta2VideoAnnotationProgress) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta2VideoAnnotationProgress
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta2VideoAnnotationResults: Annotation
// results for a single video.
type GoogleCloudVideointelligenceV1beta2VideoAnnotationResults struct {
	// Error: If set, indicates an error. Note that for a single
	// `AnnotateVideoRequest`
	// some videos may succeed and some may fail.
	Error *GoogleRpcStatus `json:"error,omitempty"`

	// ExplicitAnnotation: Explicit content annotation.
	ExplicitAnnotation *GoogleCloudVideointelligenceV1beta2ExplicitContentAnnotation `json:"explicitAnnotation,omitempty"`

	// FrameLabelAnnotations: Label annotations on frame level.
	// There is exactly one element for each unique label.
	FrameLabelAnnotations []*GoogleCloudVideointelligenceV1beta2LabelAnnotation `json:"frameLabelAnnotations,omitempty"`

	// InputUri: Video file location in
	// [Google Cloud Storage](https://cloud.google.com/storage/).
	InputUri string `json:"inputUri,omitempty"`

	// SegmentLabelAnnotations: Label annotations on video level or user
	// specified segment level.
	// There is exactly one element for each unique label.
	SegmentLabelAnnotations []*GoogleCloudVideointelligenceV1beta2LabelAnnotation `json:"segmentLabelAnnotations,omitempty"`

	// ShotAnnotations: Shot annotations. Each shot is represented as a
	// video segment.
	ShotAnnotations []*GoogleCloudVideointelligenceV1beta2VideoSegment `json:"shotAnnotations,omitempty"`

	// ShotLabelAnnotations: Label annotations on shot level.
	// There is exactly one element for each unique label.
	ShotLabelAnnotations []*GoogleCloudVideointelligenceV1beta2LabelAnnotation `json:"shotLabelAnnotations,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Error") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Error") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta2VideoAnnotationResults) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta2VideoAnnotationResults
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleCloudVideointelligenceV1beta2VideoSegment: Video segment.
type GoogleCloudVideointelligenceV1beta2VideoSegment struct {
	// EndTimeOffset: Time-offset, relative to the beginning of the
	// video,
	// corresponding to the end of the segment (inclusive).
	EndTimeOffset string `json:"endTimeOffset,omitempty"`

	// StartTimeOffset: Time-offset, relative to the beginning of the
	// video,
	// corresponding to the start of the segment (inclusive).
	StartTimeOffset string `json:"startTimeOffset,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndTimeOffset") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndTimeOffset") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GoogleCloudVideointelligenceV1beta2VideoSegment) MarshalJSON() ([]byte, error) {
	type noMethod GoogleCloudVideointelligenceV1beta2VideoSegment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleLongrunningOperation: This resource represents a long-running
// operation that is the result of a
// network API call.
type GoogleLongrunningOperation struct {
	// Done: If the value is `false`, it means the operation is still in
	// progress.
	// If `true`, the operation is completed, and either `error` or
	// `response` is
	// available.
	Done bool `json:"done,omitempty"`

	// Error: The error result of the operation in case of failure or
	// cancellation.
	Error *GoogleRpcStatus `json:"error,omitempty"`

	// Metadata: Service-specific metadata associated with the operation.
	// It typically
	// contains progress information and common metadata such as create
	// time.
	// Some services might not provide such metadata.  Any method that
	// returns a
	// long-running operation should document the metadata type, if any.
	Metadata googleapi.RawMessage `json:"metadata,omitempty"`

	// Name: The server-assigned name, which is only unique within the same
	// service that
	// originally returns it. If you use the default HTTP mapping,
	// the
	// `name` should have the format of `operations/some/unique/name`.
	Name string `json:"name,omitempty"`

	// Response: The normal response of the operation in case of success.
	// If the original
	// method returns no data on success, such as `Delete`, the response
	// is
	// `google.protobuf.Empty`.  If the original method is
	// standard
	// `Get`/`Create`/`Update`, the response should be the resource.  For
	// other
	// methods, the response should have the type `XxxResponse`, where
	// `Xxx`
	// is the original method name.  For example, if the original method
	// name
	// is `TakeSnapshot()`, the inferred response type
	// is
	// `TakeSnapshotResponse`.
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

func (s *GoogleLongrunningOperation) MarshalJSON() ([]byte, error) {
	type noMethod GoogleLongrunningOperation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GoogleRpcStatus: The `Status` type defines a logical error model that
// is suitable for different
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
type GoogleRpcStatus struct {
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

func (s *GoogleRpcStatus) MarshalJSON() ([]byte, error) {
	type noMethod GoogleRpcStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "videointelligence.videos.annotate":

type VideosAnnotateCall struct {
	s                                                       *Service
	googlecloudvideointelligencev1beta1Annotatevideorequest *GoogleCloudVideointelligenceV1beta1AnnotateVideoRequest
	urlParams_                                              gensupport.URLParams
	ctx_                                                    context.Context
	header_                                                 http.Header
}

// Annotate: Performs asynchronous video annotation. Progress and
// results can be
// retrieved through the `google.longrunning.Operations`
// interface.
// `Operation.metadata` contains `AnnotateVideoProgress`
// (progress).
// `Operation.response` contains `AnnotateVideoResponse` (results).
func (r *VideosService) Annotate(googlecloudvideointelligencev1beta1Annotatevideorequest *GoogleCloudVideointelligenceV1beta1AnnotateVideoRequest) *VideosAnnotateCall {
	c := &VideosAnnotateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.googlecloudvideointelligencev1beta1Annotatevideorequest = googlecloudvideointelligencev1beta1Annotatevideorequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *VideosAnnotateCall) Fields(s ...googleapi.Field) *VideosAnnotateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *VideosAnnotateCall) Context(ctx context.Context) *VideosAnnotateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *VideosAnnotateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *VideosAnnotateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.googlecloudvideointelligencev1beta1Annotatevideorequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1/videos:annotate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "videointelligence.videos.annotate" call.
// Exactly one of *GoogleLongrunningOperation or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *GoogleLongrunningOperation.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *VideosAnnotateCall) Do(opts ...googleapi.CallOption) (*GoogleLongrunningOperation, error) {
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
	ret := &GoogleLongrunningOperation{
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
	//   "description": "Performs asynchronous video annotation. Progress and results can be\nretrieved through the `google.longrunning.Operations` interface.\n`Operation.metadata` contains `AnnotateVideoProgress` (progress).\n`Operation.response` contains `AnnotateVideoResponse` (results).",
	//   "flatPath": "v1beta1/videos:annotate",
	//   "httpMethod": "POST",
	//   "id": "videointelligence.videos.annotate",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v1beta1/videos:annotate",
	//   "request": {
	//     "$ref": "GoogleCloudVideointelligenceV1beta1_AnnotateVideoRequest"
	//   },
	//   "response": {
	//     "$ref": "GoogleLongrunning_Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}
