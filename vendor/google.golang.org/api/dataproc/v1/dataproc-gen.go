// Package dataproc provides access to the Google Cloud Dataproc API.
//
// See https://cloud.google.com/dataproc/
//
// Usage example:
//
//   import "google.golang.org/api/dataproc/v1"
//   ...
//   dataprocService, err := dataproc.New(oauthHttpClient)
package dataproc // import "google.golang.org/api/dataproc/v1"

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

const apiId = "dataproc:v1"
const apiName = "dataproc"
const apiVersion = "v1"
const basePath = "https://dataproc.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// Administrate log data for your projects
	LoggingAdminScope = "https://www.googleapis.com/auth/logging.admin"

	// View log data for your projects
	LoggingReadScope = "https://www.googleapis.com/auth/logging.read"

	// Submit log data for your projects
	LoggingWriteScope = "https://www.googleapis.com/auth/logging.write"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Media = NewMediaService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Media *MediaService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewMediaService(s *Service) *MediaService {
	rs := &MediaService{s: s}
	return rs
}

type MediaService struct {
	s *Service
}

// DiagnoseClusterOutputLocation: The location where output from
// diagnostic command can be found.
type DiagnoseClusterOutputLocation struct {
	// OutputUri: [Output-only] The Google Cloud Storage URI of the
	// diagnostic output. This will be a plain text file with summary of
	// collected diagnostics.
	OutputUri string `json:"outputUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OutputUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DiagnoseClusterOutputLocation) MarshalJSON() ([]byte, error) {
	type noMethod DiagnoseClusterOutputLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Media: Media resource.
type Media struct {
	// ResourceName: Name of the media resource.
	ResourceName string `json:"resourceName,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ResourceName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Media) MarshalJSON() ([]byte, error) {
	type noMethod Media
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// OperationMetadata: Metadata describing the operation.
type OperationMetadata struct {
	// ClusterName: Name of the cluster for the operation.
	ClusterName string `json:"clusterName,omitempty"`

	// ClusterUuid: Cluster UUId for the operation.
	ClusterUuid string `json:"clusterUuid,omitempty"`

	// Details: A message containing any operation metadata details.
	Details string `json:"details,omitempty"`

	// EndTime: The time that the operation completed.
	EndTime string `json:"endTime,omitempty"`

	// InnerState: A message containing the detailed operation state.
	InnerState string `json:"innerState,omitempty"`

	// InsertTime: The time that the operation was requested.
	InsertTime string `json:"insertTime,omitempty"`

	// StartTime: The time that the operation was started by the server.
	StartTime string `json:"startTime,omitempty"`

	// State: A message containing the operation state.
	//
	// Possible values:
	//   "UNKNOWN"
	//   "PENDING"
	//   "RUNNING"
	//   "DONE"
	State string `json:"state,omitempty"`

	// Status: [Output-only] Current operation status.
	Status *OperationStatus `json:"status,omitempty"`

	// StatusHistory: [Output-only] Previous operation status.
	StatusHistory []*OperationStatus `json:"statusHistory,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClusterName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *OperationMetadata) MarshalJSON() ([]byte, error) {
	type noMethod OperationMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// OperationStatus: The status of the operation.
type OperationStatus struct {
	// Details: A message containing any operation metadata details.
	Details string `json:"details,omitempty"`

	// InnerState: A message containing the detailed operation state.
	InnerState string `json:"innerState,omitempty"`

	// State: A message containing the operation state.
	//
	// Possible values:
	//   "UNKNOWN"
	//   "PENDING"
	//   "RUNNING"
	//   "DONE"
	State string `json:"state,omitempty"`

	// StateStartTime: The time this state was entered.
	StateStartTime string `json:"stateStartTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Details") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *OperationStatus) MarshalJSON() ([]byte, error) {
	type noMethod OperationStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "dataproc.media.download":

type MediaDownloadCall struct {
	s            *Service
	resourceName string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Download: Method for media download. Download is supported on the URI
// `/v1/media/{+name}?alt=media`.
func (r *MediaService) Download(resourceName string) *MediaDownloadCall {
	c := &MediaDownloadCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resourceName = resourceName
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *MediaDownloadCall) QuotaUser(quotaUser string) *MediaDownloadCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MediaDownloadCall) Fields(s ...googleapi.Field) *MediaDownloadCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *MediaDownloadCall) IfNoneMatch(entityTag string) *MediaDownloadCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do and Download
// methods. Any pending HTTP request will be aborted if the provided
// context is canceled.
func (c *MediaDownloadCall) Context(ctx context.Context) *MediaDownloadCall {
	c.ctx_ = ctx
	return c
}

func (c *MediaDownloadCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/media/{+resourceName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"resourceName": c.resourceName,
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

// Download fetches the API endpoint's "media" value, instead of the normal
// API response value. If the returned error is nil, the Response is guaranteed to
// have a 2xx status code. Callers must close the Response.Body as usual.
func (c *MediaDownloadCall) Download() (*http.Response, error) {
	res, err := c.doRequest("media")
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckMediaResponse(res); err != nil {
		res.Body.Close()
		return nil, err
	}
	return res, nil
}

// Do executes the "dataproc.media.download" call.
// Exactly one of *Media or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Media.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *MediaDownloadCall) Do() (*Media, error) {
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
	ret := &Media{
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
	//   "description": "Method for media download. Download is supported on the URI `/v1/media/{+name}?alt=media`.",
	//   "httpMethod": "GET",
	//   "id": "dataproc.media.download",
	//   "parameterOrder": [
	//     "resourceName"
	//   ],
	//   "parameters": {
	//     "resourceName": {
	//       "description": "Name of the media that is being downloaded. See [][ByteStream.ReadRequest.resource_name].",
	//       "location": "path",
	//       "pattern": "^.*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/media/{+resourceName}",
	//   "response": {
	//     "$ref": "Media"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "dataproc.media.upload":

type MediaUploadCall struct {
	s                *Service
	resourceName     string
	media            *Media
	urlParams_       gensupport.URLParams
	media_           io.Reader
	resumable_       googleapi.SizeReaderAt
	mediaType_       string
	protocol_        string
	progressUpdater_ googleapi.ProgressUpdater
	ctx_             context.Context
}

// Upload: Method for media upload. Upload is supported on the URI
// `/upload/v1/media/{+name}`.
func (r *MediaService) Upload(resourceName string, media *Media) *MediaUploadCall {
	c := &MediaUploadCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resourceName = resourceName
	c.media = media
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *MediaUploadCall) QuotaUser(quotaUser string) *MediaUploadCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Media specifies the media to upload in a single chunk. At most one of
// Media and ResumableMedia may be set.
func (c *MediaUploadCall) Media(r io.Reader) *MediaUploadCall {
	c.media_ = r
	c.protocol_ = "multipart"
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx. At most one of Media and ResumableMedia may be
// set. mediaType identifies the MIME media type of the upload, such as
// "image/png". If mediaType is "", it will be auto-detected. The
// provided ctx will supersede any context previously provided to the
// Context method.
func (c *MediaUploadCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *MediaUploadCall {
	c.ctx_ = ctx
	c.resumable_ = io.NewSectionReader(r, 0, size)
	c.mediaType_ = mediaType
	c.protocol_ = "resumable"
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *MediaUploadCall) ProgressUpdater(pu googleapi.ProgressUpdater) *MediaUploadCall {
	c.progressUpdater_ = pu
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *MediaUploadCall) Fields(s ...googleapi.Field) *MediaUploadCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *MediaUploadCall) Context(ctx context.Context) *MediaUploadCall {
	c.ctx_ = ctx
	return c
}

func (c *MediaUploadCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.media)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/media/{+resourceName}")
	if c.media_ != nil || c.resumable_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.protocol_)
	}
	urls += "?" + c.urlParams_.Encode()
	if c.protocol_ != "resumable" && c.media_ != nil {
		cancel := gensupport.IncludeMedia(c.media_, &body, &ctype)
		defer cancel()
	}
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"resourceName": c.resourceName,
	})
	if c.protocol_ == "resumable" {
		if c.mediaType_ == "" {
			c.mediaType_ = gensupport.DetectMediaType(c.resumable_)
		}
		req.Header.Set("X-Upload-Content-Type", c.mediaType_)
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.media.upload" call.
// Exactly one of *Media or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Media.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *MediaUploadCall) Do() (*Media, error) {
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
	if c.protocol_ == "resumable" {
		loc := res.Header.Get("Location")
		rx := &googleapi.ResumableUpload{
			Client:        c.s.client,
			UserAgent:     c.s.userAgent(),
			URI:           loc,
			Media:         c.resumable_,
			MediaType:     c.mediaType_,
			ContentLength: c.resumable_.Size(),
			Callback: func(curr int64) {
				if c.progressUpdater_ != nil {
					c.progressUpdater_(curr, c.resumable_.Size())
				}
			},
		}
		res, err = rx.Upload(c.ctx_)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
	}
	ret := &Media{
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
	//   "description": "Method for media upload. Upload is supported on the URI `/upload/v1/media/{+name}`.",
	//   "httpMethod": "POST",
	//   "id": "dataproc.media.upload",
	//   "mediaUpload": {
	//     "accept": [
	//       "*/*"
	//     ],
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/v1/media/{+resourceName}"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/v1/media/{+resourceName}"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "resourceName"
	//   ],
	//   "parameters": {
	//     "resourceName": {
	//       "description": "Name of the media that is being downloaded. See [][ByteStream.ReadRequest.resource_name].",
	//       "location": "path",
	//       "pattern": "^.*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/media/{+resourceName}",
	//   "request": {
	//     "$ref": "Media"
	//   },
	//   "response": {
	//     "$ref": "Media"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ],
	//   "supportsMediaUpload": true
	// }

}
