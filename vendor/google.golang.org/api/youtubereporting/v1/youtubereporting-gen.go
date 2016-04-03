// Package youtubereporting provides access to the YouTube Reporting API.
//
// See https://developers.google.com/youtube/reporting/v1/reports/
//
// Usage example:
//
//   import "google.golang.org/api/youtubereporting/v1"
//   ...
//   youtubereportingService, err := youtubereporting.New(oauthHttpClient)
package youtubereporting // import "google.golang.org/api/youtubereporting/v1"

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

const apiId = "youtubereporting:v1"
const apiName = "youtubereporting"
const apiVersion = "v1"
const basePath = "https://youtubereporting.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View monetary and non-monetary YouTube Analytics reports for your
	// YouTube content
	YtAnalyticsMonetaryReadonlyScope = "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"

	// View YouTube Analytics reports for your YouTube content
	YtAnalyticsReadonlyScope = "https://www.googleapis.com/auth/yt-analytics.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Jobs = NewJobsService(s)
	s.Media = NewMediaService(s)
	s.ReportTypes = NewReportTypesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Jobs *JobsService

	Media *MediaService

	ReportTypes *ReportTypesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewJobsService(s *Service) *JobsService {
	rs := &JobsService{s: s}
	rs.Reports = NewJobsReportsService(s)
	return rs
}

type JobsService struct {
	s *Service

	Reports *JobsReportsService
}

func NewJobsReportsService(s *Service) *JobsReportsService {
	rs := &JobsReportsService{s: s}
	return rs
}

type JobsReportsService struct {
	s *Service
}

func NewMediaService(s *Service) *MediaService {
	rs := &MediaService{s: s}
	return rs
}

type MediaService struct {
	s *Service
}

func NewReportTypesService(s *Service) *ReportTypesService {
	rs := &ReportTypesService{s: s}
	return rs
}

type ReportTypesService struct {
	s *Service
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

// Job: A job creating reports of a specific type.
type Job struct {
	// CreateTime: The creation date/time of the job.
	CreateTime string `json:"createTime,omitempty"`

	// Id: The server-generated ID of the job (max. 40 characters).
	Id string `json:"id,omitempty"`

	// Name: The name of the job (max. 100 characters). TODO(lanthaler)
	// Clarify what this will actually be used for
	Name string `json:"name,omitempty"`

	// ReportTypeId: The type of reports this job creates. Corresponds to
	// the ID of a ReportType.
	ReportTypeId string `json:"reportTypeId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreateTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Job) MarshalJSON() ([]byte, error) {
	type noMethod Job
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListJobsResponse: Response message for ReportingService.ListJobs.
type ListJobsResponse struct {
	// Jobs: The list of jobs.
	Jobs []*Job `json:"jobs,omitempty"`

	// NextPageToken: A token to retrieve next page of results. Pass this
	// value in the ListJobsRequest.page_token field in the subsequent call
	// to `ListJobs` method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Jobs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListJobsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListJobsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListReportTypesResponse: Response message for
// ReportingService.ListReportTypes.
type ListReportTypesResponse struct {
	// NextPageToken: A token to retrieve next page of results. Pass this
	// value in the ListReportTypesRequest.page_token field in the
	// subsequent call to `ListReportTypes` method to retrieve the next page
	// of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ReportTypes: The list of report types.
	ReportTypes []*ReportType `json:"reportTypes,omitempty"`

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
}

func (s *ListReportTypesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListReportTypesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListReportsResponse: Response message for
// ReportingService.ListReports.
type ListReportsResponse struct {
	// NextPageToken: A token to retrieve next page of results. Pass this
	// value in the ListReportsRequest.page_token field in the subsequent
	// call to `ListReports` method to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Reports: The list of report types.
	Reports []*Report `json:"reports,omitempty"`

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
}

func (s *ListReportsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListReportsResponse
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

// Report: A report's metadata including the URL from which the report
// itself can be downloaded.
type Report struct {
	// CreateTime: The date/time when this report was created.
	CreateTime string `json:"createTime,omitempty"`

	// DownloadUrl: The URL from which the report can be downloaded (max.
	// 1000 characters).
	DownloadUrl string `json:"downloadUrl,omitempty"`

	// EndTime: The end of the time period that the report instance covers.
	// The value is exclusive.
	EndTime string `json:"endTime,omitempty"`

	// Id: The server-generated ID of the report.
	Id string `json:"id,omitempty"`

	// JobId: The ID of the job that created this report.
	JobId string `json:"jobId,omitempty"`

	// StartTime: The start of the time period that the report instance
	// covers. The value is inclusive.
	StartTime string `json:"startTime,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CreateTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Report) MarshalJSON() ([]byte, error) {
	type noMethod Report
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReportType: A report type.
type ReportType struct {
	// Id: The ID of the report type (max. 100 characters).
	Id string `json:"id,omitempty"`

	// Name: The name of the report type (max. 100 characters).
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReportType) MarshalJSON() ([]byte, error) {
	type noMethod ReportType
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "youtubereporting.jobs.create":

type JobsCreateCall struct {
	s          *Service
	job        *Job
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Creates a job and returns it.
func (r *JobsService) Create(job *Job) *JobsCreateCall {
	c := &JobsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.job = job
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The content owner's external ID on which
// behalf the user is acting on. If not set, the user is acting for
// himself (his own channel).
func (c *JobsCreateCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *JobsCreateCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *JobsCreateCall) QuotaUser(quotaUser string) *JobsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsCreateCall) Fields(s ...googleapi.Field) *JobsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsCreateCall) Context(ctx context.Context) *JobsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *JobsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.job)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/jobs")
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

// Do executes the "youtubereporting.jobs.create" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *JobsCreateCall) Do() (*Job, error) {
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
	ret := &Job{
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
	//   "description": "Creates a job and returns it.",
	//   "httpMethod": "POST",
	//   "id": "youtubereporting.jobs.create",
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel).",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/jobs",
	//   "request": {
	//     "$ref": "Job"
	//   },
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}

// method id "youtubereporting.jobs.delete":

type JobsDeleteCall struct {
	s          *Service
	jobId      string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a job.
func (r *JobsService) Delete(jobId string) *JobsDeleteCall {
	c := &JobsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.jobId = jobId
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The content owner's external ID on which
// behalf the user is acting on. If not set, the user is acting for
// himself (his own channel).
func (c *JobsDeleteCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *JobsDeleteCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *JobsDeleteCall) QuotaUser(quotaUser string) *JobsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsDeleteCall) Fields(s ...googleapi.Field) *JobsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsDeleteCall) Context(ctx context.Context) *JobsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *JobsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/jobs/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"jobId": c.jobId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "youtubereporting.jobs.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *JobsDeleteCall) Do() (*Empty, error) {
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
	//   "description": "Deletes a job.",
	//   "httpMethod": "DELETE",
	//   "id": "youtubereporting.jobs.delete",
	//   "parameterOrder": [
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "The ID of the job to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel).",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/jobs/{jobId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}

// method id "youtubereporting.jobs.get":

type JobsGetCall struct {
	s            *Service
	jobId        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets a job.
func (r *JobsService) Get(jobId string) *JobsGetCall {
	c := &JobsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.jobId = jobId
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The content owner's external ID on which
// behalf the user is acting on. If not set, the user is acting for
// himself (his own channel).
func (c *JobsGetCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *JobsGetCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *JobsGetCall) QuotaUser(quotaUser string) *JobsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsGetCall) Fields(s ...googleapi.Field) *JobsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *JobsGetCall) IfNoneMatch(entityTag string) *JobsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsGetCall) Context(ctx context.Context) *JobsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *JobsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/jobs/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"jobId": c.jobId,
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

// Do executes the "youtubereporting.jobs.get" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *JobsGetCall) Do() (*Job, error) {
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
	ret := &Job{
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
	//   "description": "Gets a job.",
	//   "httpMethod": "GET",
	//   "id": "youtubereporting.jobs.get",
	//   "parameterOrder": [
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "The ID of the job to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel).",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/jobs/{jobId}",
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}

// method id "youtubereporting.jobs.list":

type JobsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists jobs.
func (r *JobsService) List() *JobsListCall {
	c := &JobsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The content owner's external ID on which
// behalf the user is acting on. If not set, the user is acting for
// himself (his own channel).
func (c *JobsListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *JobsListCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// Server may return fewer jobs than requested. If unspecified, server
// will pick an appropriate default.
func (c *JobsListCall) PageSize(pageSize int64) *JobsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return. Typically,
// this is the value of ListReportTypesResponse.next_page_token returned
// in response to the previous call to the `ListJobs` method.
func (c *JobsListCall) PageToken(pageToken string) *JobsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *JobsListCall) QuotaUser(quotaUser string) *JobsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsListCall) Fields(s ...googleapi.Field) *JobsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *JobsListCall) IfNoneMatch(entityTag string) *JobsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsListCall) Context(ctx context.Context) *JobsListCall {
	c.ctx_ = ctx
	return c
}

func (c *JobsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/jobs")
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

// Do executes the "youtubereporting.jobs.list" call.
// Exactly one of *ListJobsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListJobsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *JobsListCall) Do() (*ListJobsResponse, error) {
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
	ret := &ListJobsResponse{
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
	//   "description": "Lists jobs.",
	//   "httpMethod": "GET",
	//   "id": "youtubereporting.jobs.list",
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. Server may return fewer jobs than requested. If unspecified, server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return. Typically, this is the value of ListReportTypesResponse.next_page_token returned in response to the previous call to the `ListJobs` method.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/jobs",
	//   "response": {
	//     "$ref": "ListJobsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}

// method id "youtubereporting.jobs.reports.get":

type JobsReportsGetCall struct {
	s            *Service
	jobId        string
	reportId     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the metadata of a specific report.
func (r *JobsReportsService) Get(jobId string, reportId string) *JobsReportsGetCall {
	c := &JobsReportsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.jobId = jobId
	c.reportId = reportId
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The content owner's external ID on which
// behalf the user is acting on. If not set, the user is acting for
// himself (his own channel).
func (c *JobsReportsGetCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *JobsReportsGetCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *JobsReportsGetCall) QuotaUser(quotaUser string) *JobsReportsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsReportsGetCall) Fields(s ...googleapi.Field) *JobsReportsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *JobsReportsGetCall) IfNoneMatch(entityTag string) *JobsReportsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsReportsGetCall) Context(ctx context.Context) *JobsReportsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *JobsReportsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/jobs/{jobId}/reports/{reportId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"jobId":    c.jobId,
		"reportId": c.reportId,
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

// Do executes the "youtubereporting.jobs.reports.get" call.
// Exactly one of *Report or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Report.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *JobsReportsGetCall) Do() (*Report, error) {
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
	ret := &Report{
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
	//   "description": "Gets the metadata of a specific report.",
	//   "httpMethod": "GET",
	//   "id": "youtubereporting.jobs.reports.get",
	//   "parameterOrder": [
	//     "jobId",
	//     "reportId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "The ID of the job.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "reportId": {
	//       "description": "The ID of the report to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/jobs/{jobId}/reports/{reportId}",
	//   "response": {
	//     "$ref": "Report"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}

// method id "youtubereporting.jobs.reports.list":

type JobsReportsListCall struct {
	s            *Service
	jobId        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists reports created by a specific job. Returns NOT_FOUND if
// the job does not exist.
func (r *JobsReportsService) List(jobId string) *JobsReportsListCall {
	c := &JobsReportsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.jobId = jobId
	return c
}

// CreatedAfter sets the optional parameter "createdAfter": If set, only
// reports created after the specified date/time are returned.
func (c *JobsReportsListCall) CreatedAfter(createdAfter string) *JobsReportsListCall {
	c.urlParams_.Set("createdAfter", createdAfter)
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The content owner's external ID on which
// behalf the user is acting on. If not set, the user is acting for
// himself (his own channel).
func (c *JobsReportsListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *JobsReportsListCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// Server may return fewer report types than requested. If unspecified,
// server will pick an appropriate default.
func (c *JobsReportsListCall) PageSize(pageSize int64) *JobsReportsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return. Typically,
// this is the value of ListReportsResponse.next_page_token returned in
// response to the previous call to the `ListReports` method.
func (c *JobsReportsListCall) PageToken(pageToken string) *JobsReportsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *JobsReportsListCall) QuotaUser(quotaUser string) *JobsReportsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *JobsReportsListCall) Fields(s ...googleapi.Field) *JobsReportsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *JobsReportsListCall) IfNoneMatch(entityTag string) *JobsReportsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *JobsReportsListCall) Context(ctx context.Context) *JobsReportsListCall {
	c.ctx_ = ctx
	return c
}

func (c *JobsReportsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/jobs/{jobId}/reports")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"jobId": c.jobId,
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

// Do executes the "youtubereporting.jobs.reports.list" call.
// Exactly one of *ListReportsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListReportsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *JobsReportsListCall) Do() (*ListReportsResponse, error) {
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
	ret := &ListReportsResponse{
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
	//   "description": "Lists reports created by a specific job. Returns NOT_FOUND if the job does not exist.",
	//   "httpMethod": "GET",
	//   "id": "youtubereporting.jobs.reports.list",
	//   "parameterOrder": [
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "createdAfter": {
	//       "description": "If set, only reports created after the specified date/time are returned.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "jobId": {
	//       "description": "The ID of the job.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. Server may return fewer report types than requested. If unspecified, server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return. Typically, this is the value of ListReportsResponse.next_page_token returned in response to the previous call to the `ListReports` method.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/jobs/{jobId}/reports",
	//   "response": {
	//     "$ref": "ListReportsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}

// method id "youtubereporting.media.download":

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

// Do executes the "youtubereporting.media.download" call.
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
	//   "id": "youtubereporting.media.download",
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
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "youtubereporting.reportTypes.list":

type ReportTypesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists report types.
func (r *ReportTypesService) List() *ReportTypesListCall {
	c := &ReportTypesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The content owner's external ID on which
// behalf the user is acting on. If not set, the user is acting for
// himself (his own channel).
func (c *ReportTypesListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *ReportTypesListCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size.
// Server may return fewer report types than requested. If unspecified,
// server will pick an appropriate default.
func (c *ReportTypesListCall) PageSize(pageSize int64) *ReportTypesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": A token
// identifying a page of results the server should return. Typically,
// this is the value of ListReportTypesResponse.next_page_token returned
// in response to the previous call to the `ListReportTypes` method.
func (c *ReportTypesListCall) PageToken(pageToken string) *ReportTypesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ReportTypesListCall) QuotaUser(quotaUser string) *ReportTypesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportTypesListCall) Fields(s ...googleapi.Field) *ReportTypesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReportTypesListCall) IfNoneMatch(entityTag string) *ReportTypesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportTypesListCall) Context(ctx context.Context) *ReportTypesListCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportTypesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/reportTypes")
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

// Do executes the "youtubereporting.reportTypes.list" call.
// Exactly one of *ListReportTypesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListReportTypesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ReportTypesListCall) Do() (*ListReportTypesResponse, error) {
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
	ret := &ListReportTypesResponse{
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
	//   "description": "Lists report types.",
	//   "httpMethod": "GET",
	//   "id": "youtubereporting.reportTypes.list",
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "The content owner's external ID on which behalf the user is acting on. If not set, the user is acting for himself (his own channel).",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size. Server may return fewer report types than requested. If unspecified, server will pick an appropriate default.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A token identifying a page of results the server should return. Typically, this is the value of ListReportTypesResponse.next_page_token returned in response to the previous call to the `ListReportTypes` method.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/reportTypes",
	//   "response": {
	//     "$ref": "ListReportTypesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}
