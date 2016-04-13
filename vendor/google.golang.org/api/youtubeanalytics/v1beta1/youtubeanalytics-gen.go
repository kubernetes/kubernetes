// Package youtubeanalytics provides access to the YouTube Analytics API.
//
// See http://developers.google.com/youtube/analytics/
//
// Usage example:
//
//   import "google.golang.org/api/youtubeanalytics/v1beta1"
//   ...
//   youtubeanalyticsService, err := youtubeanalytics.New(oauthHttpClient)
package youtubeanalytics // import "google.golang.org/api/youtubeanalytics/v1beta1"

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

const apiId = "youtubeAnalytics:v1beta1"
const apiName = "youtubeAnalytics"
const apiVersion = "v1beta1"
const basePath = "https://www.googleapis.com/youtube/analytics/v1beta1/"

// OAuth2 scopes used by this API.
const (
	// Manage your YouTube account
	YoutubeScope = "https://www.googleapis.com/auth/youtube"

	// View your YouTube account
	YoutubeReadonlyScope = "https://www.googleapis.com/auth/youtube.readonly"

	// View and manage your assets and associated content on YouTube
	YoutubepartnerScope = "https://www.googleapis.com/auth/youtubepartner"

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
	s.BatchReportDefinitions = NewBatchReportDefinitionsService(s)
	s.BatchReports = NewBatchReportsService(s)
	s.GroupItems = NewGroupItemsService(s)
	s.Groups = NewGroupsService(s)
	s.Reports = NewReportsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	BatchReportDefinitions *BatchReportDefinitionsService

	BatchReports *BatchReportsService

	GroupItems *GroupItemsService

	Groups *GroupsService

	Reports *ReportsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewBatchReportDefinitionsService(s *Service) *BatchReportDefinitionsService {
	rs := &BatchReportDefinitionsService{s: s}
	return rs
}

type BatchReportDefinitionsService struct {
	s *Service
}

func NewBatchReportsService(s *Service) *BatchReportsService {
	rs := &BatchReportsService{s: s}
	return rs
}

type BatchReportsService struct {
	s *Service
}

func NewGroupItemsService(s *Service) *GroupItemsService {
	rs := &GroupItemsService{s: s}
	return rs
}

type GroupItemsService struct {
	s *Service
}

func NewGroupsService(s *Service) *GroupsService {
	rs := &GroupsService{s: s}
	return rs
}

type GroupsService struct {
	s *Service
}

func NewReportsService(s *Service) *ReportsService {
	rs := &ReportsService{s: s}
	return rs
}

type ReportsService struct {
	s *Service
}

// BatchReport: Contains single batchReport resource.
type BatchReport struct {
	// Id: The ID that YouTube assigns and uses to uniquely identify the
	// report.
	Id string `json:"id,omitempty"`

	// Kind: This value specifies the type of data of this item. For batch
	// report the kind property value is youtubeAnalytics#batchReport.
	Kind string `json:"kind,omitempty"`

	// Outputs: Report outputs.
	Outputs []*BatchReportOutputs `json:"outputs,omitempty"`

	// ReportId: The ID of the the report definition.
	ReportId string `json:"reportId,omitempty"`

	// TimeSpan: Period included in the report. For reports containing all
	// entities endTime is not set. Both startTime and endTime are
	// inclusive.
	TimeSpan *BatchReportTimeSpan `json:"timeSpan,omitempty"`

	// TimeUpdated: The time when the report was updated.
	TimeUpdated string `json:"timeUpdated,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BatchReport) MarshalJSON() ([]byte, error) {
	type noMethod BatchReport
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type BatchReportOutputs struct {
	// DownloadUrl: Cloud storage URL to download this report. This URL is
	// valid for 30 minutes.
	DownloadUrl string `json:"downloadUrl,omitempty"`

	// Format: Format of the output.
	Format string `json:"format,omitempty"`

	// Type: Type of the output.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DownloadUrl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BatchReportOutputs) MarshalJSON() ([]byte, error) {
	type noMethod BatchReportOutputs
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// BatchReportTimeSpan: Period included in the report. For reports
// containing all entities endTime is not set. Both startTime and
// endTime are inclusive.
type BatchReportTimeSpan struct {
	// EndTime: End of the period included in the report. Inclusive. For
	// reports containing all entities endTime is not set.
	EndTime string `json:"endTime,omitempty"`

	// StartTime: Start of the period included in the report. Inclusive.
	StartTime string `json:"startTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BatchReportTimeSpan) MarshalJSON() ([]byte, error) {
	type noMethod BatchReportTimeSpan
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// BatchReportDefinition: Contains single batchReportDefinition
// resource.
type BatchReportDefinition struct {
	// Id: The ID that YouTube assigns and uses to uniquely identify the
	// report definition.
	Id string `json:"id,omitempty"`

	// Kind: This value specifies the type of data of this item. For batch
	// report definition the kind property value is
	// youtubeAnalytics#batchReportDefinition.
	Kind string `json:"kind,omitempty"`

	// Name: Name of the report definition.
	Name string `json:"name,omitempty"`

	// Status: Status of the report definition.
	Status string `json:"status,omitempty"`

	// Type: Type of the report definition.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BatchReportDefinition) MarshalJSON() ([]byte, error) {
	type noMethod BatchReportDefinition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// BatchReportDefinitionList: A paginated list of batchReportDefinition
// resources returned in response to a
// youtubeAnalytics.batchReportDefinitions.list request.
type BatchReportDefinitionList struct {
	// Items: A list of batchReportDefinition resources that match the
	// request criteria.
	Items []*BatchReportDefinition `json:"items,omitempty"`

	// Kind: This value specifies the type of data included in the API
	// response. For the list method, the kind property value is
	// youtubeAnalytics#batchReportDefinitionList.
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BatchReportDefinitionList) MarshalJSON() ([]byte, error) {
	type noMethod BatchReportDefinitionList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// BatchReportList: A paginated list of batchReport resources returned
// in response to a youtubeAnalytics.batchReport.list request.
type BatchReportList struct {
	// Items: A list of batchReport resources that match the request
	// criteria.
	Items []*BatchReport `json:"items,omitempty"`

	// Kind: This value specifies the type of data included in the API
	// response. For the list method, the kind property value is
	// youtubeAnalytics#batchReportList.
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *BatchReportList) MarshalJSON() ([]byte, error) {
	type noMethod BatchReportList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Group struct {
	ContentDetails *GroupContentDetails `json:"contentDetails,omitempty"`

	Etag string `json:"etag,omitempty"`

	Id string `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`

	Snippet *GroupSnippet `json:"snippet,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ContentDetails") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Group) MarshalJSON() ([]byte, error) {
	type noMethod Group
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type GroupContentDetails struct {
	ItemCount uint64 `json:"itemCount,omitempty,string"`

	ItemType string `json:"itemType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ItemCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GroupContentDetails) MarshalJSON() ([]byte, error) {
	type noMethod GroupContentDetails
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type GroupSnippet struct {
	PublishedAt string `json:"publishedAt,omitempty"`

	Title string `json:"title,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PublishedAt") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GroupSnippet) MarshalJSON() ([]byte, error) {
	type noMethod GroupSnippet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type GroupItem struct {
	Etag string `json:"etag,omitempty"`

	GroupId string `json:"groupId,omitempty"`

	Id string `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`

	Resource *GroupItemResource `json:"resource,omitempty"`

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
}

func (s *GroupItem) MarshalJSON() ([]byte, error) {
	type noMethod GroupItem
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type GroupItemResource struct {
	Id string `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GroupItemResource) MarshalJSON() ([]byte, error) {
	type noMethod GroupItemResource
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// GroupItemListResponse: A paginated list of grouList resources
// returned in response to a youtubeAnalytics.groupApi.list request.
type GroupItemListResponse struct {
	Etag string `json:"etag,omitempty"`

	Items []*GroupItem `json:"items,omitempty"`

	Kind string `json:"kind,omitempty"`

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
}

func (s *GroupItemListResponse) MarshalJSON() ([]byte, error) {
	type noMethod GroupItemListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// GroupListResponse: A paginated list of grouList resources returned in
// response to a youtubeAnalytics.groupApi.list request.
type GroupListResponse struct {
	Etag string `json:"etag,omitempty"`

	Items []*Group `json:"items,omitempty"`

	Kind string `json:"kind,omitempty"`

	NextPageToken string `json:"nextPageToken,omitempty"`

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
}

func (s *GroupListResponse) MarshalJSON() ([]byte, error) {
	type noMethod GroupListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ResultTable: Contains a single result table. The table is returned as
// an array of rows that contain the values for the cells of the table.
// Depending on the metric or dimension, the cell can contain a string
// (video ID, country code) or a number (number of views or number of
// likes).
type ResultTable struct {
	// ColumnHeaders: This value specifies information about the data
	// returned in the rows fields. Each item in the columnHeaders list
	// identifies a field returned in the rows value, which contains a list
	// of comma-delimited data. The columnHeaders list will begin with the
	// dimensions specified in the API request, which will be followed by
	// the metrics specified in the API request. The order of both
	// dimensions and metrics will match the ordering in the API request.
	// For example, if the API request contains the parameters
	// dimensions=ageGroup,gender&metrics=viewerPercentage, the API response
	// will return columns in this order: ageGroup,gender,viewerPercentage.
	ColumnHeaders []*ResultTableColumnHeaders `json:"columnHeaders,omitempty"`

	// Kind: This value specifies the type of data included in the API
	// response. For the query method, the kind property value will be
	// youtubeAnalytics#resultTable.
	Kind string `json:"kind,omitempty"`

	// Rows: The list contains all rows of the result table. Each item in
	// the list is an array that contains comma-delimited data corresponding
	// to a single row of data. The order of the comma-delimited data fields
	// will match the order of the columns listed in the columnHeaders
	// field. If no data is available for the given query, the rows element
	// will be omitted from the response. The response for a query with the
	// day dimension will not contain rows for the most recent days.
	Rows [][]interface{} `json:"rows,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ColumnHeaders") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ResultTable) MarshalJSON() ([]byte, error) {
	type noMethod ResultTable
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ResultTableColumnHeaders struct {
	// ColumnType: The type of the column (DIMENSION or METRIC).
	ColumnType string `json:"columnType,omitempty"`

	// DataType: The type of the data in the column (STRING, INTEGER, FLOAT,
	// etc.).
	DataType string `json:"dataType,omitempty"`

	// Name: The name of the dimension or metric.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ColumnType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ResultTableColumnHeaders) MarshalJSON() ([]byte, error) {
	type noMethod ResultTableColumnHeaders
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "youtubeAnalytics.batchReportDefinitions.list":

type BatchReportDefinitionsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves a list of available batch report definitions.
func (r *BatchReportDefinitionsService) List(onBehalfOfContentOwner string) *BatchReportDefinitionsListCall {
	c := &BatchReportDefinitionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *BatchReportDefinitionsListCall) QuotaUser(quotaUser string) *BatchReportDefinitionsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *BatchReportDefinitionsListCall) UserIP(userIP string) *BatchReportDefinitionsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BatchReportDefinitionsListCall) Fields(s ...googleapi.Field) *BatchReportDefinitionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BatchReportDefinitionsListCall) IfNoneMatch(entityTag string) *BatchReportDefinitionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BatchReportDefinitionsListCall) Context(ctx context.Context) *BatchReportDefinitionsListCall {
	c.ctx_ = ctx
	return c
}

func (c *BatchReportDefinitionsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "batchReportDefinitions")
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

// Do executes the "youtubeAnalytics.batchReportDefinitions.list" call.
// Exactly one of *BatchReportDefinitionList or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *BatchReportDefinitionList.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *BatchReportDefinitionsListCall) Do() (*BatchReportDefinitionList, error) {
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
	ret := &BatchReportDefinitionList{
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
	//   "description": "Retrieves a list of available batch report definitions.",
	//   "httpMethod": "GET",
	//   "id": "youtubeAnalytics.batchReportDefinitions.list",
	//   "parameterOrder": [
	//     "onBehalfOfContentOwner"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "The onBehalfOfContentOwner parameter identifies the content owner that the user is acting on behalf of.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "batchReportDefinitions",
	//   "response": {
	//     "$ref": "BatchReportDefinitionList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}

// method id "youtubeAnalytics.batchReports.list":

type BatchReportsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Retrieves a list of processed batch reports.
func (r *BatchReportsService) List(batchReportDefinitionId string, onBehalfOfContentOwner string) *BatchReportsListCall {
	c := &BatchReportsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("batchReportDefinitionId", batchReportDefinitionId)
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *BatchReportsListCall) QuotaUser(quotaUser string) *BatchReportsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *BatchReportsListCall) UserIP(userIP string) *BatchReportsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BatchReportsListCall) Fields(s ...googleapi.Field) *BatchReportsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BatchReportsListCall) IfNoneMatch(entityTag string) *BatchReportsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BatchReportsListCall) Context(ctx context.Context) *BatchReportsListCall {
	c.ctx_ = ctx
	return c
}

func (c *BatchReportsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "batchReports")
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

// Do executes the "youtubeAnalytics.batchReports.list" call.
// Exactly one of *BatchReportList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *BatchReportList.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *BatchReportsListCall) Do() (*BatchReportList, error) {
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
	ret := &BatchReportList{
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
	//   "description": "Retrieves a list of processed batch reports.",
	//   "httpMethod": "GET",
	//   "id": "youtubeAnalytics.batchReports.list",
	//   "parameterOrder": [
	//     "batchReportDefinitionId",
	//     "onBehalfOfContentOwner"
	//   ],
	//   "parameters": {
	//     "batchReportDefinitionId": {
	//       "description": "The batchReportDefinitionId parameter specifies the ID of the batch reportort definition for which you are retrieving reports.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "The onBehalfOfContentOwner parameter identifies the content owner that the user is acting on behalf of.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "batchReports",
	//   "response": {
	//     "$ref": "BatchReportList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}

// method id "youtubeAnalytics.groupItems.delete":

type GroupItemsDeleteCall struct {
	s          *Service
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Removes an item from a group.
func (r *GroupItemsService) Delete(id string) *GroupItemsDeleteCall {
	c := &GroupItemsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("id", id)
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner parameter indicates that the request's
// authorization credentials identify a YouTube CMS user who is acting
// on behalf of the content owner specified in the parameter value. This
// parameter is intended for YouTube content partners that own and
// manage many different YouTube channels. It allows content owners to
// authenticate once and get access to all their video and channel data,
// without having to provide authentication credentials for each
// individual channel. The CMS account that the user authenticates with
// must be linked to the specified YouTube content owner.
func (c *GroupItemsDeleteCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *GroupItemsDeleteCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GroupItemsDeleteCall) QuotaUser(quotaUser string) *GroupItemsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GroupItemsDeleteCall) UserIP(userIP string) *GroupItemsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupItemsDeleteCall) Fields(s ...googleapi.Field) *GroupItemsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupItemsDeleteCall) Context(ctx context.Context) *GroupItemsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *GroupItemsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "groupItems")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "youtubeAnalytics.groupItems.delete" call.
func (c *GroupItemsDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Removes an item from a group.",
	//   "httpMethod": "DELETE",
	//   "id": "youtubeAnalytics.groupItems.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies the YouTube group item ID for the group that is being deleted.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "groupItems",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtubeAnalytics.groupItems.insert":

type GroupItemsInsertCall struct {
	s          *Service
	groupitem  *GroupItem
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Creates a group item.
func (r *GroupItemsService) Insert(groupitem *GroupItem) *GroupItemsInsertCall {
	c := &GroupItemsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.groupitem = groupitem
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner parameter indicates that the request's
// authorization credentials identify a YouTube CMS user who is acting
// on behalf of the content owner specified in the parameter value. This
// parameter is intended for YouTube content partners that own and
// manage many different YouTube channels. It allows content owners to
// authenticate once and get access to all their video and channel data,
// without having to provide authentication credentials for each
// individual channel. The CMS account that the user authenticates with
// must be linked to the specified YouTube content owner.
func (c *GroupItemsInsertCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *GroupItemsInsertCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GroupItemsInsertCall) QuotaUser(quotaUser string) *GroupItemsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GroupItemsInsertCall) UserIP(userIP string) *GroupItemsInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupItemsInsertCall) Fields(s ...googleapi.Field) *GroupItemsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupItemsInsertCall) Context(ctx context.Context) *GroupItemsInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *GroupItemsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.groupitem)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "groupItems")
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

// Do executes the "youtubeAnalytics.groupItems.insert" call.
// Exactly one of *GroupItem or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *GroupItem.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *GroupItemsInsertCall) Do() (*GroupItem, error) {
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
	ret := &GroupItem{
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
	//   "description": "Creates a group item.",
	//   "httpMethod": "POST",
	//   "id": "youtubeAnalytics.groupItems.insert",
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "groupItems",
	//   "request": {
	//     "$ref": "GroupItem"
	//   },
	//   "response": {
	//     "$ref": "GroupItem"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtubeAnalytics.groupItems.list":

type GroupItemsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Returns a collection of group items that match the API request
// parameters.
func (r *GroupItemsService) List(groupId string) *GroupItemsListCall {
	c := &GroupItemsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("groupId", groupId)
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner parameter indicates that the request's
// authorization credentials identify a YouTube CMS user who is acting
// on behalf of the content owner specified in the parameter value. This
// parameter is intended for YouTube content partners that own and
// manage many different YouTube channels. It allows content owners to
// authenticate once and get access to all their video and channel data,
// without having to provide authentication credentials for each
// individual channel. The CMS account that the user authenticates with
// must be linked to the specified YouTube content owner.
func (c *GroupItemsListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *GroupItemsListCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GroupItemsListCall) QuotaUser(quotaUser string) *GroupItemsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GroupItemsListCall) UserIP(userIP string) *GroupItemsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupItemsListCall) Fields(s ...googleapi.Field) *GroupItemsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GroupItemsListCall) IfNoneMatch(entityTag string) *GroupItemsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupItemsListCall) Context(ctx context.Context) *GroupItemsListCall {
	c.ctx_ = ctx
	return c
}

func (c *GroupItemsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "groupItems")
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

// Do executes the "youtubeAnalytics.groupItems.list" call.
// Exactly one of *GroupItemListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GroupItemListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *GroupItemsListCall) Do() (*GroupItemListResponse, error) {
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
	ret := &GroupItemListResponse{
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
	//   "description": "Returns a collection of group items that match the API request parameters.",
	//   "httpMethod": "GET",
	//   "id": "youtubeAnalytics.groupItems.list",
	//   "parameterOrder": [
	//     "groupId"
	//   ],
	//   "parameters": {
	//     "groupId": {
	//       "description": "The id parameter specifies the unique ID of the group for which you want to retrieve group items.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "groupItems",
	//   "response": {
	//     "$ref": "GroupItemListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}

// method id "youtubeAnalytics.groups.delete":

type GroupsDeleteCall struct {
	s          *Service
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a group.
func (r *GroupsService) Delete(id string) *GroupsDeleteCall {
	c := &GroupsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("id", id)
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner parameter indicates that the request's
// authorization credentials identify a YouTube CMS user who is acting
// on behalf of the content owner specified in the parameter value. This
// parameter is intended for YouTube content partners that own and
// manage many different YouTube channels. It allows content owners to
// authenticate once and get access to all their video and channel data,
// without having to provide authentication credentials for each
// individual channel. The CMS account that the user authenticates with
// must be linked to the specified YouTube content owner.
func (c *GroupsDeleteCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *GroupsDeleteCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GroupsDeleteCall) QuotaUser(quotaUser string) *GroupsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GroupsDeleteCall) UserIP(userIP string) *GroupsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsDeleteCall) Fields(s ...googleapi.Field) *GroupsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsDeleteCall) Context(ctx context.Context) *GroupsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *GroupsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "groups")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "youtubeAnalytics.groups.delete" call.
func (c *GroupsDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes a group.",
	//   "httpMethod": "DELETE",
	//   "id": "youtubeAnalytics.groups.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies the YouTube group ID for the group that is being deleted.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "groups",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtubeAnalytics.groups.insert":

type GroupsInsertCall struct {
	s          *Service
	group      *Group
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Insert: Creates a group.
func (r *GroupsService) Insert(group *Group) *GroupsInsertCall {
	c := &GroupsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.group = group
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner parameter indicates that the request's
// authorization credentials identify a YouTube CMS user who is acting
// on behalf of the content owner specified in the parameter value. This
// parameter is intended for YouTube content partners that own and
// manage many different YouTube channels. It allows content owners to
// authenticate once and get access to all their video and channel data,
// without having to provide authentication credentials for each
// individual channel. The CMS account that the user authenticates with
// must be linked to the specified YouTube content owner.
func (c *GroupsInsertCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *GroupsInsertCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GroupsInsertCall) QuotaUser(quotaUser string) *GroupsInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GroupsInsertCall) UserIP(userIP string) *GroupsInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsInsertCall) Fields(s ...googleapi.Field) *GroupsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsInsertCall) Context(ctx context.Context) *GroupsInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *GroupsInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.group)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "groups")
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

// Do executes the "youtubeAnalytics.groups.insert" call.
// Exactly one of *Group or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Group.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *GroupsInsertCall) Do() (*Group, error) {
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
	ret := &Group{
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
	//   "description": "Creates a group.",
	//   "httpMethod": "POST",
	//   "id": "youtubeAnalytics.groups.insert",
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "groups",
	//   "request": {
	//     "$ref": "Group"
	//   },
	//   "response": {
	//     "$ref": "Group"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtubeAnalytics.groups.list":

type GroupsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Returns a collection of groups that match the API request
// parameters. For example, you can retrieve all groups that the
// authenticated user owns, or you can retrieve one or more groups by
// their unique IDs.
func (r *GroupsService) List() *GroupsListCall {
	c := &GroupsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// Id sets the optional parameter "id": The id parameter specifies a
// comma-separated list of the YouTube group ID(s) for the resource(s)
// that are being retrieved. In a group resource, the id property
// specifies the group's YouTube group ID.
func (c *GroupsListCall) Id(id string) *GroupsListCall {
	c.urlParams_.Set("id", id)
	return c
}

// Mine sets the optional parameter "mine": Set this parameter's value
// to true to instruct the API to only return groups owned by the
// authenticated user.
func (c *GroupsListCall) Mine(mine bool) *GroupsListCall {
	c.urlParams_.Set("mine", fmt.Sprint(mine))
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner parameter indicates that the request's
// authorization credentials identify a YouTube CMS user who is acting
// on behalf of the content owner specified in the parameter value. This
// parameter is intended for YouTube content partners that own and
// manage many different YouTube channels. It allows content owners to
// authenticate once and get access to all their video and channel data,
// without having to provide authentication credentials for each
// individual channel. The CMS account that the user authenticates with
// must be linked to the specified YouTube content owner.
func (c *GroupsListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *GroupsListCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// PageToken sets the optional parameter "pageToken": The pageToken
// parameter identifies a specific page in the result set that should be
// returned. In an API response, the nextPageToken property identifies
// the next page that can be retrieved.
func (c *GroupsListCall) PageToken(pageToken string) *GroupsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GroupsListCall) QuotaUser(quotaUser string) *GroupsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GroupsListCall) UserIP(userIP string) *GroupsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsListCall) Fields(s ...googleapi.Field) *GroupsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GroupsListCall) IfNoneMatch(entityTag string) *GroupsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsListCall) Context(ctx context.Context) *GroupsListCall {
	c.ctx_ = ctx
	return c
}

func (c *GroupsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "groups")
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

// Do executes the "youtubeAnalytics.groups.list" call.
// Exactly one of *GroupListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *GroupListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *GroupsListCall) Do() (*GroupListResponse, error) {
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
	ret := &GroupListResponse{
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
	//   "description": "Returns a collection of groups that match the API request parameters. For example, you can retrieve all groups that the authenticated user owns, or you can retrieve one or more groups by their unique IDs.",
	//   "httpMethod": "GET",
	//   "id": "youtubeAnalytics.groups.list",
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of the YouTube group ID(s) for the resource(s) that are being retrieved. In a group resource, the id property specifies the group's YouTube group ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "mine": {
	//       "description": "Set this parameter's value to true to instruct the API to only return groups owned by the authenticated user.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The pageToken parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken property identifies the next page that can be retrieved.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "groups",
	//   "response": {
	//     "$ref": "GroupListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}

// method id "youtubeAnalytics.groups.update":

type GroupsUpdateCall struct {
	s          *Service
	group      *Group
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Modifies a group. For example, you could change a group's
// title.
func (r *GroupsService) Update(group *Group) *GroupsUpdateCall {
	c := &GroupsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.group = group
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner parameter indicates that the request's
// authorization credentials identify a YouTube CMS user who is acting
// on behalf of the content owner specified in the parameter value. This
// parameter is intended for YouTube content partners that own and
// manage many different YouTube channels. It allows content owners to
// authenticate once and get access to all their video and channel data,
// without having to provide authentication credentials for each
// individual channel. The CMS account that the user authenticates with
// must be linked to the specified YouTube content owner.
func (c *GroupsUpdateCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *GroupsUpdateCall {
	c.urlParams_.Set("onBehalfOfContentOwner", onBehalfOfContentOwner)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GroupsUpdateCall) QuotaUser(quotaUser string) *GroupsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GroupsUpdateCall) UserIP(userIP string) *GroupsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsUpdateCall) Fields(s ...googleapi.Field) *GroupsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsUpdateCall) Context(ctx context.Context) *GroupsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *GroupsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.group)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "groups")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "youtubeAnalytics.groups.update" call.
// Exactly one of *Group or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Group.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *GroupsUpdateCall) Do() (*Group, error) {
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
	ret := &Group{
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
	//   "description": "Modifies a group. For example, you could change a group's title.",
	//   "httpMethod": "PUT",
	//   "id": "youtubeAnalytics.groups.update",
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "groups",
	//   "request": {
	//     "$ref": "Group"
	//   },
	//   "response": {
	//     "$ref": "Group"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtubeAnalytics.reports.query":

type ReportsQueryCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Query: Retrieve your YouTube Analytics reports.
func (r *ReportsService) Query(ids string, startDate string, endDate string, metrics string) *ReportsQueryCall {
	c := &ReportsQueryCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.urlParams_.Set("ids", ids)
	c.urlParams_.Set("start-date", startDate)
	c.urlParams_.Set("end-date", endDate)
	c.urlParams_.Set("metrics", metrics)
	return c
}

// Currency sets the optional parameter "currency": The currency to
// which financial metrics should be converted. The default is US Dollar
// (USD). If the result contains no financial metrics, this flag will be
// ignored. Responds with an error if the specified currency is not
// recognized.
func (c *ReportsQueryCall) Currency(currency string) *ReportsQueryCall {
	c.urlParams_.Set("currency", currency)
	return c
}

// Dimensions sets the optional parameter "dimensions": A
// comma-separated list of YouTube Analytics dimensions, such as views
// or ageGroup,gender. See the Available Reports document for a list of
// the reports that you can retrieve and the dimensions used for those
// reports. Also see the Dimensions document for definitions of those
// dimensions.
func (c *ReportsQueryCall) Dimensions(dimensions string) *ReportsQueryCall {
	c.urlParams_.Set("dimensions", dimensions)
	return c
}

// Filters sets the optional parameter "filters": A list of filters that
// should be applied when retrieving YouTube Analytics data. The
// Available Reports document identifies the dimensions that can be used
// to filter each report, and the Dimensions document defines those
// dimensions. If a request uses multiple filters, join them together
// with a semicolon (;), and the returned result table will satisfy both
// filters. For example, a filters parameter value of
// video==dMH0bHeiRNg;country==IT restricts the result set to include
// data for the given video in Italy.
func (c *ReportsQueryCall) Filters(filters string) *ReportsQueryCall {
	c.urlParams_.Set("filters", filters)
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of rows to include in the response.
func (c *ReportsQueryCall) MaxResults(maxResults int64) *ReportsQueryCall {
	c.urlParams_.Set("max-results", fmt.Sprint(maxResults))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *ReportsQueryCall) QuotaUser(quotaUser string) *ReportsQueryCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Sort sets the optional parameter "sort": A comma-separated list of
// dimensions or metrics that determine the sort order for YouTube
// Analytics data. By default the sort order is ascending. The '-'
// prefix causes descending sort order.
func (c *ReportsQueryCall) Sort(sort string) *ReportsQueryCall {
	c.urlParams_.Set("sort", sort)
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first entity to retrieve. Use this parameter as a pagination
// mechanism along with the max-results parameter (one-based,
// inclusive).
func (c *ReportsQueryCall) StartIndex(startIndex int64) *ReportsQueryCall {
	c.urlParams_.Set("start-index", fmt.Sprint(startIndex))
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *ReportsQueryCall) UserIP(userIP string) *ReportsQueryCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ReportsQueryCall) Fields(s ...googleapi.Field) *ReportsQueryCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ReportsQueryCall) IfNoneMatch(entityTag string) *ReportsQueryCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ReportsQueryCall) Context(ctx context.Context) *ReportsQueryCall {
	c.ctx_ = ctx
	return c
}

func (c *ReportsQueryCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports")
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

// Do executes the "youtubeAnalytics.reports.query" call.
// Exactly one of *ResultTable or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ResultTable.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ReportsQueryCall) Do() (*ResultTable, error) {
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
	ret := &ResultTable{
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
	//   "description": "Retrieve your YouTube Analytics reports.",
	//   "httpMethod": "GET",
	//   "id": "youtubeAnalytics.reports.query",
	//   "parameterOrder": [
	//     "ids",
	//     "start-date",
	//     "end-date",
	//     "metrics"
	//   ],
	//   "parameters": {
	//     "currency": {
	//       "description": "The currency to which financial metrics should be converted. The default is US Dollar (USD). If the result contains no financial metrics, this flag will be ignored. Responds with an error if the specified currency is not recognized.",
	//       "location": "query",
	//       "pattern": "[A-Z]{3}",
	//       "type": "string"
	//     },
	//     "dimensions": {
	//       "description": "A comma-separated list of YouTube Analytics dimensions, such as views or ageGroup,gender. See the Available Reports document for a list of the reports that you can retrieve and the dimensions used for those reports. Also see the Dimensions document for definitions of those dimensions.",
	//       "location": "query",
	//       "pattern": "[0-9a-zA-Z,]+",
	//       "type": "string"
	//     },
	//     "end-date": {
	//       "description": "The end date for fetching YouTube Analytics data. The value should be in YYYY-MM-DD format.",
	//       "location": "query",
	//       "pattern": "[0-9]{4}-[0-9]{2}-[0-9]{2}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filters": {
	//       "description": "A list of filters that should be applied when retrieving YouTube Analytics data. The Available Reports document identifies the dimensions that can be used to filter each report, and the Dimensions document defines those dimensions. If a request uses multiple filters, join them together with a semicolon (;), and the returned result table will satisfy both filters. For example, a filters parameter value of video==dMH0bHeiRNg;country==IT restricts the result set to include data for the given video in Italy.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ids": {
	//       "description": "Identifies the YouTube channel or content owner for which you are retrieving YouTube Analytics data.\n- To request data for a YouTube user, set the ids parameter value to channel==CHANNEL_ID, where CHANNEL_ID specifies the unique YouTube channel ID.\n- To request data for a YouTube CMS content owner, set the ids parameter value to contentOwner==OWNER_NAME, where OWNER_NAME is the CMS name of the content owner.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z]+==[a-zA-Z0-9_+-]+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of rows to include in the response.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "metrics": {
	//       "description": "A comma-separated list of YouTube Analytics metrics, such as views or likes,dislikes. See the Available Reports document for a list of the reports that you can retrieve and the metrics available in each report, and see the Metrics document for definitions of those metrics.",
	//       "location": "query",
	//       "pattern": "[0-9a-zA-Z,]+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sort": {
	//       "description": "A comma-separated list of dimensions or metrics that determine the sort order for YouTube Analytics data. By default the sort order is ascending. The '-' prefix causes descending sort order.",
	//       "location": "query",
	//       "pattern": "[-0-9a-zA-Z,]+",
	//       "type": "string"
	//     },
	//     "start-date": {
	//       "description": "The start date for fetching YouTube Analytics data. The value should be in YYYY-MM-DD format.",
	//       "location": "query",
	//       "pattern": "[0-9]{4}-[0-9]{2}-[0-9]{2}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "start-index": {
	//       "description": "An index of the first entity to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter (one-based, inclusive).",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "reports",
	//   "response": {
	//     "$ref": "ResultTable"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}
