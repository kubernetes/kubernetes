// Package youtubeanalytics provides access to the YouTube Analytics API.
//
// See http://developers.google.com/youtube/analytics/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/youtubeanalytics/v1beta1"
//   ...
//   youtubeanalyticsService, err := youtubeanalytics.New(oauthHttpClient)
package youtubeanalytics

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

const apiId = "youtubeAnalytics:v1beta1"
const apiName = "youtubeAnalytics"
const apiVersion = "v1beta1"
const basePath = "https://www.googleapis.com/youtube/analytics/v1beta1/"

// OAuth2 scopes used by this API.
const (
	// View YouTube Analytics monetary reports for your YouTube content
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
	s.Reports = NewReportsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	BatchReportDefinitions *BatchReportDefinitionsService

	BatchReports *BatchReportsService

	Reports *ReportsService
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

func NewReportsService(s *Service) *ReportsService {
	rs := &ReportsService{s: s}
	return rs
}

type ReportsService struct {
	s *Service
}

type BatchReportDefinitionList struct {
	// Items: A list of batchReportDefinition resources that match the
	// request criteria.
	Items []*BatchReportDefinitionTemplate `json:"items,omitempty"`

	// Kind: This value specifies the type of data included in the API
	// response. For the list method, the kind property value is
	// youtubeAnalytics#batchReportDefinitionList.
	Kind string `json:"kind,omitempty"`
}

type BatchReportDefinitionTemplate struct {
	// DefaultOutput: Default report definition's output.
	DefaultOutput []*BatchReportDefinitionTemplateDefaultOutput `json:"defaultOutput,omitempty"`

	// Id: The ID that YouTube assigns and uses to uniquely identify the
	// report definition.
	Id string `json:"id,omitempty"`

	// Name: Name of the report definition.
	Name string `json:"name,omitempty"`

	// Status: Status of the report definition.
	Status string `json:"status,omitempty"`

	// Type: Type of the report definition.
	Type string `json:"type,omitempty"`
}

type BatchReportDefinitionTemplateDefaultOutput struct {
	// Format: Format of the output.
	Format string `json:"format,omitempty"`

	// Type: Type of the output.
	Type string `json:"type,omitempty"`
}

type BatchReportList struct {
	// Items: A list of batchReport resources that match the request
	// criteria.
	Items []*BatchReportTemplate `json:"items,omitempty"`

	// Kind: This value specifies the type of data included in the API
	// response. For the list method, the kind property value is
	// youtubeAnalytics#batchReportList.
	Kind string `json:"kind,omitempty"`
}

type BatchReportTemplate struct {
	// Id: The ID that YouTube assigns and uses to uniquely identify the
	// report.
	Id string `json:"id,omitempty"`

	// Outputs: Report outputs.
	Outputs []*BatchReportTemplateOutputs `json:"outputs,omitempty"`

	// Report_id: The ID of the the report definition.
	Report_id string `json:"report_id,omitempty"`

	// TimeSpan: Period included in the report. For reports containing all
	// entities endTime is not set. Both startTime and endTime are
	// inclusive.
	TimeSpan *BatchReportTemplateTimeSpan `json:"timeSpan,omitempty"`

	// TimeUpdated: The time when the report was updated.
	TimeUpdated string `json:"timeUpdated,omitempty"`
}

type BatchReportTemplateOutputs struct {
	// DownloadUrl: Cloud storage URL to download this report. This URL is
	// valid for 30 minutes.
	DownloadUrl string `json:"downloadUrl,omitempty"`

	// Format: Format of the output.
	Format string `json:"format,omitempty"`

	// Type: Type of the output.
	Type string `json:"type,omitempty"`
}

type BatchReportTemplateTimeSpan struct {
	// EndTime: End of the period included in the report. Inclusive. For
	// reports containing all entities endTime is not set.
	EndTime string `json:"endTime,omitempty"`

	// StartTime: Start of the period included in the report. Inclusive.
	StartTime string `json:"startTime,omitempty"`
}

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
}

type ResultTableColumnHeaders struct {
	// ColumnType: The type of the column (DIMENSION or METRIC).
	ColumnType string `json:"columnType,omitempty"`

	// DataType: The type of the data in the column (STRING, INTEGER, FLOAT,
	// etc.).
	DataType string `json:"dataType,omitempty"`

	// Name: The name of the dimension or metric.
	Name string `json:"name,omitempty"`
}

// method id "youtubeAnalytics.batchReportDefinitions.list":

type BatchReportDefinitionsListCall struct {
	s                      *Service
	onBehalfOfContentOwner string
	opt_                   map[string]interface{}
}

// List: Retrieves a list of available batch report definitions.
func (r *BatchReportDefinitionsService) List(onBehalfOfContentOwner string) *BatchReportDefinitionsListCall {
	c := &BatchReportDefinitionsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.onBehalfOfContentOwner = onBehalfOfContentOwner
	return c
}

func (c *BatchReportDefinitionsListCall) Do() (*BatchReportDefinitionList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", c.onBehalfOfContentOwner))
	urls := googleapi.ResolveRelative(c.s.BasePath, "batchReportDefinitions")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(BatchReportDefinitionList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	s                       *Service
	batchReportDefinitionId string
	onBehalfOfContentOwner  string
	opt_                    map[string]interface{}
}

// List: Retrieves a list of processed batch reports.
func (r *BatchReportsService) List(batchReportDefinitionId string, onBehalfOfContentOwner string) *BatchReportsListCall {
	c := &BatchReportsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.batchReportDefinitionId = batchReportDefinitionId
	c.onBehalfOfContentOwner = onBehalfOfContentOwner
	return c
}

func (c *BatchReportsListCall) Do() (*BatchReportList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("batchReportDefinitionId", fmt.Sprintf("%v", c.batchReportDefinitionId))
	params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", c.onBehalfOfContentOwner))
	urls := googleapi.ResolveRelative(c.s.BasePath, "batchReports")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(BatchReportList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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

// method id "youtubeAnalytics.reports.query":

type ReportsQueryCall struct {
	s         *Service
	ids       string
	startDate string
	endDate   string
	metrics   string
	opt_      map[string]interface{}
}

// Query: Retrieve your YouTube Analytics reports.
func (r *ReportsService) Query(ids string, startDate string, endDate string, metrics string) *ReportsQueryCall {
	c := &ReportsQueryCall{s: r.s, opt_: make(map[string]interface{})}
	c.ids = ids
	c.startDate = startDate
	c.endDate = endDate
	c.metrics = metrics
	return c
}

// Dimensions sets the optional parameter "dimensions": A
// comma-separated list of YouTube Analytics dimensions, such as views
// or ageGroup,gender. See the Available Reports document for a list of
// the reports that you can retrieve and the dimensions used for those
// reports. Also see the Dimensions document for definitions of those
// dimensions.
func (c *ReportsQueryCall) Dimensions(dimensions string) *ReportsQueryCall {
	c.opt_["dimensions"] = dimensions
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
	c.opt_["filters"] = filters
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of rows to include in the response.
func (c *ReportsQueryCall) MaxResults(maxResults int64) *ReportsQueryCall {
	c.opt_["max-results"] = maxResults
	return c
}

// Sort sets the optional parameter "sort": A comma-separated list of
// dimensions or metrics that determine the sort order for YouTube
// Analytics data. By default the sort order is ascending. The '-'
// prefix causes descending sort order.
func (c *ReportsQueryCall) Sort(sort string) *ReportsQueryCall {
	c.opt_["sort"] = sort
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first entity to retrieve. Use this parameter as a pagination
// mechanism along with the max-results parameter (one-based,
// inclusive).
func (c *ReportsQueryCall) StartIndex(startIndex int64) *ReportsQueryCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ReportsQueryCall) Do() (*ResultTable, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("end-date", fmt.Sprintf("%v", c.endDate))
	params.Set("ids", fmt.Sprintf("%v", c.ids))
	params.Set("metrics", fmt.Sprintf("%v", c.metrics))
	params.Set("start-date", fmt.Sprintf("%v", c.startDate))
	if v, ok := c.opt_["dimensions"]; ok {
		params.Set("dimensions", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["filters"]; ok {
		params.Set("filters", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sort"]; ok {
		params.Set("sort", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ResultTable)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	//     "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
	//     "https://www.googleapis.com/auth/yt-analytics.readonly"
	//   ]
	// }

}
