// Package doubleclickbidmanager provides access to the DoubleClick Bid Manager API.
//
// See https://developers.google.com/bid-manager/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/doubleclickbidmanager/v1"
//   ...
//   doubleclickbidmanagerService, err := doubleclickbidmanager.New(oauthHttpClient)
package doubleclickbidmanager

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

const apiId = "doubleclickbidmanager:v1"
const apiName = "doubleclickbidmanager"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/doubleclickbidmanager/v1/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Lineitems = NewLineitemsService(s)
	s.Queries = NewQueriesService(s)
	s.Reports = NewReportsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Lineitems *LineitemsService

	Queries *QueriesService

	Reports *ReportsService
}

func NewLineitemsService(s *Service) *LineitemsService {
	rs := &LineitemsService{s: s}
	return rs
}

type LineitemsService struct {
	s *Service
}

func NewQueriesService(s *Service) *QueriesService {
	rs := &QueriesService{s: s}
	return rs
}

type QueriesService struct {
	s *Service
}

func NewReportsService(s *Service) *ReportsService {
	rs := &ReportsService{s: s}
	return rs
}

type ReportsService struct {
	s *Service
}

type DownloadLineItemsRequest struct {
	// FilterIds: Ids of the specified filter type used to filter line items
	// to fetch. If omitted, all the line items will be returned.
	FilterIds googleapi.Int64s `json:"filterIds,omitempty"`

	// FilterType: Filter type used to filter line items to fetch.
	FilterType string `json:"filterType,omitempty"`

	// Format: Format in which the line items will be returned. Default to
	// CSV.
	Format string `json:"format,omitempty"`
}

type DownloadLineItemsResponse struct {
	// LineItems: Retrieved line items in CSV format. Refer to  Entity Write
	// File Format for more information on file format.
	LineItems string `json:"lineItems,omitempty"`
}

type FilterPair struct {
	// Type: Filter type.
	Type string `json:"type,omitempty"`

	// Value: Filter value.
	Value string `json:"value,omitempty"`
}

type ListQueriesResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "doubleclickbidmanager#listQueriesResponse".
	Kind string `json:"kind,omitempty"`

	// Queries: Retrieved queries.
	Queries []*Query `json:"queries,omitempty"`
}

type ListReportsResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "doubleclickbidmanager#listReportsResponse".
	Kind string `json:"kind,omitempty"`

	// Reports: Retrieved reports.
	Reports []*Report `json:"reports,omitempty"`
}

type Parameters struct {
	// Filters: Filters used to match traffic data in your report.
	Filters []*FilterPair `json:"filters,omitempty"`

	// GroupBys: Data is grouped by the filters listed in this field.
	GroupBys []string `json:"groupBys,omitempty"`

	// IncludeInviteData: Whether to include data from Invite Media.
	IncludeInviteData bool `json:"includeInviteData,omitempty"`

	// Metrics: Metrics to include as columns in your report.
	Metrics []string `json:"metrics,omitempty"`

	// Type: Report type.
	Type string `json:"type,omitempty"`
}

type Query struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "doubleclickbidmanager#query".
	Kind string `json:"kind,omitempty"`

	// Metadata: Query metadata.
	Metadata *QueryMetadata `json:"metadata,omitempty"`

	// Params: Query parameters.
	Params *Parameters `json:"params,omitempty"`

	// QueryId: Query ID.
	QueryId int64 `json:"queryId,omitempty,string"`

	// ReportDataEndTimeMs: The ending time for the data that is shown in
	// the report. Note, reportDataEndTimeMs is required if
	// metadata.dataRange is CUSTOM_DATES and ignored otherwise.
	ReportDataEndTimeMs int64 `json:"reportDataEndTimeMs,omitempty,string"`

	// ReportDataStartTimeMs: The starting time for the data that is shown
	// in the report. Note, reportDataStartTimeMs is required if
	// metadata.dataRange is CUSTOM_DATES and ignored otherwise.
	ReportDataStartTimeMs int64 `json:"reportDataStartTimeMs,omitempty,string"`

	// Schedule: Information on how often and when to run a query.
	Schedule *QuerySchedule `json:"schedule,omitempty"`

	// TimezoneCode: Canonical timezone code for report data time. Defaults
	// to America/New_York.
	TimezoneCode string `json:"timezoneCode,omitempty"`
}

type QueryMetadata struct {
	// DataRange: Range of report data.
	DataRange string `json:"dataRange,omitempty"`

	// Format: Format of the generated report.
	Format string `json:"format,omitempty"`

	// GoogleCloudStoragePathForLatestReport: The path to the location in
	// Google Cloud Storage where the latest report is stored.
	GoogleCloudStoragePathForLatestReport string `json:"googleCloudStoragePathForLatestReport,omitempty"`

	// GoogleDrivePathForLatestReport: The path in Google Drive for the
	// latest report.
	GoogleDrivePathForLatestReport string `json:"googleDrivePathForLatestReport,omitempty"`

	// LatestReportRunTimeMs: The time when the latest report started to
	// run.
	LatestReportRunTimeMs int64 `json:"latestReportRunTimeMs,omitempty,string"`

	// ReportCount: Number of reports that have been generated for the
	// query.
	ReportCount int64 `json:"reportCount,omitempty"`

	// Running: Whether the latest report is currently running.
	Running bool `json:"running,omitempty"`

	// SendNotification: Whether to send an email notification when a report
	// is ready. Default to false.
	SendNotification bool `json:"sendNotification,omitempty"`

	// ShareEmailAddress: List of email addresses which are sent email
	// notifications when the report is finished. Separate from
	// sendNotification.
	ShareEmailAddress []string `json:"shareEmailAddress,omitempty"`

	// Title: Query title. It is used to name the reports generated from
	// this query.
	Title string `json:"title,omitempty"`
}

type QuerySchedule struct {
	// EndTimeMs: Datetime to periodically run the query until.
	EndTimeMs int64 `json:"endTimeMs,omitempty,string"`

	// Frequency: How often the query is run.
	Frequency string `json:"frequency,omitempty"`

	// NextRunMinuteOfDay: Time of day at which a new report will be
	// generated, represented as minutes past midnight Range is 0 to 1439.
	// Only applies to scheduled reports.
	NextRunMinuteOfDay int64 `json:"nextRunMinuteOfDay,omitempty"`

	// NextRunTimezoneCode: Canonical timezone code for report generation
	// time. Defaults to America/New_York.
	NextRunTimezoneCode string `json:"nextRunTimezoneCode,omitempty"`
}

type Report struct {
	// Key: Key used to identify a report.
	Key *ReportKey `json:"key,omitempty"`

	// Metadata: Report metadata.
	Metadata *ReportMetadata `json:"metadata,omitempty"`

	// Params: Report parameters.
	Params *Parameters `json:"params,omitempty"`
}

type ReportFailure struct {
	// ErrorCode: Error code that shows why the report was not created.
	ErrorCode string `json:"errorCode,omitempty"`
}

type ReportKey struct {
	// QueryId: Query ID.
	QueryId int64 `json:"queryId,omitempty,string"`

	// ReportId: Report ID.
	ReportId int64 `json:"reportId,omitempty,string"`
}

type ReportMetadata struct {
	// GoogleCloudStoragePath: The path to the location in Google Cloud
	// Storage where the report is stored.
	GoogleCloudStoragePath string `json:"googleCloudStoragePath,omitempty"`

	// ReportDataEndTimeMs: The ending time for the data that is shown in
	// the report.
	ReportDataEndTimeMs int64 `json:"reportDataEndTimeMs,omitempty,string"`

	// ReportDataStartTimeMs: The starting time for the data that is shown
	// in the report.
	ReportDataStartTimeMs int64 `json:"reportDataStartTimeMs,omitempty,string"`

	// Status: Report status.
	Status *ReportStatus `json:"status,omitempty"`
}

type ReportStatus struct {
	// Failure: If the report failed, this records the cause.
	Failure *ReportFailure `json:"failure,omitempty"`

	// FinishTimeMs: The time when this report either completed successfully
	// or failed.
	FinishTimeMs int64 `json:"finishTimeMs,omitempty,string"`

	// Format: The file type of the report.
	Format string `json:"format,omitempty"`

	// State: The state of the report.
	State string `json:"state,omitempty"`
}

type RowStatus struct {
	// Changed: Whether the stored entity is changed as a result of upload.
	Changed bool `json:"changed,omitempty"`

	// EntityId: Entity Id.
	EntityId int64 `json:"entityId,omitempty,string"`

	// EntityName: Entity name.
	EntityName string `json:"entityName,omitempty"`

	// Errors: Reasons why the entity can't be uploaded.
	Errors []string `json:"errors,omitempty"`

	// Persisted: Whether the entity is persisted.
	Persisted bool `json:"persisted,omitempty"`

	// RowNumber: Row number.
	RowNumber int64 `json:"rowNumber,omitempty"`
}

type RunQueryRequest struct {
	// DataRange: Report data range used to generate the report.
	DataRange string `json:"dataRange,omitempty"`

	// ReportDataEndTimeMs: The ending time for the data that is shown in
	// the report. Note, reportDataEndTimeMs is required if dataRange is
	// CUSTOM_DATES and ignored otherwise.
	ReportDataEndTimeMs int64 `json:"reportDataEndTimeMs,omitempty,string"`

	// ReportDataStartTimeMs: The starting time for the data that is shown
	// in the report. Note, reportDataStartTimeMs is required if dataRange
	// is CUSTOM_DATES and ignored otherwise.
	ReportDataStartTimeMs int64 `json:"reportDataStartTimeMs,omitempty,string"`

	// TimezoneCode: Canonical timezone code for report data time. Defaults
	// to America/New_York.
	TimezoneCode string `json:"timezoneCode,omitempty"`
}

type UploadLineItemsRequest struct {
	// DryRun: Set to true to get upload status without actually persisting
	// the line items.
	DryRun bool `json:"dryRun,omitempty"`

	// Format: Format the line items are in. Default to CSV.
	Format string `json:"format,omitempty"`

	// LineItems: Line items in CSV to upload. Refer to  Entity Write File
	// Format for more information on file format.
	LineItems string `json:"lineItems,omitempty"`
}

type UploadLineItemsResponse struct {
	// UploadStatus: Status of upload.
	UploadStatus *UploadStatus `json:"uploadStatus,omitempty"`
}

type UploadStatus struct {
	// Errors: Reasons why upload can't be completed.
	Errors []string `json:"errors,omitempty"`

	// RowStatus: Per-row upload status.
	RowStatus []*RowStatus `json:"rowStatus,omitempty"`
}

// method id "doubleclickbidmanager.lineitems.downloadlineitems":

type LineitemsDownloadlineitemsCall struct {
	s                        *Service
	downloadlineitemsrequest *DownloadLineItemsRequest
	opt_                     map[string]interface{}
}

// Downloadlineitems: Retrieves line items in CSV format.
func (r *LineitemsService) Downloadlineitems(downloadlineitemsrequest *DownloadLineItemsRequest) *LineitemsDownloadlineitemsCall {
	c := &LineitemsDownloadlineitemsCall{s: r.s, opt_: make(map[string]interface{})}
	c.downloadlineitemsrequest = downloadlineitemsrequest
	return c
}

func (c *LineitemsDownloadlineitemsCall) Do() (*DownloadLineItemsResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.downloadlineitemsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "lineitems/downloadlineitems")
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
	ret := new(DownloadLineItemsResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves line items in CSV format.",
	//   "httpMethod": "POST",
	//   "id": "doubleclickbidmanager.lineitems.downloadlineitems",
	//   "path": "lineitems/downloadlineitems",
	//   "request": {
	//     "$ref": "DownloadLineItemsRequest"
	//   },
	//   "response": {
	//     "$ref": "DownloadLineItemsResponse"
	//   }
	// }

}

// method id "doubleclickbidmanager.lineitems.uploadlineitems":

type LineitemsUploadlineitemsCall struct {
	s                      *Service
	uploadlineitemsrequest *UploadLineItemsRequest
	opt_                   map[string]interface{}
}

// Uploadlineitems: Uploads line items in CSV format.
func (r *LineitemsService) Uploadlineitems(uploadlineitemsrequest *UploadLineItemsRequest) *LineitemsUploadlineitemsCall {
	c := &LineitemsUploadlineitemsCall{s: r.s, opt_: make(map[string]interface{})}
	c.uploadlineitemsrequest = uploadlineitemsrequest
	return c
}

func (c *LineitemsUploadlineitemsCall) Do() (*UploadLineItemsResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.uploadlineitemsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "lineitems/uploadlineitems")
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
	ret := new(UploadLineItemsResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Uploads line items in CSV format.",
	//   "httpMethod": "POST",
	//   "id": "doubleclickbidmanager.lineitems.uploadlineitems",
	//   "path": "lineitems/uploadlineitems",
	//   "request": {
	//     "$ref": "UploadLineItemsRequest"
	//   },
	//   "response": {
	//     "$ref": "UploadLineItemsResponse"
	//   }
	// }

}

// method id "doubleclickbidmanager.queries.createquery":

type QueriesCreatequeryCall struct {
	s     *Service
	query *Query
	opt_  map[string]interface{}
}

// Createquery: Creates a query.
func (r *QueriesService) Createquery(query *Query) *QueriesCreatequeryCall {
	c := &QueriesCreatequeryCall{s: r.s, opt_: make(map[string]interface{})}
	c.query = query
	return c
}

func (c *QueriesCreatequeryCall) Do() (*Query, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.query)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "query")
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
	ret := new(Query)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a query.",
	//   "httpMethod": "POST",
	//   "id": "doubleclickbidmanager.queries.createquery",
	//   "path": "query",
	//   "request": {
	//     "$ref": "Query"
	//   },
	//   "response": {
	//     "$ref": "Query"
	//   }
	// }

}

// method id "doubleclickbidmanager.queries.deletequery":

type QueriesDeletequeryCall struct {
	s       *Service
	queryId int64
	opt_    map[string]interface{}
}

// Deletequery: Deletes a stored query as well as the associated stored
// reports.
func (r *QueriesService) Deletequery(queryId int64) *QueriesDeletequeryCall {
	c := &QueriesDeletequeryCall{s: r.s, opt_: make(map[string]interface{})}
	c.queryId = queryId
	return c
}

func (c *QueriesDeletequeryCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "query/{queryId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{queryId}", strconv.FormatInt(c.queryId, 10), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes a stored query as well as the associated stored reports.",
	//   "httpMethod": "DELETE",
	//   "id": "doubleclickbidmanager.queries.deletequery",
	//   "parameterOrder": [
	//     "queryId"
	//   ],
	//   "parameters": {
	//     "queryId": {
	//       "description": "Query ID to delete.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "query/{queryId}"
	// }

}

// method id "doubleclickbidmanager.queries.getquery":

type QueriesGetqueryCall struct {
	s       *Service
	queryId int64
	opt_    map[string]interface{}
}

// Getquery: Retrieves a stored query.
func (r *QueriesService) Getquery(queryId int64) *QueriesGetqueryCall {
	c := &QueriesGetqueryCall{s: r.s, opt_: make(map[string]interface{})}
	c.queryId = queryId
	return c
}

func (c *QueriesGetqueryCall) Do() (*Query, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "query/{queryId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{queryId}", strconv.FormatInt(c.queryId, 10), 1)
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
	ret := new(Query)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a stored query.",
	//   "httpMethod": "GET",
	//   "id": "doubleclickbidmanager.queries.getquery",
	//   "parameterOrder": [
	//     "queryId"
	//   ],
	//   "parameters": {
	//     "queryId": {
	//       "description": "Query ID to retrieve.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "query/{queryId}",
	//   "response": {
	//     "$ref": "Query"
	//   }
	// }

}

// method id "doubleclickbidmanager.queries.listqueries":

type QueriesListqueriesCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// Listqueries: Retrieves stored queries.
func (r *QueriesService) Listqueries() *QueriesListqueriesCall {
	c := &QueriesListqueriesCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *QueriesListqueriesCall) Do() (*ListQueriesResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "queries")
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
	ret := new(ListQueriesResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves stored queries.",
	//   "httpMethod": "GET",
	//   "id": "doubleclickbidmanager.queries.listqueries",
	//   "path": "queries",
	//   "response": {
	//     "$ref": "ListQueriesResponse"
	//   }
	// }

}

// method id "doubleclickbidmanager.queries.runquery":

type QueriesRunqueryCall struct {
	s               *Service
	queryId         int64
	runqueryrequest *RunQueryRequest
	opt_            map[string]interface{}
}

// Runquery: Runs a stored query to generate a report.
func (r *QueriesService) Runquery(queryId int64, runqueryrequest *RunQueryRequest) *QueriesRunqueryCall {
	c := &QueriesRunqueryCall{s: r.s, opt_: make(map[string]interface{})}
	c.queryId = queryId
	c.runqueryrequest = runqueryrequest
	return c
}

func (c *QueriesRunqueryCall) Do() error {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.runqueryrequest)
	if err != nil {
		return err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "query/{queryId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{queryId}", strconv.FormatInt(c.queryId, 10), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Runs a stored query to generate a report.",
	//   "httpMethod": "POST",
	//   "id": "doubleclickbidmanager.queries.runquery",
	//   "parameterOrder": [
	//     "queryId"
	//   ],
	//   "parameters": {
	//     "queryId": {
	//       "description": "Query ID to run.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "query/{queryId}",
	//   "request": {
	//     "$ref": "RunQueryRequest"
	//   }
	// }

}

// method id "doubleclickbidmanager.reports.listreports":

type ReportsListreportsCall struct {
	s       *Service
	queryId int64
	opt_    map[string]interface{}
}

// Listreports: Retrieves stored reports.
func (r *ReportsService) Listreports(queryId int64) *ReportsListreportsCall {
	c := &ReportsListreportsCall{s: r.s, opt_: make(map[string]interface{})}
	c.queryId = queryId
	return c
}

func (c *ReportsListreportsCall) Do() (*ListReportsResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "queries/{queryId}/reports")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{queryId}", strconv.FormatInt(c.queryId, 10), 1)
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
	ret := new(ListReportsResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves stored reports.",
	//   "httpMethod": "GET",
	//   "id": "doubleclickbidmanager.reports.listreports",
	//   "parameterOrder": [
	//     "queryId"
	//   ],
	//   "parameters": {
	//     "queryId": {
	//       "description": "Query ID with which the reports are associated.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "queries/{queryId}/reports",
	//   "response": {
	//     "$ref": "ListReportsResponse"
	//   }
	// }

}
