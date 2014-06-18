// Package doubleclicksearch provides access to the DoubleClick Search API.
//
// See https://developers.google.com/doubleclick-search/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/doubleclicksearch/v2"
//   ...
//   doubleclicksearchService, err := doubleclicksearch.New(oauthHttpClient)
package doubleclicksearch

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

const apiId = "doubleclicksearch:v2"
const apiName = "doubleclicksearch"
const apiVersion = "v2"
const basePath = "https://www.googleapis.com/doubleclicksearch/v2/"

// OAuth2 scopes used by this API.
const (
	// View and manage your advertising data in DoubleClick Search
	DoubleclicksearchScope = "https://www.googleapis.com/auth/doubleclicksearch"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Conversion = NewConversionService(s)
	s.Reports = NewReportsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Conversion *ConversionService

	Reports *ReportsService
}

func NewConversionService(s *Service) *ConversionService {
	rs := &ConversionService{s: s}
	return rs
}

type ConversionService struct {
	s *Service
}

func NewReportsService(s *Service) *ReportsService {
	rs := &ReportsService{s: s}
	return rs
}

type ReportsService struct {
	s *Service
}

type Availability struct {
	// AdvertiserId: DS advertiser ID.
	AdvertiserId int64 `json:"advertiserId,omitempty,string"`

	// AgencyId: DS agency ID.
	AgencyId int64 `json:"agencyId,omitempty,string"`

	// AvailabilityTimestamp: The time by which all conversions have been
	// uploaded, in epoch millis UTC.
	AvailabilityTimestamp uint64 `json:"availabilityTimestamp,omitempty,string"`

	// SegmentationId: The numeric segmentation identifier (for example,
	// DoubleClick Search Floodlight activity ID).
	SegmentationId int64 `json:"segmentationId,omitempty,string"`

	// SegmentationName: The friendly segmentation identifier (for example,
	// DoubleClick Search Floodlight activity name).
	SegmentationName string `json:"segmentationName,omitempty"`

	// SegmentationType: The segmentation type that this availability is for
	// (its default value is FLOODLIGHT).
	SegmentationType string `json:"segmentationType,omitempty"`
}

type Conversion struct {
	// AdGroupId: DS ad group ID.
	AdGroupId int64 `json:"adGroupId,omitempty,string"`

	// AdId: DS ad ID.
	AdId int64 `json:"adId,omitempty,string"`

	// AdvertiserId: DS advertiser ID.
	AdvertiserId int64 `json:"advertiserId,omitempty,string"`

	// AgencyId: DS agency ID.
	AgencyId int64 `json:"agencyId,omitempty,string"`

	// CampaignId: DS campaign ID.
	CampaignId int64 `json:"campaignId,omitempty,string"`

	// ClickId: DS click ID for the conversion.
	ClickId string `json:"clickId,omitempty"`

	// ConversionId: Advertiser-provided ID for the conversion, also known
	// as the order ID.
	ConversionId string `json:"conversionId,omitempty"`

	// ConversionModifiedTimestamp: The time at which the conversion was
	// last modified, in epoch millis UTC.
	ConversionModifiedTimestamp uint64 `json:"conversionModifiedTimestamp,omitempty,string"`

	// ConversionTimestamp: The time at which the conversion took place, in
	// epoch millis UTC.
	ConversionTimestamp uint64 `json:"conversionTimestamp,omitempty,string"`

	// CriterionId: DS criterion (keyword) ID.
	CriterionId int64 `json:"criterionId,omitempty,string"`

	// CurrencyCode: The currency code for the conversion's revenue. Should
	// be in ISO 4217 alphabetic (3-char) format.
	CurrencyCode string `json:"currencyCode,omitempty"`

	// CustomDimension: Custom dimensions for the conversion, which can be
	// used to filter data in a report.
	CustomDimension []*CustomDimension `json:"customDimension,omitempty"`

	// CustomMetric: Custom metrics for the conversion.
	CustomMetric []*CustomMetric `json:"customMetric,omitempty"`

	// DsConversionId: DS conversion ID.
	DsConversionId int64 `json:"dsConversionId,omitempty,string"`

	// EngineAccountId: DS engine account ID.
	EngineAccountId int64 `json:"engineAccountId,omitempty,string"`

	// FloodlightOrderId: The advertiser-provided order id for the
	// conversion.
	FloodlightOrderId string `json:"floodlightOrderId,omitempty"`

	// QuantityMillis: The quantity of this conversion, in millis.
	QuantityMillis int64 `json:"quantityMillis,omitempty,string"`

	// RevenueMicros: The revenue amount of this TRANSACTION conversion, in
	// micros.
	RevenueMicros int64 `json:"revenueMicros,omitempty,string"`

	// SegmentationId: The numeric segmentation identifier (for example,
	// DoubleClick Search Floodlight activity ID).
	SegmentationId int64 `json:"segmentationId,omitempty,string"`

	// SegmentationName: The friendly segmentation identifier (for example,
	// DoubleClick Search Floodlight activity name).
	SegmentationName string `json:"segmentationName,omitempty"`

	// SegmentationType: The segmentation type of this conversion (for
	// example, FLOODLIGHT).
	SegmentationType string `json:"segmentationType,omitempty"`

	// State: The state of the conversion, that is, either ACTIVE or
	// DELETED.
	State string `json:"state,omitempty"`

	// Type: The type of the conversion, that is, either ACTION or
	// TRANSACTION. An ACTION conversion is an action by the user that has
	// no monetarily quantifiable value, while a TRANSACTION conversion is
	// an action that does have a monetarily quantifiable value. Examples
	// are email list signups (ACTION) versus ecommerce purchases
	// (TRANSACTION).
	Type string `json:"type,omitempty"`
}

type ConversionList struct {
	// Conversion: The conversions being requested.
	Conversion []*Conversion `json:"conversion,omitempty"`

	// Kind: Identifies this as a ConversionList resource. Value: the fixed
	// string doubleclicksearch#conversionList.
	Kind string `json:"kind,omitempty"`
}

type CustomDimension struct {
	// Name: Custom dimension name.
	Name string `json:"name,omitempty"`

	// Value: Custom dimension value.
	Value string `json:"value,omitempty"`
}

type CustomMetric struct {
	// Name: Custom metric name.
	Name string `json:"name,omitempty"`

	// Value: Custom metric numeric value.
	Value float64 `json:"value,omitempty"`
}

type Report struct {
	// Files: Asynchronous report only. Contains a list of generated report
	// files once the report has succesfully completed.
	Files []*ReportFiles `json:"files,omitempty"`

	// Id: Asynchronous report only. Id of the report.
	Id string `json:"id,omitempty"`

	// IsReportReady: Asynchronous report only. True if and only if the
	// report has completed successfully and the report files are ready to
	// be downloaded.
	IsReportReady bool `json:"isReportReady,omitempty"`

	// Kind: Identifies this as a Report resource. Value: the fixed string
	// doubleclicksearch#report.
	Kind string `json:"kind,omitempty"`

	// Request: The request that created the report. Optional fields not
	// specified in the original request are filled with default values.
	Request *ReportRequest `json:"request,omitempty"`

	// RowCount: The number of report rows generated by the report, not
	// including headers.
	RowCount int64 `json:"rowCount,omitempty"`

	// Rows: Synchronous report only. Generated report rows.
	Rows []*ReportRow `json:"rows,omitempty"`

	// StatisticsCurrencyCode: The currency code of all monetary values
	// produced in the report, including values that are set by users (e.g.,
	// keyword bid settings) and metrics (e.g., cost and revenue). The
	// currency code of a report is determined by the statisticsCurrency
	// field of the report request.
	StatisticsCurrencyCode string `json:"statisticsCurrencyCode,omitempty"`

	// StatisticsTimeZone: If all statistics of the report are sourced from
	// the same time zone, this would be it. Otherwise the field is unset.
	StatisticsTimeZone string `json:"statisticsTimeZone,omitempty"`
}

type ReportFiles struct {
	// ByteCount: The size of this report file in bytes.
	ByteCount int64 `json:"byteCount,omitempty,string"`

	// Url: Use this url to download the report file.
	Url string `json:"url,omitempty"`
}

type ReportRequest struct {
	// Columns: The columns to include in the report. This includes both
	// DoubleClick Search columns and saved columns. For DoubleClick Search
	// columns, only the columnName parameter is required. For saved columns
	// only the savedColumnName parameter is required. Both columnName and
	// savedColumnName cannot be set in the same stanza.
	Columns []*ReportRequestColumns `json:"columns,omitempty"`

	// DownloadFormat: Format that the report should be returned in.
	// Currently csv or tsv is supported.
	DownloadFormat string `json:"downloadFormat,omitempty"`

	// Filters: A list of filters to be applied to the report.
	Filters []*ReportRequestFilters `json:"filters,omitempty"`

	// IncludeDeletedEntities: Determines if removed entities should be
	// included in the report. Deprecated, please use includeRemovedEntities
	// instead. Defaults to false.
	IncludeDeletedEntities bool `json:"includeDeletedEntities,omitempty"`

	// IncludeRemovedEntities: Determines if removed entities should be
	// included in the report. Defaults to false.
	IncludeRemovedEntities bool `json:"includeRemovedEntities,omitempty"`

	// MaxRowsPerFile: Asynchronous report only. The maximum number of rows
	// per report file. A large report is split into many files based on
	// this field. Acceptable values are 1000000 to 100000000, inclusive.
	MaxRowsPerFile int64 `json:"maxRowsPerFile,omitempty"`

	// OrderBy: Synchronous report only. A list of columns and directions
	// defining sorting to be performed on the report rows.
	OrderBy []*ReportRequestOrderBy `json:"orderBy,omitempty"`

	// ReportScope: The reportScope is a set of IDs that are used to
	// determine which subset of entities will be returned in the report.
	// The full lineage of IDs from the lowest scoped level desired up
	// through agency is required.
	ReportScope *ReportRequestReportScope `json:"reportScope,omitempty"`

	// ReportType: Determines the type of rows that are returned in the
	// report. For example, if you specify reportType: keyword, each row in
	// the report will contain data about a keyword. See the Types of
	// Reports reference for the columns that are available for each type.
	ReportType string `json:"reportType,omitempty"`

	// RowCount: Synchronous report only. The maxinum number of rows to
	// return; additional rows are dropped. Acceptable values are 0 to
	// 10000, inclusive. Defaults to 10000.
	RowCount int64 `json:"rowCount,omitempty"`

	// StartRow: Synchronous report only. Zero-based index of the first row
	// to return. Acceptable values are 0 to 50000, inclusive. Defaults to
	// 0.
	StartRow int64 `json:"startRow,omitempty"`

	// StatisticsCurrency: Specifies the currency in which monetary will be
	// returned. Possible values are: usd, agency (valid if the report is
	// scoped to agency or lower), advertiser (valid if the report is scoped
	// to * advertiser or lower), or account (valid if the report is scoped
	// to engine account or lower).
	StatisticsCurrency string `json:"statisticsCurrency,omitempty"`

	// TimeRange: If metrics are requested in a report, this argument will
	// be used to restrict the metrics to a specific time range.
	TimeRange *ReportRequestTimeRange `json:"timeRange,omitempty"`

	// VerifySingleTimeZone: If true, the report would only be created if
	// all the requested stat data are sourced from a single timezone.
	// Defaults to false.
	VerifySingleTimeZone bool `json:"verifySingleTimeZone,omitempty"`
}

type ReportRequestColumns struct {
	// ColumnName: Name of a DoubleClick Search column to include in the
	// report.
	ColumnName string `json:"columnName,omitempty"`

	// EndDate: Inclusive day in YYYY-MM-DD format. When provided, this
	// overrides the overall time range of the report for this column only.
	// Must be provided together with startDate.
	EndDate string `json:"endDate,omitempty"`

	// GroupByColumn: Synchronous report only. Set to true to group by this
	// column. Defaults to false.
	GroupByColumn bool `json:"groupByColumn,omitempty"`

	// HeaderText: Text used to identify this column in the report output;
	// defaults to columnName or savedColumnName when not specified. This
	// can be used to prevent collisions between DoubleClick Search columns
	// and saved columns with the same name.
	HeaderText string `json:"headerText,omitempty"`

	// SavedColumnName: Name of a saved column to include in the report. The
	// report must be scoped at advertiser or lower, and this saved column
	// must already be created in the DoubleClick Search UI.
	SavedColumnName string `json:"savedColumnName,omitempty"`

	// StartDate: Inclusive date in YYYY-MM-DD format. When provided, this
	// overrides the overall time range of the report for this column only.
	// Must be provided together with endDate.
	StartDate string `json:"startDate,omitempty"`
}

type ReportRequestFilters struct {
	// Column: Column to perform the filter on. This can be a DoubleClick
	// Search column or a saved column.
	Column *ReportRequestFiltersColumn `json:"column,omitempty"`

	// Operator: Operator to use in the filter. See the filter reference for
	// a list of available operators.
	Operator string `json:"operator,omitempty"`

	// Values: A list of values to filter the column value against.
	Values []interface{} `json:"values,omitempty"`
}

type ReportRequestFiltersColumn struct {
	// ColumnName: Name of a DoubleClick Search column to filter on.
	ColumnName string `json:"columnName,omitempty"`

	// SavedColumnName: Name of a saved column to filter on.
	SavedColumnName string `json:"savedColumnName,omitempty"`
}

type ReportRequestOrderBy struct {
	// Column: Column to perform the sort on. This can be a DoubleClick
	// Search-defined column or a saved column.
	Column *ReportRequestOrderByColumn `json:"column,omitempty"`

	// SortOrder: The sort direction, which is either ascending or
	// descending.
	SortOrder string `json:"sortOrder,omitempty"`
}

type ReportRequestOrderByColumn struct {
	// ColumnName: Name of a DoubleClick Search column to sort by.
	ColumnName string `json:"columnName,omitempty"`

	// SavedColumnName: Name of a saved column to sort by.
	SavedColumnName string `json:"savedColumnName,omitempty"`
}

type ReportRequestReportScope struct {
	// AdGroupId: DS ad group ID.
	AdGroupId int64 `json:"adGroupId,omitempty,string"`

	// AdId: DS ad ID.
	AdId int64 `json:"adId,omitempty,string"`

	// AdvertiserId: DS advertiser ID.
	AdvertiserId int64 `json:"advertiserId,omitempty,string"`

	// AgencyId: DS agency ID.
	AgencyId int64 `json:"agencyId,omitempty,string"`

	// CampaignId: DS campaign ID.
	CampaignId int64 `json:"campaignId,omitempty,string"`

	// EngineAccountId: DS engine account ID.
	EngineAccountId int64 `json:"engineAccountId,omitempty,string"`

	// KeywordId: DS keyword ID.
	KeywordId int64 `json:"keywordId,omitempty,string"`
}

type ReportRequestTimeRange struct {
	// ChangedAttributesSinceTimestamp: Inclusive UTC timestamp in RFC
	// format, e.g., 2013-07-16T10:16:23.555Z. See additional references on
	// how changed attribute reports work.
	ChangedAttributesSinceTimestamp string `json:"changedAttributesSinceTimestamp,omitempty"`

	// ChangedMetricsSinceTimestamp: Inclusive UTC timestamp in RFC format,
	// e.g., 2013-07-16T10:16:23.555Z. See additional references on how
	// changed metrics reports work.
	ChangedMetricsSinceTimestamp string `json:"changedMetricsSinceTimestamp,omitempty"`

	// EndDate: Inclusive date in YYYY-MM-DD format.
	EndDate string `json:"endDate,omitempty"`

	// StartDate: Inclusive date in YYYY-MM-DD format.
	StartDate string `json:"startDate,omitempty"`
}

type ReportRow struct {
}

type UpdateAvailabilityRequest struct {
	// Availabilities: The availabilities being requested.
	Availabilities []*Availability `json:"availabilities,omitempty"`
}

type UpdateAvailabilityResponse struct {
	// Availabilities: The availabilities being returned.
	Availabilities []*Availability `json:"availabilities,omitempty"`
}

// method id "doubleclicksearch.conversion.get":

type ConversionGetCall struct {
	s               *Service
	agencyId        int64
	advertiserId    int64
	engineAccountId int64
	endDate         int64
	rowCount        int64
	startDate       int64
	startRow        int64
	opt_            map[string]interface{}
}

// Get: Retrieves a list of conversions from a DoubleClick Search engine
// account.
func (r *ConversionService) Get(agencyId int64, advertiserId int64, engineAccountId int64, endDate int64, rowCount int64, startDate int64, startRow int64) *ConversionGetCall {
	c := &ConversionGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.agencyId = agencyId
	c.advertiserId = advertiserId
	c.engineAccountId = engineAccountId
	c.endDate = endDate
	c.rowCount = rowCount
	c.startDate = startDate
	c.startRow = startRow
	return c
}

// AdGroupId sets the optional parameter "adGroupId": Numeric ID of the
// ad group.
func (c *ConversionGetCall) AdGroupId(adGroupId int64) *ConversionGetCall {
	c.opt_["adGroupId"] = adGroupId
	return c
}

// AdId sets the optional parameter "adId": Numeric ID of the ad.
func (c *ConversionGetCall) AdId(adId int64) *ConversionGetCall {
	c.opt_["adId"] = adId
	return c
}

// CampaignId sets the optional parameter "campaignId": Numeric ID of
// the campaign.
func (c *ConversionGetCall) CampaignId(campaignId int64) *ConversionGetCall {
	c.opt_["campaignId"] = campaignId
	return c
}

// CriterionId sets the optional parameter "criterionId": Numeric ID of
// the criterion.
func (c *ConversionGetCall) CriterionId(criterionId int64) *ConversionGetCall {
	c.opt_["criterionId"] = criterionId
	return c
}

func (c *ConversionGetCall) Do() (*ConversionList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("endDate", fmt.Sprintf("%v", c.endDate))
	params.Set("rowCount", fmt.Sprintf("%v", c.rowCount))
	params.Set("startDate", fmt.Sprintf("%v", c.startDate))
	params.Set("startRow", fmt.Sprintf("%v", c.startRow))
	if v, ok := c.opt_["adGroupId"]; ok {
		params.Set("adGroupId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["adId"]; ok {
		params.Set("adId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["campaignId"]; ok {
		params.Set("campaignId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["criterionId"]; ok {
		params.Set("criterionId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "agency/{agencyId}/advertiser/{advertiserId}/engine/{engineAccountId}/conversion")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{agencyId}", strconv.FormatInt(c.agencyId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{advertiserId}", strconv.FormatInt(c.advertiserId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{engineAccountId}", strconv.FormatInt(c.engineAccountId, 10), 1)
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
	ret := new(ConversionList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of conversions from a DoubleClick Search engine account.",
	//   "httpMethod": "GET",
	//   "id": "doubleclicksearch.conversion.get",
	//   "parameterOrder": [
	//     "agencyId",
	//     "advertiserId",
	//     "engineAccountId",
	//     "endDate",
	//     "rowCount",
	//     "startDate",
	//     "startRow"
	//   ],
	//   "parameters": {
	//     "adGroupId": {
	//       "description": "Numeric ID of the ad group.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "adId": {
	//       "description": "Numeric ID of the ad.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "advertiserId": {
	//       "description": "Numeric ID of the advertiser.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "agencyId": {
	//       "description": "Numeric ID of the agency.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "campaignId": {
	//       "description": "Numeric ID of the campaign.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "criterionId": {
	//       "description": "Numeric ID of the criterion.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "endDate": {
	//       "description": "Last date (inclusive) on which to retrieve conversions. Format is yyyymmdd.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "99991231",
	//       "minimum": "20091101",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "engineAccountId": {
	//       "description": "Numeric ID of the engine account.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "rowCount": {
	//       "description": "The number of conversions to return per call.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "startDate": {
	//       "description": "First date (inclusive) on which to retrieve conversions. Format is yyyymmdd.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "99991231",
	//       "minimum": "20091101",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "startRow": {
	//       "description": "The 0-based starting index for retrieving conversions results.",
	//       "format": "uint32",
	//       "location": "query",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "agency/{agencyId}/advertiser/{advertiserId}/engine/{engineAccountId}/conversion",
	//   "response": {
	//     "$ref": "ConversionList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/doubleclicksearch"
	//   ]
	// }

}

// method id "doubleclicksearch.conversion.insert":

type ConversionInsertCall struct {
	s              *Service
	conversionlist *ConversionList
	opt_           map[string]interface{}
}

// Insert: Inserts a batch of new conversions into DoubleClick Search.
func (r *ConversionService) Insert(conversionlist *ConversionList) *ConversionInsertCall {
	c := &ConversionInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.conversionlist = conversionlist
	return c
}

func (c *ConversionInsertCall) Do() (*ConversionList, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.conversionlist)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "conversion")
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
	ret := new(ConversionList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a batch of new conversions into DoubleClick Search.",
	//   "httpMethod": "POST",
	//   "id": "doubleclicksearch.conversion.insert",
	//   "path": "conversion",
	//   "request": {
	//     "$ref": "ConversionList"
	//   },
	//   "response": {
	//     "$ref": "ConversionList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/doubleclicksearch"
	//   ]
	// }

}

// method id "doubleclicksearch.conversion.patch":

type ConversionPatchCall struct {
	s               *Service
	advertiserId    int64
	agencyId        int64
	endDate         int64
	engineAccountId int64
	rowCount        int64
	startDate       int64
	startRow        int64
	conversionlist  *ConversionList
	opt_            map[string]interface{}
}

// Patch: Updates a batch of conversions in DoubleClick Search. This
// method supports patch semantics.
func (r *ConversionService) Patch(advertiserId int64, agencyId int64, endDate int64, engineAccountId int64, rowCount int64, startDate int64, startRow int64, conversionlist *ConversionList) *ConversionPatchCall {
	c := &ConversionPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.advertiserId = advertiserId
	c.agencyId = agencyId
	c.endDate = endDate
	c.engineAccountId = engineAccountId
	c.rowCount = rowCount
	c.startDate = startDate
	c.startRow = startRow
	c.conversionlist = conversionlist
	return c
}

func (c *ConversionPatchCall) Do() (*ConversionList, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.conversionlist)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("advertiserId", fmt.Sprintf("%v", c.advertiserId))
	params.Set("agencyId", fmt.Sprintf("%v", c.agencyId))
	params.Set("endDate", fmt.Sprintf("%v", c.endDate))
	params.Set("engineAccountId", fmt.Sprintf("%v", c.engineAccountId))
	params.Set("rowCount", fmt.Sprintf("%v", c.rowCount))
	params.Set("startDate", fmt.Sprintf("%v", c.startDate))
	params.Set("startRow", fmt.Sprintf("%v", c.startRow))
	urls := googleapi.ResolveRelative(c.s.BasePath, "conversion")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
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
	ret := new(ConversionList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a batch of conversions in DoubleClick Search. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "doubleclicksearch.conversion.patch",
	//   "parameterOrder": [
	//     "advertiserId",
	//     "agencyId",
	//     "endDate",
	//     "engineAccountId",
	//     "rowCount",
	//     "startDate",
	//     "startRow"
	//   ],
	//   "parameters": {
	//     "advertiserId": {
	//       "description": "Numeric ID of the advertiser.",
	//       "format": "int64",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "agencyId": {
	//       "description": "Numeric ID of the agency.",
	//       "format": "int64",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "endDate": {
	//       "description": "Last date (inclusive) on which to retrieve conversions. Format is yyyymmdd.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "99991231",
	//       "minimum": "20091101",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "engineAccountId": {
	//       "description": "Numeric ID of the engine account.",
	//       "format": "int64",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "rowCount": {
	//       "description": "The number of conversions to return per call.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "1",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "startDate": {
	//       "description": "First date (inclusive) on which to retrieve conversions. Format is yyyymmdd.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "99991231",
	//       "minimum": "20091101",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "startRow": {
	//       "description": "The 0-based starting index for retrieving conversions results.",
	//       "format": "uint32",
	//       "location": "query",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "conversion",
	//   "request": {
	//     "$ref": "ConversionList"
	//   },
	//   "response": {
	//     "$ref": "ConversionList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/doubleclicksearch"
	//   ]
	// }

}

// method id "doubleclicksearch.conversion.update":

type ConversionUpdateCall struct {
	s              *Service
	conversionlist *ConversionList
	opt_           map[string]interface{}
}

// Update: Updates a batch of conversions in DoubleClick Search.
func (r *ConversionService) Update(conversionlist *ConversionList) *ConversionUpdateCall {
	c := &ConversionUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.conversionlist = conversionlist
	return c
}

func (c *ConversionUpdateCall) Do() (*ConversionList, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.conversionlist)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "conversion")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
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
	ret := new(ConversionList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a batch of conversions in DoubleClick Search.",
	//   "httpMethod": "PUT",
	//   "id": "doubleclicksearch.conversion.update",
	//   "path": "conversion",
	//   "request": {
	//     "$ref": "ConversionList"
	//   },
	//   "response": {
	//     "$ref": "ConversionList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/doubleclicksearch"
	//   ]
	// }

}

// method id "doubleclicksearch.conversion.updateAvailability":

type ConversionUpdateAvailabilityCall struct {
	s                         *Service
	updateavailabilityrequest *UpdateAvailabilityRequest
	opt_                      map[string]interface{}
}

// UpdateAvailability: Updates the availabilities of a batch of
// floodlight activities in DoubleClick Search.
func (r *ConversionService) UpdateAvailability(updateavailabilityrequest *UpdateAvailabilityRequest) *ConversionUpdateAvailabilityCall {
	c := &ConversionUpdateAvailabilityCall{s: r.s, opt_: make(map[string]interface{})}
	c.updateavailabilityrequest = updateavailabilityrequest
	return c
}

func (c *ConversionUpdateAvailabilityCall) Do() (*UpdateAvailabilityResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.updateavailabilityrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "conversion/updateAvailability")
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
	ret := new(UpdateAvailabilityResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the availabilities of a batch of floodlight activities in DoubleClick Search.",
	//   "httpMethod": "POST",
	//   "id": "doubleclicksearch.conversion.updateAvailability",
	//   "path": "conversion/updateAvailability",
	//   "request": {
	//     "$ref": "UpdateAvailabilityRequest",
	//     "parameterName": "empty"
	//   },
	//   "response": {
	//     "$ref": "UpdateAvailabilityResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/doubleclicksearch"
	//   ]
	// }

}

// method id "doubleclicksearch.reports.generate":

type ReportsGenerateCall struct {
	s             *Service
	reportrequest *ReportRequest
	opt_          map[string]interface{}
}

// Generate: Generates and returns a report immediately.
func (r *ReportsService) Generate(reportrequest *ReportRequest) *ReportsGenerateCall {
	c := &ReportsGenerateCall{s: r.s, opt_: make(map[string]interface{})}
	c.reportrequest = reportrequest
	return c
}

func (c *ReportsGenerateCall) Do() (*Report, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.reportrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports/generate")
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
	ret := new(Report)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Generates and returns a report immediately.",
	//   "httpMethod": "POST",
	//   "id": "doubleclicksearch.reports.generate",
	//   "path": "reports/generate",
	//   "request": {
	//     "$ref": "ReportRequest",
	//     "parameterName": "reportRequest"
	//   },
	//   "response": {
	//     "$ref": "Report"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/doubleclicksearch"
	//   ]
	// }

}

// method id "doubleclicksearch.reports.get":

type ReportsGetCall struct {
	s        *Service
	reportId string
	opt_     map[string]interface{}
}

// Get: Polls for the status of a report request.
func (r *ReportsService) Get(reportId string) *ReportsGetCall {
	c := &ReportsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.reportId = reportId
	return c
}

func (c *ReportsGetCall) Do() (*Report, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports/{reportId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportId}", url.QueryEscape(c.reportId), 1)
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
	ret := new(Report)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Polls for the status of a report request.",
	//   "httpMethod": "GET",
	//   "id": "doubleclicksearch.reports.get",
	//   "parameterOrder": [
	//     "reportId"
	//   ],
	//   "parameters": {
	//     "reportId": {
	//       "description": "ID of the report request being polled.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "reports/{reportId}",
	//   "response": {
	//     "$ref": "Report"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/doubleclicksearch"
	//   ]
	// }

}

// method id "doubleclicksearch.reports.getFile":

type ReportsGetFileCall struct {
	s              *Service
	reportId       string
	reportFragment int64
	opt_           map[string]interface{}
}

// GetFile: Downloads a report file.
func (r *ReportsService) GetFile(reportId string, reportFragment int64) *ReportsGetFileCall {
	c := &ReportsGetFileCall{s: r.s, opt_: make(map[string]interface{})}
	c.reportId = reportId
	c.reportFragment = reportFragment
	return c
}

func (c *ReportsGetFileCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports/{reportId}/files/{reportFragment}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportId}", url.QueryEscape(c.reportId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportFragment}", strconv.FormatInt(c.reportFragment, 10), 1)
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
	//   "description": "Downloads a report file.",
	//   "httpMethod": "GET",
	//   "id": "doubleclicksearch.reports.getFile",
	//   "parameterOrder": [
	//     "reportId",
	//     "reportFragment"
	//   ],
	//   "parameters": {
	//     "reportFragment": {
	//       "description": "The index of the report fragment to download.",
	//       "format": "int32",
	//       "location": "path",
	//       "minimum": "0",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "reportId": {
	//       "description": "ID of the report.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "reports/{reportId}/files/{reportFragment}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/doubleclicksearch"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "doubleclicksearch.reports.request":

type ReportsRequestCall struct {
	s             *Service
	reportrequest *ReportRequest
	opt_          map[string]interface{}
}

// Request: Inserts a report request into the reporting system.
func (r *ReportsService) Request(reportrequest *ReportRequest) *ReportsRequestCall {
	c := &ReportsRequestCall{s: r.s, opt_: make(map[string]interface{})}
	c.reportrequest = reportrequest
	return c
}

func (c *ReportsRequestCall) Do() (*Report, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.reportrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports")
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
	ret := new(Report)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a report request into the reporting system.",
	//   "httpMethod": "POST",
	//   "id": "doubleclicksearch.reports.request",
	//   "path": "reports",
	//   "request": {
	//     "$ref": "ReportRequest",
	//     "parameterName": "reportRequest"
	//   },
	//   "response": {
	//     "$ref": "Report"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/doubleclicksearch"
	//   ]
	// }

}
