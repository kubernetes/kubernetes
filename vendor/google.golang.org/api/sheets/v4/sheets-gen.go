// Package sheets provides access to the Google Sheets API.
//
// See https://developers.google.com/sheets/
//
// Usage example:
//
//   import "google.golang.org/api/sheets/v4"
//   ...
//   sheetsService, err := sheets.New(oauthHttpClient)
package sheets // import "google.golang.org/api/sheets/v4"

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

const apiId = "sheets:v4"
const apiName = "sheets"
const apiVersion = "v4"
const basePath = "https://sheets.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage the files in your Google Drive
	DriveScope = "https://www.googleapis.com/auth/drive"

	// View and manage Google Drive files and folders that you have opened
	// or created with this app
	DriveFileScope = "https://www.googleapis.com/auth/drive.file"

	// View the files in your Google Drive
	DriveReadonlyScope = "https://www.googleapis.com/auth/drive.readonly"

	// View and manage your spreadsheets in Google Drive
	SpreadsheetsScope = "https://www.googleapis.com/auth/spreadsheets"

	// View your Google Spreadsheets
	SpreadsheetsReadonlyScope = "https://www.googleapis.com/auth/spreadsheets.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Spreadsheets = NewSpreadsheetsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Spreadsheets *SpreadsheetsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewSpreadsheetsService(s *Service) *SpreadsheetsService {
	rs := &SpreadsheetsService{s: s}
	rs.Sheets = NewSpreadsheetsSheetsService(s)
	rs.Values = NewSpreadsheetsValuesService(s)
	return rs
}

type SpreadsheetsService struct {
	s *Service

	Sheets *SpreadsheetsSheetsService

	Values *SpreadsheetsValuesService
}

func NewSpreadsheetsSheetsService(s *Service) *SpreadsheetsSheetsService {
	rs := &SpreadsheetsSheetsService{s: s}
	return rs
}

type SpreadsheetsSheetsService struct {
	s *Service
}

func NewSpreadsheetsValuesService(s *Service) *SpreadsheetsValuesService {
	rs := &SpreadsheetsValuesService{s: s}
	return rs
}

type SpreadsheetsValuesService struct {
	s *Service
}

// AddBandingRequest: Adds a new banded range to the spreadsheet.
type AddBandingRequest struct {
	// BandedRange: The banded range to add. The bandedRangeId
	// field is optional; if one is not set, an id will be randomly
	// generated. (It
	// is an error to specify the ID of a range that already exists.)
	BandedRange *BandedRange `json:"bandedRange,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BandedRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BandedRange") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddBandingRequest) MarshalJSON() ([]byte, error) {
	type noMethod AddBandingRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddBandingResponse: The result of adding a banded range.
type AddBandingResponse struct {
	// BandedRange: The banded range that was added.
	BandedRange *BandedRange `json:"bandedRange,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BandedRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BandedRange") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddBandingResponse) MarshalJSON() ([]byte, error) {
	type noMethod AddBandingResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddChartRequest: Adds a chart to a sheet in the spreadsheet.
type AddChartRequest struct {
	// Chart: The chart that should be added to the spreadsheet, including
	// the position
	// where it should be placed. The chartId
	// field is optional; if one is not set, an id will be randomly
	// generated. (It
	// is an error to specify the ID of a chart that already exists.)
	Chart *EmbeddedChart `json:"chart,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Chart") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Chart") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddChartRequest) MarshalJSON() ([]byte, error) {
	type noMethod AddChartRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddChartResponse: The result of adding a chart to a spreadsheet.
type AddChartResponse struct {
	// Chart: The newly added chart.
	Chart *EmbeddedChart `json:"chart,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Chart") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Chart") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddChartResponse) MarshalJSON() ([]byte, error) {
	type noMethod AddChartResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddConditionalFormatRuleRequest: Adds a new conditional format rule
// at the given index.
// All subsequent rules' indexes are incremented.
type AddConditionalFormatRuleRequest struct {
	// Index: The zero-based index where the rule should be inserted.
	Index int64 `json:"index,omitempty"`

	// Rule: The rule to add.
	Rule *ConditionalFormatRule `json:"rule,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Index") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Index") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddConditionalFormatRuleRequest) MarshalJSON() ([]byte, error) {
	type noMethod AddConditionalFormatRuleRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddFilterViewRequest: Adds a filter view.
type AddFilterViewRequest struct {
	// Filter: The filter to add. The filterViewId
	// field is optional; if one is not set, an id will be randomly
	// generated. (It
	// is an error to specify the ID of a filter that already exists.)
	Filter *FilterView `json:"filter,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Filter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Filter") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddFilterViewRequest) MarshalJSON() ([]byte, error) {
	type noMethod AddFilterViewRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddFilterViewResponse: The result of adding a filter view.
type AddFilterViewResponse struct {
	// Filter: The newly added filter view.
	Filter *FilterView `json:"filter,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Filter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Filter") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddFilterViewResponse) MarshalJSON() ([]byte, error) {
	type noMethod AddFilterViewResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddNamedRangeRequest: Adds a named range to the spreadsheet.
type AddNamedRangeRequest struct {
	// NamedRange: The named range to add. The namedRangeId
	// field is optional; if one is not set, an id will be randomly
	// generated. (It
	// is an error to specify the ID of a range that already exists.)
	NamedRange *NamedRange `json:"namedRange,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NamedRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NamedRange") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddNamedRangeRequest) MarshalJSON() ([]byte, error) {
	type noMethod AddNamedRangeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddNamedRangeResponse: The result of adding a named range.
type AddNamedRangeResponse struct {
	// NamedRange: The named range to add.
	NamedRange *NamedRange `json:"namedRange,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NamedRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NamedRange") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddNamedRangeResponse) MarshalJSON() ([]byte, error) {
	type noMethod AddNamedRangeResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddProtectedRangeRequest: Adds a new protected range.
type AddProtectedRangeRequest struct {
	// ProtectedRange: The protected range to be added. The
	// protectedRangeId field is optional; if
	// one is not set, an id will be randomly generated. (It is an error
	// to
	// specify the ID of a range that already exists.)
	ProtectedRange *ProtectedRange `json:"protectedRange,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ProtectedRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ProtectedRange") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AddProtectedRangeRequest) MarshalJSON() ([]byte, error) {
	type noMethod AddProtectedRangeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddProtectedRangeResponse: The result of adding a new protected
// range.
type AddProtectedRangeResponse struct {
	// ProtectedRange: The newly added protected range.
	ProtectedRange *ProtectedRange `json:"protectedRange,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ProtectedRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ProtectedRange") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AddProtectedRangeResponse) MarshalJSON() ([]byte, error) {
	type noMethod AddProtectedRangeResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddSheetRequest: Adds a new sheet.
// When a sheet is added at a given index,
// all subsequent sheets' indexes are incremented.
// To add an object sheet, use AddChartRequest instead and
// specify
// EmbeddedObjectPosition.sheetId or
// EmbeddedObjectPosition.newSheet.
type AddSheetRequest struct {
	// Properties: The properties the new sheet should have.
	// All properties are optional.
	// The sheetId field is optional; if one is not
	// set, an id will be randomly generated. (It is an error to specify the
	// ID
	// of a sheet that already exists.)
	Properties *SheetProperties `json:"properties,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Properties") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Properties") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddSheetRequest) MarshalJSON() ([]byte, error) {
	type noMethod AddSheetRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AddSheetResponse: The result of adding a sheet.
type AddSheetResponse struct {
	// Properties: The properties of the newly added sheet.
	Properties *SheetProperties `json:"properties,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Properties") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Properties") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AddSheetResponse) MarshalJSON() ([]byte, error) {
	type noMethod AddSheetResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppendCellsRequest: Adds new cells after the last row with data in a
// sheet,
// inserting new rows into the sheet if necessary.
type AppendCellsRequest struct {
	// Fields: The fields of CellData that should be updated.
	// At least one field must be specified.
	// The root is the CellData; 'row.values.' should not be specified.
	// A single "*" can be used as short-hand for listing every field.
	Fields string `json:"fields,omitempty"`

	// Rows: The data to append.
	Rows []*RowData `json:"rows,omitempty"`

	// SheetId: The sheet ID to append the data to.
	SheetId int64 `json:"sheetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fields") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fields") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AppendCellsRequest) MarshalJSON() ([]byte, error) {
	type noMethod AppendCellsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppendDimensionRequest: Appends rows or columns to the end of a
// sheet.
type AppendDimensionRequest struct {
	// Dimension: Whether rows or columns should be appended.
	//
	// Possible values:
	//   "DIMENSION_UNSPECIFIED" - The default value, do not use.
	//   "ROWS" - Operates on the rows of a sheet.
	//   "COLUMNS" - Operates on the columns of a sheet.
	Dimension string `json:"dimension,omitempty"`

	// Length: The number of rows or columns to append.
	Length int64 `json:"length,omitempty"`

	// SheetId: The sheet to append rows or columns to.
	SheetId int64 `json:"sheetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Dimension") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Dimension") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AppendDimensionRequest) MarshalJSON() ([]byte, error) {
	type noMethod AppendDimensionRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AppendValuesResponse: The response when updating a range of values in
// a spreadsheet.
type AppendValuesResponse struct {
	// SpreadsheetId: The spreadsheet the updates were applied to.
	SpreadsheetId string `json:"spreadsheetId,omitempty"`

	// TableRange: The range (in A1 notation) of the table that values are
	// being appended to
	// (before the values were appended).
	// Empty if no table was found.
	TableRange string `json:"tableRange,omitempty"`

	// Updates: Information about the updates that were applied.
	Updates *UpdateValuesResponse `json:"updates,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "SpreadsheetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SpreadsheetId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AppendValuesResponse) MarshalJSON() ([]byte, error) {
	type noMethod AppendValuesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AutoFillRequest: Fills in more data based on existing data.
type AutoFillRequest struct {
	// Range: The range to autofill. This will examine the range and
	// detect
	// the location that has data and automatically fill that data
	// in to the rest of the range.
	Range *GridRange `json:"range,omitempty"`

	// SourceAndDestination: The source and destination areas to
	// autofill.
	// This explicitly lists the source of the autofill and where to
	// extend that data.
	SourceAndDestination *SourceAndDestination `json:"sourceAndDestination,omitempty"`

	// UseAlternateSeries: True if we should generate data with the
	// "alternate" series.
	// This differs based on the type and amount of source data.
	UseAlternateSeries bool `json:"useAlternateSeries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Range") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Range") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AutoFillRequest) MarshalJSON() ([]byte, error) {
	type noMethod AutoFillRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AutoResizeDimensionsRequest: Automatically resizes one or more
// dimensions based on the contents
// of the cells in that dimension.
type AutoResizeDimensionsRequest struct {
	// Dimensions: The dimensions to automatically resize.
	Dimensions *DimensionRange `json:"dimensions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Dimensions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Dimensions") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AutoResizeDimensionsRequest) MarshalJSON() ([]byte, error) {
	type noMethod AutoResizeDimensionsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BandedRange: A banded (alternating colors) range in a sheet.
type BandedRange struct {
	// BandedRangeId: The id of the banded range.
	BandedRangeId int64 `json:"bandedRangeId,omitempty"`

	// ColumnProperties: Properties for column bands. These properties will
	// be applied on a column-
	// by-column basis throughout all the columns in the range. At least one
	// of
	// row_properties or column_properties must be specified.
	ColumnProperties *BandingProperties `json:"columnProperties,omitempty"`

	// Range: The range over which these properties are applied.
	Range *GridRange `json:"range,omitempty"`

	// RowProperties: Properties for row bands. These properties will be
	// applied on a row-by-row
	// basis throughout all the rows in the range. At least one
	// of
	// row_properties or column_properties must be specified.
	RowProperties *BandingProperties `json:"rowProperties,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BandedRangeId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BandedRangeId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BandedRange) MarshalJSON() ([]byte, error) {
	type noMethod BandedRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BandingProperties: Properties referring a single dimension (either
// row or column). If both
// BandedRange.row_properties and BandedRange.column_properties are
// set, the fill colors are applied to cells according to the following
// rules:
//
// * header_color and footer_color take priority over band colors.
// * first_band_color takes priority over second_band_color.
// * row_properties takes priority over column_properties.
//
// For example, the first row color takes priority over the first
// column
// color, but the first column color takes priority over the second row
// color.
// Similarly, the row header takes priority over the column header in
// the
// top left cell, but the column header takes priority over the first
// row
// color if the row header is not set.
type BandingProperties struct {
	// FirstBandColor: The first color that is alternating. (Required)
	FirstBandColor *Color `json:"firstBandColor,omitempty"`

	// FooterColor: The color of the last row or column. If this field is
	// not set, the last
	// row or column will be filled with either first_band_color
	// or
	// second_band_color, depending on the color of the previous row
	// or
	// column.
	FooterColor *Color `json:"footerColor,omitempty"`

	// HeaderColor: The color of the first row or column. If this field is
	// set, the first
	// row or column will be filled with this color and the colors
	// will
	// alternate between first_band_color and second_band_color
	// starting
	// from the second row or column. Otherwise, the first row or column
	// will be
	// filled with first_band_color and the colors will proceed to
	// alternate
	// as they normally would.
	HeaderColor *Color `json:"headerColor,omitempty"`

	// SecondBandColor: The second color that is alternating. (Required)
	SecondBandColor *Color `json:"secondBandColor,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FirstBandColor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FirstBandColor") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BandingProperties) MarshalJSON() ([]byte, error) {
	type noMethod BandingProperties
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BasicChartAxis: An axis of the chart.
// A chart may not have more than one axis per
// axis position.
type BasicChartAxis struct {
	// Format: The format of the title.
	// Only valid if the axis is not associated with the domain.
	Format *TextFormat `json:"format,omitempty"`

	// Position: The position of this axis.
	//
	// Possible values:
	//   "BASIC_CHART_AXIS_POSITION_UNSPECIFIED" - Default value, do not
	// use.
	//   "BOTTOM_AXIS" - The axis rendered at the bottom of a chart.
	// For most charts, this is the standard major axis.
	// For bar charts, this is a minor axis.
	//   "LEFT_AXIS" - The axis rendered at the left of a chart.
	// For most charts, this is a minor axis.
	// For bar charts, this is the standard major axis.
	//   "RIGHT_AXIS" - The axis rendered at the right of a chart.
	// For most charts, this is a minor axis.
	// For bar charts, this is an unusual major axis.
	Position string `json:"position,omitempty"`

	// Title: The title of this axis. If set, this overrides any title
	// inferred
	// from headers of the data.
	Title string `json:"title,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Format") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Format") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BasicChartAxis) MarshalJSON() ([]byte, error) {
	type noMethod BasicChartAxis
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BasicChartDomain: The domain of a chart.
// For example, if charting stock prices over time, this would be the
// date.
type BasicChartDomain struct {
	// Domain: The data of the domain. For example, if charting stock prices
	// over time,
	// this is the data representing the dates.
	Domain *ChartData `json:"domain,omitempty"`

	// Reversed: True to reverse the order of the domain values (horizontal
	// axis).
	Reversed bool `json:"reversed,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Domain") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Domain") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BasicChartDomain) MarshalJSON() ([]byte, error) {
	type noMethod BasicChartDomain
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BasicChartSeries: A single series of data in a chart.
// For example, if charting stock prices over time, multiple series may
// exist,
// one for the "Open Price", "High Price", "Low Price" and "Close
// Price".
type BasicChartSeries struct {
	// Series: The data being visualized in this chart series.
	Series *ChartData `json:"series,omitempty"`

	// TargetAxis: The minor axis that will specify the range of values for
	// this series.
	// For example, if charting stocks over time, the "Volume" series
	// may want to be pinned to the right with the prices pinned to the
	// left,
	// because the scale of trading volume is different than the scale
	// of
	// prices.
	// It is an error to specify an axis that isn't a valid minor axis
	// for the chart's type.
	//
	// Possible values:
	//   "BASIC_CHART_AXIS_POSITION_UNSPECIFIED" - Default value, do not
	// use.
	//   "BOTTOM_AXIS" - The axis rendered at the bottom of a chart.
	// For most charts, this is the standard major axis.
	// For bar charts, this is a minor axis.
	//   "LEFT_AXIS" - The axis rendered at the left of a chart.
	// For most charts, this is a minor axis.
	// For bar charts, this is the standard major axis.
	//   "RIGHT_AXIS" - The axis rendered at the right of a chart.
	// For most charts, this is a minor axis.
	// For bar charts, this is an unusual major axis.
	TargetAxis string `json:"targetAxis,omitempty"`

	// Type: The type of this series. Valid only if the
	// chartType is
	// COMBO.
	// Different types will change the way the series is visualized.
	// Only LINE, AREA,
	// and COLUMN are supported.
	//
	// Possible values:
	//   "BASIC_CHART_TYPE_UNSPECIFIED" - Default value, do not use.
	//   "BAR" - A <a href="/chart/interactive/docs/gallery/barchart">bar
	// chart</a>.
	//   "LINE" - A <a href="/chart/interactive/docs/gallery/linechart">line
	// chart</a>.
	//   "AREA" - An <a
	// href="/chart/interactive/docs/gallery/areachart">area chart</a>.
	//   "COLUMN" - A <a
	// href="/chart/interactive/docs/gallery/columnchart">column chart</a>.
	//   "SCATTER" - A <a
	// href="/chart/interactive/docs/gallery/scatterchart">scatter
	// chart</a>.
	//   "COMBO" - A <a
	// href="/chart/interactive/docs/gallery/combochart">combo chart</a>.
	//   "STEPPED_AREA" - A <a
	// href="/chart/interactive/docs/gallery/steppedareachart">stepped area
	// chart</a>.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Series") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Series") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BasicChartSeries) MarshalJSON() ([]byte, error) {
	type noMethod BasicChartSeries
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BasicChartSpec: The specification for a basic chart.  See
// BasicChartType for the list
// of charts this supports.
type BasicChartSpec struct {
	// Axis: The axis on the chart.
	Axis []*BasicChartAxis `json:"axis,omitempty"`

	// ChartType: The type of the chart.
	//
	// Possible values:
	//   "BASIC_CHART_TYPE_UNSPECIFIED" - Default value, do not use.
	//   "BAR" - A <a href="/chart/interactive/docs/gallery/barchart">bar
	// chart</a>.
	//   "LINE" - A <a href="/chart/interactive/docs/gallery/linechart">line
	// chart</a>.
	//   "AREA" - An <a
	// href="/chart/interactive/docs/gallery/areachart">area chart</a>.
	//   "COLUMN" - A <a
	// href="/chart/interactive/docs/gallery/columnchart">column chart</a>.
	//   "SCATTER" - A <a
	// href="/chart/interactive/docs/gallery/scatterchart">scatter
	// chart</a>.
	//   "COMBO" - A <a
	// href="/chart/interactive/docs/gallery/combochart">combo chart</a>.
	//   "STEPPED_AREA" - A <a
	// href="/chart/interactive/docs/gallery/steppedareachart">stepped area
	// chart</a>.
	ChartType string `json:"chartType,omitempty"`

	// Domains: The domain of data this is charting.
	// Only a single domain is supported.
	Domains []*BasicChartDomain `json:"domains,omitempty"`

	// HeaderCount: The number of rows or columns in the data that are
	// "headers".
	// If not set, Google Sheets will guess how many rows are headers
	// based
	// on the data.
	//
	// (Note that BasicChartAxis.title may override the axis title
	//  inferred from the header values.)
	HeaderCount int64 `json:"headerCount,omitempty"`

	// InterpolateNulls: If some values in a series are missing, gaps may
	// appear in the chart (e.g,
	// segments of lines in a line chart will be missing).  To eliminate
	// these
	// gaps set this to true.
	// Applies to Line, Area, and Combo charts.
	InterpolateNulls bool `json:"interpolateNulls,omitempty"`

	// LegendPosition: The position of the chart legend.
	//
	// Possible values:
	//   "BASIC_CHART_LEGEND_POSITION_UNSPECIFIED" - Default value, do not
	// use.
	//   "BOTTOM_LEGEND" - The legend is rendered on the bottom of the
	// chart.
	//   "LEFT_LEGEND" - The legend is rendered on the left of the chart.
	//   "RIGHT_LEGEND" - The legend is rendered on the right of the chart.
	//   "TOP_LEGEND" - The legend is rendered on the top of the chart.
	//   "NO_LEGEND" - No legend is rendered.
	LegendPosition string `json:"legendPosition,omitempty"`

	// LineSmoothing: Gets whether all lines should be rendered smooth or
	// straight by default.
	// Applies to Line charts.
	LineSmoothing bool `json:"lineSmoothing,omitempty"`

	// Series: The data this chart is visualizing.
	Series []*BasicChartSeries `json:"series,omitempty"`

	// StackedType: The stacked type for charts that support vertical
	// stacking.
	// Applies to Area, Bar, Column, and Stepped Area charts.
	//
	// Possible values:
	//   "BASIC_CHART_STACKED_TYPE_UNSPECIFIED" - Default value, do not use.
	//   "NOT_STACKED" - Series are not stacked.
	//   "STACKED" - Series values are stacked, each value is rendered
	// vertically beginning
	// from the top of the value below it.
	//   "PERCENT_STACKED" - Vertical stacks are stretched to reach the top
	// of the chart, with
	// values laid out as percentages of each other.
	StackedType string `json:"stackedType,omitempty"`

	// ThreeDimensional: True to make the chart 3D.
	// Applies to Bar and Column charts.
	ThreeDimensional bool `json:"threeDimensional,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Axis") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Axis") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BasicChartSpec) MarshalJSON() ([]byte, error) {
	type noMethod BasicChartSpec
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BasicFilter: The default filter associated with a sheet.
type BasicFilter struct {
	// Criteria: The criteria for showing/hiding values per column.
	// The map's key is the column index, and the value is the criteria
	// for
	// that column.
	Criteria map[string]FilterCriteria `json:"criteria,omitempty"`

	// Range: The range the filter covers.
	Range *GridRange `json:"range,omitempty"`

	// SortSpecs: The sort order per column. Later specifications are used
	// when values
	// are equal in the earlier specifications.
	SortSpecs []*SortSpec `json:"sortSpecs,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Criteria") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Criteria") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BasicFilter) MarshalJSON() ([]byte, error) {
	type noMethod BasicFilter
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchClearValuesRequest: The request for clearing more than one range
// of values in a spreadsheet.
type BatchClearValuesRequest struct {
	// Ranges: The ranges to clear, in A1 notation.
	Ranges []string `json:"ranges,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Ranges") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Ranges") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchClearValuesRequest) MarshalJSON() ([]byte, error) {
	type noMethod BatchClearValuesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchClearValuesResponse: The response when clearing a range of
// values in a spreadsheet.
type BatchClearValuesResponse struct {
	// ClearedRanges: The ranges that were cleared, in A1 notation.
	// (If the requests were for an unbounded range or a ranger larger
	//  than the bounds of the sheet, this will be the actual ranges
	//  that were cleared, bounded to the sheet's limits.)
	ClearedRanges []string `json:"clearedRanges,omitempty"`

	// SpreadsheetId: The spreadsheet the updates were applied to.
	SpreadsheetId string `json:"spreadsheetId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ClearedRanges") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClearedRanges") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchClearValuesResponse) MarshalJSON() ([]byte, error) {
	type noMethod BatchClearValuesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchGetValuesResponse: The response when retrieving more than one
// range of values in a spreadsheet.
type BatchGetValuesResponse struct {
	// SpreadsheetId: The ID of the spreadsheet the data was retrieved from.
	SpreadsheetId string `json:"spreadsheetId,omitempty"`

	// ValueRanges: The requested values. The order of the ValueRanges is
	// the same as the
	// order of the requested ranges.
	ValueRanges []*ValueRange `json:"valueRanges,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "SpreadsheetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SpreadsheetId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchGetValuesResponse) MarshalJSON() ([]byte, error) {
	type noMethod BatchGetValuesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchUpdateSpreadsheetRequest: The request for updating any aspect of
// a spreadsheet.
type BatchUpdateSpreadsheetRequest struct {
	// IncludeSpreadsheetInResponse: Determines if the update response
	// should include the spreadsheet
	// resource.
	IncludeSpreadsheetInResponse bool `json:"includeSpreadsheetInResponse,omitempty"`

	// Requests: A list of updates to apply to the spreadsheet.
	// Requests will be applied in the order they are specified.
	// If any request is not valid, no requests will be applied.
	Requests []*Request `json:"requests,omitempty"`

	// ResponseIncludeGridData: True if grid data should be returned.
	// Meaningful only if
	// if include_spreadsheet_response is 'true'.
	// This parameter is ignored if a field mask was set in the request.
	ResponseIncludeGridData bool `json:"responseIncludeGridData,omitempty"`

	// ResponseRanges: Limits the ranges included in the response
	// spreadsheet.
	// Meaningful only if include_spreadsheet_response is 'true'.
	ResponseRanges []string `json:"responseRanges,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "IncludeSpreadsheetInResponse") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g.
	// "IncludeSpreadsheetInResponse") to include in API requests with the
	// JSON null value. By default, fields with empty values are omitted
	// from API requests. However, any field with an empty value appearing
	// in NullFields will be sent to the server as null. It is an error if a
	// field in this list has a non-empty value. This may be used to include
	// null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchUpdateSpreadsheetRequest) MarshalJSON() ([]byte, error) {
	type noMethod BatchUpdateSpreadsheetRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchUpdateSpreadsheetResponse: The reply for batch updating a
// spreadsheet.
type BatchUpdateSpreadsheetResponse struct {
	// Replies: The reply of the updates.  This maps 1:1 with the updates,
	// although
	// replies to some requests may be empty.
	Replies []*Response `json:"replies,omitempty"`

	// SpreadsheetId: The spreadsheet the updates were applied to.
	SpreadsheetId string `json:"spreadsheetId,omitempty"`

	// UpdatedSpreadsheet: The spreadsheet after updates were applied. This
	// is only set
	// if
	// [BatchUpdateSpreadsheetRequest.include_spreadsheet_in_response] is
	// `true`.
	UpdatedSpreadsheet *Spreadsheet `json:"updatedSpreadsheet,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Replies") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Replies") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchUpdateSpreadsheetResponse) MarshalJSON() ([]byte, error) {
	type noMethod BatchUpdateSpreadsheetResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchUpdateValuesRequest: The request for updating more than one
// range of values in a spreadsheet.
type BatchUpdateValuesRequest struct {
	// Data: The new values to apply to the spreadsheet.
	Data []*ValueRange `json:"data,omitempty"`

	// IncludeValuesInResponse: Determines if the update response should
	// include the values
	// of the cells that were updated. By default, responses
	// do not include the updated values. The `updatedData` field
	// within
	// each of the BatchUpdateValuesResponse.responses will contain
	// the updated values. If the range to write was larger than than the
	// range
	// actually written, the response will include all values in the
	// requested
	// range (excluding trailing empty rows and columns).
	IncludeValuesInResponse bool `json:"includeValuesInResponse,omitempty"`

	// ResponseDateTimeRenderOption: Determines how dates, times, and
	// durations in the response should be
	// rendered. This is ignored if response_value_render_option
	// is
	// FORMATTED_VALUE.
	// The default dateTime render option
	// is
	// DateTimeRenderOption.SERIAL_NUMBER.
	//
	// Possible values:
	//   "SERIAL_NUMBER" - Instructs date, time, datetime, and duration
	// fields to be output
	// as doubles in "serial number" format, as popularized by Lotus
	// 1-2-3.
	// The whole number portion of the value (left of the decimal)
	// counts
	// the days since December 30th 1899. The fractional portion (right
	// of
	// the decimal) counts the time as a fraction of the day. For
	// example,
	// January 1st 1900 at noon would be 2.5, 2 because it's 2 days
	// after
	// December 30st 1899, and .5 because noon is half a day.  February
	// 1st
	// 1900 at 3pm would be 33.625. This correctly treats the year 1900
	// as
	// not a leap year.
	//   "FORMATTED_STRING" - Instructs date, time, datetime, and duration
	// fields to be output
	// as strings in their given number format (which is dependent
	// on the spreadsheet locale).
	ResponseDateTimeRenderOption string `json:"responseDateTimeRenderOption,omitempty"`

	// ResponseValueRenderOption: Determines how values in the response
	// should be rendered.
	// The default render option is ValueRenderOption.FORMATTED_VALUE.
	//
	// Possible values:
	//   "FORMATTED_VALUE" - Values will be calculated & formatted in the
	// reply according to the
	// cell's formatting.  Formatting is based on the spreadsheet's
	// locale,
	// not the requesting user's locale.
	// For example, if `A1` is `1.23` and `A2` is `=A1` and formatted as
	// currency,
	// then `A2` would return "$1.23".
	//   "UNFORMATTED_VALUE" - Values will be calculated, but not formatted
	// in the reply.
	// For example, if `A1` is `1.23` and `A2` is `=A1` and formatted as
	// currency,
	// then `A2` would return the number `1.23`.
	//   "FORMULA" - Values will not be calculated.  The reply will include
	// the formulas.
	// For example, if `A1` is `1.23` and `A2` is `=A1` and formatted as
	// currency,
	// then A2 would return "=A1".
	ResponseValueRenderOption string `json:"responseValueRenderOption,omitempty"`

	// ValueInputOption: How the input data should be interpreted.
	//
	// Possible values:
	//   "INPUT_VALUE_OPTION_UNSPECIFIED" - Default input value. This value
	// must not be used.
	//   "RAW" - The values the user has entered will not be parsed and will
	// be stored
	// as-is.
	//   "USER_ENTERED" - The values will be parsed as if the user typed
	// them into the UI.
	// Numbers will stay as numbers, but strings may be converted to
	// numbers,
	// dates, etc. following the same rules that are applied when
	// entering
	// text into a cell via the Google Sheets UI.
	ValueInputOption string `json:"valueInputOption,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Data") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Data") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchUpdateValuesRequest) MarshalJSON() ([]byte, error) {
	type noMethod BatchUpdateValuesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BatchUpdateValuesResponse: The response when updating a range of
// values in a spreadsheet.
type BatchUpdateValuesResponse struct {
	// Responses: One UpdateValuesResponse per requested range, in the same
	// order as
	// the requests appeared.
	Responses []*UpdateValuesResponse `json:"responses,omitempty"`

	// SpreadsheetId: The spreadsheet the updates were applied to.
	SpreadsheetId string `json:"spreadsheetId,omitempty"`

	// TotalUpdatedCells: The total number of cells updated.
	TotalUpdatedCells int64 `json:"totalUpdatedCells,omitempty"`

	// TotalUpdatedColumns: The total number of columns where at least one
	// cell in the column was
	// updated.
	TotalUpdatedColumns int64 `json:"totalUpdatedColumns,omitempty"`

	// TotalUpdatedRows: The total number of rows where at least one cell in
	// the row was updated.
	TotalUpdatedRows int64 `json:"totalUpdatedRows,omitempty"`

	// TotalUpdatedSheets: The total number of sheets where at least one
	// cell in the sheet was
	// updated.
	TotalUpdatedSheets int64 `json:"totalUpdatedSheets,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Responses") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Responses") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BatchUpdateValuesResponse) MarshalJSON() ([]byte, error) {
	type noMethod BatchUpdateValuesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BooleanCondition: A condition that can evaluate to true or
// false.
// BooleanConditions are used by conditional formatting,
// data validation, and the criteria in filters.
type BooleanCondition struct {
	// Type: The type of condition.
	//
	// Possible values:
	//   "CONDITION_TYPE_UNSPECIFIED" - The default value, do not use.
	//   "NUMBER_GREATER" - The cell's value must be greater than the
	// condition's value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue.
	//   "NUMBER_GREATER_THAN_EQ" - The cell's value must be greater than or
	// equal to the condition's value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue.
	//   "NUMBER_LESS" - The cell's value must be less than the condition's
	// value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue.
	//   "NUMBER_LESS_THAN_EQ" - The cell's value must be less than or equal
	// to the condition's value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue.
	//   "NUMBER_EQ" - The cell's value must be equal to the condition's
	// value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue.
	//   "NUMBER_NOT_EQ" - The cell's value must be not equal to the
	// condition's value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue.
	//   "NUMBER_BETWEEN" - The cell's value must be between the two
	// condition values.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires exactly two ConditionValues.
	//   "NUMBER_NOT_BETWEEN" - The cell's value must not be between the two
	// condition values.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires exactly two ConditionValues.
	//   "TEXT_CONTAINS" - The cell's value must contain the condition's
	// value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue.
	//   "TEXT_NOT_CONTAINS" - The cell's value must not contain the
	// condition's value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue.
	//   "TEXT_STARTS_WITH" - The cell's value must start with the
	// condition's value.
	// Supported by conditional formatting and filters.
	// Requires a single ConditionValue.
	//   "TEXT_ENDS_WITH" - The cell's value must end with the condition's
	// value.
	// Supported by conditional formatting and filters.
	// Requires a single ConditionValue.
	//   "TEXT_EQ" - The cell's value must be exactly the condition's
	// value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue.
	//   "TEXT_IS_EMAIL" - The cell's value must be a valid email
	// address.
	// Supported by data validation.
	// Requires no ConditionValues.
	//   "TEXT_IS_URL" - The cell's value must be a valid URL.
	// Supported by data validation.
	// Requires no ConditionValues.
	//   "DATE_EQ" - The cell's value must be the same date as the
	// condition's value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue.
	//   "DATE_BEFORE" - The cell's value must be before the date of the
	// condition's value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue
	// that may be a relative date.
	//   "DATE_AFTER" - The cell's value must be after the date of the
	// condition's value.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue
	// that may be a relative date.
	//   "DATE_ON_OR_BEFORE" - The cell's value must be on or before the
	// date of the condition's value.
	// Supported by data validation.
	// Requires a single ConditionValue
	// that may be a relative date.
	//   "DATE_ON_OR_AFTER" - The cell's value must be on or after the date
	// of the condition's value.
	// Supported by data validation.
	// Requires a single ConditionValue
	// that may be a relative date.
	//   "DATE_BETWEEN" - The cell's value must be between the dates of the
	// two condition values.
	// Supported by data validation.
	// Requires exactly two ConditionValues.
	//   "DATE_NOT_BETWEEN" - The cell's value must be outside the dates of
	// the two condition values.
	// Supported by data validation.
	// Requires exactly two ConditionValues.
	//   "DATE_IS_VALID" - The cell's value must be a date.
	// Supported by data validation.
	// Requires no ConditionValues.
	//   "ONE_OF_RANGE" - The cell's value must be listed in the grid in
	// condition value's range.
	// Supported by data validation.
	// Requires a single ConditionValue,
	// and the value must be a valid range in A1 notation.
	//   "ONE_OF_LIST" - The cell's value must in the list of condition
	// values.
	// Supported by data validation.
	// Supports any number of condition values,
	// one per item in the list.
	// Formulas are not supported in the values.
	//   "BLANK" - The cell's value must be empty.
	// Supported by conditional formatting and filters.
	// Requires no ConditionValues.
	//   "NOT_BLANK" - The cell's value must not be empty.
	// Supported by conditional formatting and filters.
	// Requires no ConditionValues.
	//   "CUSTOM_FORMULA" - The condition's formula must evaluate to
	// true.
	// Supported by data validation, conditional formatting and
	// filters.
	// Requires a single ConditionValue.
	Type string `json:"type,omitempty"`

	// Values: The values of the condition. The number of supported values
	// depends
	// on the condition type.  Some support zero values,
	// others one or two values,
	// and ConditionType.ONE_OF_LIST supports an arbitrary number of values.
	Values []*ConditionValue `json:"values,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Type") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Type") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BooleanCondition) MarshalJSON() ([]byte, error) {
	type noMethod BooleanCondition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BooleanRule: A rule that may or may not match, depending on the
// condition.
type BooleanRule struct {
	// Condition: The condition of the rule. If the condition evaluates to
	// true,
	// the format will be applied.
	Condition *BooleanCondition `json:"condition,omitempty"`

	// Format: The format to apply.
	// Conditional formatting can only apply a subset of formatting:
	// bold, italic,
	// strikethrough,
	// foreground color &
	// background color.
	Format *CellFormat `json:"format,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Condition") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Condition") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BooleanRule) MarshalJSON() ([]byte, error) {
	type noMethod BooleanRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Border: A border along a cell.
type Border struct {
	// Color: The color of the border.
	Color *Color `json:"color,omitempty"`

	// Style: The style of the border.
	//
	// Possible values:
	//   "STYLE_UNSPECIFIED" - The style is not specified. Do not use this.
	//   "DOTTED" - The border is dotted.
	//   "DASHED" - The border is dashed.
	//   "SOLID" - The border is a thin solid line.
	//   "SOLID_MEDIUM" - The border is a medium solid line.
	//   "SOLID_THICK" - The border is a thick solid line.
	//   "NONE" - No border.
	// Used only when updating a border in order to erase it.
	//   "DOUBLE" - The border is two solid lines.
	Style string `json:"style,omitempty"`

	// Width: The width of the border, in pixels.
	// Deprecated; the width is determined by the "style" field.
	Width int64 `json:"width,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Color") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Color") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Border) MarshalJSON() ([]byte, error) {
	type noMethod Border
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Borders: The borders of the cell.
type Borders struct {
	// Bottom: The bottom border of the cell.
	Bottom *Border `json:"bottom,omitempty"`

	// Left: The left border of the cell.
	Left *Border `json:"left,omitempty"`

	// Right: The right border of the cell.
	Right *Border `json:"right,omitempty"`

	// Top: The top border of the cell.
	Top *Border `json:"top,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Bottom") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Bottom") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Borders) MarshalJSON() ([]byte, error) {
	type noMethod Borders
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BubbleChartSpec: A <a
// href="/chart/interactive/docs/gallery/bubblechart">bubble chart</a>.
type BubbleChartSpec struct {
	// BubbleBorderColor: The bubble border color.
	BubbleBorderColor *Color `json:"bubbleBorderColor,omitempty"`

	// BubbleLabels: The data containing the bubble labels.  These do not
	// need to be unique.
	BubbleLabels *ChartData `json:"bubbleLabels,omitempty"`

	// BubbleMaxRadiusSize: The max radius size of the bubbles, in
	// pixels.
	// If specified, the field must be a positive value.
	BubbleMaxRadiusSize int64 `json:"bubbleMaxRadiusSize,omitempty"`

	// BubbleMinRadiusSize: The minimum radius size of the bubbles, in
	// pixels.
	// If specific, the field must be a positive value.
	BubbleMinRadiusSize int64 `json:"bubbleMinRadiusSize,omitempty"`

	// BubbleOpacity: The opacity of the bubbles between 0 and 1.0.
	// 0 is fully transparent and 1 is fully opaque.
	BubbleOpacity float64 `json:"bubbleOpacity,omitempty"`

	// BubbleSizes: The data contianing the bubble sizes.  Bubble sizes are
	// used to draw
	// the bubbles at different sizes relative to each other.
	// If specified, group_ids must also be specified.  This field
	// is
	// optional.
	BubbleSizes *ChartData `json:"bubbleSizes,omitempty"`

	// BubbleTextStyle: The format of the text inside the bubbles.
	// Underline and Strikethrough are not supported.
	BubbleTextStyle *TextFormat `json:"bubbleTextStyle,omitempty"`

	// Domain: The data containing the bubble x-values.  These values locate
	// the bubbles
	// in the chart horizontally.
	Domain *ChartData `json:"domain,omitempty"`

	// GroupIds: The data containing the bubble group IDs. All bubbles with
	// the same group
	// ID will be drawn in the same color. If bubble_sizes is specified
	// then
	// this field must also be specified but may contain blank values.
	// This field is optional.
	GroupIds *ChartData `json:"groupIds,omitempty"`

	// LegendPosition: Where the legend of the chart should be drawn.
	//
	// Possible values:
	//   "BUBBLE_CHART_LEGEND_POSITION_UNSPECIFIED" - Default value, do not
	// use.
	//   "BOTTOM_LEGEND" - The legend is rendered on the bottom of the
	// chart.
	//   "LEFT_LEGEND" - The legend is rendered on the left of the chart.
	//   "RIGHT_LEGEND" - The legend is rendered on the right of the chart.
	//   "TOP_LEGEND" - The legend is rendered on the top of the chart.
	//   "NO_LEGEND" - No legend is rendered.
	//   "INSIDE_LEGEND" - The legend is rendered inside the chart area.
	LegendPosition string `json:"legendPosition,omitempty"`

	// Series: The data contianing the bubble y-values.  These values locate
	// the bubbles
	// in the chart vertically.
	Series *ChartData `json:"series,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BubbleBorderColor")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BubbleBorderColor") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BubbleChartSpec) MarshalJSON() ([]byte, error) {
	type noMethod BubbleChartSpec
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *BubbleChartSpec) UnmarshalJSON(data []byte) error {
	type noMethod BubbleChartSpec
	var s1 struct {
		BubbleOpacity gensupport.JSONFloat64 `json:"bubbleOpacity"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.BubbleOpacity = float64(s1.BubbleOpacity)
	return nil
}

// CandlestickChartSpec: A <a
// href="/chart/interactive/docs/gallery/candlestickchart">candlestick
// chart</a>.
type CandlestickChartSpec struct {
	// Data: The Candlestick chart data.
	// Only one CandlestickData is supported.
	Data []*CandlestickData `json:"data,omitempty"`

	// Domain: The domain data (horizontal axis) for the candlestick chart.
	// String data
	// will be treated as discrete labels, other data will be treated
	// as
	// continuous values.
	Domain *CandlestickDomain `json:"domain,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Data") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Data") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CandlestickChartSpec) MarshalJSON() ([]byte, error) {
	type noMethod CandlestickChartSpec
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CandlestickData: The Candlestick chart data, each containing the low,
// open, close, and high
// values for a series.
type CandlestickData struct {
	// CloseSeries: The range data (vertical axis) for the close/final value
	// for each candle.
	// This is the top of the candle body.  If greater than the open value
	// the
	// candle will be filled.  Otherwise the candle will be hollow.
	CloseSeries *CandlestickSeries `json:"closeSeries,omitempty"`

	// HighSeries: The range data (vertical axis) for the high/maximum value
	// for each
	// candle. This is the top of the candle's center line.
	HighSeries *CandlestickSeries `json:"highSeries,omitempty"`

	// LowSeries: The range data (vertical axis) for the low/minimum value
	// for each candle.
	// This is the bottom of the candle's center line.
	LowSeries *CandlestickSeries `json:"lowSeries,omitempty"`

	// OpenSeries: The range data (vertical axis) for the open/initial value
	// for each
	// candle. This is the bottom of the candle body.  If less than the
	// close
	// value the candle will be filled.  Otherwise the candle will be
	// hollow.
	OpenSeries *CandlestickSeries `json:"openSeries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CloseSeries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CloseSeries") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CandlestickData) MarshalJSON() ([]byte, error) {
	type noMethod CandlestickData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CandlestickDomain: The domain of a CandlestickChart.
type CandlestickDomain struct {
	// Data: The data of the CandlestickDomain.
	Data *ChartData `json:"data,omitempty"`

	// Reversed: True to reverse the order of the domain values (horizontal
	// axis).
	Reversed bool `json:"reversed,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Data") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Data") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CandlestickDomain) MarshalJSON() ([]byte, error) {
	type noMethod CandlestickDomain
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CandlestickSeries: The series of a CandlestickData.
type CandlestickSeries struct {
	// Data: The data of the CandlestickSeries.
	Data *ChartData `json:"data,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Data") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Data") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CandlestickSeries) MarshalJSON() ([]byte, error) {
	type noMethod CandlestickSeries
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CellData: Data about a specific cell.
type CellData struct {
	// DataValidation: A data validation rule on the cell, if any.
	//
	// When writing, the new data validation rule will overwrite any prior
	// rule.
	DataValidation *DataValidationRule `json:"dataValidation,omitempty"`

	// EffectiveFormat: The effective format being used by the cell.
	// This includes the results of applying any conditional formatting
	// and,
	// if the cell contains a formula, the computed number format.
	// If the effective format is the default format, effective format
	// will
	// not be written.
	// This field is read-only.
	EffectiveFormat *CellFormat `json:"effectiveFormat,omitempty"`

	// EffectiveValue: The effective value of the cell. For cells with
	// formulas, this will be
	// the calculated value.  For cells with literals, this will be
	// the same as the user_entered_value.
	// This field is read-only.
	EffectiveValue *ExtendedValue `json:"effectiveValue,omitempty"`

	// FormattedValue: The formatted value of the cell.
	// This is the value as it's shown to the user.
	// This field is read-only.
	FormattedValue string `json:"formattedValue,omitempty"`

	// Hyperlink: A hyperlink this cell points to, if any.
	// This field is read-only.  (To set it, use a `=HYPERLINK` formula
	// in the userEnteredValue.formulaValue
	// field.)
	Hyperlink string `json:"hyperlink,omitempty"`

	// Note: Any note on the cell.
	Note string `json:"note,omitempty"`

	// PivotTable: A pivot table anchored at this cell. The size of pivot
	// table itself
	// is computed dynamically based on its data, grouping, filters,
	// values,
	// etc. Only the top-left cell of the pivot table contains the pivot
	// table
	// definition. The other cells will contain the calculated values of
	// the
	// results of the pivot in their effective_value fields.
	PivotTable *PivotTable `json:"pivotTable,omitempty"`

	// TextFormatRuns: Runs of rich text applied to subsections of the cell.
	//  Runs are only valid
	// on user entered strings, not formulas, bools, or numbers.
	// Runs start at specific indexes in the text and continue until the
	// next
	// run. Properties of a run will continue unless explicitly changed
	// in a subsequent run (and properties of the first run will
	// continue
	// the properties of the cell unless explicitly changed).
	//
	// When writing, the new runs will overwrite any prior runs.  When
	// writing a
	// new user_entered_value, previous runs will be erased.
	TextFormatRuns []*TextFormatRun `json:"textFormatRuns,omitempty"`

	// UserEnteredFormat: The format the user entered for the cell.
	//
	// When writing, the new format will be merged with the existing format.
	UserEnteredFormat *CellFormat `json:"userEnteredFormat,omitempty"`

	// UserEnteredValue: The value the user entered in the cell. e.g,
	// `1234`, `'Hello'`, or `=NOW()`
	// Note: Dates, Times and DateTimes are represented as doubles in
	// serial number format.
	UserEnteredValue *ExtendedValue `json:"userEnteredValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DataValidation") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DataValidation") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *CellData) MarshalJSON() ([]byte, error) {
	type noMethod CellData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CellFormat: The format of a cell.
type CellFormat struct {
	// BackgroundColor: The background color of the cell.
	BackgroundColor *Color `json:"backgroundColor,omitempty"`

	// Borders: The borders of the cell.
	Borders *Borders `json:"borders,omitempty"`

	// HorizontalAlignment: The horizontal alignment of the value in the
	// cell.
	//
	// Possible values:
	//   "HORIZONTAL_ALIGN_UNSPECIFIED" - The horizontal alignment is not
	// specified. Do not use this.
	//   "LEFT" - The text is explicitly aligned to the left of the cell.
	//   "CENTER" - The text is explicitly aligned to the center of the
	// cell.
	//   "RIGHT" - The text is explicitly aligned to the right of the cell.
	HorizontalAlignment string `json:"horizontalAlignment,omitempty"`

	// HyperlinkDisplayType: How a hyperlink, if it exists, should be
	// displayed in the cell.
	//
	// Possible values:
	//   "HYPERLINK_DISPLAY_TYPE_UNSPECIFIED" - The default value: the
	// hyperlink is rendered. Do not use this.
	//   "LINKED" - A hyperlink should be explicitly rendered.
	//   "PLAIN_TEXT" - A hyperlink should not be rendered.
	HyperlinkDisplayType string `json:"hyperlinkDisplayType,omitempty"`

	// NumberFormat: A format describing how number values should be
	// represented to the user.
	NumberFormat *NumberFormat `json:"numberFormat,omitempty"`

	// Padding: The padding of the cell.
	Padding *Padding `json:"padding,omitempty"`

	// TextDirection: The direction of the text in the cell.
	//
	// Possible values:
	//   "TEXT_DIRECTION_UNSPECIFIED" - The text direction is not specified.
	// Do not use this.
	//   "LEFT_TO_RIGHT" - The text direction of left-to-right was set by
	// the user.
	//   "RIGHT_TO_LEFT" - The text direction of right-to-left was set by
	// the user.
	TextDirection string `json:"textDirection,omitempty"`

	// TextFormat: The format of the text in the cell (unless overridden by
	// a format run).
	TextFormat *TextFormat `json:"textFormat,omitempty"`

	// TextRotation: The rotation applied to text in a cell
	TextRotation *TextRotation `json:"textRotation,omitempty"`

	// VerticalAlignment: The vertical alignment of the value in the cell.
	//
	// Possible values:
	//   "VERTICAL_ALIGN_UNSPECIFIED" - The vertical alignment is not
	// specified.  Do not use this.
	//   "TOP" - The text is explicitly aligned to the top of the cell.
	//   "MIDDLE" - The text is explicitly aligned to the middle of the
	// cell.
	//   "BOTTOM" - The text is explicitly aligned to the bottom of the
	// cell.
	VerticalAlignment string `json:"verticalAlignment,omitempty"`

	// WrapStrategy: The wrap strategy for the value in the cell.
	//
	// Possible values:
	//   "WRAP_STRATEGY_UNSPECIFIED" - The default value, do not use.
	//   "OVERFLOW_CELL" - Lines that are longer than the cell width will be
	// written in the next
	// cell over, so long as that cell is empty. If the next cell over
	// is
	// non-empty, this behaves the same as CLIP. The text will never wrap
	// to the next line unless the user manually inserts a new
	// line.
	// Example:
	//
	//     | First sentence. |
	//     | Manual newline that is very long. <- Text continues into next
	// cell
	//     | Next newline.   |
	//   "LEGACY_WRAP" - This wrap strategy represents the old Google Sheets
	// wrap strategy where
	// words that are longer than a line are clipped rather than broken.
	// This
	// strategy is not supported on all platforms and is being phased
	// out.
	// Example:
	//
	//     | Cell has a |
	//     | loooooooooo| <- Word is clipped.
	//     | word.      |
	//   "CLIP" - Lines that are longer than the cell width will be
	// clipped.
	// The text will never wrap to the next line unless the user
	// manually
	// inserts a new line.
	// Example:
	//
	//     | First sentence. |
	//     | Manual newline t| <- Text is clipped
	//     | Next newline.   |
	//   "WRAP" - Words that are longer than a line are wrapped at the
	// character level
	// rather than clipped.
	// Example:
	//
	//     | Cell has a |
	//     | loooooooooo| <- Word is broken.
	//     | ong word.  |
	WrapStrategy string `json:"wrapStrategy,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BackgroundColor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BackgroundColor") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *CellFormat) MarshalJSON() ([]byte, error) {
	type noMethod CellFormat
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ChartData: The data included in a domain or series.
type ChartData struct {
	// SourceRange: The source ranges of the data.
	SourceRange *ChartSourceRange `json:"sourceRange,omitempty"`

	// ForceSendFields is a list of field names (e.g. "SourceRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SourceRange") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ChartData) MarshalJSON() ([]byte, error) {
	type noMethod ChartData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ChartSourceRange: Source ranges for a chart.
type ChartSourceRange struct {
	// Sources: The ranges of data for a series or domain.
	// Exactly one dimension must have a length of 1,
	// and all sources in the list must have the same dimension
	// with length 1.
	// The domain (if it exists) & all series must have the same number
	// of source ranges. If using more than one source range, then the
	// source
	// range at a given offset must be contiguous across the domain and
	// series.
	//
	// For example, these are valid configurations:
	//
	//     domain sources: A1:A5
	//     series1 sources: B1:B5
	//     series2 sources: D6:D10
	//
	//     domain sources: A1:A5, C10:C12
	//     series1 sources: B1:B5, D10:D12
	//     series2 sources: C1:C5, E10:E12
	Sources []*GridRange `json:"sources,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Sources") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Sources") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ChartSourceRange) MarshalJSON() ([]byte, error) {
	type noMethod ChartSourceRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ChartSpec: The specifications of a chart.
type ChartSpec struct {
	// AltText: The alternative text that describes the chart.  This is
	// often used
	// for accessibility.
	AltText string `json:"altText,omitempty"`

	// BackgroundColor: The background color of the entire chart.
	// Not applicable to Org charts.
	BackgroundColor *Color `json:"backgroundColor,omitempty"`

	// BasicChart: A basic chart specification, can be one of many kinds of
	// charts.
	// See BasicChartType for the list of all
	// charts this supports.
	BasicChart *BasicChartSpec `json:"basicChart,omitempty"`

	// BubbleChart: A bubble chart specification.
	BubbleChart *BubbleChartSpec `json:"bubbleChart,omitempty"`

	// CandlestickChart: A candlestick chart specification.
	CandlestickChart *CandlestickChartSpec `json:"candlestickChart,omitempty"`

	// FontName: The name of the font to use by default for all chart text
	// (e.g. title,
	// axis labels, legend).  If a font is specified for a specific part of
	// the
	// chart it will override this font name.
	FontName string `json:"fontName,omitempty"`

	// HiddenDimensionStrategy: Determines how the charts will use hidden
	// rows or columns.
	//
	// Possible values:
	//   "CHART_HIDDEN_DIMENSION_STRATEGY_UNSPECIFIED" - Default value, do
	// not use.
	//   "SKIP_HIDDEN_ROWS_AND_COLUMNS" - Charts will skip hidden rows and
	// columns.
	//   "SKIP_HIDDEN_ROWS" - Charts will skip hidden rows only.
	//   "SKIP_HIDDEN_COLUMNS" - Charts will skip hidden columns only.
	//   "SHOW_ALL" - Charts will not skip any hidden rows or columns.
	HiddenDimensionStrategy string `json:"hiddenDimensionStrategy,omitempty"`

	// HistogramChart: A histogram chart specification.
	HistogramChart *HistogramChartSpec `json:"histogramChart,omitempty"`

	// Maximized: True to make a chart fill the entire space in which it's
	// rendered with
	// minimum padding.  False to use the default padding.
	// (Not applicable to Geo and Org charts.)
	Maximized bool `json:"maximized,omitempty"`

	// OrgChart: An org chart specification.
	OrgChart *OrgChartSpec `json:"orgChart,omitempty"`

	// PieChart: A pie chart specification.
	PieChart *PieChartSpec `json:"pieChart,omitempty"`

	// Title: The title of the chart.
	Title string `json:"title,omitempty"`

	// TitleTextFormat: The title text format.
	// Strikethrough and underline are not supported.
	TitleTextFormat *TextFormat `json:"titleTextFormat,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AltText") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AltText") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ChartSpec) MarshalJSON() ([]byte, error) {
	type noMethod ChartSpec
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClearBasicFilterRequest: Clears the basic filter, if any exists on
// the sheet.
type ClearBasicFilterRequest struct {
	// SheetId: The sheet ID on which the basic filter should be cleared.
	SheetId int64 `json:"sheetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "SheetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SheetId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ClearBasicFilterRequest) MarshalJSON() ([]byte, error) {
	type noMethod ClearBasicFilterRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClearValuesRequest: The request for clearing a range of values in a
// spreadsheet.
type ClearValuesRequest struct {
}

// ClearValuesResponse: The response when clearing a range of values in
// a spreadsheet.
type ClearValuesResponse struct {
	// ClearedRange: The range (in A1 notation) that was cleared.
	// (If the request was for an unbounded range or a ranger larger
	//  than the bounds of the sheet, this will be the actual range
	//  that was cleared, bounded to the sheet's limits.)
	ClearedRange string `json:"clearedRange,omitempty"`

	// SpreadsheetId: The spreadsheet the updates were applied to.
	SpreadsheetId string `json:"spreadsheetId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ClearedRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClearedRange") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ClearValuesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ClearValuesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Color: Represents a color in the RGBA color space. This
// representation is designed
// for simplicity of conversion to/from color representations in
// various
// languages over compactness; for example, the fields of this
// representation
// can be trivially provided to the constructor of "java.awt.Color" in
// Java; it
// can also be trivially provided to UIColor's
// "+colorWithRed:green:blue:alpha"
// method in iOS; and, with just a little work, it can be easily
// formatted into
// a CSS "rgba()" string in JavaScript, as well. Here are some
// examples:
//
// Example (Java):
//
//      import com.google.type.Color;
//
//      // ...
//      public static java.awt.Color fromProto(Color protocolor) {
//        float alpha = protocolor.hasAlpha()
//            ? protocolor.getAlpha().getValue()
//            : 1.0;
//
//        return new java.awt.Color(
//            protocolor.getRed(),
//            protocolor.getGreen(),
//            protocolor.getBlue(),
//            alpha);
//      }
//
//      public static Color toProto(java.awt.Color color) {
//        float red = (float) color.getRed();
//        float green = (float) color.getGreen();
//        float blue = (float) color.getBlue();
//        float denominator = 255.0;
//        Color.Builder resultBuilder =
//            Color
//                .newBuilder()
//                .setRed(red / denominator)
//                .setGreen(green / denominator)
//                .setBlue(blue / denominator);
//        int alpha = color.getAlpha();
//        if (alpha != 255) {
//          result.setAlpha(
//              FloatValue
//                  .newBuilder()
//                  .setValue(((float) alpha) / denominator)
//                  .build());
//        }
//        return resultBuilder.build();
//      }
//      // ...
//
// Example (iOS / Obj-C):
//
//      // ...
//      static UIColor* fromProto(Color* protocolor) {
//         float red = [protocolor red];
//         float green = [protocolor green];
//         float blue = [protocolor blue];
//         FloatValue* alpha_wrapper = [protocolor alpha];
//         float alpha = 1.0;
//         if (alpha_wrapper != nil) {
//           alpha = [alpha_wrapper value];
//         }
//         return [UIColor colorWithRed:red green:green blue:blue
// alpha:alpha];
//      }
//
//      static Color* toProto(UIColor* color) {
//          CGFloat red, green, blue, alpha;
//          if (![color getRed:&red green:&green blue:&blue
// alpha:&alpha]) {
//            return nil;
//          }
//          Color* result = [Color alloc] init];
//          [result setRed:red];
//          [result setGreen:green];
//          [result setBlue:blue];
//          if (alpha <= 0.9999) {
//            [result setAlpha:floatWrapperWithValue(alpha)];
//          }
//          [result autorelease];
//          return result;
//     }
//     // ...
//
//  Example (JavaScript):
//
//     // ...
//
//     var protoToCssColor = function(rgb_color) {
//        var redFrac = rgb_color.red || 0.0;
//        var greenFrac = rgb_color.green || 0.0;
//        var blueFrac = rgb_color.blue || 0.0;
//        var red = Math.floor(redFrac * 255);
//        var green = Math.floor(greenFrac * 255);
//        var blue = Math.floor(blueFrac * 255);
//
//        if (!('alpha' in rgb_color)) {
//           return rgbToCssColor_(red, green, blue);
//        }
//
//        var alphaFrac = rgb_color.alpha.value || 0.0;
//        var rgbParams = [red, green, blue].join(',');
//        return ['rgba(', rgbParams, ',', alphaFrac, ')'].join('');
//     };
//
//     var rgbToCssColor_ = function(red, green, blue) {
//       var rgbNumber = new Number((red << 16) | (green << 8) | blue);
//       var hexString = rgbNumber.toString(16);
//       var missingZeros = 6 - hexString.length;
//       var resultBuilder = ['#'];
//       for (var i = 0; i < missingZeros; i++) {
//          resultBuilder.push('0');
//       }
//       resultBuilder.push(hexString);
//       return resultBuilder.join('');
//     };
//
//     // ...
type Color struct {
	// Alpha: The fraction of this color that should be applied to the
	// pixel. That is,
	// the final pixel color is defined by the equation:
	//
	//   pixel color = alpha * (this color) + (1.0 - alpha) * (background
	// color)
	//
	// This means that a value of 1.0 corresponds to a solid color,
	// whereas
	// a value of 0.0 corresponds to a completely transparent color.
	// This
	// uses a wrapper message rather than a simple float scalar so that it
	// is
	// possible to distinguish between a default value and the value being
	// unset.
	// If omitted, this color object is to be rendered as a solid color
	// (as if the alpha value had been explicitly given with a value of
	// 1.0).
	Alpha float64 `json:"alpha,omitempty"`

	// Blue: The amount of blue in the color as a value in the interval [0,
	// 1].
	Blue float64 `json:"blue,omitempty"`

	// Green: The amount of green in the color as a value in the interval
	// [0, 1].
	Green float64 `json:"green,omitempty"`

	// Red: The amount of red in the color as a value in the interval [0,
	// 1].
	Red float64 `json:"red,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Alpha") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Alpha") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Color) MarshalJSON() ([]byte, error) {
	type noMethod Color
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *Color) UnmarshalJSON(data []byte) error {
	type noMethod Color
	var s1 struct {
		Alpha gensupport.JSONFloat64 `json:"alpha"`
		Blue  gensupport.JSONFloat64 `json:"blue"`
		Green gensupport.JSONFloat64 `json:"green"`
		Red   gensupport.JSONFloat64 `json:"red"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Alpha = float64(s1.Alpha)
	s.Blue = float64(s1.Blue)
	s.Green = float64(s1.Green)
	s.Red = float64(s1.Red)
	return nil
}

// ConditionValue: The value of the condition.
type ConditionValue struct {
	// RelativeDate: A relative date (based on the current date).
	// Valid only if the type is
	// DATE_BEFORE,
	// DATE_AFTER,
	// DATE_ON_OR_BEFORE or
	// DATE_ON_OR_AFTER.
	//
	// Relative dates are not supported in data validation.
	// They are supported only in conditional formatting and
	// conditional filters.
	//
	// Possible values:
	//   "RELATIVE_DATE_UNSPECIFIED" - Default value, do not use.
	//   "PAST_YEAR" - The value is one year before today.
	//   "PAST_MONTH" - The value is one month before today.
	//   "PAST_WEEK" - The value is one week before today.
	//   "YESTERDAY" - The value is yesterday.
	//   "TODAY" - The value is today.
	//   "TOMORROW" - The value is tomorrow.
	RelativeDate string `json:"relativeDate,omitempty"`

	// UserEnteredValue: A value the condition is based on.
	// The value will be parsed as if the user typed into a cell.
	// Formulas are supported (and must begin with an `=`).
	UserEnteredValue string `json:"userEnteredValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "RelativeDate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "RelativeDate") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ConditionValue) MarshalJSON() ([]byte, error) {
	type noMethod ConditionValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ConditionalFormatRule: A rule describing a conditional format.
type ConditionalFormatRule struct {
	// BooleanRule: The formatting is either "on" or "off" according to the
	// rule.
	BooleanRule *BooleanRule `json:"booleanRule,omitempty"`

	// GradientRule: The formatting will vary based on the gradients in the
	// rule.
	GradientRule *GradientRule `json:"gradientRule,omitempty"`

	// Ranges: The ranges that will be formatted if the condition is
	// true.
	// All the ranges must be on the same grid.
	Ranges []*GridRange `json:"ranges,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BooleanRule") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BooleanRule") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ConditionalFormatRule) MarshalJSON() ([]byte, error) {
	type noMethod ConditionalFormatRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CopyPasteRequest: Copies data from the source to the destination.
type CopyPasteRequest struct {
	// Destination: The location to paste to. If the range covers a span
	// that's
	// a multiple of the source's height or width, then the
	// data will be repeated to fill in the destination range.
	// If the range is smaller than the source range, the entire
	// source data will still be copied (beyond the end of the destination
	// range).
	Destination *GridRange `json:"destination,omitempty"`

	// PasteOrientation: How that data should be oriented when pasting.
	//
	// Possible values:
	//   "NORMAL" - Paste normally.
	//   "TRANSPOSE" - Paste transposed, where all rows become columns and
	// vice versa.
	PasteOrientation string `json:"pasteOrientation,omitempty"`

	// PasteType: What kind of data to paste.
	//
	// Possible values:
	//   "PASTE_NORMAL" - Paste values, formulas, formats, and merges.
	//   "PASTE_VALUES" - Paste the values ONLY without formats, formulas,
	// or merges.
	//   "PASTE_FORMAT" - Paste the format and data validation only.
	//   "PASTE_NO_BORDERS" - Like PASTE_NORMAL but without borders.
	//   "PASTE_FORMULA" - Paste the formulas only.
	//   "PASTE_DATA_VALIDATION" - Paste the data validation only.
	//   "PASTE_CONDITIONAL_FORMATTING" - Paste the conditional formatting
	// rules only.
	PasteType string `json:"pasteType,omitempty"`

	// Source: The source range to copy.
	Source *GridRange `json:"source,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Destination") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Destination") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CopyPasteRequest) MarshalJSON() ([]byte, error) {
	type noMethod CopyPasteRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CopySheetToAnotherSpreadsheetRequest: The request to copy a sheet
// across spreadsheets.
type CopySheetToAnotherSpreadsheetRequest struct {
	// DestinationSpreadsheetId: The ID of the spreadsheet to copy the sheet
	// to.
	DestinationSpreadsheetId string `json:"destinationSpreadsheetId,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "DestinationSpreadsheetId") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DestinationSpreadsheetId")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *CopySheetToAnotherSpreadsheetRequest) MarshalJSON() ([]byte, error) {
	type noMethod CopySheetToAnotherSpreadsheetRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CutPasteRequest: Moves data from the source to the destination.
type CutPasteRequest struct {
	// Destination: The top-left coordinate where the data should be pasted.
	Destination *GridCoordinate `json:"destination,omitempty"`

	// PasteType: What kind of data to paste.  All the source data will be
	// cut, regardless
	// of what is pasted.
	//
	// Possible values:
	//   "PASTE_NORMAL" - Paste values, formulas, formats, and merges.
	//   "PASTE_VALUES" - Paste the values ONLY without formats, formulas,
	// or merges.
	//   "PASTE_FORMAT" - Paste the format and data validation only.
	//   "PASTE_NO_BORDERS" - Like PASTE_NORMAL but without borders.
	//   "PASTE_FORMULA" - Paste the formulas only.
	//   "PASTE_DATA_VALIDATION" - Paste the data validation only.
	//   "PASTE_CONDITIONAL_FORMATTING" - Paste the conditional formatting
	// rules only.
	PasteType string `json:"pasteType,omitempty"`

	// Source: The source data to cut.
	Source *GridRange `json:"source,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Destination") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Destination") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CutPasteRequest) MarshalJSON() ([]byte, error) {
	type noMethod CutPasteRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DataValidationRule: A data validation rule.
type DataValidationRule struct {
	// Condition: The condition that data in the cell must match.
	Condition *BooleanCondition `json:"condition,omitempty"`

	// InputMessage: A message to show the user when adding data to the
	// cell.
	InputMessage string `json:"inputMessage,omitempty"`

	// ShowCustomUi: True if the UI should be customized based on the kind
	// of condition.
	// If true, "List" conditions will show a dropdown.
	ShowCustomUi bool `json:"showCustomUi,omitempty"`

	// Strict: True if invalid data should be rejected.
	Strict bool `json:"strict,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Condition") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Condition") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DataValidationRule) MarshalJSON() ([]byte, error) {
	type noMethod DataValidationRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeleteBandingRequest: Removes the banded range with the given ID from
// the spreadsheet.
type DeleteBandingRequest struct {
	// BandedRangeId: The ID of the banded range to delete.
	BandedRangeId int64 `json:"bandedRangeId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BandedRangeId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BandedRangeId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeleteBandingRequest) MarshalJSON() ([]byte, error) {
	type noMethod DeleteBandingRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeleteConditionalFormatRuleRequest: Deletes a conditional format rule
// at the given index.
// All subsequent rules' indexes are decremented.
type DeleteConditionalFormatRuleRequest struct {
	// Index: The zero-based index of the rule to be deleted.
	Index int64 `json:"index,omitempty"`

	// SheetId: The sheet the rule is being deleted from.
	SheetId int64 `json:"sheetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Index") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Index") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeleteConditionalFormatRuleRequest) MarshalJSON() ([]byte, error) {
	type noMethod DeleteConditionalFormatRuleRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeleteConditionalFormatRuleResponse: The result of deleting a
// conditional format rule.
type DeleteConditionalFormatRuleResponse struct {
	// Rule: The rule that was deleted.
	Rule *ConditionalFormatRule `json:"rule,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Rule") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Rule") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeleteConditionalFormatRuleResponse) MarshalJSON() ([]byte, error) {
	type noMethod DeleteConditionalFormatRuleResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeleteDimensionRequest: Deletes the dimensions from the sheet.
type DeleteDimensionRequest struct {
	// Range: The dimensions to delete from the sheet.
	Range *DimensionRange `json:"range,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Range") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Range") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeleteDimensionRequest) MarshalJSON() ([]byte, error) {
	type noMethod DeleteDimensionRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeleteEmbeddedObjectRequest: Deletes the embedded object with the
// given ID.
type DeleteEmbeddedObjectRequest struct {
	// ObjectId: The ID of the embedded object to delete.
	ObjectId int64 `json:"objectId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ObjectId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ObjectId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeleteEmbeddedObjectRequest) MarshalJSON() ([]byte, error) {
	type noMethod DeleteEmbeddedObjectRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeleteFilterViewRequest: Deletes a particular filter view.
type DeleteFilterViewRequest struct {
	// FilterId: The ID of the filter to delete.
	FilterId int64 `json:"filterId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FilterId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FilterId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeleteFilterViewRequest) MarshalJSON() ([]byte, error) {
	type noMethod DeleteFilterViewRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeleteNamedRangeRequest: Removes the named range with the given ID
// from the spreadsheet.
type DeleteNamedRangeRequest struct {
	// NamedRangeId: The ID of the named range to delete.
	NamedRangeId string `json:"namedRangeId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NamedRangeId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NamedRangeId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeleteNamedRangeRequest) MarshalJSON() ([]byte, error) {
	type noMethod DeleteNamedRangeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeleteProtectedRangeRequest: Deletes the protected range with the
// given ID.
type DeleteProtectedRangeRequest struct {
	// ProtectedRangeId: The ID of the protected range to delete.
	ProtectedRangeId int64 `json:"protectedRangeId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ProtectedRangeId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ProtectedRangeId") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DeleteProtectedRangeRequest) MarshalJSON() ([]byte, error) {
	type noMethod DeleteProtectedRangeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeleteRangeRequest: Deletes a range of cells, shifting other cells
// into the deleted area.
type DeleteRangeRequest struct {
	// Range: The range of cells to delete.
	Range *GridRange `json:"range,omitempty"`

	// ShiftDimension: The dimension from which deleted cells will be
	// replaced with.
	// If ROWS, existing cells will be shifted upward to
	// replace the deleted cells. If COLUMNS, existing cells
	// will be shifted left to replace the deleted cells.
	//
	// Possible values:
	//   "DIMENSION_UNSPECIFIED" - The default value, do not use.
	//   "ROWS" - Operates on the rows of a sheet.
	//   "COLUMNS" - Operates on the columns of a sheet.
	ShiftDimension string `json:"shiftDimension,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Range") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Range") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeleteRangeRequest) MarshalJSON() ([]byte, error) {
	type noMethod DeleteRangeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DeleteSheetRequest: Deletes the requested sheet.
type DeleteSheetRequest struct {
	// SheetId: The ID of the sheet to delete.
	SheetId int64 `json:"sheetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "SheetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SheetId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DeleteSheetRequest) MarshalJSON() ([]byte, error) {
	type noMethod DeleteSheetRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DimensionProperties: Properties about a dimension.
type DimensionProperties struct {
	// HiddenByFilter: True if this dimension is being filtered.
	// This field is read-only.
	HiddenByFilter bool `json:"hiddenByFilter,omitempty"`

	// HiddenByUser: True if this dimension is explicitly hidden.
	HiddenByUser bool `json:"hiddenByUser,omitempty"`

	// PixelSize: The height (if a row) or width (if a column) of the
	// dimension in pixels.
	PixelSize int64 `json:"pixelSize,omitempty"`

	// ForceSendFields is a list of field names (e.g. "HiddenByFilter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "HiddenByFilter") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DimensionProperties) MarshalJSON() ([]byte, error) {
	type noMethod DimensionProperties
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DimensionRange: A range along a single dimension on a sheet.
// All indexes are zero-based.
// Indexes are half open: the start index is inclusive
// and the end index is exclusive.
// Missing indexes indicate the range is unbounded on that side.
type DimensionRange struct {
	// Dimension: The dimension of the span.
	//
	// Possible values:
	//   "DIMENSION_UNSPECIFIED" - The default value, do not use.
	//   "ROWS" - Operates on the rows of a sheet.
	//   "COLUMNS" - Operates on the columns of a sheet.
	Dimension string `json:"dimension,omitempty"`

	// EndIndex: The end (exclusive) of the span, or not set if unbounded.
	EndIndex int64 `json:"endIndex,omitempty"`

	// SheetId: The sheet this span is on.
	SheetId int64 `json:"sheetId,omitempty"`

	// StartIndex: The start (inclusive) of the span, or not set if
	// unbounded.
	StartIndex int64 `json:"startIndex,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Dimension") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Dimension") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DimensionRange) MarshalJSON() ([]byte, error) {
	type noMethod DimensionRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DuplicateFilterViewRequest: Duplicates a particular filter view.
type DuplicateFilterViewRequest struct {
	// FilterId: The ID of the filter being duplicated.
	FilterId int64 `json:"filterId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FilterId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FilterId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DuplicateFilterViewRequest) MarshalJSON() ([]byte, error) {
	type noMethod DuplicateFilterViewRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DuplicateFilterViewResponse: The result of a filter view being
// duplicated.
type DuplicateFilterViewResponse struct {
	// Filter: The newly created filter.
	Filter *FilterView `json:"filter,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Filter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Filter") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DuplicateFilterViewResponse) MarshalJSON() ([]byte, error) {
	type noMethod DuplicateFilterViewResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DuplicateSheetRequest: Duplicates the contents of a sheet.
type DuplicateSheetRequest struct {
	// InsertSheetIndex: The zero-based index where the new sheet should be
	// inserted.
	// The index of all sheets after this are incremented.
	InsertSheetIndex int64 `json:"insertSheetIndex,omitempty"`

	// NewSheetId: If set, the ID of the new sheet. If not set, an ID is
	// chosen.
	// If set, the ID must not conflict with any existing sheet ID.
	// If set, it must be non-negative.
	NewSheetId int64 `json:"newSheetId,omitempty"`

	// NewSheetName: The name of the new sheet.  If empty, a new name is
	// chosen for you.
	NewSheetName string `json:"newSheetName,omitempty"`

	// SourceSheetId: The sheet to duplicate.
	SourceSheetId int64 `json:"sourceSheetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InsertSheetIndex") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InsertSheetIndex") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DuplicateSheetRequest) MarshalJSON() ([]byte, error) {
	type noMethod DuplicateSheetRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DuplicateSheetResponse: The result of duplicating a sheet.
type DuplicateSheetResponse struct {
	// Properties: The properties of the duplicate sheet.
	Properties *SheetProperties `json:"properties,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Properties") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Properties") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DuplicateSheetResponse) MarshalJSON() ([]byte, error) {
	type noMethod DuplicateSheetResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Editors: The editors of a protected range.
type Editors struct {
	// DomainUsersCanEdit: True if anyone in the document's domain has edit
	// access to the protected
	// range.  Domain protection is only supported on documents within a
	// domain.
	DomainUsersCanEdit bool `json:"domainUsersCanEdit,omitempty"`

	// Groups: The email addresses of groups with edit access to the
	// protected range.
	Groups []string `json:"groups,omitempty"`

	// Users: The email addresses of users with edit access to the protected
	// range.
	Users []string `json:"users,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DomainUsersCanEdit")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DomainUsersCanEdit") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Editors) MarshalJSON() ([]byte, error) {
	type noMethod Editors
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// EmbeddedChart: A chart embedded in a sheet.
type EmbeddedChart struct {
	// ChartId: The ID of the chart.
	ChartId int64 `json:"chartId,omitempty"`

	// Position: The position of the chart.
	Position *EmbeddedObjectPosition `json:"position,omitempty"`

	// Spec: The specification of the chart.
	Spec *ChartSpec `json:"spec,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ChartId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ChartId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EmbeddedChart) MarshalJSON() ([]byte, error) {
	type noMethod EmbeddedChart
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// EmbeddedObjectPosition: The position of an embedded object such as a
// chart.
type EmbeddedObjectPosition struct {
	// NewSheet: If true, the embedded object will be put on a new sheet
	// whose ID
	// is chosen for you. Used only when writing.
	NewSheet bool `json:"newSheet,omitempty"`

	// OverlayPosition: The position at which the object is overlaid on top
	// of a grid.
	OverlayPosition *OverlayPosition `json:"overlayPosition,omitempty"`

	// SheetId: The sheet this is on. Set only if the embedded object
	// is on its own sheet. Must be non-negative.
	SheetId int64 `json:"sheetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NewSheet") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NewSheet") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *EmbeddedObjectPosition) MarshalJSON() ([]byte, error) {
	type noMethod EmbeddedObjectPosition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ErrorValue: An error in a cell.
type ErrorValue struct {
	// Message: A message with more information about the error
	// (in the spreadsheet's locale).
	Message string `json:"message,omitempty"`

	// Type: The type of error.
	//
	// Possible values:
	//   "ERROR_TYPE_UNSPECIFIED" - The default error type, do not use this.
	//   "ERROR" - Corresponds to the `#ERROR!` error.
	//   "NULL_VALUE" - Corresponds to the `#NULL!` error.
	//   "DIVIDE_BY_ZERO" - Corresponds to the `#DIV/0` error.
	//   "VALUE" - Corresponds to the `#VALUE!` error.
	//   "REF" - Corresponds to the `#REF!` error.
	//   "NAME" - Corresponds to the `#NAME?` error.
	//   "NUM" - Corresponds to the `#NUM`! error.
	//   "N_A" - Corresponds to the `#N/A` error.
	//   "LOADING" - Corresponds to the `Loading...` state.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Message") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Message") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ErrorValue) MarshalJSON() ([]byte, error) {
	type noMethod ErrorValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ExtendedValue: The kinds of value that a cell in a spreadsheet can
// have.
type ExtendedValue struct {
	// BoolValue: Represents a boolean value.
	BoolValue bool `json:"boolValue,omitempty"`

	// ErrorValue: Represents an error.
	// This field is read-only.
	ErrorValue *ErrorValue `json:"errorValue,omitempty"`

	// FormulaValue: Represents a formula.
	FormulaValue string `json:"formulaValue,omitempty"`

	// NumberValue: Represents a double value.
	// Note: Dates, Times and DateTimes are represented as doubles
	// in
	// "serial number" format.
	NumberValue float64 `json:"numberValue,omitempty"`

	// StringValue: Represents a string value.
	// Leading single quotes are not included. For example, if the user
	// typed
	// `'123` into the UI, this would be represented as a `stringValue`
	// of
	// "123".
	StringValue string `json:"stringValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BoolValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BoolValue") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ExtendedValue) MarshalJSON() ([]byte, error) {
	type noMethod ExtendedValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *ExtendedValue) UnmarshalJSON(data []byte) error {
	type noMethod ExtendedValue
	var s1 struct {
		NumberValue gensupport.JSONFloat64 `json:"numberValue"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.NumberValue = float64(s1.NumberValue)
	return nil
}

// FilterCriteria: Criteria for showing/hiding rows in a filter or
// filter view.
type FilterCriteria struct {
	// Condition: A condition that must be true for values to be
	// shown.
	// (This does not override hiddenValues -- if a value is listed there,
	//  it will still be hidden.)
	Condition *BooleanCondition `json:"condition,omitempty"`

	// HiddenValues: Values that should be hidden.
	HiddenValues []string `json:"hiddenValues,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Condition") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Condition") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FilterCriteria) MarshalJSON() ([]byte, error) {
	type noMethod FilterCriteria
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FilterView: A filter view.
type FilterView struct {
	// Criteria: The criteria for showing/hiding values per column.
	// The map's key is the column index, and the value is the criteria
	// for
	// that column.
	Criteria map[string]FilterCriteria `json:"criteria,omitempty"`

	// FilterViewId: The ID of the filter view.
	FilterViewId int64 `json:"filterViewId,omitempty"`

	// NamedRangeId: The named range this filter view is backed by, if
	// any.
	//
	// When writing, only one of range or named_range_id
	// may be set.
	NamedRangeId string `json:"namedRangeId,omitempty"`

	// Range: The range this filter view covers.
	//
	// When writing, only one of range or named_range_id
	// may be set.
	Range *GridRange `json:"range,omitempty"`

	// SortSpecs: The sort order per column. Later specifications are used
	// when values
	// are equal in the earlier specifications.
	SortSpecs []*SortSpec `json:"sortSpecs,omitempty"`

	// Title: The name of the filter view.
	Title string `json:"title,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Criteria") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Criteria") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FilterView) MarshalJSON() ([]byte, error) {
	type noMethod FilterView
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FindReplaceRequest: Finds and replaces data in cells over a range,
// sheet, or all sheets.
type FindReplaceRequest struct {
	// AllSheets: True to find/replace over all sheets.
	AllSheets bool `json:"allSheets,omitempty"`

	// Find: The value to search.
	Find string `json:"find,omitempty"`

	// IncludeFormulas: True if the search should include cells with
	// formulas.
	// False to skip cells with formulas.
	IncludeFormulas bool `json:"includeFormulas,omitempty"`

	// MatchCase: True if the search is case sensitive.
	MatchCase bool `json:"matchCase,omitempty"`

	// MatchEntireCell: True if the find value should match the entire cell.
	MatchEntireCell bool `json:"matchEntireCell,omitempty"`

	// Range: The range to find/replace over.
	Range *GridRange `json:"range,omitempty"`

	// Replacement: The value to use as the replacement.
	Replacement string `json:"replacement,omitempty"`

	// SearchByRegex: True if the find value is a regex.
	// The regular expression and replacement should follow Java regex
	// rules
	// at
	// https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html.
	// The replacement string is allowed to refer to capturing groups.
	// For example, if one cell has the contents "Google Sheets" and
	// another
	// has "Google Docs", then searching for "o.* (.*)" with a
	// replacement of
	// "$1 Rocks" would change the contents of the cells to
	// "GSheets Rocks" and "GDocs Rocks" respectively.
	SearchByRegex bool `json:"searchByRegex,omitempty"`

	// SheetId: The sheet to find/replace over.
	SheetId int64 `json:"sheetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AllSheets") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AllSheets") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FindReplaceRequest) MarshalJSON() ([]byte, error) {
	type noMethod FindReplaceRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FindReplaceResponse: The result of the find/replace.
type FindReplaceResponse struct {
	// FormulasChanged: The number of formula cells changed.
	FormulasChanged int64 `json:"formulasChanged,omitempty"`

	// OccurrencesChanged: The number of occurrences (possibly multiple
	// within a cell) changed.
	// For example, if replacing "e" with "o" in "Google Sheets", this
	// would
	// be "3" because "Google Sheets" -> "Googlo Shoots".
	OccurrencesChanged int64 `json:"occurrencesChanged,omitempty"`

	// RowsChanged: The number of rows changed.
	RowsChanged int64 `json:"rowsChanged,omitempty"`

	// SheetsChanged: The number of sheets changed.
	SheetsChanged int64 `json:"sheetsChanged,omitempty"`

	// ValuesChanged: The number of non-formula cells changed.
	ValuesChanged int64 `json:"valuesChanged,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FormulasChanged") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FormulasChanged") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *FindReplaceResponse) MarshalJSON() ([]byte, error) {
	type noMethod FindReplaceResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GradientRule: A rule that applies a gradient color scale format,
// based on
// the interpolation points listed. The format of a cell will vary
// based on its contents as compared to the values of the
// interpolation
// points.
type GradientRule struct {
	// Maxpoint: The final interpolation point.
	Maxpoint *InterpolationPoint `json:"maxpoint,omitempty"`

	// Midpoint: An optional midway interpolation point.
	Midpoint *InterpolationPoint `json:"midpoint,omitempty"`

	// Minpoint: The starting interpolation point.
	Minpoint *InterpolationPoint `json:"minpoint,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Maxpoint") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Maxpoint") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GradientRule) MarshalJSON() ([]byte, error) {
	type noMethod GradientRule
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GridCoordinate: A coordinate in a sheet.
// All indexes are zero-based.
type GridCoordinate struct {
	// ColumnIndex: The column index of the coordinate.
	ColumnIndex int64 `json:"columnIndex,omitempty"`

	// RowIndex: The row index of the coordinate.
	RowIndex int64 `json:"rowIndex,omitempty"`

	// SheetId: The sheet this coordinate is on.
	SheetId int64 `json:"sheetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ColumnIndex") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ColumnIndex") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GridCoordinate) MarshalJSON() ([]byte, error) {
	type noMethod GridCoordinate
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GridData: Data in the grid, as well as metadata about the dimensions.
type GridData struct {
	// ColumnMetadata: Metadata about the requested columns in the grid,
	// starting with the column
	// in start_column.
	ColumnMetadata []*DimensionProperties `json:"columnMetadata,omitempty"`

	// RowData: The data in the grid, one entry per row,
	// starting with the row in startRow.
	// The values in RowData will correspond to columns starting
	// at start_column.
	RowData []*RowData `json:"rowData,omitempty"`

	// RowMetadata: Metadata about the requested rows in the grid, starting
	// with the row
	// in start_row.
	RowMetadata []*DimensionProperties `json:"rowMetadata,omitempty"`

	// StartColumn: The first column this GridData refers to, zero-based.
	StartColumn int64 `json:"startColumn,omitempty"`

	// StartRow: The first row this GridData refers to, zero-based.
	StartRow int64 `json:"startRow,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ColumnMetadata") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ColumnMetadata") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GridData) MarshalJSON() ([]byte, error) {
	type noMethod GridData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GridProperties: Properties of a grid.
type GridProperties struct {
	// ColumnCount: The number of columns in the grid.
	ColumnCount int64 `json:"columnCount,omitempty"`

	// FrozenColumnCount: The number of columns that are frozen in the grid.
	FrozenColumnCount int64 `json:"frozenColumnCount,omitempty"`

	// FrozenRowCount: The number of rows that are frozen in the grid.
	FrozenRowCount int64 `json:"frozenRowCount,omitempty"`

	// HideGridlines: True if the grid isn't showing gridlines in the UI.
	HideGridlines bool `json:"hideGridlines,omitempty"`

	// RowCount: The number of rows in the grid.
	RowCount int64 `json:"rowCount,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ColumnCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ColumnCount") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GridProperties) MarshalJSON() ([]byte, error) {
	type noMethod GridProperties
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GridRange: A range on a sheet.
// All indexes are zero-based.
// Indexes are half open, e.g the start index is inclusive
// and the end index is exclusive -- [start_index, end_index).
// Missing indexes indicate the range is unbounded on that side.
//
// For example, if "Sheet1" is sheet ID 0, then:
//
//   `Sheet1!A1:A1 == sheet_id: 0,
//                   start_row_index: 0, end_row_index: 1,
//                   start_column_index: 0, end_column_index: 1`
//
//   `Sheet1!A3:B4 == sheet_id: 0,
//                   start_row_index: 2, end_row_index: 4,
//                   start_column_index: 0, end_column_index: 2`
//
//   `Sheet1!A:B == sheet_id: 0,
//                 start_column_index: 0, end_column_index: 2`
//
//   `Sheet1!A5:B == sheet_id: 0,
//                  start_row_index: 4,
//                  start_column_index: 0, end_column_index: 2`
//
//   `Sheet1 == sheet_id:0`
//
// The start index must always be less than or equal to the end
// index.
// If the start index equals the end index, then the range is
// empty.
// Empty ranges are typically not meaningful and are usually rendered in
// the
// UI as `#REF!`.
type GridRange struct {
	// EndColumnIndex: The end column (exclusive) of the range, or not set
	// if unbounded.
	EndColumnIndex int64 `json:"endColumnIndex,omitempty"`

	// EndRowIndex: The end row (exclusive) of the range, or not set if
	// unbounded.
	EndRowIndex int64 `json:"endRowIndex,omitempty"`

	// SheetId: The sheet this range is on.
	SheetId int64 `json:"sheetId,omitempty"`

	// StartColumnIndex: The start column (inclusive) of the range, or not
	// set if unbounded.
	StartColumnIndex int64 `json:"startColumnIndex,omitempty"`

	// StartRowIndex: The start row (inclusive) of the range, or not set if
	// unbounded.
	StartRowIndex int64 `json:"startRowIndex,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EndColumnIndex") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndColumnIndex") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GridRange) MarshalJSON() ([]byte, error) {
	type noMethod GridRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// HistogramChartSpec: A <a
// href="/chart/interactive/docs/gallery/histogram">histogram
// chart</a>.
// A histogram chart groups data items into bins, displaying each bin as
// a
// column of stacked items.  Histograms are used to display the
// distribution
// of a dataset.  Each column of items represents a range into which
// those
// items fall.  The number of bins can be chosen automatically or
// specified
// explicitly.
type HistogramChartSpec struct {
	// BucketSize: By default the bucket size (the range of values stacked
	// in a single
	// column) is chosen automatically, but it may be overridden here.
	// E.g., A bucket size of 1.5 results in buckets from 0 - 1.5, 1.5 -
	// 3.0, etc.
	// Cannot be negative.
	// This field is optional.
	BucketSize float64 `json:"bucketSize,omitempty"`

	// LegendPosition: The position of the chart legend.
	//
	// Possible values:
	//   "HISTOGRAM_CHART_LEGEND_POSITION_UNSPECIFIED" - Default value, do
	// not use.
	//   "BOTTOM_LEGEND" - The legend is rendered on the bottom of the
	// chart.
	//   "LEFT_LEGEND" - The legend is rendered on the left of the chart.
	//   "RIGHT_LEGEND" - The legend is rendered on the right of the chart.
	//   "TOP_LEGEND" - The legend is rendered on the top of the chart.
	//   "NO_LEGEND" - No legend is rendered.
	//   "INSIDE_LEGEND" - The legend is rendered inside the chart area.
	LegendPosition string `json:"legendPosition,omitempty"`

	// OutlierPercentile: The outlier percentile is used to ensure that
	// outliers do not adversely
	// affect the calculation of bucket sizes.  For example, setting an
	// outlier
	// percentile of 0.05 indicates that the top and bottom 5% of values
	// when
	// calculating buckets.  The values are still included in the chart,
	// they will
	// be added to the first or last buckets instead of their own
	// buckets.
	// Must be between 0.0 and 0.5.
	OutlierPercentile float64 `json:"outlierPercentile,omitempty"`

	// Series: The series for a histogram may be either a single series of
	// values to be
	// bucketed or multiple series, each of the same length, containing the
	// name
	// of the series followed by the values to be bucketed for that series.
	Series []*HistogramSeries `json:"series,omitempty"`

	// ShowItemDividers: Whether horizontal divider lines should be
	// displayed between items in each
	// column.
	ShowItemDividers bool `json:"showItemDividers,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BucketSize") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BucketSize") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *HistogramChartSpec) MarshalJSON() ([]byte, error) {
	type noMethod HistogramChartSpec
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *HistogramChartSpec) UnmarshalJSON(data []byte) error {
	type noMethod HistogramChartSpec
	var s1 struct {
		BucketSize        gensupport.JSONFloat64 `json:"bucketSize"`
		OutlierPercentile gensupport.JSONFloat64 `json:"outlierPercentile"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.BucketSize = float64(s1.BucketSize)
	s.OutlierPercentile = float64(s1.OutlierPercentile)
	return nil
}

// HistogramSeries: A histogram series containing the series color and
// data.
type HistogramSeries struct {
	// BarColor: The color of the column representing this series in each
	// bucket.
	// This field is optional.
	BarColor *Color `json:"barColor,omitempty"`

	// Data: The data for this histogram series.
	Data *ChartData `json:"data,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BarColor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BarColor") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *HistogramSeries) MarshalJSON() ([]byte, error) {
	type noMethod HistogramSeries
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InsertDimensionRequest: Inserts rows or columns in a sheet at a
// particular index.
type InsertDimensionRequest struct {
	// InheritFromBefore: Whether dimension properties should be extended
	// from the dimensions
	// before or after the newly inserted dimensions.
	// True to inherit from the dimensions before (in which case the
	// start
	// index must be greater than 0), and false to inherit from the
	// dimensions
	// after.
	//
	// For example, if row index 0 has red background and row index 1
	// has a green background, then inserting 2 rows at index 1 can
	// inherit
	// either the green or red background.  If `inheritFromBefore` is
	// true,
	// the two new rows will be red (because the row before the insertion
	// point
	// was red), whereas if `inheritFromBefore` is false, the two new rows
	// will
	// be green (because the row after the insertion point was green).
	InheritFromBefore bool `json:"inheritFromBefore,omitempty"`

	// Range: The dimensions to insert.  Both the start and end indexes must
	// be bounded.
	Range *DimensionRange `json:"range,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InheritFromBefore")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InheritFromBefore") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *InsertDimensionRequest) MarshalJSON() ([]byte, error) {
	type noMethod InsertDimensionRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InsertRangeRequest: Inserts cells into a range, shifting the existing
// cells over or down.
type InsertRangeRequest struct {
	// Range: The range to insert new cells into.
	Range *GridRange `json:"range,omitempty"`

	// ShiftDimension: The dimension which will be shifted when inserting
	// cells.
	// If ROWS, existing cells will be shifted down.
	// If COLUMNS, existing cells will be shifted right.
	//
	// Possible values:
	//   "DIMENSION_UNSPECIFIED" - The default value, do not use.
	//   "ROWS" - Operates on the rows of a sheet.
	//   "COLUMNS" - Operates on the columns of a sheet.
	ShiftDimension string `json:"shiftDimension,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Range") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Range") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InsertRangeRequest) MarshalJSON() ([]byte, error) {
	type noMethod InsertRangeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InterpolationPoint: A single interpolation point on a gradient
// conditional format.
// These pin the gradient color scale according to the color,
// type and value chosen.
type InterpolationPoint struct {
	// Color: The color this interpolation point should use.
	Color *Color `json:"color,omitempty"`

	// Type: How the value should be interpreted.
	//
	// Possible values:
	//   "INTERPOLATION_POINT_TYPE_UNSPECIFIED" - The default value, do not
	// use.
	//   "MIN" - The interpolation point will use the minimum value in
	// the
	// cells over the range of the conditional format.
	//   "MAX" - The interpolation point will use the maximum value in
	// the
	// cells over the range of the conditional format.
	//   "NUMBER" - The interpolation point will use exactly the value
	// in
	// InterpolationPoint.value.
	//   "PERCENT" - The interpolation point will be the given percentage
	// over
	// all the cells in the range of the conditional format.
	// This is equivalent to NUMBER if the value was:
	// `=(MAX(FLATTEN(range)) * (value / 100))
	//   + (MIN(FLATTEN(range)) * (1 - (value / 100)))`
	// (where errors in the range are ignored when flattening).
	//   "PERCENTILE" - The interpolation point will be the given
	// percentile
	// over all the cells in the range of the conditional format.
	// This is equivalent to NUMBER if the value
	// was:
	// `=PERCENTILE(FLATTEN(range), value / 100)`
	// (where errors in the range are ignored when flattening).
	Type string `json:"type,omitempty"`

	// Value: The value this interpolation point uses.  May be a
	// formula.
	// Unused if type is MIN or
	// MAX.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Color") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Color") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InterpolationPoint) MarshalJSON() ([]byte, error) {
	type noMethod InterpolationPoint
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// IterativeCalculationSettings: Settings to control how circular
// dependencies are resolved with iterative
// calculation.
type IterativeCalculationSettings struct {
	// ConvergenceThreshold: When iterative calculation is enabled and
	// successive results differ by
	// less than this threshold value, the calculation rounds stop.
	ConvergenceThreshold float64 `json:"convergenceThreshold,omitempty"`

	// MaxIterations: When iterative calculation is enabled, the maximum
	// number of calculation
	// rounds to perform.
	MaxIterations int64 `json:"maxIterations,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ConvergenceThreshold") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConvergenceThreshold") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *IterativeCalculationSettings) MarshalJSON() ([]byte, error) {
	type noMethod IterativeCalculationSettings
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *IterativeCalculationSettings) UnmarshalJSON(data []byte) error {
	type noMethod IterativeCalculationSettings
	var s1 struct {
		ConvergenceThreshold gensupport.JSONFloat64 `json:"convergenceThreshold"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.ConvergenceThreshold = float64(s1.ConvergenceThreshold)
	return nil
}

// MergeCellsRequest: Merges all cells in the range.
type MergeCellsRequest struct {
	// MergeType: How the cells should be merged.
	//
	// Possible values:
	//   "MERGE_ALL" - Create a single merge from the range
	//   "MERGE_COLUMNS" - Create a merge for each column in the range
	//   "MERGE_ROWS" - Create a merge for each row in the range
	MergeType string `json:"mergeType,omitempty"`

	// Range: The range of cells to merge.
	Range *GridRange `json:"range,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MergeType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MergeType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MergeCellsRequest) MarshalJSON() ([]byte, error) {
	type noMethod MergeCellsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MoveDimensionRequest: Moves one or more rows or columns.
type MoveDimensionRequest struct {
	// DestinationIndex: The zero-based start index of where to move the
	// source data to,
	// based on the coordinates *before* the source data is removed
	// from the grid.  Existing data will be shifted down or
	// right
	// (depending on the dimension) to make room for the moved
	// dimensions.
	// The source dimensions are removed from the grid, so the
	// the data may end up in a different index than specified.
	//
	// For example, given `A1..A5` of `0, 1, 2, 3, 4` and wanting to
	// move
	// "1" and "2" to between "3" and "4", the source would be
	// `ROWS [1..3)`,and the destination index would be "4"
	// (the zero-based index of row 5).
	// The end result would be `A1..A5` of `0, 3, 1, 2, 4`.
	DestinationIndex int64 `json:"destinationIndex,omitempty"`

	// Source: The source dimensions to move.
	Source *DimensionRange `json:"source,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DestinationIndex") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DestinationIndex") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *MoveDimensionRequest) MarshalJSON() ([]byte, error) {
	type noMethod MoveDimensionRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NamedRange: A named range.
type NamedRange struct {
	// Name: The name of the named range.
	Name string `json:"name,omitempty"`

	// NamedRangeId: The ID of the named range.
	NamedRangeId string `json:"namedRangeId,omitempty"`

	// Range: The range this represents.
	Range *GridRange `json:"range,omitempty"`

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

func (s *NamedRange) MarshalJSON() ([]byte, error) {
	type noMethod NamedRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NumberFormat: The number format of a cell.
type NumberFormat struct {
	// Pattern: Pattern string used for formatting.  If not set, a default
	// pattern based on
	// the user's locale will be used if necessary for the given type.
	// See the [Date and Number Formats guide](/sheets/api/guides/formats)
	// for more
	// information about the supported patterns.
	Pattern string `json:"pattern,omitempty"`

	// Type: The type of the number format.
	// When writing, this field must be set.
	//
	// Possible values:
	//   "NUMBER_FORMAT_TYPE_UNSPECIFIED" - The number format is not
	// specified
	// and is based on the contents of the cell.
	// Do not explicitly use this.
	//   "TEXT" - Text formatting, e.g `1000.12`
	//   "NUMBER" - Number formatting, e.g, `1,000.12`
	//   "PERCENT" - Percent formatting, e.g `10.12%`
	//   "CURRENCY" - Currency formatting, e.g `$1,000.12`
	//   "DATE" - Date formatting, e.g `9/26/2008`
	//   "TIME" - Time formatting, e.g `3:59:00 PM`
	//   "DATE_TIME" - Date+Time formatting, e.g `9/26/08 15:59:00`
	//   "SCIENTIFIC" - Scientific number formatting, e.g `1.01E+03`
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Pattern") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Pattern") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *NumberFormat) MarshalJSON() ([]byte, error) {
	type noMethod NumberFormat
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OrgChartSpec: An <a
// href="/chart/interactive/docs/gallery/orgchart">org chart</a>.
// Org charts require a unique set of labels in labels and
// may
// optionally include parent_labels and tooltips.
// parent_labels contain, for each node, the label identifying the
// parent
// node.  tooltips contain, for each node, an optional tooltip.
//
// For example, to describe an OrgChart with Alice as the CEO, Bob as
// the
// President (reporting to Alice) and Cathy as VP of Sales (also
// reporting to
// Alice), have labels contain "Alice", "Bob", "Cathy",
// parent_labels contain "", "Alice", "Alice" and tooltips
// contain
// "CEO", "President", "VP Sales".
type OrgChartSpec struct {
	// Labels: The data containing the labels for all the nodes in the
	// chart.  Labels
	// must be unique.
	Labels *ChartData `json:"labels,omitempty"`

	// NodeColor: The color of the org chart nodes.
	NodeColor *Color `json:"nodeColor,omitempty"`

	// NodeSize: The size of the org chart nodes.
	//
	// Possible values:
	//   "ORG_CHART_LABEL_SIZE_UNSPECIFIED" - Default value, do not use.
	//   "SMALL" - The small org chart node size.
	//   "MEDIUM" - The medium org chart node size.
	//   "LARGE" - The large org chart node size.
	NodeSize string `json:"nodeSize,omitempty"`

	// ParentLabels: The data containing the label of the parent for the
	// corresponding node.
	// A blank value indicates that the node has no parent and is a
	// top-level
	// node.
	// This field is optional.
	ParentLabels *ChartData `json:"parentLabels,omitempty"`

	// SelectedNodeColor: The color of the selected org chart nodes.
	SelectedNodeColor *Color `json:"selectedNodeColor,omitempty"`

	// Tooltips: The data containing the tooltip for the corresponding node.
	//  A blank value
	// results in no tooltip being displayed for the node.
	// This field is optional.
	Tooltips *ChartData `json:"tooltips,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Labels") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Labels") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OrgChartSpec) MarshalJSON() ([]byte, error) {
	type noMethod OrgChartSpec
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OverlayPosition: The location an object is overlaid on top of a grid.
type OverlayPosition struct {
	// AnchorCell: The cell the object is anchored to.
	AnchorCell *GridCoordinate `json:"anchorCell,omitempty"`

	// HeightPixels: The height of the object, in pixels. Defaults to 371.
	HeightPixels int64 `json:"heightPixels,omitempty"`

	// OffsetXPixels: The horizontal offset, in pixels, that the object is
	// offset
	// from the anchor cell.
	OffsetXPixels int64 `json:"offsetXPixels,omitempty"`

	// OffsetYPixels: The vertical offset, in pixels, that the object is
	// offset
	// from the anchor cell.
	OffsetYPixels int64 `json:"offsetYPixels,omitempty"`

	// WidthPixels: The width of the object, in pixels. Defaults to 600.
	WidthPixels int64 `json:"widthPixels,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AnchorCell") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AnchorCell") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OverlayPosition) MarshalJSON() ([]byte, error) {
	type noMethod OverlayPosition
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Padding: The amount of padding around the cell, in pixels.
// When updating padding, every field must be specified.
type Padding struct {
	// Bottom: The bottom padding of the cell.
	Bottom int64 `json:"bottom,omitempty"`

	// Left: The left padding of the cell.
	Left int64 `json:"left,omitempty"`

	// Right: The right padding of the cell.
	Right int64 `json:"right,omitempty"`

	// Top: The top padding of the cell.
	Top int64 `json:"top,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Bottom") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Bottom") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Padding) MarshalJSON() ([]byte, error) {
	type noMethod Padding
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PasteDataRequest: Inserts data into the spreadsheet starting at the
// specified coordinate.
type PasteDataRequest struct {
	// Coordinate: The coordinate at which the data should start being
	// inserted.
	Coordinate *GridCoordinate `json:"coordinate,omitempty"`

	// Data: The data to insert.
	Data string `json:"data,omitempty"`

	// Delimiter: The delimiter in the data.
	Delimiter string `json:"delimiter,omitempty"`

	// Html: True if the data is HTML.
	Html bool `json:"html,omitempty"`

	// Type: How the data should be pasted.
	//
	// Possible values:
	//   "PASTE_NORMAL" - Paste values, formulas, formats, and merges.
	//   "PASTE_VALUES" - Paste the values ONLY without formats, formulas,
	// or merges.
	//   "PASTE_FORMAT" - Paste the format and data validation only.
	//   "PASTE_NO_BORDERS" - Like PASTE_NORMAL but without borders.
	//   "PASTE_FORMULA" - Paste the formulas only.
	//   "PASTE_DATA_VALIDATION" - Paste the data validation only.
	//   "PASTE_CONDITIONAL_FORMATTING" - Paste the conditional formatting
	// rules only.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Coordinate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Coordinate") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PasteDataRequest) MarshalJSON() ([]byte, error) {
	type noMethod PasteDataRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PieChartSpec: A <a
// href="/chart/interactive/docs/gallery/piechart">pie chart</a>.
type PieChartSpec struct {
	// Domain: The data that covers the domain of the pie chart.
	Domain *ChartData `json:"domain,omitempty"`

	// LegendPosition: Where the legend of the pie chart should be drawn.
	//
	// Possible values:
	//   "PIE_CHART_LEGEND_POSITION_UNSPECIFIED" - Default value, do not
	// use.
	//   "BOTTOM_LEGEND" - The legend is rendered on the bottom of the
	// chart.
	//   "LEFT_LEGEND" - The legend is rendered on the left of the chart.
	//   "RIGHT_LEGEND" - The legend is rendered on the right of the chart.
	//   "TOP_LEGEND" - The legend is rendered on the top of the chart.
	//   "NO_LEGEND" - No legend is rendered.
	//   "LABELED_LEGEND" - Each pie slice has a label attached to it.
	LegendPosition string `json:"legendPosition,omitempty"`

	// PieHole: The size of the hole in the pie chart.
	PieHole float64 `json:"pieHole,omitempty"`

	// Series: The data that covers the one and only series of the pie
	// chart.
	Series *ChartData `json:"series,omitempty"`

	// ThreeDimensional: True if the pie is three dimensional.
	ThreeDimensional bool `json:"threeDimensional,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Domain") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Domain") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PieChartSpec) MarshalJSON() ([]byte, error) {
	type noMethod PieChartSpec
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *PieChartSpec) UnmarshalJSON(data []byte) error {
	type noMethod PieChartSpec
	var s1 struct {
		PieHole gensupport.JSONFloat64 `json:"pieHole"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.PieHole = float64(s1.PieHole)
	return nil
}

// PivotFilterCriteria: Criteria for showing/hiding rows in a pivot
// table.
type PivotFilterCriteria struct {
	// VisibleValues: Values that should be included.  Values not listed
	// here are excluded.
	VisibleValues []string `json:"visibleValues,omitempty"`

	// ForceSendFields is a list of field names (e.g. "VisibleValues") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "VisibleValues") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PivotFilterCriteria) MarshalJSON() ([]byte, error) {
	type noMethod PivotFilterCriteria
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PivotGroup: A single grouping (either row or column) in a pivot
// table.
type PivotGroup struct {
	// ShowTotals: True if the pivot table should include the totals for
	// this grouping.
	ShowTotals bool `json:"showTotals,omitempty"`

	// SortOrder: The order the values in this group should be sorted.
	//
	// Possible values:
	//   "SORT_ORDER_UNSPECIFIED" - Default value, do not use this.
	//   "ASCENDING" - Sort ascending.
	//   "DESCENDING" - Sort descending.
	SortOrder string `json:"sortOrder,omitempty"`

	// SourceColumnOffset: The column offset of the source range that this
	// grouping is based on.
	//
	// For example, if the source was `C10:E15`, a `sourceColumnOffset` of
	// `0`
	// means this group refers to column `C`, whereas the offset `1` would
	// refer
	// to column `D`.
	SourceColumnOffset int64 `json:"sourceColumnOffset,omitempty"`

	// ValueBucket: The bucket of the opposite pivot group to sort by.
	// If not specified, sorting is alphabetical by this group's values.
	ValueBucket *PivotGroupSortValueBucket `json:"valueBucket,omitempty"`

	// ValueMetadata: Metadata about values in the grouping.
	ValueMetadata []*PivotGroupValueMetadata `json:"valueMetadata,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ShowTotals") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ShowTotals") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PivotGroup) MarshalJSON() ([]byte, error) {
	type noMethod PivotGroup
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PivotGroupSortValueBucket: Information about which values in a pivot
// group should be used for sorting.
type PivotGroupSortValueBucket struct {
	// Buckets: Determines the bucket from which values are chosen to
	// sort.
	//
	// For example, in a pivot table with one row group & two column
	// groups,
	// the row group can list up to two values. The first value
	// corresponds
	// to a value within the first column group, and the second
	// value
	// corresponds to a value in the second column group.  If no values
	// are listed, this would indicate that the row should be sorted
	// according
	// to the "Grand Total" over the column groups. If a single value is
	// listed,
	// this would correspond to using the "Total" of that bucket.
	Buckets []*ExtendedValue `json:"buckets,omitempty"`

	// ValuesIndex: The offset in the PivotTable.values list which the
	// values in this
	// grouping should be sorted by.
	ValuesIndex int64 `json:"valuesIndex,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Buckets") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Buckets") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PivotGroupSortValueBucket) MarshalJSON() ([]byte, error) {
	type noMethod PivotGroupSortValueBucket
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PivotGroupValueMetadata: Metadata about a value in a pivot grouping.
type PivotGroupValueMetadata struct {
	// Collapsed: True if the data corresponding to the value is collapsed.
	Collapsed bool `json:"collapsed,omitempty"`

	// Value: The calculated value the metadata corresponds to.
	// (Note that formulaValue is not valid,
	//  because the values will be calculated.)
	Value *ExtendedValue `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Collapsed") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Collapsed") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PivotGroupValueMetadata) MarshalJSON() ([]byte, error) {
	type noMethod PivotGroupValueMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PivotTable: A pivot table.
type PivotTable struct {
	// Columns: Each column grouping in the pivot table.
	Columns []*PivotGroup `json:"columns,omitempty"`

	// Criteria: An optional mapping of filters per source column
	// offset.
	//
	// The filters will be applied before aggregating data into the pivot
	// table.
	// The map's key is the column offset of the source range that you want
	// to
	// filter, and the value is the criteria for that column.
	//
	// For example, if the source was `C10:E15`, a key of `0` will have the
	// filter
	// for column `C`, whereas the key `1` is for column `D`.
	Criteria map[string]PivotFilterCriteria `json:"criteria,omitempty"`

	// Rows: Each row grouping in the pivot table.
	Rows []*PivotGroup `json:"rows,omitempty"`

	// Source: The range the pivot table is reading data from.
	Source *GridRange `json:"source,omitempty"`

	// ValueLayout: Whether values should be listed horizontally (as
	// columns)
	// or vertically (as rows).
	//
	// Possible values:
	//   "HORIZONTAL" - Values are laid out horizontally (as columns).
	//   "VERTICAL" - Values are laid out vertically (as rows).
	ValueLayout string `json:"valueLayout,omitempty"`

	// Values: A list of values to include in the pivot table.
	Values []*PivotValue `json:"values,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Columns") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Columns") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PivotTable) MarshalJSON() ([]byte, error) {
	type noMethod PivotTable
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PivotValue: The definition of how a value in a pivot table should be
// calculated.
type PivotValue struct {
	// Formula: A custom formula to calculate the value.  The formula must
	// start
	// with an `=` character.
	Formula string `json:"formula,omitempty"`

	// Name: A name to use for the value. This is only used if formula was
	// set.
	// Otherwise, the column name is used.
	Name string `json:"name,omitempty"`

	// SourceColumnOffset: The column offset of the source range that this
	// value reads from.
	//
	// For example, if the source was `C10:E15`, a `sourceColumnOffset` of
	// `0`
	// means this value refers to column `C`, whereas the offset `1`
	// would
	// refer to column `D`.
	SourceColumnOffset int64 `json:"sourceColumnOffset,omitempty"`

	// SummarizeFunction: A function to summarize the value.
	// If formula is set, the only supported values are
	// SUM and
	// CUSTOM.
	// If sourceColumnOffset is set, then `CUSTOM`
	// is not supported.
	//
	// Possible values:
	//   "PIVOT_STANDARD_VALUE_FUNCTION_UNSPECIFIED" - The default, do not
	// use.
	//   "SUM" - Corresponds to the `SUM` function.
	//   "COUNTA" - Corresponds to the `COUNTA` function.
	//   "COUNT" - Corresponds to the `COUNT` function.
	//   "COUNTUNIQUE" - Corresponds to the `COUNTUNIQUE` function.
	//   "AVERAGE" - Corresponds to the `AVERAGE` function.
	//   "MAX" - Corresponds to the `MAX` function.
	//   "MIN" - Corresponds to the `MIN` function.
	//   "MEDIAN" - Corresponds to the `MEDIAN` function.
	//   "PRODUCT" - Corresponds to the `PRODUCT` function.
	//   "STDEV" - Corresponds to the `STDEV` function.
	//   "STDEVP" - Corresponds to the `STDEVP` function.
	//   "VAR" - Corresponds to the `VAR` function.
	//   "VARP" - Corresponds to the `VARP` function.
	//   "CUSTOM" - Indicates the formula should be used as-is.
	// Only valid if PivotValue.formula was set.
	SummarizeFunction string `json:"summarizeFunction,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Formula") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Formula") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PivotValue) MarshalJSON() ([]byte, error) {
	type noMethod PivotValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ProtectedRange: A protected range.
type ProtectedRange struct {
	// Description: The description of this protected range.
	Description string `json:"description,omitempty"`

	// Editors: The users and groups with edit access to the protected
	// range.
	// This field is only visible to users with edit access to the
	// protected
	// range and the document.
	// Editors are not supported with warning_only protection.
	Editors *Editors `json:"editors,omitempty"`

	// NamedRangeId: The named range this protected range is backed by, if
	// any.
	//
	// When writing, only one of range or named_range_id
	// may be set.
	NamedRangeId string `json:"namedRangeId,omitempty"`

	// ProtectedRangeId: The ID of the protected range.
	// This field is read-only.
	ProtectedRangeId int64 `json:"protectedRangeId,omitempty"`

	// Range: The range that is being protected.
	// The range may be fully unbounded, in which case this is considered
	// a protected sheet.
	//
	// When writing, only one of range or named_range_id
	// may be set.
	Range *GridRange `json:"range,omitempty"`

	// RequestingUserCanEdit: True if the user who requested this protected
	// range can edit the
	// protected area.
	// This field is read-only.
	RequestingUserCanEdit bool `json:"requestingUserCanEdit,omitempty"`

	// UnprotectedRanges: The list of unprotected ranges within a protected
	// sheet.
	// Unprotected ranges are only supported on protected sheets.
	UnprotectedRanges []*GridRange `json:"unprotectedRanges,omitempty"`

	// WarningOnly: True if this protected range will show a warning when
	// editing.
	// Warning-based protection means that every user can edit data in
	// the
	// protected range, except editing will prompt a warning asking the
	// user
	// to confirm the edit.
	//
	// When writing: if this field is true, then editors is
	// ignored.
	// Additionally, if this field is changed from true to false and
	// the
	// `editors` field is not set (nor included in the field mask), then
	// the editors will be set to all the editors in the document.
	WarningOnly bool `json:"warningOnly,omitempty"`

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

func (s *ProtectedRange) MarshalJSON() ([]byte, error) {
	type noMethod ProtectedRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RandomizeRangeRequest: Randomizes the order of the rows in a range.
type RandomizeRangeRequest struct {
	// Range: The range to randomize.
	Range *GridRange `json:"range,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Range") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Range") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RandomizeRangeRequest) MarshalJSON() ([]byte, error) {
	type noMethod RandomizeRangeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RepeatCellRequest: Updates all cells in the range to the values in
// the given Cell object.
// Only the fields listed in the fields field are updated; others
// are
// unchanged.
//
// If writing a cell with a formula, the formula's ranges will
// automatically
// increment for each field in the range.
// For example, if writing a cell with formula `=A1` into range
// B2:C4,
// B2 would be `=A1`, B3 would be `=A2`, B4 would be `=A3`,
// C2 would be `=B1`, C3 would be `=B2`, C4 would be `=B3`.
//
// To keep the formula's ranges static, use the `$` indicator.
// For example, use the formula `=$A$1` to prevent both the row and
// the
// column from incrementing.
type RepeatCellRequest struct {
	// Cell: The data to write.
	Cell *CellData `json:"cell,omitempty"`

	// Fields: The fields that should be updated.  At least one field must
	// be specified.
	// The root `cell` is implied and should not be specified.
	// A single "*" can be used as short-hand for listing every field.
	Fields string `json:"fields,omitempty"`

	// Range: The range to repeat the cell in.
	Range *GridRange `json:"range,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Cell") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Cell") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RepeatCellRequest) MarshalJSON() ([]byte, error) {
	type noMethod RepeatCellRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Request: A single kind of update to apply to a spreadsheet.
type Request struct {
	// AddBanding: Adds a new banded range
	AddBanding *AddBandingRequest `json:"addBanding,omitempty"`

	// AddChart: Adds a chart.
	AddChart *AddChartRequest `json:"addChart,omitempty"`

	// AddConditionalFormatRule: Adds a new conditional format rule.
	AddConditionalFormatRule *AddConditionalFormatRuleRequest `json:"addConditionalFormatRule,omitempty"`

	// AddFilterView: Adds a filter view.
	AddFilterView *AddFilterViewRequest `json:"addFilterView,omitempty"`

	// AddNamedRange: Adds a named range.
	AddNamedRange *AddNamedRangeRequest `json:"addNamedRange,omitempty"`

	// AddProtectedRange: Adds a protected range.
	AddProtectedRange *AddProtectedRangeRequest `json:"addProtectedRange,omitempty"`

	// AddSheet: Adds a sheet.
	AddSheet *AddSheetRequest `json:"addSheet,omitempty"`

	// AppendCells: Appends cells after the last row with data in a sheet.
	AppendCells *AppendCellsRequest `json:"appendCells,omitempty"`

	// AppendDimension: Appends dimensions to the end of a sheet.
	AppendDimension *AppendDimensionRequest `json:"appendDimension,omitempty"`

	// AutoFill: Automatically fills in more data based on existing data.
	AutoFill *AutoFillRequest `json:"autoFill,omitempty"`

	// AutoResizeDimensions: Automatically resizes one or more dimensions
	// based on the contents
	// of the cells in that dimension.
	AutoResizeDimensions *AutoResizeDimensionsRequest `json:"autoResizeDimensions,omitempty"`

	// ClearBasicFilter: Clears the basic filter on a sheet.
	ClearBasicFilter *ClearBasicFilterRequest `json:"clearBasicFilter,omitempty"`

	// CopyPaste: Copies data from one area and pastes it to another.
	CopyPaste *CopyPasteRequest `json:"copyPaste,omitempty"`

	// CutPaste: Cuts data from one area and pastes it to another.
	CutPaste *CutPasteRequest `json:"cutPaste,omitempty"`

	// DeleteBanding: Removes a banded range
	DeleteBanding *DeleteBandingRequest `json:"deleteBanding,omitempty"`

	// DeleteConditionalFormatRule: Deletes an existing conditional format
	// rule.
	DeleteConditionalFormatRule *DeleteConditionalFormatRuleRequest `json:"deleteConditionalFormatRule,omitempty"`

	// DeleteDimension: Deletes rows or columns in a sheet.
	DeleteDimension *DeleteDimensionRequest `json:"deleteDimension,omitempty"`

	// DeleteEmbeddedObject: Deletes an embedded object (e.g, chart, image)
	// in a sheet.
	DeleteEmbeddedObject *DeleteEmbeddedObjectRequest `json:"deleteEmbeddedObject,omitempty"`

	// DeleteFilterView: Deletes a filter view from a sheet.
	DeleteFilterView *DeleteFilterViewRequest `json:"deleteFilterView,omitempty"`

	// DeleteNamedRange: Deletes a named range.
	DeleteNamedRange *DeleteNamedRangeRequest `json:"deleteNamedRange,omitempty"`

	// DeleteProtectedRange: Deletes a protected range.
	DeleteProtectedRange *DeleteProtectedRangeRequest `json:"deleteProtectedRange,omitempty"`

	// DeleteRange: Deletes a range of cells from a sheet, shifting the
	// remaining cells.
	DeleteRange *DeleteRangeRequest `json:"deleteRange,omitempty"`

	// DeleteSheet: Deletes a sheet.
	DeleteSheet *DeleteSheetRequest `json:"deleteSheet,omitempty"`

	// DuplicateFilterView: Duplicates a filter view.
	DuplicateFilterView *DuplicateFilterViewRequest `json:"duplicateFilterView,omitempty"`

	// DuplicateSheet: Duplicates a sheet.
	DuplicateSheet *DuplicateSheetRequest `json:"duplicateSheet,omitempty"`

	// FindReplace: Finds and replaces occurrences of some text with other
	// text.
	FindReplace *FindReplaceRequest `json:"findReplace,omitempty"`

	// InsertDimension: Inserts new rows or columns in a sheet.
	InsertDimension *InsertDimensionRequest `json:"insertDimension,omitempty"`

	// InsertRange: Inserts new cells in a sheet, shifting the existing
	// cells.
	InsertRange *InsertRangeRequest `json:"insertRange,omitempty"`

	// MergeCells: Merges cells together.
	MergeCells *MergeCellsRequest `json:"mergeCells,omitempty"`

	// MoveDimension: Moves rows or columns to another location in a sheet.
	MoveDimension *MoveDimensionRequest `json:"moveDimension,omitempty"`

	// PasteData: Pastes data (HTML or delimited) into a sheet.
	PasteData *PasteDataRequest `json:"pasteData,omitempty"`

	// RandomizeRange: Randomizes the order of the rows in a range.
	RandomizeRange *RandomizeRangeRequest `json:"randomizeRange,omitempty"`

	// RepeatCell: Repeats a single cell across a range.
	RepeatCell *RepeatCellRequest `json:"repeatCell,omitempty"`

	// SetBasicFilter: Sets the basic filter on a sheet.
	SetBasicFilter *SetBasicFilterRequest `json:"setBasicFilter,omitempty"`

	// SetDataValidation: Sets data validation for one or more cells.
	SetDataValidation *SetDataValidationRequest `json:"setDataValidation,omitempty"`

	// SortRange: Sorts data in a range.
	SortRange *SortRangeRequest `json:"sortRange,omitempty"`

	// TextToColumns: Converts a column of text into many columns of text.
	TextToColumns *TextToColumnsRequest `json:"textToColumns,omitempty"`

	// UnmergeCells: Unmerges merged cells.
	UnmergeCells *UnmergeCellsRequest `json:"unmergeCells,omitempty"`

	// UpdateBanding: Updates a banded range
	UpdateBanding *UpdateBandingRequest `json:"updateBanding,omitempty"`

	// UpdateBorders: Updates the borders in a range of cells.
	UpdateBorders *UpdateBordersRequest `json:"updateBorders,omitempty"`

	// UpdateCells: Updates many cells at once.
	UpdateCells *UpdateCellsRequest `json:"updateCells,omitempty"`

	// UpdateChartSpec: Updates a chart's specifications.
	UpdateChartSpec *UpdateChartSpecRequest `json:"updateChartSpec,omitempty"`

	// UpdateConditionalFormatRule: Updates an existing conditional format
	// rule.
	UpdateConditionalFormatRule *UpdateConditionalFormatRuleRequest `json:"updateConditionalFormatRule,omitempty"`

	// UpdateDimensionProperties: Updates dimensions' properties.
	UpdateDimensionProperties *UpdateDimensionPropertiesRequest `json:"updateDimensionProperties,omitempty"`

	// UpdateEmbeddedObjectPosition: Updates an embedded object's (e.g.
	// chart, image) position.
	UpdateEmbeddedObjectPosition *UpdateEmbeddedObjectPositionRequest `json:"updateEmbeddedObjectPosition,omitempty"`

	// UpdateFilterView: Updates the properties of a filter view.
	UpdateFilterView *UpdateFilterViewRequest `json:"updateFilterView,omitempty"`

	// UpdateNamedRange: Updates a named range.
	UpdateNamedRange *UpdateNamedRangeRequest `json:"updateNamedRange,omitempty"`

	// UpdateProtectedRange: Updates a protected range.
	UpdateProtectedRange *UpdateProtectedRangeRequest `json:"updateProtectedRange,omitempty"`

	// UpdateSheetProperties: Updates a sheet's properties.
	UpdateSheetProperties *UpdateSheetPropertiesRequest `json:"updateSheetProperties,omitempty"`

	// UpdateSpreadsheetProperties: Updates the spreadsheet's properties.
	UpdateSpreadsheetProperties *UpdateSpreadsheetPropertiesRequest `json:"updateSpreadsheetProperties,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AddBanding") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AddBanding") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Request) MarshalJSON() ([]byte, error) {
	type noMethod Request
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Response: A single response from an update.
type Response struct {
	// AddBanding: A reply from adding a banded range.
	AddBanding *AddBandingResponse `json:"addBanding,omitempty"`

	// AddChart: A reply from adding a chart.
	AddChart *AddChartResponse `json:"addChart,omitempty"`

	// AddFilterView: A reply from adding a filter view.
	AddFilterView *AddFilterViewResponse `json:"addFilterView,omitempty"`

	// AddNamedRange: A reply from adding a named range.
	AddNamedRange *AddNamedRangeResponse `json:"addNamedRange,omitempty"`

	// AddProtectedRange: A reply from adding a protected range.
	AddProtectedRange *AddProtectedRangeResponse `json:"addProtectedRange,omitempty"`

	// AddSheet: A reply from adding a sheet.
	AddSheet *AddSheetResponse `json:"addSheet,omitempty"`

	// DeleteConditionalFormatRule: A reply from deleting a conditional
	// format rule.
	DeleteConditionalFormatRule *DeleteConditionalFormatRuleResponse `json:"deleteConditionalFormatRule,omitempty"`

	// DuplicateFilterView: A reply from duplicating a filter view.
	DuplicateFilterView *DuplicateFilterViewResponse `json:"duplicateFilterView,omitempty"`

	// DuplicateSheet: A reply from duplicating a sheet.
	DuplicateSheet *DuplicateSheetResponse `json:"duplicateSheet,omitempty"`

	// FindReplace: A reply from doing a find/replace.
	FindReplace *FindReplaceResponse `json:"findReplace,omitempty"`

	// UpdateConditionalFormatRule: A reply from updating a conditional
	// format rule.
	UpdateConditionalFormatRule *UpdateConditionalFormatRuleResponse `json:"updateConditionalFormatRule,omitempty"`

	// UpdateEmbeddedObjectPosition: A reply from updating an embedded
	// object's position.
	UpdateEmbeddedObjectPosition *UpdateEmbeddedObjectPositionResponse `json:"updateEmbeddedObjectPosition,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AddBanding") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AddBanding") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Response) MarshalJSON() ([]byte, error) {
	type noMethod Response
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RowData: Data about each cell in a row.
type RowData struct {
	// Values: The values in the row, one per column.
	Values []*CellData `json:"values,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Values") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Values") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RowData) MarshalJSON() ([]byte, error) {
	type noMethod RowData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SetBasicFilterRequest: Sets the basic filter associated with a sheet.
type SetBasicFilterRequest struct {
	// Filter: The filter to set.
	Filter *BasicFilter `json:"filter,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Filter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Filter") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SetBasicFilterRequest) MarshalJSON() ([]byte, error) {
	type noMethod SetBasicFilterRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SetDataValidationRequest: Sets a data validation rule to every cell
// in the range.
// To clear validation in a range, call this with no rule specified.
type SetDataValidationRequest struct {
	// Range: The range the data validation rule should apply to.
	Range *GridRange `json:"range,omitempty"`

	// Rule: The data validation rule to set on each cell in the range,
	// or empty to clear the data validation in the range.
	Rule *DataValidationRule `json:"rule,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Range") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Range") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SetDataValidationRequest) MarshalJSON() ([]byte, error) {
	type noMethod SetDataValidationRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Sheet: A sheet in a spreadsheet.
type Sheet struct {
	// BandedRanges: The banded (i.e. alternating colors) ranges on this
	// sheet.
	BandedRanges []*BandedRange `json:"bandedRanges,omitempty"`

	// BasicFilter: The filter on this sheet, if any.
	BasicFilter *BasicFilter `json:"basicFilter,omitempty"`

	// Charts: The specifications of every chart on this sheet.
	Charts []*EmbeddedChart `json:"charts,omitempty"`

	// ConditionalFormats: The conditional format rules in this sheet.
	ConditionalFormats []*ConditionalFormatRule `json:"conditionalFormats,omitempty"`

	// Data: Data in the grid, if this is a grid sheet.
	// The number of GridData objects returned is dependent on the number
	// of
	// ranges requested on this sheet. For example, if this is
	// representing
	// `Sheet1`, and the spreadsheet was requested with
	// ranges
	// `Sheet1!A1:C10` and `Sheet1!D15:E20`, then the first GridData will
	// have a
	// startRow/startColumn of `0`,
	// while the second one will have `startRow 14` (zero-based row 15),
	// and `startColumn 3` (zero-based column D).
	Data []*GridData `json:"data,omitempty"`

	// FilterViews: The filter views in this sheet.
	FilterViews []*FilterView `json:"filterViews,omitempty"`

	// Merges: The ranges that are merged together.
	Merges []*GridRange `json:"merges,omitempty"`

	// Properties: The properties of the sheet.
	Properties *SheetProperties `json:"properties,omitempty"`

	// ProtectedRanges: The protected ranges in this sheet.
	ProtectedRanges []*ProtectedRange `json:"protectedRanges,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BandedRanges") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BandedRanges") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Sheet) MarshalJSON() ([]byte, error) {
	type noMethod Sheet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SheetProperties: Properties of a sheet.
type SheetProperties struct {
	// GridProperties: Additional properties of the sheet if this sheet is a
	// grid.
	// (If the sheet is an object sheet, containing a chart or image,
	// then
	// this field will be absent.)
	// When writing it is an error to set any grid properties on non-grid
	// sheets.
	GridProperties *GridProperties `json:"gridProperties,omitempty"`

	// Hidden: True if the sheet is hidden in the UI, false if it's visible.
	Hidden bool `json:"hidden,omitempty"`

	// Index: The index of the sheet within the spreadsheet.
	// When adding or updating sheet properties, if this field
	// is excluded then the sheet will be added or moved to the end
	// of the sheet list. When updating sheet indices or inserting
	// sheets, movement is considered in "before the move" indexes.
	// For example, if there were 3 sheets (S1, S2, S3) in order to
	// move S1 ahead of S2 the index would have to be set to 2. A
	// sheet
	// index update request will be ignored if the requested index
	// is
	// identical to the sheets current index or if the requested new
	// index is equal to the current sheet index + 1.
	Index int64 `json:"index,omitempty"`

	// RightToLeft: True if the sheet is an RTL sheet instead of an LTR
	// sheet.
	RightToLeft bool `json:"rightToLeft,omitempty"`

	// SheetId: The ID of the sheet. Must be non-negative.
	// This field cannot be changed once set.
	SheetId int64 `json:"sheetId,omitempty"`

	// SheetType: The type of sheet. Defaults to GRID.
	// This field cannot be changed once set.
	//
	// Possible values:
	//   "SHEET_TYPE_UNSPECIFIED" - Default value, do not use.
	//   "GRID" - The sheet is a grid.
	//   "OBJECT" - The sheet has no grid and instead has an object like a
	// chart or image.
	SheetType string `json:"sheetType,omitempty"`

	// TabColor: The color of the tab in the UI.
	TabColor *Color `json:"tabColor,omitempty"`

	// Title: The name of the sheet.
	Title string `json:"title,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "GridProperties") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "GridProperties") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SheetProperties) MarshalJSON() ([]byte, error) {
	type noMethod SheetProperties
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SortRangeRequest: Sorts data in rows based on a sort order per
// column.
type SortRangeRequest struct {
	// Range: The range to sort.
	Range *GridRange `json:"range,omitempty"`

	// SortSpecs: The sort order per column. Later specifications are used
	// when values
	// are equal in the earlier specifications.
	SortSpecs []*SortSpec `json:"sortSpecs,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Range") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Range") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SortRangeRequest) MarshalJSON() ([]byte, error) {
	type noMethod SortRangeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SortSpec: A sort order associated with a specific column or row.
type SortSpec struct {
	// DimensionIndex: The dimension the sort should be applied to.
	DimensionIndex int64 `json:"dimensionIndex,omitempty"`

	// SortOrder: The order data should be sorted.
	//
	// Possible values:
	//   "SORT_ORDER_UNSPECIFIED" - Default value, do not use this.
	//   "ASCENDING" - Sort ascending.
	//   "DESCENDING" - Sort descending.
	SortOrder string `json:"sortOrder,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DimensionIndex") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DimensionIndex") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *SortSpec) MarshalJSON() ([]byte, error) {
	type noMethod SortSpec
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SourceAndDestination: A combination of a source range and how to
// extend that source.
type SourceAndDestination struct {
	// Dimension: The dimension that data should be filled into.
	//
	// Possible values:
	//   "DIMENSION_UNSPECIFIED" - The default value, do not use.
	//   "ROWS" - Operates on the rows of a sheet.
	//   "COLUMNS" - Operates on the columns of a sheet.
	Dimension string `json:"dimension,omitempty"`

	// FillLength: The number of rows or columns that data should be filled
	// into.
	// Positive numbers expand beyond the last row or last column
	// of the source.  Negative numbers expand before the first row
	// or first column of the source.
	FillLength int64 `json:"fillLength,omitempty"`

	// Source: The location of the data to use as the source of the
	// autofill.
	Source *GridRange `json:"source,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Dimension") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Dimension") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SourceAndDestination) MarshalJSON() ([]byte, error) {
	type noMethod SourceAndDestination
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Spreadsheet: Resource that represents a spreadsheet.
type Spreadsheet struct {
	// NamedRanges: The named ranges defined in a spreadsheet.
	NamedRanges []*NamedRange `json:"namedRanges,omitempty"`

	// Properties: Overall properties of a spreadsheet.
	Properties *SpreadsheetProperties `json:"properties,omitempty"`

	// Sheets: The sheets that are part of a spreadsheet.
	Sheets []*Sheet `json:"sheets,omitempty"`

	// SpreadsheetId: The ID of the spreadsheet.
	// This field is read-only.
	SpreadsheetId string `json:"spreadsheetId,omitempty"`

	// SpreadsheetUrl: The url of the spreadsheet.
	// This field is read-only.
	SpreadsheetUrl string `json:"spreadsheetUrl,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NamedRanges") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NamedRanges") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Spreadsheet) MarshalJSON() ([]byte, error) {
	type noMethod Spreadsheet
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SpreadsheetProperties: Properties of a spreadsheet.
type SpreadsheetProperties struct {
	// AutoRecalc: The amount of time to wait before volatile functions are
	// recalculated.
	//
	// Possible values:
	//   "RECALCULATION_INTERVAL_UNSPECIFIED" - Default value. This value
	// must not be used.
	//   "ON_CHANGE" - Volatile functions are updated on every change.
	//   "MINUTE" - Volatile functions are updated on every change and every
	// minute.
	//   "HOUR" - Volatile functions are updated on every change and hourly.
	AutoRecalc string `json:"autoRecalc,omitempty"`

	// DefaultFormat: The default format of all cells in the
	// spreadsheet.
	// CellData.effectiveFormat will not be set if the
	// cell's format is equal to this default format.
	// This field is read-only.
	DefaultFormat *CellFormat `json:"defaultFormat,omitempty"`

	// IterativeCalculationSettings: Determines whether and how circular
	// references are resolved with iterative
	// calculation.  Absence of this field means that circular references
	// will
	// result in calculation errors.
	IterativeCalculationSettings *IterativeCalculationSettings `json:"iterativeCalculationSettings,omitempty"`

	// Locale: The locale of the spreadsheet in one of the following
	// formats:
	//
	// * an ISO 639-1 language code such as `en`
	//
	// * an ISO 639-2 language code such as `fil`, if no 639-1 code
	// exists
	//
	// * a combination of the ISO language code and country code, such as
	// `en_US`
	//
	// Note: when updating this field, not all locales/languages are
	// supported.
	Locale string `json:"locale,omitempty"`

	// TimeZone: The time zone of the spreadsheet, in CLDR format such
	// as
	// `America/New_York`. If the time zone isn't recognized, this may
	// be a custom time zone such as `GMT-07:00`.
	TimeZone string `json:"timeZone,omitempty"`

	// Title: The title of the spreadsheet.
	Title string `json:"title,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AutoRecalc") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AutoRecalc") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SpreadsheetProperties) MarshalJSON() ([]byte, error) {
	type noMethod SpreadsheetProperties
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TextFormat: The format of a run of text in a cell.
// Absent values indicate that the field isn't specified.
type TextFormat struct {
	// Bold: True if the text is bold.
	Bold bool `json:"bold,omitempty"`

	// FontFamily: The font family.
	FontFamily string `json:"fontFamily,omitempty"`

	// FontSize: The size of the font.
	FontSize int64 `json:"fontSize,omitempty"`

	// ForegroundColor: The foreground color of the text.
	ForegroundColor *Color `json:"foregroundColor,omitempty"`

	// Italic: True if the text is italicized.
	Italic bool `json:"italic,omitempty"`

	// Strikethrough: True if the text has a strikethrough.
	Strikethrough bool `json:"strikethrough,omitempty"`

	// Underline: True if the text is underlined.
	Underline bool `json:"underline,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Bold") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Bold") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TextFormat) MarshalJSON() ([]byte, error) {
	type noMethod TextFormat
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TextFormatRun: A run of a text format. The format of this run
// continues until the start
// index of the next run.
// When updating, all fields must be set.
type TextFormatRun struct {
	// Format: The format of this run.  Absent values inherit the cell's
	// format.
	Format *TextFormat `json:"format,omitempty"`

	// StartIndex: The character index where this run starts.
	StartIndex int64 `json:"startIndex,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Format") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Format") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TextFormatRun) MarshalJSON() ([]byte, error) {
	type noMethod TextFormatRun
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TextRotation: The rotation applied to text in a cell.
type TextRotation struct {
	// Angle: The angle between the standard orientation and the desired
	// orientation.
	// Measured in degrees. Valid values are between -90 and 90.
	// Positive
	// angles are angled upwards, negative are angled downwards.
	//
	// Note: For LTR text direction positive angles are in the
	// counterclockwise
	// direction, whereas for RTL they are in the clockwise direction
	Angle int64 `json:"angle,omitempty"`

	// Vertical: If true, text reads top to bottom, but the orientation of
	// individual
	// characters is unchanged.
	// For example:
	//
	//     | V |
	//     | e |
	//     | r |
	//     | t |
	//     | i |
	//     | c |
	//     | a |
	//     | l |
	Vertical bool `json:"vertical,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Angle") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Angle") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TextRotation) MarshalJSON() ([]byte, error) {
	type noMethod TextRotation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TextToColumnsRequest: Splits a column of text into multiple
// columns,
// based on a delimiter in each cell.
type TextToColumnsRequest struct {
	// Delimiter: The delimiter to use. Used only if delimiterType
	// is
	// CUSTOM.
	Delimiter string `json:"delimiter,omitempty"`

	// DelimiterType: The delimiter type to use.
	//
	// Possible values:
	//   "DELIMITER_TYPE_UNSPECIFIED" - Default value. This value must not
	// be used.
	//   "COMMA" - ","
	//   "SEMICOLON" - ";"
	//   "PERIOD" - "."
	//   "SPACE" - " "
	//   "CUSTOM" - A custom value as defined in delimiter.
	DelimiterType string `json:"delimiterType,omitempty"`

	// Source: The source data range.  This must span exactly one column.
	Source *GridRange `json:"source,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Delimiter") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Delimiter") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TextToColumnsRequest) MarshalJSON() ([]byte, error) {
	type noMethod TextToColumnsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UnmergeCellsRequest: Unmerges cells in the given range.
type UnmergeCellsRequest struct {
	// Range: The range within which all cells should be unmerged.
	// If the range spans multiple merges, all will be unmerged.
	// The range must not partially span any merge.
	Range *GridRange `json:"range,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Range") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Range") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UnmergeCellsRequest) MarshalJSON() ([]byte, error) {
	type noMethod UnmergeCellsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateBandingRequest: Updates properties of the supplied banded
// range.
type UpdateBandingRequest struct {
	// BandedRange: The banded range to update with the new properties.
	BandedRange *BandedRange `json:"bandedRange,omitempty"`

	// Fields: The fields that should be updated.  At least one field must
	// be specified.
	// The root `bandedRange` is implied and should not be specified.
	// A single "*" can be used as short-hand for listing every field.
	Fields string `json:"fields,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BandedRange") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BandedRange") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateBandingRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateBandingRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateBordersRequest: Updates the borders of a range.
// If a field is not set in the request, that means the border remains
// as-is.
// For example, with two subsequent UpdateBordersRequest:
//
//  1. range: A1:A5 `{ top: RED, bottom: WHITE }`
//  2. range: A1:A5 `{ left: BLUE }`
//
// That would result in A1:A5 having a borders of
// `{ top: RED, bottom: WHITE, left: BLUE }`.
// If you want to clear a border, explicitly set the style to
// NONE.
type UpdateBordersRequest struct {
	// Bottom: The border to put at the bottom of the range.
	Bottom *Border `json:"bottom,omitempty"`

	// InnerHorizontal: The horizontal border to put within the range.
	InnerHorizontal *Border `json:"innerHorizontal,omitempty"`

	// InnerVertical: The vertical border to put within the range.
	InnerVertical *Border `json:"innerVertical,omitempty"`

	// Left: The border to put at the left of the range.
	Left *Border `json:"left,omitempty"`

	// Range: The range whose borders should be updated.
	Range *GridRange `json:"range,omitempty"`

	// Right: The border to put at the right of the range.
	Right *Border `json:"right,omitempty"`

	// Top: The border to put at the top of the range.
	Top *Border `json:"top,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Bottom") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Bottom") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateBordersRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateBordersRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateCellsRequest: Updates all cells in a range with new data.
type UpdateCellsRequest struct {
	// Fields: The fields of CellData that should be updated.
	// At least one field must be specified.
	// The root is the CellData; 'row.values.' should not be specified.
	// A single "*" can be used as short-hand for listing every field.
	Fields string `json:"fields,omitempty"`

	// Range: The range to write data to.
	//
	// If the data in rows does not cover the entire requested range,
	// the fields matching those set in fields will be cleared.
	Range *GridRange `json:"range,omitempty"`

	// Rows: The data to write.
	Rows []*RowData `json:"rows,omitempty"`

	// Start: The coordinate to start writing data at.
	// Any number of rows and columns (including a different number
	// of
	// columns per row) may be written.
	Start *GridCoordinate `json:"start,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fields") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fields") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateCellsRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateCellsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateChartSpecRequest: Updates a chart's specifications.
// (This does not move or resize a chart. To move or resize a chart,
// use
//  UpdateEmbeddedObjectPositionRequest.)
type UpdateChartSpecRequest struct {
	// ChartId: The ID of the chart to update.
	ChartId int64 `json:"chartId,omitempty"`

	// Spec: The specification to apply to the chart.
	Spec *ChartSpec `json:"spec,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ChartId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ChartId") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateChartSpecRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateChartSpecRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateConditionalFormatRuleRequest: Updates a conditional format rule
// at the given index,
// or moves a conditional format rule to another index.
type UpdateConditionalFormatRuleRequest struct {
	// Index: The zero-based index of the rule that should be replaced or
	// moved.
	Index int64 `json:"index,omitempty"`

	// NewIndex: The zero-based new index the rule should end up at.
	NewIndex int64 `json:"newIndex,omitempty"`

	// Rule: The rule that should replace the rule at the given index.
	Rule *ConditionalFormatRule `json:"rule,omitempty"`

	// SheetId: The sheet of the rule to move.  Required if new_index is
	// set,
	// unused otherwise.
	SheetId int64 `json:"sheetId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Index") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Index") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateConditionalFormatRuleRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateConditionalFormatRuleRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateConditionalFormatRuleResponse: The result of updating a
// conditional format rule.
type UpdateConditionalFormatRuleResponse struct {
	// NewIndex: The index of the new rule.
	NewIndex int64 `json:"newIndex,omitempty"`

	// NewRule: The new rule that replaced the old rule (if replacing),
	// or the rule that was moved (if moved)
	NewRule *ConditionalFormatRule `json:"newRule,omitempty"`

	// OldIndex: The old index of the rule. Not set if a rule was
	// replaced
	// (because it is the same as new_index).
	OldIndex int64 `json:"oldIndex,omitempty"`

	// OldRule: The old (deleted) rule. Not set if a rule was moved
	// (because it is the same as new_rule).
	OldRule *ConditionalFormatRule `json:"oldRule,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NewIndex") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "NewIndex") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateConditionalFormatRuleResponse) MarshalJSON() ([]byte, error) {
	type noMethod UpdateConditionalFormatRuleResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateDimensionPropertiesRequest: Updates properties of dimensions
// within the specified range.
type UpdateDimensionPropertiesRequest struct {
	// Fields: The fields that should be updated.  At least one field must
	// be specified.
	// The root `properties` is implied and should not be specified.
	// A single "*" can be used as short-hand for listing every field.
	Fields string `json:"fields,omitempty"`

	// Properties: Properties to update.
	Properties *DimensionProperties `json:"properties,omitempty"`

	// Range: The rows or columns to update.
	Range *DimensionRange `json:"range,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fields") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fields") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateDimensionPropertiesRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateDimensionPropertiesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateEmbeddedObjectPositionRequest: Update an embedded object's
// position (such as a moving or resizing a
// chart or image).
type UpdateEmbeddedObjectPositionRequest struct {
	// Fields: The fields of OverlayPosition
	// that should be updated when setting a new position. Used only
	// if
	// newPosition.overlayPosition
	// is set, in which case at least one field must
	// be specified.  The root `newPosition.overlayPosition` is implied
	// and
	// should not be specified.
	// A single "*" can be used as short-hand for listing every field.
	Fields string `json:"fields,omitempty"`

	// NewPosition: An explicit position to move the embedded object to.
	// If newPosition.sheetId is set,
	// a new sheet with that ID will be created.
	// If newPosition.newSheet is set to true,
	// a new sheet will be created with an ID that will be chosen for you.
	NewPosition *EmbeddedObjectPosition `json:"newPosition,omitempty"`

	// ObjectId: The ID of the object to moved.
	ObjectId int64 `json:"objectId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fields") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fields") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateEmbeddedObjectPositionRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateEmbeddedObjectPositionRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateEmbeddedObjectPositionResponse: The result of updating an
// embedded object's position.
type UpdateEmbeddedObjectPositionResponse struct {
	// Position: The new position of the embedded object.
	Position *EmbeddedObjectPosition `json:"position,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Position") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Position") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateEmbeddedObjectPositionResponse) MarshalJSON() ([]byte, error) {
	type noMethod UpdateEmbeddedObjectPositionResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateFilterViewRequest: Updates properties of the filter view.
type UpdateFilterViewRequest struct {
	// Fields: The fields that should be updated.  At least one field must
	// be specified.
	// The root `filter` is implied and should not be specified.
	// A single "*" can be used as short-hand for listing every field.
	Fields string `json:"fields,omitempty"`

	// Filter: The new properties of the filter view.
	Filter *FilterView `json:"filter,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fields") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fields") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateFilterViewRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateFilterViewRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateNamedRangeRequest: Updates properties of the named range with
// the specified
// namedRangeId.
type UpdateNamedRangeRequest struct {
	// Fields: The fields that should be updated.  At least one field must
	// be specified.
	// The root `namedRange` is implied and should not be specified.
	// A single "*" can be used as short-hand for listing every field.
	Fields string `json:"fields,omitempty"`

	// NamedRange: The named range to update with the new properties.
	NamedRange *NamedRange `json:"namedRange,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fields") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fields") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateNamedRangeRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateNamedRangeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateProtectedRangeRequest: Updates an existing protected range with
// the specified
// protectedRangeId.
type UpdateProtectedRangeRequest struct {
	// Fields: The fields that should be updated.  At least one field must
	// be specified.
	// The root `protectedRange` is implied and should not be specified.
	// A single "*" can be used as short-hand for listing every field.
	Fields string `json:"fields,omitempty"`

	// ProtectedRange: The protected range to update with the new
	// properties.
	ProtectedRange *ProtectedRange `json:"protectedRange,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fields") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fields") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateProtectedRangeRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateProtectedRangeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateSheetPropertiesRequest: Updates properties of the sheet with
// the specified
// sheetId.
type UpdateSheetPropertiesRequest struct {
	// Fields: The fields that should be updated.  At least one field must
	// be specified.
	// The root `properties` is implied and should not be specified.
	// A single "*" can be used as short-hand for listing every field.
	Fields string `json:"fields,omitempty"`

	// Properties: The properties to update.
	Properties *SheetProperties `json:"properties,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fields") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fields") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateSheetPropertiesRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateSheetPropertiesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateSpreadsheetPropertiesRequest: Updates properties of a
// spreadsheet.
type UpdateSpreadsheetPropertiesRequest struct {
	// Fields: The fields that should be updated.  At least one field must
	// be specified.
	// The root 'properties' is implied and should not be specified.
	// A single "*" can be used as short-hand for listing every field.
	Fields string `json:"fields,omitempty"`

	// Properties: The properties to update.
	Properties *SpreadsheetProperties `json:"properties,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Fields") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Fields") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateSpreadsheetPropertiesRequest) MarshalJSON() ([]byte, error) {
	type noMethod UpdateSpreadsheetPropertiesRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UpdateValuesResponse: The response when updating a range of values in
// a spreadsheet.
type UpdateValuesResponse struct {
	// SpreadsheetId: The spreadsheet the updates were applied to.
	SpreadsheetId string `json:"spreadsheetId,omitempty"`

	// UpdatedCells: The number of cells updated.
	UpdatedCells int64 `json:"updatedCells,omitempty"`

	// UpdatedColumns: The number of columns where at least one cell in the
	// column was updated.
	UpdatedColumns int64 `json:"updatedColumns,omitempty"`

	// UpdatedData: The values of the cells after updates were applied.
	// This is only included if the request's `includeValuesInResponse`
	// field
	// was `true`.
	UpdatedData *ValueRange `json:"updatedData,omitempty"`

	// UpdatedRange: The range (in A1 notation) that updates were applied
	// to.
	UpdatedRange string `json:"updatedRange,omitempty"`

	// UpdatedRows: The number of rows where at least one cell in the row
	// was updated.
	UpdatedRows int64 `json:"updatedRows,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "SpreadsheetId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SpreadsheetId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UpdateValuesResponse) MarshalJSON() ([]byte, error) {
	type noMethod UpdateValuesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ValueRange: Data within a range of the spreadsheet.
type ValueRange struct {
	// MajorDimension: The major dimension of the values.
	//
	// For output, if the spreadsheet data is: `A1=1,B1=2,A2=3,B2=4`,
	// then requesting `range=A1:B2,majorDimension=ROWS` will
	// return
	// `[[1,2],[3,4]]`,
	// whereas requesting `range=A1:B2,majorDimension=COLUMNS` will
	// return
	// `[[1,3],[2,4]]`.
	//
	// For input, with `range=A1:B2,majorDimension=ROWS` then
	// `[[1,2],[3,4]]`
	// will set `A1=1,B1=2,A2=3,B2=4`. With
	// `range=A1:B2,majorDimension=COLUMNS`
	// then `[[1,2],[3,4]]` will set `A1=1,B1=3,A2=2,B2=4`.
	//
	// When writing, if this field is not set, it defaults to ROWS.
	//
	// Possible values:
	//   "DIMENSION_UNSPECIFIED" - The default value, do not use.
	//   "ROWS" - Operates on the rows of a sheet.
	//   "COLUMNS" - Operates on the columns of a sheet.
	MajorDimension string `json:"majorDimension,omitempty"`

	// Range: The range the values cover, in A1 notation.
	// For output, this range indicates the entire requested range,
	// even though the values will exclude trailing rows and columns.
	// When appending values, this field represents the range to search for
	// a
	// table, after which values will be appended.
	Range string `json:"range,omitempty"`

	// Values: The data that was read or to be written.  This is an array of
	// arrays,
	// the outer array representing all the data and each inner
	// array
	// representing a major dimension. Each item in the inner
	// array
	// corresponds with one cell.
	//
	// For output, empty trailing rows and columns will not be
	// included.
	//
	// For input, supported value types are: bool, string, and double.
	// Null values will be skipped.
	// To set a cell to an empty value, set the string value to an empty
	// string.
	Values [][]interface{} `json:"values,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "MajorDimension") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MajorDimension") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ValueRange) MarshalJSON() ([]byte, error) {
	type noMethod ValueRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "sheets.spreadsheets.batchUpdate":

type SpreadsheetsBatchUpdateCall struct {
	s                             *Service
	spreadsheetId                 string
	batchupdatespreadsheetrequest *BatchUpdateSpreadsheetRequest
	urlParams_                    gensupport.URLParams
	ctx_                          context.Context
	header_                       http.Header
}

// BatchUpdate: Applies one or more updates to the spreadsheet.
//
// Each request is validated before
// being applied. If any request is not valid then the entire request
// will
// fail and nothing will be applied.
//
// Some requests have replies to
// give you some information about how
// they are applied. The replies will mirror the requests.  For
// example,
// if you applied 4 updates and the 3rd one had a reply, then
// the
// response will have 2 empty replies, the actual reply, and another
// empty
// reply, in that order.
//
// Due to the collaborative nature of spreadsheets, it is not guaranteed
// that
// the spreadsheet will reflect exactly your changes after this
// completes,
// however it is guaranteed that the updates in the request will
// be
// applied together atomically. Your changes may be altered with respect
// to
// collaborator changes. If there are no collaborators, the
// spreadsheet
// should reflect your changes.
func (r *SpreadsheetsService) BatchUpdate(spreadsheetId string, batchupdatespreadsheetrequest *BatchUpdateSpreadsheetRequest) *SpreadsheetsBatchUpdateCall {
	c := &SpreadsheetsBatchUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.spreadsheetId = spreadsheetId
	c.batchupdatespreadsheetrequest = batchupdatespreadsheetrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SpreadsheetsBatchUpdateCall) Fields(s ...googleapi.Field) *SpreadsheetsBatchUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SpreadsheetsBatchUpdateCall) Context(ctx context.Context) *SpreadsheetsBatchUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SpreadsheetsBatchUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SpreadsheetsBatchUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.batchupdatespreadsheetrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/spreadsheets/{spreadsheetId}:batchUpdate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"spreadsheetId": c.spreadsheetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sheets.spreadsheets.batchUpdate" call.
// Exactly one of *BatchUpdateSpreadsheetResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *BatchUpdateSpreadsheetResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SpreadsheetsBatchUpdateCall) Do(opts ...googleapi.CallOption) (*BatchUpdateSpreadsheetResponse, error) {
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
	ret := &BatchUpdateSpreadsheetResponse{
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
	//   "description": "Applies one or more updates to the spreadsheet.\n\nEach request is validated before\nbeing applied. If any request is not valid then the entire request will\nfail and nothing will be applied.\n\nSome requests have replies to\ngive you some information about how\nthey are applied. The replies will mirror the requests.  For example,\nif you applied 4 updates and the 3rd one had a reply, then the\nresponse will have 2 empty replies, the actual reply, and another empty\nreply, in that order.\n\nDue to the collaborative nature of spreadsheets, it is not guaranteed that\nthe spreadsheet will reflect exactly your changes after this completes,\nhowever it is guaranteed that the updates in the request will be\napplied together atomically. Your changes may be altered with respect to\ncollaborator changes. If there are no collaborators, the spreadsheet\nshould reflect your changes.",
	//   "flatPath": "v4/spreadsheets/{spreadsheetId}:batchUpdate",
	//   "httpMethod": "POST",
	//   "id": "sheets.spreadsheets.batchUpdate",
	//   "parameterOrder": [
	//     "spreadsheetId"
	//   ],
	//   "parameters": {
	//     "spreadsheetId": {
	//       "description": "The spreadsheet to apply the updates to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/spreadsheets/{spreadsheetId}:batchUpdate",
	//   "request": {
	//     "$ref": "BatchUpdateSpreadsheetRequest"
	//   },
	//   "response": {
	//     "$ref": "BatchUpdateSpreadsheetResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/spreadsheets"
	//   ]
	// }

}

// method id "sheets.spreadsheets.create":

type SpreadsheetsCreateCall struct {
	s           *Service
	spreadsheet *Spreadsheet
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Create: Creates a spreadsheet, returning the newly created
// spreadsheet.
func (r *SpreadsheetsService) Create(spreadsheet *Spreadsheet) *SpreadsheetsCreateCall {
	c := &SpreadsheetsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.spreadsheet = spreadsheet
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SpreadsheetsCreateCall) Fields(s ...googleapi.Field) *SpreadsheetsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SpreadsheetsCreateCall) Context(ctx context.Context) *SpreadsheetsCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SpreadsheetsCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SpreadsheetsCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.spreadsheet)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/spreadsheets")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sheets.spreadsheets.create" call.
// Exactly one of *Spreadsheet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Spreadsheet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SpreadsheetsCreateCall) Do(opts ...googleapi.CallOption) (*Spreadsheet, error) {
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
	ret := &Spreadsheet{
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
	//   "description": "Creates a spreadsheet, returning the newly created spreadsheet.",
	//   "flatPath": "v4/spreadsheets",
	//   "httpMethod": "POST",
	//   "id": "sheets.spreadsheets.create",
	//   "parameterOrder": [],
	//   "parameters": {},
	//   "path": "v4/spreadsheets",
	//   "request": {
	//     "$ref": "Spreadsheet"
	//   },
	//   "response": {
	//     "$ref": "Spreadsheet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/spreadsheets"
	//   ]
	// }

}

// method id "sheets.spreadsheets.get":

type SpreadsheetsGetCall struct {
	s             *Service
	spreadsheetId string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
	header_       http.Header
}

// Get: Returns the spreadsheet at the given ID.
// The caller must specify the spreadsheet ID.
//
// By default, data within grids will not be returned.
// You can include grid data one of two ways:
//
// * Specify a field mask listing your desired fields using the `fields`
// URL
// parameter in HTTP
//
// * Set the includeGridData
// URL parameter to true.  If a field mask is set, the
// `includeGridData`
// parameter is ignored
//
// For large spreadsheets, it is recommended to retrieve only the
// specific
// fields of the spreadsheet that you want.
//
// To retrieve only subsets of the spreadsheet, use the
// ranges URL parameter.
// Multiple ranges can be specified.  Limiting the range will
// return only the portions of the spreadsheet that intersect the
// requested
// ranges. Ranges are specified using A1 notation.
func (r *SpreadsheetsService) Get(spreadsheetId string) *SpreadsheetsGetCall {
	c := &SpreadsheetsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.spreadsheetId = spreadsheetId
	return c
}

// IncludeGridData sets the optional parameter "includeGridData": True
// if grid data should be returned.
// This parameter is ignored if a field mask was set in the request.
func (c *SpreadsheetsGetCall) IncludeGridData(includeGridData bool) *SpreadsheetsGetCall {
	c.urlParams_.Set("includeGridData", fmt.Sprint(includeGridData))
	return c
}

// Ranges sets the optional parameter "ranges": The ranges to retrieve
// from the spreadsheet.
func (c *SpreadsheetsGetCall) Ranges(ranges ...string) *SpreadsheetsGetCall {
	c.urlParams_.SetMulti("ranges", append([]string{}, ranges...))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SpreadsheetsGetCall) Fields(s ...googleapi.Field) *SpreadsheetsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *SpreadsheetsGetCall) IfNoneMatch(entityTag string) *SpreadsheetsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SpreadsheetsGetCall) Context(ctx context.Context) *SpreadsheetsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SpreadsheetsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SpreadsheetsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/spreadsheets/{spreadsheetId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"spreadsheetId": c.spreadsheetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sheets.spreadsheets.get" call.
// Exactly one of *Spreadsheet or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Spreadsheet.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SpreadsheetsGetCall) Do(opts ...googleapi.CallOption) (*Spreadsheet, error) {
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
	ret := &Spreadsheet{
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
	//   "description": "Returns the spreadsheet at the given ID.\nThe caller must specify the spreadsheet ID.\n\nBy default, data within grids will not be returned.\nYou can include grid data one of two ways:\n\n* Specify a field mask listing your desired fields using the `fields` URL\nparameter in HTTP\n\n* Set the includeGridData\nURL parameter to true.  If a field mask is set, the `includeGridData`\nparameter is ignored\n\nFor large spreadsheets, it is recommended to retrieve only the specific\nfields of the spreadsheet that you want.\n\nTo retrieve only subsets of the spreadsheet, use the\nranges URL parameter.\nMultiple ranges can be specified.  Limiting the range will\nreturn only the portions of the spreadsheet that intersect the requested\nranges. Ranges are specified using A1 notation.",
	//   "flatPath": "v4/spreadsheets/{spreadsheetId}",
	//   "httpMethod": "GET",
	//   "id": "sheets.spreadsheets.get",
	//   "parameterOrder": [
	//     "spreadsheetId"
	//   ],
	//   "parameters": {
	//     "includeGridData": {
	//       "description": "True if grid data should be returned.\nThis parameter is ignored if a field mask was set in the request.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "ranges": {
	//       "description": "The ranges to retrieve from the spreadsheet.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "spreadsheetId": {
	//       "description": "The spreadsheet to request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/spreadsheets/{spreadsheetId}",
	//   "response": {
	//     "$ref": "Spreadsheet"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.readonly",
	//     "https://www.googleapis.com/auth/spreadsheets",
	//     "https://www.googleapis.com/auth/spreadsheets.readonly"
	//   ]
	// }

}

// method id "sheets.spreadsheets.sheets.copyTo":

type SpreadsheetsSheetsCopyToCall struct {
	s                                    *Service
	spreadsheetId                        string
	sheetId                              int64
	copysheettoanotherspreadsheetrequest *CopySheetToAnotherSpreadsheetRequest
	urlParams_                           gensupport.URLParams
	ctx_                                 context.Context
	header_                              http.Header
}

// CopyTo: Copies a single sheet from a spreadsheet to another
// spreadsheet.
// Returns the properties of the newly created sheet.
func (r *SpreadsheetsSheetsService) CopyTo(spreadsheetId string, sheetId int64, copysheettoanotherspreadsheetrequest *CopySheetToAnotherSpreadsheetRequest) *SpreadsheetsSheetsCopyToCall {
	c := &SpreadsheetsSheetsCopyToCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.spreadsheetId = spreadsheetId
	c.sheetId = sheetId
	c.copysheettoanotherspreadsheetrequest = copysheettoanotherspreadsheetrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SpreadsheetsSheetsCopyToCall) Fields(s ...googleapi.Field) *SpreadsheetsSheetsCopyToCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SpreadsheetsSheetsCopyToCall) Context(ctx context.Context) *SpreadsheetsSheetsCopyToCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SpreadsheetsSheetsCopyToCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SpreadsheetsSheetsCopyToCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.copysheettoanotherspreadsheetrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/spreadsheets/{spreadsheetId}/sheets/{sheetId}:copyTo")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"spreadsheetId": c.spreadsheetId,
		"sheetId":       strconv.FormatInt(c.sheetId, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sheets.spreadsheets.sheets.copyTo" call.
// Exactly one of *SheetProperties or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *SheetProperties.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SpreadsheetsSheetsCopyToCall) Do(opts ...googleapi.CallOption) (*SheetProperties, error) {
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
	ret := &SheetProperties{
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
	//   "description": "Copies a single sheet from a spreadsheet to another spreadsheet.\nReturns the properties of the newly created sheet.",
	//   "flatPath": "v4/spreadsheets/{spreadsheetId}/sheets/{sheetId}:copyTo",
	//   "httpMethod": "POST",
	//   "id": "sheets.spreadsheets.sheets.copyTo",
	//   "parameterOrder": [
	//     "spreadsheetId",
	//     "sheetId"
	//   ],
	//   "parameters": {
	//     "sheetId": {
	//       "description": "The ID of the sheet to copy.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "spreadsheetId": {
	//       "description": "The ID of the spreadsheet containing the sheet to copy.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/spreadsheets/{spreadsheetId}/sheets/{sheetId}:copyTo",
	//   "request": {
	//     "$ref": "CopySheetToAnotherSpreadsheetRequest"
	//   },
	//   "response": {
	//     "$ref": "SheetProperties"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/spreadsheets"
	//   ]
	// }

}

// method id "sheets.spreadsheets.values.append":

type SpreadsheetsValuesAppendCall struct {
	s             *Service
	spreadsheetId string
	range_        string
	valuerange    *ValueRange
	urlParams_    gensupport.URLParams
	ctx_          context.Context
	header_       http.Header
}

// Append: Appends values to a spreadsheet. The input range is used to
// search for
// existing data and find a "table" within that range. Values will
// be
// appended to the next row of the table, starting with the first column
// of
// the table. See
// the
// [guide](/sheets/api/guides/values#appending_values)
// and
// [sample code](/sheets/api/samples/writing#append_values)
// for specific details of how tables are detected and data is
// appended.
//
// The caller must specify the spreadsheet ID, range, and
// a valueInputOption.  The `valueInputOption` only
// controls how the input data will be added to the sheet (column-wise
// or
// row-wise), it does not influence what cell the data starts being
// written
// to.
func (r *SpreadsheetsValuesService) Append(spreadsheetId string, range_ string, valuerange *ValueRange) *SpreadsheetsValuesAppendCall {
	c := &SpreadsheetsValuesAppendCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.spreadsheetId = spreadsheetId
	c.range_ = range_
	c.valuerange = valuerange
	return c
}

// IncludeValuesInResponse sets the optional parameter
// "includeValuesInResponse": Determines if the update response should
// include the values
// of the cells that were appended. By default, responses
// do not include the updated values.
func (c *SpreadsheetsValuesAppendCall) IncludeValuesInResponse(includeValuesInResponse bool) *SpreadsheetsValuesAppendCall {
	c.urlParams_.Set("includeValuesInResponse", fmt.Sprint(includeValuesInResponse))
	return c
}

// InsertDataOption sets the optional parameter "insertDataOption": How
// the input data should be inserted.
//
// Possible values:
//   "OVERWRITE"
//   "INSERT_ROWS"
func (c *SpreadsheetsValuesAppendCall) InsertDataOption(insertDataOption string) *SpreadsheetsValuesAppendCall {
	c.urlParams_.Set("insertDataOption", insertDataOption)
	return c
}

// ResponseDateTimeRenderOption sets the optional parameter
// "responseDateTimeRenderOption": Determines how dates, times, and
// durations in the response should be
// rendered. This is ignored if response_value_render_option
// is
// FORMATTED_VALUE.
// The default dateTime render option is
// [DateTimeRenderOption.SERIAL_NUMBER].
//
// Possible values:
//   "SERIAL_NUMBER"
//   "FORMATTED_STRING"
func (c *SpreadsheetsValuesAppendCall) ResponseDateTimeRenderOption(responseDateTimeRenderOption string) *SpreadsheetsValuesAppendCall {
	c.urlParams_.Set("responseDateTimeRenderOption", responseDateTimeRenderOption)
	return c
}

// ResponseValueRenderOption sets the optional parameter
// "responseValueRenderOption": Determines how values in the response
// should be rendered.
// The default render option is ValueRenderOption.FORMATTED_VALUE.
//
// Possible values:
//   "FORMATTED_VALUE"
//   "UNFORMATTED_VALUE"
//   "FORMULA"
func (c *SpreadsheetsValuesAppendCall) ResponseValueRenderOption(responseValueRenderOption string) *SpreadsheetsValuesAppendCall {
	c.urlParams_.Set("responseValueRenderOption", responseValueRenderOption)
	return c
}

// ValueInputOption sets the optional parameter "valueInputOption": How
// the input data should be interpreted.
//
// Possible values:
//   "INPUT_VALUE_OPTION_UNSPECIFIED"
//   "RAW"
//   "USER_ENTERED"
func (c *SpreadsheetsValuesAppendCall) ValueInputOption(valueInputOption string) *SpreadsheetsValuesAppendCall {
	c.urlParams_.Set("valueInputOption", valueInputOption)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SpreadsheetsValuesAppendCall) Fields(s ...googleapi.Field) *SpreadsheetsValuesAppendCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SpreadsheetsValuesAppendCall) Context(ctx context.Context) *SpreadsheetsValuesAppendCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SpreadsheetsValuesAppendCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SpreadsheetsValuesAppendCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.valuerange)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/spreadsheets/{spreadsheetId}/values/{range}:append")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"spreadsheetId": c.spreadsheetId,
		"range":         c.range_,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sheets.spreadsheets.values.append" call.
// Exactly one of *AppendValuesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AppendValuesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SpreadsheetsValuesAppendCall) Do(opts ...googleapi.CallOption) (*AppendValuesResponse, error) {
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
	ret := &AppendValuesResponse{
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
	//   "description": "Appends values to a spreadsheet. The input range is used to search for\nexisting data and find a \"table\" within that range. Values will be\nappended to the next row of the table, starting with the first column of\nthe table. See the\n[guide](/sheets/api/guides/values#appending_values)\nand\n[sample code](/sheets/api/samples/writing#append_values)\nfor specific details of how tables are detected and data is appended.\n\nThe caller must specify the spreadsheet ID, range, and\na valueInputOption.  The `valueInputOption` only\ncontrols how the input data will be added to the sheet (column-wise or\nrow-wise), it does not influence what cell the data starts being written\nto.",
	//   "flatPath": "v4/spreadsheets/{spreadsheetId}/values/{range}:append",
	//   "httpMethod": "POST",
	//   "id": "sheets.spreadsheets.values.append",
	//   "parameterOrder": [
	//     "spreadsheetId",
	//     "range"
	//   ],
	//   "parameters": {
	//     "includeValuesInResponse": {
	//       "description": "Determines if the update response should include the values\nof the cells that were appended. By default, responses\ndo not include the updated values.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "insertDataOption": {
	//       "description": "How the input data should be inserted.",
	//       "enum": [
	//         "OVERWRITE",
	//         "INSERT_ROWS"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "range": {
	//       "description": "The A1 notation of a range to search for a logical table of data.\nValues will be appended after the last row of the table.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "responseDateTimeRenderOption": {
	//       "description": "Determines how dates, times, and durations in the response should be\nrendered. This is ignored if response_value_render_option is\nFORMATTED_VALUE.\nThe default dateTime render option is [DateTimeRenderOption.SERIAL_NUMBER].",
	//       "enum": [
	//         "SERIAL_NUMBER",
	//         "FORMATTED_STRING"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "responseValueRenderOption": {
	//       "description": "Determines how values in the response should be rendered.\nThe default render option is ValueRenderOption.FORMATTED_VALUE.",
	//       "enum": [
	//         "FORMATTED_VALUE",
	//         "UNFORMATTED_VALUE",
	//         "FORMULA"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "spreadsheetId": {
	//       "description": "The ID of the spreadsheet to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "valueInputOption": {
	//       "description": "How the input data should be interpreted.",
	//       "enum": [
	//         "INPUT_VALUE_OPTION_UNSPECIFIED",
	//         "RAW",
	//         "USER_ENTERED"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/spreadsheets/{spreadsheetId}/values/{range}:append",
	//   "request": {
	//     "$ref": "ValueRange"
	//   },
	//   "response": {
	//     "$ref": "AppendValuesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/spreadsheets"
	//   ]
	// }

}

// method id "sheets.spreadsheets.values.batchClear":

type SpreadsheetsValuesBatchClearCall struct {
	s                       *Service
	spreadsheetId           string
	batchclearvaluesrequest *BatchClearValuesRequest
	urlParams_              gensupport.URLParams
	ctx_                    context.Context
	header_                 http.Header
}

// BatchClear: Clears one or more ranges of values from a
// spreadsheet.
// The caller must specify the spreadsheet ID and one or more
// ranges.
// Only values are cleared -- all other properties of the cell (such
// as
// formatting, data validation, etc..) are kept.
func (r *SpreadsheetsValuesService) BatchClear(spreadsheetId string, batchclearvaluesrequest *BatchClearValuesRequest) *SpreadsheetsValuesBatchClearCall {
	c := &SpreadsheetsValuesBatchClearCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.spreadsheetId = spreadsheetId
	c.batchclearvaluesrequest = batchclearvaluesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SpreadsheetsValuesBatchClearCall) Fields(s ...googleapi.Field) *SpreadsheetsValuesBatchClearCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SpreadsheetsValuesBatchClearCall) Context(ctx context.Context) *SpreadsheetsValuesBatchClearCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SpreadsheetsValuesBatchClearCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SpreadsheetsValuesBatchClearCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.batchclearvaluesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/spreadsheets/{spreadsheetId}/values:batchClear")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"spreadsheetId": c.spreadsheetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sheets.spreadsheets.values.batchClear" call.
// Exactly one of *BatchClearValuesResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *BatchClearValuesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SpreadsheetsValuesBatchClearCall) Do(opts ...googleapi.CallOption) (*BatchClearValuesResponse, error) {
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
	ret := &BatchClearValuesResponse{
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
	//   "description": "Clears one or more ranges of values from a spreadsheet.\nThe caller must specify the spreadsheet ID and one or more ranges.\nOnly values are cleared -- all other properties of the cell (such as\nformatting, data validation, etc..) are kept.",
	//   "flatPath": "v4/spreadsheets/{spreadsheetId}/values:batchClear",
	//   "httpMethod": "POST",
	//   "id": "sheets.spreadsheets.values.batchClear",
	//   "parameterOrder": [
	//     "spreadsheetId"
	//   ],
	//   "parameters": {
	//     "spreadsheetId": {
	//       "description": "The ID of the spreadsheet to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/spreadsheets/{spreadsheetId}/values:batchClear",
	//   "request": {
	//     "$ref": "BatchClearValuesRequest"
	//   },
	//   "response": {
	//     "$ref": "BatchClearValuesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/spreadsheets"
	//   ]
	// }

}

// method id "sheets.spreadsheets.values.batchGet":

type SpreadsheetsValuesBatchGetCall struct {
	s             *Service
	spreadsheetId string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
	header_       http.Header
}

// BatchGet: Returns one or more ranges of values from a
// spreadsheet.
// The caller must specify the spreadsheet ID and one or more ranges.
func (r *SpreadsheetsValuesService) BatchGet(spreadsheetId string) *SpreadsheetsValuesBatchGetCall {
	c := &SpreadsheetsValuesBatchGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.spreadsheetId = spreadsheetId
	return c
}

// DateTimeRenderOption sets the optional parameter
// "dateTimeRenderOption": How dates, times, and durations should be
// represented in the output.
// This is ignored if value_render_option is
// FORMATTED_VALUE.
// The default dateTime render option is
// [DateTimeRenderOption.SERIAL_NUMBER].
//
// Possible values:
//   "SERIAL_NUMBER"
//   "FORMATTED_STRING"
func (c *SpreadsheetsValuesBatchGetCall) DateTimeRenderOption(dateTimeRenderOption string) *SpreadsheetsValuesBatchGetCall {
	c.urlParams_.Set("dateTimeRenderOption", dateTimeRenderOption)
	return c
}

// MajorDimension sets the optional parameter "majorDimension": The
// major dimension that results should use.
//
// For example, if the spreadsheet data is: `A1=1,B1=2,A2=3,B2=4`,
// then requesting `range=A1:B2,majorDimension=ROWS` will
// return
// `[[1,2],[3,4]]`,
// whereas requesting `range=A1:B2,majorDimension=COLUMNS` will
// return
// `[[1,3],[2,4]]`.
//
// Possible values:
//   "DIMENSION_UNSPECIFIED"
//   "ROWS"
//   "COLUMNS"
func (c *SpreadsheetsValuesBatchGetCall) MajorDimension(majorDimension string) *SpreadsheetsValuesBatchGetCall {
	c.urlParams_.Set("majorDimension", majorDimension)
	return c
}

// Ranges sets the optional parameter "ranges": The A1 notation of the
// values to retrieve.
func (c *SpreadsheetsValuesBatchGetCall) Ranges(ranges ...string) *SpreadsheetsValuesBatchGetCall {
	c.urlParams_.SetMulti("ranges", append([]string{}, ranges...))
	return c
}

// ValueRenderOption sets the optional parameter "valueRenderOption":
// How values should be represented in the output.
// The default render option is ValueRenderOption.FORMATTED_VALUE.
//
// Possible values:
//   "FORMATTED_VALUE"
//   "UNFORMATTED_VALUE"
//   "FORMULA"
func (c *SpreadsheetsValuesBatchGetCall) ValueRenderOption(valueRenderOption string) *SpreadsheetsValuesBatchGetCall {
	c.urlParams_.Set("valueRenderOption", valueRenderOption)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SpreadsheetsValuesBatchGetCall) Fields(s ...googleapi.Field) *SpreadsheetsValuesBatchGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *SpreadsheetsValuesBatchGetCall) IfNoneMatch(entityTag string) *SpreadsheetsValuesBatchGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SpreadsheetsValuesBatchGetCall) Context(ctx context.Context) *SpreadsheetsValuesBatchGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SpreadsheetsValuesBatchGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SpreadsheetsValuesBatchGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/spreadsheets/{spreadsheetId}/values:batchGet")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"spreadsheetId": c.spreadsheetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sheets.spreadsheets.values.batchGet" call.
// Exactly one of *BatchGetValuesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *BatchGetValuesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SpreadsheetsValuesBatchGetCall) Do(opts ...googleapi.CallOption) (*BatchGetValuesResponse, error) {
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
	ret := &BatchGetValuesResponse{
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
	//   "description": "Returns one or more ranges of values from a spreadsheet.\nThe caller must specify the spreadsheet ID and one or more ranges.",
	//   "flatPath": "v4/spreadsheets/{spreadsheetId}/values:batchGet",
	//   "httpMethod": "GET",
	//   "id": "sheets.spreadsheets.values.batchGet",
	//   "parameterOrder": [
	//     "spreadsheetId"
	//   ],
	//   "parameters": {
	//     "dateTimeRenderOption": {
	//       "description": "How dates, times, and durations should be represented in the output.\nThis is ignored if value_render_option is\nFORMATTED_VALUE.\nThe default dateTime render option is [DateTimeRenderOption.SERIAL_NUMBER].",
	//       "enum": [
	//         "SERIAL_NUMBER",
	//         "FORMATTED_STRING"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "majorDimension": {
	//       "description": "The major dimension that results should use.\n\nFor example, if the spreadsheet data is: `A1=1,B1=2,A2=3,B2=4`,\nthen requesting `range=A1:B2,majorDimension=ROWS` will return\n`[[1,2],[3,4]]`,\nwhereas requesting `range=A1:B2,majorDimension=COLUMNS` will return\n`[[1,3],[2,4]]`.",
	//       "enum": [
	//         "DIMENSION_UNSPECIFIED",
	//         "ROWS",
	//         "COLUMNS"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "ranges": {
	//       "description": "The A1 notation of the values to retrieve.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "spreadsheetId": {
	//       "description": "The ID of the spreadsheet to retrieve data from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "valueRenderOption": {
	//       "description": "How values should be represented in the output.\nThe default render option is ValueRenderOption.FORMATTED_VALUE.",
	//       "enum": [
	//         "FORMATTED_VALUE",
	//         "UNFORMATTED_VALUE",
	//         "FORMULA"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/spreadsheets/{spreadsheetId}/values:batchGet",
	//   "response": {
	//     "$ref": "BatchGetValuesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.readonly",
	//     "https://www.googleapis.com/auth/spreadsheets",
	//     "https://www.googleapis.com/auth/spreadsheets.readonly"
	//   ]
	// }

}

// method id "sheets.spreadsheets.values.batchUpdate":

type SpreadsheetsValuesBatchUpdateCall struct {
	s                        *Service
	spreadsheetId            string
	batchupdatevaluesrequest *BatchUpdateValuesRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// BatchUpdate: Sets values in one or more ranges of a spreadsheet.
// The caller must specify the spreadsheet ID,
// a valueInputOption, and one or more
// ValueRanges.
func (r *SpreadsheetsValuesService) BatchUpdate(spreadsheetId string, batchupdatevaluesrequest *BatchUpdateValuesRequest) *SpreadsheetsValuesBatchUpdateCall {
	c := &SpreadsheetsValuesBatchUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.spreadsheetId = spreadsheetId
	c.batchupdatevaluesrequest = batchupdatevaluesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SpreadsheetsValuesBatchUpdateCall) Fields(s ...googleapi.Field) *SpreadsheetsValuesBatchUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SpreadsheetsValuesBatchUpdateCall) Context(ctx context.Context) *SpreadsheetsValuesBatchUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SpreadsheetsValuesBatchUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SpreadsheetsValuesBatchUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.batchupdatevaluesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/spreadsheets/{spreadsheetId}/values:batchUpdate")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"spreadsheetId": c.spreadsheetId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sheets.spreadsheets.values.batchUpdate" call.
// Exactly one of *BatchUpdateValuesResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *BatchUpdateValuesResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SpreadsheetsValuesBatchUpdateCall) Do(opts ...googleapi.CallOption) (*BatchUpdateValuesResponse, error) {
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
	ret := &BatchUpdateValuesResponse{
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
	//   "description": "Sets values in one or more ranges of a spreadsheet.\nThe caller must specify the spreadsheet ID,\na valueInputOption, and one or more\nValueRanges.",
	//   "flatPath": "v4/spreadsheets/{spreadsheetId}/values:batchUpdate",
	//   "httpMethod": "POST",
	//   "id": "sheets.spreadsheets.values.batchUpdate",
	//   "parameterOrder": [
	//     "spreadsheetId"
	//   ],
	//   "parameters": {
	//     "spreadsheetId": {
	//       "description": "The ID of the spreadsheet to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/spreadsheets/{spreadsheetId}/values:batchUpdate",
	//   "request": {
	//     "$ref": "BatchUpdateValuesRequest"
	//   },
	//   "response": {
	//     "$ref": "BatchUpdateValuesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/spreadsheets"
	//   ]
	// }

}

// method id "sheets.spreadsheets.values.clear":

type SpreadsheetsValuesClearCall struct {
	s                  *Service
	spreadsheetId      string
	range_             string
	clearvaluesrequest *ClearValuesRequest
	urlParams_         gensupport.URLParams
	ctx_               context.Context
	header_            http.Header
}

// Clear: Clears values from a spreadsheet.
// The caller must specify the spreadsheet ID and range.
// Only values are cleared -- all other properties of the cell (such
// as
// formatting, data validation, etc..) are kept.
func (r *SpreadsheetsValuesService) Clear(spreadsheetId string, range_ string, clearvaluesrequest *ClearValuesRequest) *SpreadsheetsValuesClearCall {
	c := &SpreadsheetsValuesClearCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.spreadsheetId = spreadsheetId
	c.range_ = range_
	c.clearvaluesrequest = clearvaluesrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SpreadsheetsValuesClearCall) Fields(s ...googleapi.Field) *SpreadsheetsValuesClearCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SpreadsheetsValuesClearCall) Context(ctx context.Context) *SpreadsheetsValuesClearCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SpreadsheetsValuesClearCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SpreadsheetsValuesClearCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.clearvaluesrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/spreadsheets/{spreadsheetId}/values/{range}:clear")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"spreadsheetId": c.spreadsheetId,
		"range":         c.range_,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sheets.spreadsheets.values.clear" call.
// Exactly one of *ClearValuesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ClearValuesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SpreadsheetsValuesClearCall) Do(opts ...googleapi.CallOption) (*ClearValuesResponse, error) {
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
	ret := &ClearValuesResponse{
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
	//   "description": "Clears values from a spreadsheet.\nThe caller must specify the spreadsheet ID and range.\nOnly values are cleared -- all other properties of the cell (such as\nformatting, data validation, etc..) are kept.",
	//   "flatPath": "v4/spreadsheets/{spreadsheetId}/values/{range}:clear",
	//   "httpMethod": "POST",
	//   "id": "sheets.spreadsheets.values.clear",
	//   "parameterOrder": [
	//     "spreadsheetId",
	//     "range"
	//   ],
	//   "parameters": {
	//     "range": {
	//       "description": "The A1 notation of the values to clear.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "spreadsheetId": {
	//       "description": "The ID of the spreadsheet to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/spreadsheets/{spreadsheetId}/values/{range}:clear",
	//   "request": {
	//     "$ref": "ClearValuesRequest"
	//   },
	//   "response": {
	//     "$ref": "ClearValuesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/spreadsheets"
	//   ]
	// }

}

// method id "sheets.spreadsheets.values.get":

type SpreadsheetsValuesGetCall struct {
	s             *Service
	spreadsheetId string
	range_        string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
	header_       http.Header
}

// Get: Returns a range of values from a spreadsheet.
// The caller must specify the spreadsheet ID and a range.
func (r *SpreadsheetsValuesService) Get(spreadsheetId string, range_ string) *SpreadsheetsValuesGetCall {
	c := &SpreadsheetsValuesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.spreadsheetId = spreadsheetId
	c.range_ = range_
	return c
}

// DateTimeRenderOption sets the optional parameter
// "dateTimeRenderOption": How dates, times, and durations should be
// represented in the output.
// This is ignored if value_render_option is
// FORMATTED_VALUE.
// The default dateTime render option is
// [DateTimeRenderOption.SERIAL_NUMBER].
//
// Possible values:
//   "SERIAL_NUMBER"
//   "FORMATTED_STRING"
func (c *SpreadsheetsValuesGetCall) DateTimeRenderOption(dateTimeRenderOption string) *SpreadsheetsValuesGetCall {
	c.urlParams_.Set("dateTimeRenderOption", dateTimeRenderOption)
	return c
}

// MajorDimension sets the optional parameter "majorDimension": The
// major dimension that results should use.
//
// For example, if the spreadsheet data is: `A1=1,B1=2,A2=3,B2=4`,
// then requesting `range=A1:B2,majorDimension=ROWS` will
// return
// `[[1,2],[3,4]]`,
// whereas requesting `range=A1:B2,majorDimension=COLUMNS` will
// return
// `[[1,3],[2,4]]`.
//
// Possible values:
//   "DIMENSION_UNSPECIFIED"
//   "ROWS"
//   "COLUMNS"
func (c *SpreadsheetsValuesGetCall) MajorDimension(majorDimension string) *SpreadsheetsValuesGetCall {
	c.urlParams_.Set("majorDimension", majorDimension)
	return c
}

// ValueRenderOption sets the optional parameter "valueRenderOption":
// How values should be represented in the output.
// The default render option is ValueRenderOption.FORMATTED_VALUE.
//
// Possible values:
//   "FORMATTED_VALUE"
//   "UNFORMATTED_VALUE"
//   "FORMULA"
func (c *SpreadsheetsValuesGetCall) ValueRenderOption(valueRenderOption string) *SpreadsheetsValuesGetCall {
	c.urlParams_.Set("valueRenderOption", valueRenderOption)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SpreadsheetsValuesGetCall) Fields(s ...googleapi.Field) *SpreadsheetsValuesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *SpreadsheetsValuesGetCall) IfNoneMatch(entityTag string) *SpreadsheetsValuesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SpreadsheetsValuesGetCall) Context(ctx context.Context) *SpreadsheetsValuesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SpreadsheetsValuesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SpreadsheetsValuesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/spreadsheets/{spreadsheetId}/values/{range}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"spreadsheetId": c.spreadsheetId,
		"range":         c.range_,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sheets.spreadsheets.values.get" call.
// Exactly one of *ValueRange or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *ValueRange.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SpreadsheetsValuesGetCall) Do(opts ...googleapi.CallOption) (*ValueRange, error) {
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
	ret := &ValueRange{
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
	//   "description": "Returns a range of values from a spreadsheet.\nThe caller must specify the spreadsheet ID and a range.",
	//   "flatPath": "v4/spreadsheets/{spreadsheetId}/values/{range}",
	//   "httpMethod": "GET",
	//   "id": "sheets.spreadsheets.values.get",
	//   "parameterOrder": [
	//     "spreadsheetId",
	//     "range"
	//   ],
	//   "parameters": {
	//     "dateTimeRenderOption": {
	//       "description": "How dates, times, and durations should be represented in the output.\nThis is ignored if value_render_option is\nFORMATTED_VALUE.\nThe default dateTime render option is [DateTimeRenderOption.SERIAL_NUMBER].",
	//       "enum": [
	//         "SERIAL_NUMBER",
	//         "FORMATTED_STRING"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "majorDimension": {
	//       "description": "The major dimension that results should use.\n\nFor example, if the spreadsheet data is: `A1=1,B1=2,A2=3,B2=4`,\nthen requesting `range=A1:B2,majorDimension=ROWS` will return\n`[[1,2],[3,4]]`,\nwhereas requesting `range=A1:B2,majorDimension=COLUMNS` will return\n`[[1,3],[2,4]]`.",
	//       "enum": [
	//         "DIMENSION_UNSPECIFIED",
	//         "ROWS",
	//         "COLUMNS"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "range": {
	//       "description": "The A1 notation of the values to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "spreadsheetId": {
	//       "description": "The ID of the spreadsheet to retrieve data from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "valueRenderOption": {
	//       "description": "How values should be represented in the output.\nThe default render option is ValueRenderOption.FORMATTED_VALUE.",
	//       "enum": [
	//         "FORMATTED_VALUE",
	//         "UNFORMATTED_VALUE",
	//         "FORMULA"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/spreadsheets/{spreadsheetId}/values/{range}",
	//   "response": {
	//     "$ref": "ValueRange"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/drive.readonly",
	//     "https://www.googleapis.com/auth/spreadsheets",
	//     "https://www.googleapis.com/auth/spreadsheets.readonly"
	//   ]
	// }

}

// method id "sheets.spreadsheets.values.update":

type SpreadsheetsValuesUpdateCall struct {
	s             *Service
	spreadsheetId string
	range_        string
	valuerange    *ValueRange
	urlParams_    gensupport.URLParams
	ctx_          context.Context
	header_       http.Header
}

// Update: Sets values in a range of a spreadsheet.
// The caller must specify the spreadsheet ID, range, and
// a valueInputOption.
func (r *SpreadsheetsValuesService) Update(spreadsheetId string, range_ string, valuerange *ValueRange) *SpreadsheetsValuesUpdateCall {
	c := &SpreadsheetsValuesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.spreadsheetId = spreadsheetId
	c.range_ = range_
	c.valuerange = valuerange
	return c
}

// IncludeValuesInResponse sets the optional parameter
// "includeValuesInResponse": Determines if the update response should
// include the values
// of the cells that were updated. By default, responses
// do not include the updated values.
// If the range to write was larger than than the range actually
// written,
// the response will include all values in the requested range
// (excluding
// trailing empty rows and columns).
func (c *SpreadsheetsValuesUpdateCall) IncludeValuesInResponse(includeValuesInResponse bool) *SpreadsheetsValuesUpdateCall {
	c.urlParams_.Set("includeValuesInResponse", fmt.Sprint(includeValuesInResponse))
	return c
}

// ResponseDateTimeRenderOption sets the optional parameter
// "responseDateTimeRenderOption": Determines how dates, times, and
// durations in the response should be
// rendered. This is ignored if response_value_render_option
// is
// FORMATTED_VALUE.
// The default dateTime render option is
// [DateTimeRenderOption.SERIAL_NUMBER].
//
// Possible values:
//   "SERIAL_NUMBER"
//   "FORMATTED_STRING"
func (c *SpreadsheetsValuesUpdateCall) ResponseDateTimeRenderOption(responseDateTimeRenderOption string) *SpreadsheetsValuesUpdateCall {
	c.urlParams_.Set("responseDateTimeRenderOption", responseDateTimeRenderOption)
	return c
}

// ResponseValueRenderOption sets the optional parameter
// "responseValueRenderOption": Determines how values in the response
// should be rendered.
// The default render option is ValueRenderOption.FORMATTED_VALUE.
//
// Possible values:
//   "FORMATTED_VALUE"
//   "UNFORMATTED_VALUE"
//   "FORMULA"
func (c *SpreadsheetsValuesUpdateCall) ResponseValueRenderOption(responseValueRenderOption string) *SpreadsheetsValuesUpdateCall {
	c.urlParams_.Set("responseValueRenderOption", responseValueRenderOption)
	return c
}

// ValueInputOption sets the optional parameter "valueInputOption": How
// the input data should be interpreted.
//
// Possible values:
//   "INPUT_VALUE_OPTION_UNSPECIFIED"
//   "RAW"
//   "USER_ENTERED"
func (c *SpreadsheetsValuesUpdateCall) ValueInputOption(valueInputOption string) *SpreadsheetsValuesUpdateCall {
	c.urlParams_.Set("valueInputOption", valueInputOption)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SpreadsheetsValuesUpdateCall) Fields(s ...googleapi.Field) *SpreadsheetsValuesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SpreadsheetsValuesUpdateCall) Context(ctx context.Context) *SpreadsheetsValuesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SpreadsheetsValuesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SpreadsheetsValuesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.valuerange)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v4/spreadsheets/{spreadsheetId}/values/{range}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"spreadsheetId": c.spreadsheetId,
		"range":         c.range_,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sheets.spreadsheets.values.update" call.
// Exactly one of *UpdateValuesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *UpdateValuesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SpreadsheetsValuesUpdateCall) Do(opts ...googleapi.CallOption) (*UpdateValuesResponse, error) {
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
	ret := &UpdateValuesResponse{
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
	//   "description": "Sets values in a range of a spreadsheet.\nThe caller must specify the spreadsheet ID, range, and\na valueInputOption.",
	//   "flatPath": "v4/spreadsheets/{spreadsheetId}/values/{range}",
	//   "httpMethod": "PUT",
	//   "id": "sheets.spreadsheets.values.update",
	//   "parameterOrder": [
	//     "spreadsheetId",
	//     "range"
	//   ],
	//   "parameters": {
	//     "includeValuesInResponse": {
	//       "description": "Determines if the update response should include the values\nof the cells that were updated. By default, responses\ndo not include the updated values.\nIf the range to write was larger than than the range actually written,\nthe response will include all values in the requested range (excluding\ntrailing empty rows and columns).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "range": {
	//       "description": "The A1 notation of the values to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "responseDateTimeRenderOption": {
	//       "description": "Determines how dates, times, and durations in the response should be\nrendered. This is ignored if response_value_render_option is\nFORMATTED_VALUE.\nThe default dateTime render option is [DateTimeRenderOption.SERIAL_NUMBER].",
	//       "enum": [
	//         "SERIAL_NUMBER",
	//         "FORMATTED_STRING"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "responseValueRenderOption": {
	//       "description": "Determines how values in the response should be rendered.\nThe default render option is ValueRenderOption.FORMATTED_VALUE.",
	//       "enum": [
	//         "FORMATTED_VALUE",
	//         "UNFORMATTED_VALUE",
	//         "FORMULA"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "spreadsheetId": {
	//       "description": "The ID of the spreadsheet to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "valueInputOption": {
	//       "description": "How the input data should be interpreted.",
	//       "enum": [
	//         "INPUT_VALUE_OPTION_UNSPECIFIED",
	//         "RAW",
	//         "USER_ENTERED"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v4/spreadsheets/{spreadsheetId}/values/{range}",
	//   "request": {
	//     "$ref": "ValueRange"
	//   },
	//   "response": {
	//     "$ref": "UpdateValuesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/drive",
	//     "https://www.googleapis.com/auth/drive.file",
	//     "https://www.googleapis.com/auth/spreadsheets"
	//   ]
	// }

}
