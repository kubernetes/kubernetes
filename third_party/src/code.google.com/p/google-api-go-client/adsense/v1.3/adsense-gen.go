// Package adsense provides access to the AdSense Management API.
//
// See https://developers.google.com/adsense/management/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/adsense/v1.3"
//   ...
//   adsenseService, err := adsense.New(oauthHttpClient)
package adsense

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

const apiId = "adsense:v1.3"
const apiName = "adsense"
const apiVersion = "v1.3"
const basePath = "https://www.googleapis.com/adsense/v1.3/"

// OAuth2 scopes used by this API.
const (
	// View and manage your AdSense data
	AdsenseScope = "https://www.googleapis.com/auth/adsense"

	// View your AdSense data
	AdsenseReadonlyScope = "https://www.googleapis.com/auth/adsense.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Accounts = NewAccountsService(s)
	s.Adclients = NewAdclientsService(s)
	s.Adunits = NewAdunitsService(s)
	s.Alerts = NewAlertsService(s)
	s.Customchannels = NewCustomchannelsService(s)
	s.Metadata = NewMetadataService(s)
	s.Reports = NewReportsService(s)
	s.Savedadstyles = NewSavedadstylesService(s)
	s.Urlchannels = NewUrlchannelsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Accounts *AccountsService

	Adclients *AdclientsService

	Adunits *AdunitsService

	Alerts *AlertsService

	Customchannels *CustomchannelsService

	Metadata *MetadataService

	Reports *ReportsService

	Savedadstyles *SavedadstylesService

	Urlchannels *UrlchannelsService
}

func NewAccountsService(s *Service) *AccountsService {
	rs := &AccountsService{s: s}
	rs.Adclients = NewAccountsAdclientsService(s)
	rs.Adunits = NewAccountsAdunitsService(s)
	rs.Alerts = NewAccountsAlertsService(s)
	rs.Customchannels = NewAccountsCustomchannelsService(s)
	rs.Reports = NewAccountsReportsService(s)
	rs.Savedadstyles = NewAccountsSavedadstylesService(s)
	rs.Urlchannels = NewAccountsUrlchannelsService(s)
	return rs
}

type AccountsService struct {
	s *Service

	Adclients *AccountsAdclientsService

	Adunits *AccountsAdunitsService

	Alerts *AccountsAlertsService

	Customchannels *AccountsCustomchannelsService

	Reports *AccountsReportsService

	Savedadstyles *AccountsSavedadstylesService

	Urlchannels *AccountsUrlchannelsService
}

func NewAccountsAdclientsService(s *Service) *AccountsAdclientsService {
	rs := &AccountsAdclientsService{s: s}
	return rs
}

type AccountsAdclientsService struct {
	s *Service
}

func NewAccountsAdunitsService(s *Service) *AccountsAdunitsService {
	rs := &AccountsAdunitsService{s: s}
	rs.Customchannels = NewAccountsAdunitsCustomchannelsService(s)
	return rs
}

type AccountsAdunitsService struct {
	s *Service

	Customchannels *AccountsAdunitsCustomchannelsService
}

func NewAccountsAdunitsCustomchannelsService(s *Service) *AccountsAdunitsCustomchannelsService {
	rs := &AccountsAdunitsCustomchannelsService{s: s}
	return rs
}

type AccountsAdunitsCustomchannelsService struct {
	s *Service
}

func NewAccountsAlertsService(s *Service) *AccountsAlertsService {
	rs := &AccountsAlertsService{s: s}
	return rs
}

type AccountsAlertsService struct {
	s *Service
}

func NewAccountsCustomchannelsService(s *Service) *AccountsCustomchannelsService {
	rs := &AccountsCustomchannelsService{s: s}
	rs.Adunits = NewAccountsCustomchannelsAdunitsService(s)
	return rs
}

type AccountsCustomchannelsService struct {
	s *Service

	Adunits *AccountsCustomchannelsAdunitsService
}

func NewAccountsCustomchannelsAdunitsService(s *Service) *AccountsCustomchannelsAdunitsService {
	rs := &AccountsCustomchannelsAdunitsService{s: s}
	return rs
}

type AccountsCustomchannelsAdunitsService struct {
	s *Service
}

func NewAccountsReportsService(s *Service) *AccountsReportsService {
	rs := &AccountsReportsService{s: s}
	rs.Saved = NewAccountsReportsSavedService(s)
	return rs
}

type AccountsReportsService struct {
	s *Service

	Saved *AccountsReportsSavedService
}

func NewAccountsReportsSavedService(s *Service) *AccountsReportsSavedService {
	rs := &AccountsReportsSavedService{s: s}
	return rs
}

type AccountsReportsSavedService struct {
	s *Service
}

func NewAccountsSavedadstylesService(s *Service) *AccountsSavedadstylesService {
	rs := &AccountsSavedadstylesService{s: s}
	return rs
}

type AccountsSavedadstylesService struct {
	s *Service
}

func NewAccountsUrlchannelsService(s *Service) *AccountsUrlchannelsService {
	rs := &AccountsUrlchannelsService{s: s}
	return rs
}

type AccountsUrlchannelsService struct {
	s *Service
}

func NewAdclientsService(s *Service) *AdclientsService {
	rs := &AdclientsService{s: s}
	return rs
}

type AdclientsService struct {
	s *Service
}

func NewAdunitsService(s *Service) *AdunitsService {
	rs := &AdunitsService{s: s}
	rs.Customchannels = NewAdunitsCustomchannelsService(s)
	return rs
}

type AdunitsService struct {
	s *Service

	Customchannels *AdunitsCustomchannelsService
}

func NewAdunitsCustomchannelsService(s *Service) *AdunitsCustomchannelsService {
	rs := &AdunitsCustomchannelsService{s: s}
	return rs
}

type AdunitsCustomchannelsService struct {
	s *Service
}

func NewAlertsService(s *Service) *AlertsService {
	rs := &AlertsService{s: s}
	return rs
}

type AlertsService struct {
	s *Service
}

func NewCustomchannelsService(s *Service) *CustomchannelsService {
	rs := &CustomchannelsService{s: s}
	rs.Adunits = NewCustomchannelsAdunitsService(s)
	return rs
}

type CustomchannelsService struct {
	s *Service

	Adunits *CustomchannelsAdunitsService
}

func NewCustomchannelsAdunitsService(s *Service) *CustomchannelsAdunitsService {
	rs := &CustomchannelsAdunitsService{s: s}
	return rs
}

type CustomchannelsAdunitsService struct {
	s *Service
}

func NewMetadataService(s *Service) *MetadataService {
	rs := &MetadataService{s: s}
	rs.Dimensions = NewMetadataDimensionsService(s)
	rs.Metrics = NewMetadataMetricsService(s)
	return rs
}

type MetadataService struct {
	s *Service

	Dimensions *MetadataDimensionsService

	Metrics *MetadataMetricsService
}

func NewMetadataDimensionsService(s *Service) *MetadataDimensionsService {
	rs := &MetadataDimensionsService{s: s}
	return rs
}

type MetadataDimensionsService struct {
	s *Service
}

func NewMetadataMetricsService(s *Service) *MetadataMetricsService {
	rs := &MetadataMetricsService{s: s}
	return rs
}

type MetadataMetricsService struct {
	s *Service
}

func NewReportsService(s *Service) *ReportsService {
	rs := &ReportsService{s: s}
	rs.Saved = NewReportsSavedService(s)
	return rs
}

type ReportsService struct {
	s *Service

	Saved *ReportsSavedService
}

func NewReportsSavedService(s *Service) *ReportsSavedService {
	rs := &ReportsSavedService{s: s}
	return rs
}

type ReportsSavedService struct {
	s *Service
}

func NewSavedadstylesService(s *Service) *SavedadstylesService {
	rs := &SavedadstylesService{s: s}
	return rs
}

type SavedadstylesService struct {
	s *Service
}

func NewUrlchannelsService(s *Service) *UrlchannelsService {
	rs := &UrlchannelsService{s: s}
	return rs
}

type UrlchannelsService struct {
	s *Service
}

type Account struct {
	// Id: Unique identifier of this account.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#account.
	Kind string `json:"kind,omitempty"`

	// Name: Name of this account.
	Name string `json:"name,omitempty"`

	// Premium: Whether this account is premium.
	Premium bool `json:"premium,omitempty"`

	// SubAccounts: Sub accounts of the this account.
	SubAccounts []*Account `json:"subAccounts,omitempty"`
}

type Accounts struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The accounts returned in this list response.
	Items []*Account `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adsense#accounts.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through accounts. To
	// retrieve the next page of results, set the next request's "pageToken"
	// value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type AdClient struct {
	// ArcOptIn: Whether this ad client is opted in to ARC.
	ArcOptIn bool `json:"arcOptIn,omitempty"`

	// Id: Unique identifier of this ad client.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#adClient.
	Kind string `json:"kind,omitempty"`

	// ProductCode: This ad client's product code, which corresponds to the
	// PRODUCT_CODE report dimension.
	ProductCode string `json:"productCode,omitempty"`

	// SupportsReporting: Whether this ad client supports being reported on.
	SupportsReporting bool `json:"supportsReporting,omitempty"`
}

type AdClients struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The ad clients returned in this list response.
	Items []*AdClient `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adsense#adClients.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through ad clients. To
	// retrieve the next page of results, set the next request's "pageToken"
	// value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type AdCode struct {
	// AdCode: The ad code snippet.
	AdCode string `json:"adCode,omitempty"`

	// Kind: Kind this is, in this case adsense#adCode.
	Kind string `json:"kind,omitempty"`
}

type AdStyle struct {
	// Colors: The colors which are included in the style. These are
	// represented as six hexadecimal characters, similar to HTML color
	// codes, but without the leading hash.
	Colors *AdStyleColors `json:"colors,omitempty"`

	// Corners: The style of the corners in the ad.
	Corners string `json:"corners,omitempty"`

	// Font: The font which is included in the style.
	Font *AdStyleFont `json:"font,omitempty"`

	// Kind: Kind this is, in this case adsense#adStyle.
	Kind string `json:"kind,omitempty"`
}

type AdStyleColors struct {
	// Background: The color of the ad background.
	Background string `json:"background,omitempty"`

	// Border: The color of the ad border.
	Border string `json:"border,omitempty"`

	// Text: The color of the ad text.
	Text string `json:"text,omitempty"`

	// Title: The color of the ad title.
	Title string `json:"title,omitempty"`

	// Url: The color of the ad url.
	Url string `json:"url,omitempty"`
}

type AdStyleFont struct {
	// Family: The family of the font.
	Family string `json:"family,omitempty"`

	// Size: The size of the font.
	Size string `json:"size,omitempty"`
}

type AdUnit struct {
	// Code: Identity code of this ad unit, not necessarily unique across ad
	// clients.
	Code string `json:"code,omitempty"`

	// ContentAdsSettings: Settings specific to content ads (AFC) and
	// highend mobile content ads (AFMC).
	ContentAdsSettings *AdUnitContentAdsSettings `json:"contentAdsSettings,omitempty"`

	// CustomStyle: Custom style information specific to this ad unit.
	CustomStyle *AdStyle `json:"customStyle,omitempty"`

	// FeedAdsSettings: Settings specific to feed ads (AFF).
	FeedAdsSettings *AdUnitFeedAdsSettings `json:"feedAdsSettings,omitempty"`

	// Id: Unique identifier of this ad unit. This should be considered an
	// opaque identifier; it is not safe to rely on it being in any
	// particular format.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#adUnit.
	Kind string `json:"kind,omitempty"`

	// MobileContentAdsSettings: Settings specific to WAP mobile content ads
	// (AFMC).
	MobileContentAdsSettings *AdUnitMobileContentAdsSettings `json:"mobileContentAdsSettings,omitempty"`

	// Name: Name of this ad unit.
	Name string `json:"name,omitempty"`

	// SavedStyleId: ID of the saved ad style which holds this ad unit's
	// style information.
	SavedStyleId string `json:"savedStyleId,omitempty"`

	// Status: Status of this ad unit. Possible values are:
	// NEW: Indicates
	// that the ad unit was created within the last seven days and does not
	// yet have any activity associated with it.
	//
	// ACTIVE: Indicates that
	// there has been activity on this ad unit in the last seven
	// days.
	//
	// INACTIVE: Indicates that there has been no activity on this ad
	// unit in the last seven days.
	Status string `json:"status,omitempty"`
}

type AdUnitContentAdsSettings struct {
	// BackupOption: The backup option to be used in instances where no ad
	// is available.
	BackupOption *AdUnitContentAdsSettingsBackupOption `json:"backupOption,omitempty"`

	// Size: Size of this ad unit.
	Size string `json:"size,omitempty"`

	// Type: Type of this ad unit.
	Type string `json:"type,omitempty"`
}

type AdUnitContentAdsSettingsBackupOption struct {
	// Color: Color to use when type is set to COLOR.
	Color string `json:"color,omitempty"`

	// Type: Type of the backup option. Possible values are BLANK, COLOR and
	// URL.
	Type string `json:"type,omitempty"`

	// Url: URL to use when type is set to URL.
	Url string `json:"url,omitempty"`
}

type AdUnitFeedAdsSettings struct {
	// AdPosition: The position of the ads relative to the feed entries.
	AdPosition string `json:"adPosition,omitempty"`

	// Frequency: The frequency at which ads should appear in the feed (i.e.
	// every N entries).
	Frequency int64 `json:"frequency,omitempty"`

	// MinimumWordCount: The minimum length an entry should be in order to
	// have attached ads.
	MinimumWordCount int64 `json:"minimumWordCount,omitempty"`

	// Type: The type of ads which should appear.
	Type string `json:"type,omitempty"`
}

type AdUnitMobileContentAdsSettings struct {
	// MarkupLanguage: The markup language to use for this ad unit.
	MarkupLanguage string `json:"markupLanguage,omitempty"`

	// ScriptingLanguage: The scripting language to use for this ad unit.
	ScriptingLanguage string `json:"scriptingLanguage,omitempty"`

	// Size: Size of this ad unit.
	Size string `json:"size,omitempty"`

	// Type: Type of this ad unit.
	Type string `json:"type,omitempty"`
}

type AdUnits struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The ad units returned in this list response.
	Items []*AdUnit `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adsense#adUnits.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through ad units. To
	// retrieve the next page of results, set the next request's "pageToken"
	// value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type AdsenseReportsGenerateResponse struct {
	// Averages: The averages of the report. This is the same length as any
	// other row in the report; cells corresponding to dimension columns are
	// empty.
	Averages []string `json:"averages,omitempty"`

	// Headers: The header information of the columns requested in the
	// report. This is a list of headers; one for each dimension in the
	// request, followed by one for each metric in the request.
	Headers []*AdsenseReportsGenerateResponseHeaders `json:"headers,omitempty"`

	// Kind: Kind this is, in this case adsense#report.
	Kind string `json:"kind,omitempty"`

	// Rows: The output rows of the report. Each row is a list of cells; one
	// for each dimension in the request, followed by one for each metric in
	// the request. The dimension cells contain strings, and the metric
	// cells contain numbers.
	Rows [][]string `json:"rows,omitempty"`

	// TotalMatchedRows: The total number of rows matched by the report
	// request. Fewer rows may be returned in the response due to being
	// limited by the row count requested or the report row limit.
	TotalMatchedRows int64 `json:"totalMatchedRows,omitempty,string"`

	// Totals: The totals of the report. This is the same length as any
	// other row in the report; cells corresponding to dimension columns are
	// empty.
	Totals []string `json:"totals,omitempty"`

	// Warnings: Any warnings associated with generation of the report.
	Warnings []string `json:"warnings,omitempty"`
}

type AdsenseReportsGenerateResponseHeaders struct {
	// Currency: The currency of this column. Only present if the header
	// type is METRIC_CURRENCY.
	Currency string `json:"currency,omitempty"`

	// Name: The name of the header.
	Name string `json:"name,omitempty"`

	// Type: The type of the header; one of DIMENSION, METRIC_TALLY,
	// METRIC_RATIO, or METRIC_CURRENCY.
	Type string `json:"type,omitempty"`
}

type Alert struct {
	// Id: Unique identifier of this alert. This should be considered an
	// opaque identifier; it is not safe to rely on it being in any
	// particular format.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#alert.
	Kind string `json:"kind,omitempty"`

	// Message: The localized alert message.
	Message string `json:"message,omitempty"`

	// Severity: Severity of this alert. Possible values: INFO, WARNING,
	// SEVERE.
	Severity string `json:"severity,omitempty"`

	// Type: Type of this alert. Possible values: SELF_HOLD,
	// MIGRATED_TO_BILLING3, ADDRESS_PIN_VERIFICATION,
	// PHONE_PIN_VERIFICATION, CORPORATE_ENTITY, GRAYLISTED_PUBLISHER,
	// API_HOLD.
	Type string `json:"type,omitempty"`
}

type Alerts struct {
	// Items: The alerts returned in this list response.
	Items []*Alert `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adsense#alerts.
	Kind string `json:"kind,omitempty"`
}

type CustomChannel struct {
	// Code: Code of this custom channel, not necessarily unique across ad
	// clients.
	Code string `json:"code,omitempty"`

	// Id: Unique identifier of this custom channel. This should be
	// considered an opaque identifier; it is not safe to rely on it being
	// in any particular format.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#customChannel.
	Kind string `json:"kind,omitempty"`

	// Name: Name of this custom channel.
	Name string `json:"name,omitempty"`

	// TargetingInfo: The targeting information of this custom channel, if
	// activated.
	TargetingInfo *CustomChannelTargetingInfo `json:"targetingInfo,omitempty"`
}

type CustomChannelTargetingInfo struct {
	// AdsAppearOn: The name used to describe this channel externally.
	AdsAppearOn string `json:"adsAppearOn,omitempty"`

	// Description: The external description of the channel.
	Description string `json:"description,omitempty"`

	// Location: The locations in which ads appear. (Only valid for content
	// and mobile content ads). Acceptable values for content ads are:
	// TOP_LEFT, TOP_CENTER, TOP_RIGHT, MIDDLE_LEFT, MIDDLE_CENTER,
	// MIDDLE_RIGHT, BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT,
	// MULTIPLE_LOCATIONS. Acceptable values for mobile content ads are:
	// TOP, MIDDLE, BOTTOM, MULTIPLE_LOCATIONS.
	Location string `json:"location,omitempty"`

	// SiteLanguage: The language of the sites ads will be displayed on.
	SiteLanguage string `json:"siteLanguage,omitempty"`
}

type CustomChannels struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The custom channels returned in this list response.
	Items []*CustomChannel `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adsense#customChannels.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through custom
	// channels. To retrieve the next page of results, set the next
	// request's "pageToken" value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type Metadata struct {
	Items []*ReportingMetadataEntry `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adsense#metadata.
	Kind string `json:"kind,omitempty"`
}

type ReportingMetadataEntry struct {
	// CompatibleDimensions: For metrics this is a list of dimension IDs
	// which the metric is compatible with, for dimensions it is a list of
	// compatibility groups the dimension belongs to.
	CompatibleDimensions []string `json:"compatibleDimensions,omitempty"`

	// CompatibleMetrics: The names of the metrics the dimension or metric
	// this reporting metadata entry describes is compatible with.
	CompatibleMetrics []string `json:"compatibleMetrics,omitempty"`

	// Id: Unique identifier of this reporting metadata entry, corresponding
	// to the name of the appropriate dimension or metric.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case
	// adsense#reportingMetadataEntry.
	Kind string `json:"kind,omitempty"`

	// RequiredDimensions: The names of the dimensions which the dimension
	// or metric this reporting metadata entry describes requires to also be
	// present in order for the report to be valid. Omitting these will not
	// cause an error or warning, but may result in data which cannot be
	// correctly interpreted.
	RequiredDimensions []string `json:"requiredDimensions,omitempty"`

	// RequiredMetrics: The names of the metrics which the dimension or
	// metric this reporting metadata entry describes requires to also be
	// present in order for the report to be valid. Omitting these will not
	// cause an error or warning, but may result in data which cannot be
	// correctly interpreted.
	RequiredMetrics []string `json:"requiredMetrics,omitempty"`

	// SupportedProducts: The codes of the projects supported by the
	// dimension or metric this reporting metadata entry describes.
	SupportedProducts []string `json:"supportedProducts,omitempty"`
}

type SavedAdStyle struct {
	// AdStyle: The AdStyle itself.
	AdStyle *AdStyle `json:"adStyle,omitempty"`

	// Id: Unique identifier of this saved ad style. This should be
	// considered an opaque identifier; it is not safe to rely on it being
	// in any particular format.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#savedAdStyle.
	Kind string `json:"kind,omitempty"`

	// Name: The user selected name of this SavedAdStyle.
	Name string `json:"name,omitempty"`
}

type SavedAdStyles struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The saved ad styles returned in this list response.
	Items []*SavedAdStyle `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adsense#savedAdStyles.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through ad units. To
	// retrieve the next page of results, set the next request's "pageToken"
	// value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type SavedReport struct {
	// Id: Unique identifier of this saved report.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#savedReport.
	Kind string `json:"kind,omitempty"`

	// Name: This saved report's name.
	Name string `json:"name,omitempty"`
}

type SavedReports struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The saved reports returned in this list response.
	Items []*SavedReport `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adsense#savedReports.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through saved reports.
	// To retrieve the next page of results, set the next request's
	// "pageToken" value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type UrlChannel struct {
	// Id: Unique identifier of this URL channel. This should be considered
	// an opaque identifier; it is not safe to rely on it being in any
	// particular format.
	Id string `json:"id,omitempty"`

	// Kind: Kind of resource this is, in this case adsense#urlChannel.
	Kind string `json:"kind,omitempty"`

	// UrlPattern: URL Pattern of this URL channel. Does not include
	// "http://" or "https://". Example: www.example.com/home
	UrlPattern string `json:"urlPattern,omitempty"`
}

type UrlChannels struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The URL channels returned in this list response.
	Items []*UrlChannel `json:"items,omitempty"`

	// Kind: Kind of list this is, in this case adsense#urlChannels.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Continuation token used to page through URL channels.
	// To retrieve the next page of results, set the next request's
	// "pageToken" value to this.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

// method id "adsense.accounts.get":

type AccountsGetCall struct {
	s         *Service
	accountId string
	opt_      map[string]interface{}
}

// Get: Get information about the selected AdSense account.
func (r *AccountsService) Get(accountId string) *AccountsGetCall {
	c := &AccountsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	return c
}

// Tree sets the optional parameter "tree": Whether the tree of sub
// accounts should be returned.
func (c *AccountsGetCall) Tree(tree bool) *AccountsGetCall {
	c.opt_["tree"] = tree
	return c
}

func (c *AccountsGetCall) Do() (*Account, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["tree"]; ok {
		params.Set("tree", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
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
	ret := new(Account)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get information about the selected AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.get",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to get information about.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tree": {
	//       "description": "Whether the tree of sub accounts should be returned.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "accounts/{accountId}",
	//   "response": {
	//     "$ref": "Account"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.list":

type AccountsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List all accounts available to this AdSense account.
func (r *AccountsService) List() *AccountsListCall {
	c := &AccountsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of accounts to include in the response, used for paging.
func (c *AccountsListCall) MaxResults(maxResults int64) *AccountsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through accounts. To retrieve the next page, set
// this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AccountsListCall) PageToken(pageToken string) *AccountsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsListCall) Do() (*Accounts, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts")
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
	ret := new(Accounts)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all accounts available to this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of accounts to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through accounts. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts",
	//   "response": {
	//     "$ref": "Accounts"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.adclients.list":

type AccountsAdclientsListCall struct {
	s         *Service
	accountId string
	opt_      map[string]interface{}
}

// List: List all ad clients in the specified account.
func (r *AccountsAdclientsService) List(accountId string) *AccountsAdclientsListCall {
	c := &AccountsAdclientsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad clients to include in the response, used for paging.
func (c *AccountsAdclientsListCall) MaxResults(maxResults int64) *AccountsAdclientsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad clients. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AccountsAdclientsListCall) PageToken(pageToken string) *AccountsAdclientsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsAdclientsListCall) Do() (*AdClients, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/adclients")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
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
	ret := new(AdClients)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad clients in the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.adclients.list",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account for which to list ad clients.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of ad clients to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through ad clients. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/adclients",
	//   "response": {
	//     "$ref": "AdClients"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.adunits.get":

type AccountsAdunitsGetCall struct {
	s          *Service
	accountId  string
	adClientId string
	adUnitId   string
	opt_       map[string]interface{}
}

// Get: Gets the specified ad unit in the specified ad client for the
// specified account.
func (r *AccountsAdunitsService) Get(accountId string, adClientId string, adUnitId string) *AccountsAdunitsGetCall {
	c := &AccountsAdunitsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	c.adUnitId = adUnitId
	return c
}

func (c *AccountsAdunitsGetCall) Do() (*AdUnit, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/adclients/{adClientId}/adunits/{adUnitId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adUnitId}", url.QueryEscape(c.adUnitId), 1)
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
	ret := new(AdUnit)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the specified ad unit in the specified ad client for the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.adunits.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId",
	//     "adUnitId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adClientId": {
	//       "description": "Ad client for which to get the ad unit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adUnitId": {
	//       "description": "Ad unit to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/adclients/{adClientId}/adunits/{adUnitId}",
	//   "response": {
	//     "$ref": "AdUnit"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.adunits.getAdCode":

type AccountsAdunitsGetAdCodeCall struct {
	s          *Service
	accountId  string
	adClientId string
	adUnitId   string
	opt_       map[string]interface{}
}

// GetAdCode: Get ad code for the specified ad unit.
func (r *AccountsAdunitsService) GetAdCode(accountId string, adClientId string, adUnitId string) *AccountsAdunitsGetAdCodeCall {
	c := &AccountsAdunitsGetAdCodeCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	c.adUnitId = adUnitId
	return c
}

func (c *AccountsAdunitsGetAdCodeCall) Do() (*AdCode, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/adclients/{adClientId}/adunits/{adUnitId}/adcode")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adUnitId}", url.QueryEscape(c.adUnitId), 1)
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
	ret := new(AdCode)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get ad code for the specified ad unit.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.adunits.getAdCode",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId",
	//     "adUnitId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account which contains the ad client.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adClientId": {
	//       "description": "Ad client with contains the ad unit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adUnitId": {
	//       "description": "Ad unit to get the code for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/adclients/{adClientId}/adunits/{adUnitId}/adcode",
	//   "response": {
	//     "$ref": "AdCode"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.adunits.list":

type AccountsAdunitsListCall struct {
	s          *Service
	accountId  string
	adClientId string
	opt_       map[string]interface{}
}

// List: List all ad units in the specified ad client for the specified
// account.
func (r *AccountsAdunitsService) List(accountId string, adClientId string) *AccountsAdunitsListCall {
	c := &AccountsAdunitsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	return c
}

// IncludeInactive sets the optional parameter "includeInactive":
// Whether to include inactive ad units. Default: true.
func (c *AccountsAdunitsListCall) IncludeInactive(includeInactive bool) *AccountsAdunitsListCall {
	c.opt_["includeInactive"] = includeInactive
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad units to include in the response, used for paging.
func (c *AccountsAdunitsListCall) MaxResults(maxResults int64) *AccountsAdunitsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad units. To retrieve the next page, set
// this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AccountsAdunitsListCall) PageToken(pageToken string) *AccountsAdunitsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsAdunitsListCall) Do() (*AdUnits, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["includeInactive"]; ok {
		params.Set("includeInactive", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/adclients/{adClientId}/adunits")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
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
	ret := new(AdUnits)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad units in the specified ad client for the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.adunits.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adClientId": {
	//       "description": "Ad client for which to list ad units.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "includeInactive": {
	//       "description": "Whether to include inactive ad units. Default: true.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of ad units to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through ad units. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/adclients/{adClientId}/adunits",
	//   "response": {
	//     "$ref": "AdUnits"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.adunits.customchannels.list":

type AccountsAdunitsCustomchannelsListCall struct {
	s          *Service
	accountId  string
	adClientId string
	adUnitId   string
	opt_       map[string]interface{}
}

// List: List all custom channels which the specified ad unit belongs
// to.
func (r *AccountsAdunitsCustomchannelsService) List(accountId string, adClientId string, adUnitId string) *AccountsAdunitsCustomchannelsListCall {
	c := &AccountsAdunitsCustomchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	c.adUnitId = adUnitId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of custom channels to include in the response, used for
// paging.
func (c *AccountsAdunitsCustomchannelsListCall) MaxResults(maxResults int64) *AccountsAdunitsCustomchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through custom channels. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *AccountsAdunitsCustomchannelsListCall) PageToken(pageToken string) *AccountsAdunitsCustomchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsAdunitsCustomchannelsListCall) Do() (*CustomChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/adclients/{adClientId}/adunits/{adUnitId}/customchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adUnitId}", url.QueryEscape(c.adUnitId), 1)
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
	ret := new(CustomChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all custom channels which the specified ad unit belongs to.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.adunits.customchannels.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId",
	//     "adUnitId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adClientId": {
	//       "description": "Ad client which contains the ad unit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adUnitId": {
	//       "description": "Ad unit for which to list custom channels.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of custom channels to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through custom channels. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/adclients/{adClientId}/adunits/{adUnitId}/customchannels",
	//   "response": {
	//     "$ref": "CustomChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.alerts.list":

type AccountsAlertsListCall struct {
	s         *Service
	accountId string
	opt_      map[string]interface{}
}

// List: List the alerts for the specified AdSense account.
func (r *AccountsAlertsService) List(accountId string) *AccountsAlertsListCall {
	c := &AccountsAlertsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	return c
}

// Locale sets the optional parameter "locale": The locale to use for
// translating alert messages. The account locale will be used if this
// is not supplied. The AdSense default (English) will be used if the
// supplied locale is invalid or unsupported.
func (c *AccountsAlertsListCall) Locale(locale string) *AccountsAlertsListCall {
	c.opt_["locale"] = locale
	return c
}

func (c *AccountsAlertsListCall) Do() (*Alerts, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/alerts")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
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
	ret := new(Alerts)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List the alerts for the specified AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.alerts.list",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account for which to retrieve the alerts.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "The locale to use for translating alert messages. The account locale will be used if this is not supplied. The AdSense default (English) will be used if the supplied locale is invalid or unsupported.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/alerts",
	//   "response": {
	//     "$ref": "Alerts"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.customchannels.get":

type AccountsCustomchannelsGetCall struct {
	s               *Service
	accountId       string
	adClientId      string
	customChannelId string
	opt_            map[string]interface{}
}

// Get: Get the specified custom channel from the specified ad client
// for the specified account.
func (r *AccountsCustomchannelsService) Get(accountId string, adClientId string, customChannelId string) *AccountsCustomchannelsGetCall {
	c := &AccountsCustomchannelsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	c.customChannelId = customChannelId
	return c
}

func (c *AccountsCustomchannelsGetCall) Do() (*CustomChannel, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/adclients/{adClientId}/customchannels/{customChannelId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customChannelId}", url.QueryEscape(c.customChannelId), 1)
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
	ret := new(CustomChannel)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get the specified custom channel from the specified ad client for the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.customchannels.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId",
	//     "customChannelId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adClientId": {
	//       "description": "Ad client which contains the custom channel.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customChannelId": {
	//       "description": "Custom channel to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/adclients/{adClientId}/customchannels/{customChannelId}",
	//   "response": {
	//     "$ref": "CustomChannel"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.customchannels.list":

type AccountsCustomchannelsListCall struct {
	s          *Service
	accountId  string
	adClientId string
	opt_       map[string]interface{}
}

// List: List all custom channels in the specified ad client for the
// specified account.
func (r *AccountsCustomchannelsService) List(accountId string, adClientId string) *AccountsCustomchannelsListCall {
	c := &AccountsCustomchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of custom channels to include in the response, used for
// paging.
func (c *AccountsCustomchannelsListCall) MaxResults(maxResults int64) *AccountsCustomchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through custom channels. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *AccountsCustomchannelsListCall) PageToken(pageToken string) *AccountsCustomchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsCustomchannelsListCall) Do() (*CustomChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/adclients/{adClientId}/customchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
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
	ret := new(CustomChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all custom channels in the specified ad client for the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.customchannels.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adClientId": {
	//       "description": "Ad client for which to list custom channels.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of custom channels to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through custom channels. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/adclients/{adClientId}/customchannels",
	//   "response": {
	//     "$ref": "CustomChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.customchannels.adunits.list":

type AccountsCustomchannelsAdunitsListCall struct {
	s               *Service
	accountId       string
	adClientId      string
	customChannelId string
	opt_            map[string]interface{}
}

// List: List all ad units in the specified custom channel.
func (r *AccountsCustomchannelsAdunitsService) List(accountId string, adClientId string, customChannelId string) *AccountsCustomchannelsAdunitsListCall {
	c := &AccountsCustomchannelsAdunitsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	c.customChannelId = customChannelId
	return c
}

// IncludeInactive sets the optional parameter "includeInactive":
// Whether to include inactive ad units. Default: true.
func (c *AccountsCustomchannelsAdunitsListCall) IncludeInactive(includeInactive bool) *AccountsCustomchannelsAdunitsListCall {
	c.opt_["includeInactive"] = includeInactive
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad units to include in the response, used for paging.
func (c *AccountsCustomchannelsAdunitsListCall) MaxResults(maxResults int64) *AccountsCustomchannelsAdunitsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad units. To retrieve the next page, set
// this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AccountsCustomchannelsAdunitsListCall) PageToken(pageToken string) *AccountsCustomchannelsAdunitsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsCustomchannelsAdunitsListCall) Do() (*AdUnits, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["includeInactive"]; ok {
		params.Set("includeInactive", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/adclients/{adClientId}/customchannels/{customChannelId}/adunits")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customChannelId}", url.QueryEscape(c.customChannelId), 1)
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
	ret := new(AdUnits)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad units in the specified custom channel.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.customchannels.adunits.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId",
	//     "customChannelId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adClientId": {
	//       "description": "Ad client which contains the custom channel.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customChannelId": {
	//       "description": "Custom channel for which to list ad units.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "includeInactive": {
	//       "description": "Whether to include inactive ad units. Default: true.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of ad units to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through ad units. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/adclients/{adClientId}/customchannels/{customChannelId}/adunits",
	//   "response": {
	//     "$ref": "AdUnits"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.reports.generate":

type AccountsReportsGenerateCall struct {
	s         *Service
	accountId string
	startDate string
	endDate   string
	opt_      map[string]interface{}
}

// Generate: Generate an AdSense report based on the report request sent
// in the query parameters. Returns the result as JSON; to retrieve
// output in CSV format specify "alt=csv" as a query parameter.
func (r *AccountsReportsService) Generate(accountId string, startDate string, endDate string) *AccountsReportsGenerateCall {
	c := &AccountsReportsGenerateCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.startDate = startDate
	c.endDate = endDate
	return c
}

// Currency sets the optional parameter "currency": Optional currency to
// use when reporting on monetary metrics. Defaults to the account's
// currency if not set.
func (c *AccountsReportsGenerateCall) Currency(currency string) *AccountsReportsGenerateCall {
	c.opt_["currency"] = currency
	return c
}

// Dimension sets the optional parameter "dimension": Dimensions to base
// the report on.
func (c *AccountsReportsGenerateCall) Dimension(dimension string) *AccountsReportsGenerateCall {
	c.opt_["dimension"] = dimension
	return c
}

// Filter sets the optional parameter "filter": Filters to be run on the
// report.
func (c *AccountsReportsGenerateCall) Filter(filter string) *AccountsReportsGenerateCall {
	c.opt_["filter"] = filter
	return c
}

// Locale sets the optional parameter "locale": Optional locale to use
// for translating report output to a local language. Defaults to
// "en_US" if not specified.
func (c *AccountsReportsGenerateCall) Locale(locale string) *AccountsReportsGenerateCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of rows of report data to return.
func (c *AccountsReportsGenerateCall) MaxResults(maxResults int64) *AccountsReportsGenerateCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Metric sets the optional parameter "metric": Numeric columns to
// include in the report.
func (c *AccountsReportsGenerateCall) Metric(metric string) *AccountsReportsGenerateCall {
	c.opt_["metric"] = metric
	return c
}

// Sort sets the optional parameter "sort": The name of a dimension or
// metric to sort the resulting report on, optionally prefixed with "+"
// to sort ascending or "-" to sort descending. If no prefix is
// specified, the column is sorted ascending.
func (c *AccountsReportsGenerateCall) Sort(sort string) *AccountsReportsGenerateCall {
	c.opt_["sort"] = sort
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first row of report data to return.
func (c *AccountsReportsGenerateCall) StartIndex(startIndex int64) *AccountsReportsGenerateCall {
	c.opt_["startIndex"] = startIndex
	return c
}

// UseTimezoneReporting sets the optional parameter
// "useTimezoneReporting": Whether the report should be generated in the
// AdSense account's local timezone. If false default PST/PDT timezone
// will be used.
func (c *AccountsReportsGenerateCall) UseTimezoneReporting(useTimezoneReporting bool) *AccountsReportsGenerateCall {
	c.opt_["useTimezoneReporting"] = useTimezoneReporting
	return c
}

func (c *AccountsReportsGenerateCall) Do() (*AdsenseReportsGenerateResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("endDate", fmt.Sprintf("%v", c.endDate))
	params.Set("startDate", fmt.Sprintf("%v", c.startDate))
	if v, ok := c.opt_["currency"]; ok {
		params.Set("currency", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["dimension"]; ok {
		params.Set("dimension", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["metric"]; ok {
		params.Set("metric", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sort"]; ok {
		params.Set("sort", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["useTimezoneReporting"]; ok {
		params.Set("useTimezoneReporting", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/reports")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
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
	ret := new(AdsenseReportsGenerateResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Generate an AdSense report based on the report request sent in the query parameters. Returns the result as JSON; to retrieve output in CSV format specify \"alt=csv\" as a query parameter.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.reports.generate",
	//   "parameterOrder": [
	//     "accountId",
	//     "startDate",
	//     "endDate"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account upon which to report.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "currency": {
	//       "description": "Optional currency to use when reporting on monetary metrics. Defaults to the account's currency if not set.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z]+",
	//       "type": "string"
	//     },
	//     "dimension": {
	//       "description": "Dimensions to base the report on.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "endDate": {
	//       "description": "End of the date range to report on in \"YYYY-MM-DD\" format, inclusive.",
	//       "location": "query",
	//       "pattern": "\\d{4}-\\d{2}-\\d{2}|(today|startOfMonth|startOfYear)(([\\-\\+]\\d+[dwmy]){0,3}?)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filter": {
	//       "description": "Filters to be run on the report.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+(==|=@).+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "Optional locale to use for translating report output to a local language. Defaults to \"en_US\" if not specified.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of rows of report data to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "50000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "metric": {
	//       "description": "Numeric columns to include in the report.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "sort": {
	//       "description": "The name of a dimension or metric to sort the resulting report on, optionally prefixed with \"+\" to sort ascending or \"-\" to sort descending. If no prefix is specified, the column is sorted ascending.",
	//       "location": "query",
	//       "pattern": "(\\+|-)?[a-zA-Z_]+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "startDate": {
	//       "description": "Start of the date range to report on in \"YYYY-MM-DD\" format, inclusive.",
	//       "location": "query",
	//       "pattern": "\\d{4}-\\d{2}-\\d{2}|(today|startOfMonth|startOfYear)(([\\-\\+]\\d+[dwmy]){0,3}?)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first row of report data to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "5000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "useTimezoneReporting": {
	//       "description": "Whether the report should be generated in the AdSense account's local timezone. If false default PST/PDT timezone will be used.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "accounts/{accountId}/reports",
	//   "response": {
	//     "$ref": "AdsenseReportsGenerateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "adsense.accounts.reports.saved.generate":

type AccountsReportsSavedGenerateCall struct {
	s             *Service
	accountId     string
	savedReportId string
	opt_          map[string]interface{}
}

// Generate: Generate an AdSense report based on the saved report ID
// sent in the query parameters.
func (r *AccountsReportsSavedService) Generate(accountId string, savedReportId string) *AccountsReportsSavedGenerateCall {
	c := &AccountsReportsSavedGenerateCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.savedReportId = savedReportId
	return c
}

// Locale sets the optional parameter "locale": Optional locale to use
// for translating report output to a local language. Defaults to
// "en_US" if not specified.
func (c *AccountsReportsSavedGenerateCall) Locale(locale string) *AccountsReportsSavedGenerateCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of rows of report data to return.
func (c *AccountsReportsSavedGenerateCall) MaxResults(maxResults int64) *AccountsReportsSavedGenerateCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first row of report data to return.
func (c *AccountsReportsSavedGenerateCall) StartIndex(startIndex int64) *AccountsReportsSavedGenerateCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *AccountsReportsSavedGenerateCall) Do() (*AdsenseReportsGenerateResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/reports/{savedReportId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{savedReportId}", url.QueryEscape(c.savedReportId), 1)
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
	ret := new(AdsenseReportsGenerateResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Generate an AdSense report based on the saved report ID sent in the query parameters.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.reports.saved.generate",
	//   "parameterOrder": [
	//     "accountId",
	//     "savedReportId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the saved reports belong.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "Optional locale to use for translating report output to a local language. Defaults to \"en_US\" if not specified.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of rows of report data to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "50000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "savedReportId": {
	//       "description": "The saved report to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first row of report data to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "5000",
	//       "minimum": "0",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "accounts/{accountId}/reports/{savedReportId}",
	//   "response": {
	//     "$ref": "AdsenseReportsGenerateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.reports.saved.list":

type AccountsReportsSavedListCall struct {
	s         *Service
	accountId string
	opt_      map[string]interface{}
}

// List: List all saved reports in the specified AdSense account.
func (r *AccountsReportsSavedService) List(accountId string) *AccountsReportsSavedListCall {
	c := &AccountsReportsSavedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of saved reports to include in the response, used for paging.
func (c *AccountsReportsSavedListCall) MaxResults(maxResults int64) *AccountsReportsSavedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through saved reports. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AccountsReportsSavedListCall) PageToken(pageToken string) *AccountsReportsSavedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsReportsSavedListCall) Do() (*SavedReports, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/reports/saved")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
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
	ret := new(SavedReports)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all saved reports in the specified AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.reports.saved.list",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the saved reports belong.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of saved reports to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through saved reports. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/reports/saved",
	//   "response": {
	//     "$ref": "SavedReports"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.savedadstyles.get":

type AccountsSavedadstylesGetCall struct {
	s              *Service
	accountId      string
	savedAdStyleId string
	opt_           map[string]interface{}
}

// Get: List a specific saved ad style for the specified account.
func (r *AccountsSavedadstylesService) Get(accountId string, savedAdStyleId string) *AccountsSavedadstylesGetCall {
	c := &AccountsSavedadstylesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.savedAdStyleId = savedAdStyleId
	return c
}

func (c *AccountsSavedadstylesGetCall) Do() (*SavedAdStyle, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/savedadstyles/{savedAdStyleId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{savedAdStyleId}", url.QueryEscape(c.savedAdStyleId), 1)
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
	ret := new(SavedAdStyle)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List a specific saved ad style for the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.savedadstyles.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "savedAdStyleId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account for which to get the saved ad style.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "savedAdStyleId": {
	//       "description": "Saved ad style to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/savedadstyles/{savedAdStyleId}",
	//   "response": {
	//     "$ref": "SavedAdStyle"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.savedadstyles.list":

type AccountsSavedadstylesListCall struct {
	s         *Service
	accountId string
	opt_      map[string]interface{}
}

// List: List all saved ad styles in the specified account.
func (r *AccountsSavedadstylesService) List(accountId string) *AccountsSavedadstylesListCall {
	c := &AccountsSavedadstylesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of saved ad styles to include in the response, used for
// paging.
func (c *AccountsSavedadstylesListCall) MaxResults(maxResults int64) *AccountsSavedadstylesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through saved ad styles. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *AccountsSavedadstylesListCall) PageToken(pageToken string) *AccountsSavedadstylesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsSavedadstylesListCall) Do() (*SavedAdStyles, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/savedadstyles")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
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
	ret := new(SavedAdStyles)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all saved ad styles in the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.savedadstyles.list",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account for which to list saved ad styles.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of saved ad styles to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through saved ad styles. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/savedadstyles",
	//   "response": {
	//     "$ref": "SavedAdStyles"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.accounts.urlchannels.list":

type AccountsUrlchannelsListCall struct {
	s          *Service
	accountId  string
	adClientId string
	opt_       map[string]interface{}
}

// List: List all URL channels in the specified ad client for the
// specified account.
func (r *AccountsUrlchannelsService) List(accountId string, adClientId string) *AccountsUrlchannelsListCall {
	c := &AccountsUrlchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.adClientId = adClientId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of URL channels to include in the response, used for paging.
func (c *AccountsUrlchannelsListCall) MaxResults(maxResults int64) *AccountsUrlchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through URL channels. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AccountsUrlchannelsListCall) PageToken(pageToken string) *AccountsUrlchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AccountsUrlchannelsListCall) Do() (*UrlChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{accountId}/adclients/{adClientId}/urlchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
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
	ret := new(UrlChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all URL channels in the specified ad client for the specified account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.accounts.urlchannels.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account to which the ad client belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adClientId": {
	//       "description": "Ad client for which to list URL channels.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of URL channels to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through URL channels. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{accountId}/adclients/{adClientId}/urlchannels",
	//   "response": {
	//     "$ref": "UrlChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.adclients.list":

type AdclientsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List all ad clients in this AdSense account.
func (r *AdclientsService) List() *AdclientsListCall {
	c := &AdclientsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad clients to include in the response, used for paging.
func (c *AdclientsListCall) MaxResults(maxResults int64) *AdclientsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad clients. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AdclientsListCall) PageToken(pageToken string) *AdclientsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AdclientsListCall) Do() (*AdClients, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients")
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
	ret := new(AdClients)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad clients in this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.adclients.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of ad clients to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through ad clients. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients",
	//   "response": {
	//     "$ref": "AdClients"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.adunits.get":

type AdunitsGetCall struct {
	s          *Service
	adClientId string
	adUnitId   string
	opt_       map[string]interface{}
}

// Get: Gets the specified ad unit in the specified ad client.
func (r *AdunitsService) Get(adClientId string, adUnitId string) *AdunitsGetCall {
	c := &AdunitsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	c.adUnitId = adUnitId
	return c
}

func (c *AdunitsGetCall) Do() (*AdUnit, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/adunits/{adUnitId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adUnitId}", url.QueryEscape(c.adUnitId), 1)
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
	ret := new(AdUnit)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the specified ad unit in the specified ad client.",
	//   "httpMethod": "GET",
	//   "id": "adsense.adunits.get",
	//   "parameterOrder": [
	//     "adClientId",
	//     "adUnitId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client for which to get the ad unit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adUnitId": {
	//       "description": "Ad unit to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/adunits/{adUnitId}",
	//   "response": {
	//     "$ref": "AdUnit"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.adunits.getAdCode":

type AdunitsGetAdCodeCall struct {
	s          *Service
	adClientId string
	adUnitId   string
	opt_       map[string]interface{}
}

// GetAdCode: Get ad code for the specified ad unit.
func (r *AdunitsService) GetAdCode(adClientId string, adUnitId string) *AdunitsGetAdCodeCall {
	c := &AdunitsGetAdCodeCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	c.adUnitId = adUnitId
	return c
}

func (c *AdunitsGetAdCodeCall) Do() (*AdCode, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/adunits/{adUnitId}/adcode")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adUnitId}", url.QueryEscape(c.adUnitId), 1)
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
	ret := new(AdCode)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get ad code for the specified ad unit.",
	//   "httpMethod": "GET",
	//   "id": "adsense.adunits.getAdCode",
	//   "parameterOrder": [
	//     "adClientId",
	//     "adUnitId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client with contains the ad unit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adUnitId": {
	//       "description": "Ad unit to get the code for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/adunits/{adUnitId}/adcode",
	//   "response": {
	//     "$ref": "AdCode"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.adunits.list":

type AdunitsListCall struct {
	s          *Service
	adClientId string
	opt_       map[string]interface{}
}

// List: List all ad units in the specified ad client for this AdSense
// account.
func (r *AdunitsService) List(adClientId string) *AdunitsListCall {
	c := &AdunitsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	return c
}

// IncludeInactive sets the optional parameter "includeInactive":
// Whether to include inactive ad units. Default: true.
func (c *AdunitsListCall) IncludeInactive(includeInactive bool) *AdunitsListCall {
	c.opt_["includeInactive"] = includeInactive
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad units to include in the response, used for paging.
func (c *AdunitsListCall) MaxResults(maxResults int64) *AdunitsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad units. To retrieve the next page, set
// this parameter to the value of "nextPageToken" from the previous
// response.
func (c *AdunitsListCall) PageToken(pageToken string) *AdunitsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AdunitsListCall) Do() (*AdUnits, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["includeInactive"]; ok {
		params.Set("includeInactive", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/adunits")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
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
	ret := new(AdUnits)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad units in the specified ad client for this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.adunits.list",
	//   "parameterOrder": [
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client for which to list ad units.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "includeInactive": {
	//       "description": "Whether to include inactive ad units. Default: true.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of ad units to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through ad units. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/adunits",
	//   "response": {
	//     "$ref": "AdUnits"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.adunits.customchannels.list":

type AdunitsCustomchannelsListCall struct {
	s          *Service
	adClientId string
	adUnitId   string
	opt_       map[string]interface{}
}

// List: List all custom channels which the specified ad unit belongs
// to.
func (r *AdunitsCustomchannelsService) List(adClientId string, adUnitId string) *AdunitsCustomchannelsListCall {
	c := &AdunitsCustomchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	c.adUnitId = adUnitId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of custom channels to include in the response, used for
// paging.
func (c *AdunitsCustomchannelsListCall) MaxResults(maxResults int64) *AdunitsCustomchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through custom channels. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *AdunitsCustomchannelsListCall) PageToken(pageToken string) *AdunitsCustomchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AdunitsCustomchannelsListCall) Do() (*CustomChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/adunits/{adUnitId}/customchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{adUnitId}", url.QueryEscape(c.adUnitId), 1)
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
	ret := new(CustomChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all custom channels which the specified ad unit belongs to.",
	//   "httpMethod": "GET",
	//   "id": "adsense.adunits.customchannels.list",
	//   "parameterOrder": [
	//     "adClientId",
	//     "adUnitId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client which contains the ad unit.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "adUnitId": {
	//       "description": "Ad unit for which to list custom channels.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of custom channels to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through custom channels. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/adunits/{adUnitId}/customchannels",
	//   "response": {
	//     "$ref": "CustomChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.alerts.list":

type AlertsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List the alerts for this AdSense account.
func (r *AlertsService) List() *AlertsListCall {
	c := &AlertsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// Locale sets the optional parameter "locale": The locale to use for
// translating alert messages. The account locale will be used if this
// is not supplied. The AdSense default (English) will be used if the
// supplied locale is invalid or unsupported.
func (c *AlertsListCall) Locale(locale string) *AlertsListCall {
	c.opt_["locale"] = locale
	return c
}

func (c *AlertsListCall) Do() (*Alerts, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "alerts")
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
	ret := new(Alerts)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List the alerts for this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.alerts.list",
	//   "parameters": {
	//     "locale": {
	//       "description": "The locale to use for translating alert messages. The account locale will be used if this is not supplied. The AdSense default (English) will be used if the supplied locale is invalid or unsupported.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "alerts",
	//   "response": {
	//     "$ref": "Alerts"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.customchannels.get":

type CustomchannelsGetCall struct {
	s               *Service
	adClientId      string
	customChannelId string
	opt_            map[string]interface{}
}

// Get: Get the specified custom channel from the specified ad client.
func (r *CustomchannelsService) Get(adClientId string, customChannelId string) *CustomchannelsGetCall {
	c := &CustomchannelsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	c.customChannelId = customChannelId
	return c
}

func (c *CustomchannelsGetCall) Do() (*CustomChannel, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/customchannels/{customChannelId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customChannelId}", url.QueryEscape(c.customChannelId), 1)
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
	ret := new(CustomChannel)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get the specified custom channel from the specified ad client.",
	//   "httpMethod": "GET",
	//   "id": "adsense.customchannels.get",
	//   "parameterOrder": [
	//     "adClientId",
	//     "customChannelId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client which contains the custom channel.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customChannelId": {
	//       "description": "Custom channel to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/customchannels/{customChannelId}",
	//   "response": {
	//     "$ref": "CustomChannel"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.customchannels.list":

type CustomchannelsListCall struct {
	s          *Service
	adClientId string
	opt_       map[string]interface{}
}

// List: List all custom channels in the specified ad client for this
// AdSense account.
func (r *CustomchannelsService) List(adClientId string) *CustomchannelsListCall {
	c := &CustomchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of custom channels to include in the response, used for
// paging.
func (c *CustomchannelsListCall) MaxResults(maxResults int64) *CustomchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through custom channels. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *CustomchannelsListCall) PageToken(pageToken string) *CustomchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *CustomchannelsListCall) Do() (*CustomChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/customchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
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
	ret := new(CustomChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all custom channels in the specified ad client for this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.customchannels.list",
	//   "parameterOrder": [
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client for which to list custom channels.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of custom channels to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through custom channels. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/customchannels",
	//   "response": {
	//     "$ref": "CustomChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.customchannels.adunits.list":

type CustomchannelsAdunitsListCall struct {
	s               *Service
	adClientId      string
	customChannelId string
	opt_            map[string]interface{}
}

// List: List all ad units in the specified custom channel.
func (r *CustomchannelsAdunitsService) List(adClientId string, customChannelId string) *CustomchannelsAdunitsListCall {
	c := &CustomchannelsAdunitsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	c.customChannelId = customChannelId
	return c
}

// IncludeInactive sets the optional parameter "includeInactive":
// Whether to include inactive ad units. Default: true.
func (c *CustomchannelsAdunitsListCall) IncludeInactive(includeInactive bool) *CustomchannelsAdunitsListCall {
	c.opt_["includeInactive"] = includeInactive
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of ad units to include in the response, used for paging.
func (c *CustomchannelsAdunitsListCall) MaxResults(maxResults int64) *CustomchannelsAdunitsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through ad units. To retrieve the next page, set
// this parameter to the value of "nextPageToken" from the previous
// response.
func (c *CustomchannelsAdunitsListCall) PageToken(pageToken string) *CustomchannelsAdunitsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *CustomchannelsAdunitsListCall) Do() (*AdUnits, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["includeInactive"]; ok {
		params.Set("includeInactive", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/customchannels/{customChannelId}/adunits")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customChannelId}", url.QueryEscape(c.customChannelId), 1)
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
	ret := new(AdUnits)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all ad units in the specified custom channel.",
	//   "httpMethod": "GET",
	//   "id": "adsense.customchannels.adunits.list",
	//   "parameterOrder": [
	//     "adClientId",
	//     "customChannelId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client which contains the custom channel.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customChannelId": {
	//       "description": "Custom channel for which to list ad units.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "includeInactive": {
	//       "description": "Whether to include inactive ad units. Default: true.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of ad units to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through ad units. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/customchannels/{customChannelId}/adunits",
	//   "response": {
	//     "$ref": "AdUnits"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.metadata.dimensions.list":

type MetadataDimensionsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List the metadata for the dimensions available to this AdSense
// account.
func (r *MetadataDimensionsService) List() *MetadataDimensionsListCall {
	c := &MetadataDimensionsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *MetadataDimensionsListCall) Do() (*Metadata, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "metadata/dimensions")
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
	ret := new(Metadata)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List the metadata for the dimensions available to this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.metadata.dimensions.list",
	//   "path": "metadata/dimensions",
	//   "response": {
	//     "$ref": "Metadata"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.metadata.metrics.list":

type MetadataMetricsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List the metadata for the metrics available to this AdSense
// account.
func (r *MetadataMetricsService) List() *MetadataMetricsListCall {
	c := &MetadataMetricsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *MetadataMetricsListCall) Do() (*Metadata, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "metadata/metrics")
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
	ret := new(Metadata)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List the metadata for the metrics available to this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.metadata.metrics.list",
	//   "path": "metadata/metrics",
	//   "response": {
	//     "$ref": "Metadata"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.reports.generate":

type ReportsGenerateCall struct {
	s         *Service
	startDate string
	endDate   string
	opt_      map[string]interface{}
}

// Generate: Generate an AdSense report based on the report request sent
// in the query parameters. Returns the result as JSON; to retrieve
// output in CSV format specify "alt=csv" as a query parameter.
func (r *ReportsService) Generate(startDate string, endDate string) *ReportsGenerateCall {
	c := &ReportsGenerateCall{s: r.s, opt_: make(map[string]interface{})}
	c.startDate = startDate
	c.endDate = endDate
	return c
}

// AccountId sets the optional parameter "accountId": Accounts upon
// which to report.
func (c *ReportsGenerateCall) AccountId(accountId string) *ReportsGenerateCall {
	c.opt_["accountId"] = accountId
	return c
}

// Currency sets the optional parameter "currency": Optional currency to
// use when reporting on monetary metrics. Defaults to the account's
// currency if not set.
func (c *ReportsGenerateCall) Currency(currency string) *ReportsGenerateCall {
	c.opt_["currency"] = currency
	return c
}

// Dimension sets the optional parameter "dimension": Dimensions to base
// the report on.
func (c *ReportsGenerateCall) Dimension(dimension string) *ReportsGenerateCall {
	c.opt_["dimension"] = dimension
	return c
}

// Filter sets the optional parameter "filter": Filters to be run on the
// report.
func (c *ReportsGenerateCall) Filter(filter string) *ReportsGenerateCall {
	c.opt_["filter"] = filter
	return c
}

// Locale sets the optional parameter "locale": Optional locale to use
// for translating report output to a local language. Defaults to
// "en_US" if not specified.
func (c *ReportsGenerateCall) Locale(locale string) *ReportsGenerateCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of rows of report data to return.
func (c *ReportsGenerateCall) MaxResults(maxResults int64) *ReportsGenerateCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Metric sets the optional parameter "metric": Numeric columns to
// include in the report.
func (c *ReportsGenerateCall) Metric(metric string) *ReportsGenerateCall {
	c.opt_["metric"] = metric
	return c
}

// Sort sets the optional parameter "sort": The name of a dimension or
// metric to sort the resulting report on, optionally prefixed with "+"
// to sort ascending or "-" to sort descending. If no prefix is
// specified, the column is sorted ascending.
func (c *ReportsGenerateCall) Sort(sort string) *ReportsGenerateCall {
	c.opt_["sort"] = sort
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first row of report data to return.
func (c *ReportsGenerateCall) StartIndex(startIndex int64) *ReportsGenerateCall {
	c.opt_["startIndex"] = startIndex
	return c
}

// UseTimezoneReporting sets the optional parameter
// "useTimezoneReporting": Whether the report should be generated in the
// AdSense account's local timezone. If false default PST/PDT timezone
// will be used.
func (c *ReportsGenerateCall) UseTimezoneReporting(useTimezoneReporting bool) *ReportsGenerateCall {
	c.opt_["useTimezoneReporting"] = useTimezoneReporting
	return c
}

func (c *ReportsGenerateCall) Do() (*AdsenseReportsGenerateResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("endDate", fmt.Sprintf("%v", c.endDate))
	params.Set("startDate", fmt.Sprintf("%v", c.startDate))
	if v, ok := c.opt_["accountId"]; ok {
		params.Set("accountId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["currency"]; ok {
		params.Set("currency", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["dimension"]; ok {
		params.Set("dimension", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["filter"]; ok {
		params.Set("filter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["metric"]; ok {
		params.Set("metric", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sort"]; ok {
		params.Set("sort", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["useTimezoneReporting"]; ok {
		params.Set("useTimezoneReporting", fmt.Sprintf("%v", v))
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
	ret := new(AdsenseReportsGenerateResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Generate an AdSense report based on the report request sent in the query parameters. Returns the result as JSON; to retrieve output in CSV format specify \"alt=csv\" as a query parameter.",
	//   "httpMethod": "GET",
	//   "id": "adsense.reports.generate",
	//   "parameterOrder": [
	//     "startDate",
	//     "endDate"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Accounts upon which to report.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "currency": {
	//       "description": "Optional currency to use when reporting on monetary metrics. Defaults to the account's currency if not set.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z]+",
	//       "type": "string"
	//     },
	//     "dimension": {
	//       "description": "Dimensions to base the report on.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "endDate": {
	//       "description": "End of the date range to report on in \"YYYY-MM-DD\" format, inclusive.",
	//       "location": "query",
	//       "pattern": "\\d{4}-\\d{2}-\\d{2}|(today|startOfMonth|startOfYear)(([\\-\\+]\\d+[dwmy]){0,3}?)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filter": {
	//       "description": "Filters to be run on the report.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+(==|=@).+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "Optional locale to use for translating report output to a local language. Defaults to \"en_US\" if not specified.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of rows of report data to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "50000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "metric": {
	//       "description": "Numeric columns to include in the report.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "sort": {
	//       "description": "The name of a dimension or metric to sort the resulting report on, optionally prefixed with \"+\" to sort ascending or \"-\" to sort descending. If no prefix is specified, the column is sorted ascending.",
	//       "location": "query",
	//       "pattern": "(\\+|-)?[a-zA-Z_]+",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "startDate": {
	//       "description": "Start of the date range to report on in \"YYYY-MM-DD\" format, inclusive.",
	//       "location": "query",
	//       "pattern": "\\d{4}-\\d{2}-\\d{2}|(today|startOfMonth|startOfYear)(([\\-\\+]\\d+[dwmy]){0,3}?)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first row of report data to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "5000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "useTimezoneReporting": {
	//       "description": "Whether the report should be generated in the AdSense account's local timezone. If false default PST/PDT timezone will be used.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "reports",
	//   "response": {
	//     "$ref": "AdsenseReportsGenerateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "adsense.reports.saved.generate":

type ReportsSavedGenerateCall struct {
	s             *Service
	savedReportId string
	opt_          map[string]interface{}
}

// Generate: Generate an AdSense report based on the saved report ID
// sent in the query parameters.
func (r *ReportsSavedService) Generate(savedReportId string) *ReportsSavedGenerateCall {
	c := &ReportsSavedGenerateCall{s: r.s, opt_: make(map[string]interface{})}
	c.savedReportId = savedReportId
	return c
}

// Locale sets the optional parameter "locale": Optional locale to use
// for translating report output to a local language. Defaults to
// "en_US" if not specified.
func (c *ReportsSavedGenerateCall) Locale(locale string) *ReportsSavedGenerateCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of rows of report data to return.
func (c *ReportsSavedGenerateCall) MaxResults(maxResults int64) *ReportsSavedGenerateCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first row of report data to return.
func (c *ReportsSavedGenerateCall) StartIndex(startIndex int64) *ReportsSavedGenerateCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *ReportsSavedGenerateCall) Do() (*AdsenseReportsGenerateResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports/{savedReportId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{savedReportId}", url.QueryEscape(c.savedReportId), 1)
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
	ret := new(AdsenseReportsGenerateResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Generate an AdSense report based on the saved report ID sent in the query parameters.",
	//   "httpMethod": "GET",
	//   "id": "adsense.reports.saved.generate",
	//   "parameterOrder": [
	//     "savedReportId"
	//   ],
	//   "parameters": {
	//     "locale": {
	//       "description": "Optional locale to use for translating report output to a local language. Defaults to \"en_US\" if not specified.",
	//       "location": "query",
	//       "pattern": "[a-zA-Z_]+",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of rows of report data to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "50000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "savedReportId": {
	//       "description": "The saved report to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first row of report data to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "5000",
	//       "minimum": "0",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "reports/{savedReportId}",
	//   "response": {
	//     "$ref": "AdsenseReportsGenerateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.reports.saved.list":

type ReportsSavedListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List all saved reports in this AdSense account.
func (r *ReportsSavedService) List() *ReportsSavedListCall {
	c := &ReportsSavedListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of saved reports to include in the response, used for paging.
func (c *ReportsSavedListCall) MaxResults(maxResults int64) *ReportsSavedListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through saved reports. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *ReportsSavedListCall) PageToken(pageToken string) *ReportsSavedListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *ReportsSavedListCall) Do() (*SavedReports, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "reports/saved")
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
	ret := new(SavedReports)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all saved reports in this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.reports.saved.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of saved reports to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through saved reports. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "reports/saved",
	//   "response": {
	//     "$ref": "SavedReports"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.savedadstyles.get":

type SavedadstylesGetCall struct {
	s              *Service
	savedAdStyleId string
	opt_           map[string]interface{}
}

// Get: Get a specific saved ad style from the user's account.
func (r *SavedadstylesService) Get(savedAdStyleId string) *SavedadstylesGetCall {
	c := &SavedadstylesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.savedAdStyleId = savedAdStyleId
	return c
}

func (c *SavedadstylesGetCall) Do() (*SavedAdStyle, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "savedadstyles/{savedAdStyleId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{savedAdStyleId}", url.QueryEscape(c.savedAdStyleId), 1)
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
	ret := new(SavedAdStyle)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get a specific saved ad style from the user's account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.savedadstyles.get",
	//   "parameterOrder": [
	//     "savedAdStyleId"
	//   ],
	//   "parameters": {
	//     "savedAdStyleId": {
	//       "description": "Saved ad style to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "savedadstyles/{savedAdStyleId}",
	//   "response": {
	//     "$ref": "SavedAdStyle"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.savedadstyles.list":

type SavedadstylesListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: List all saved ad styles in the user's account.
func (r *SavedadstylesService) List() *SavedadstylesListCall {
	c := &SavedadstylesListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of saved ad styles to include in the response, used for
// paging.
func (c *SavedadstylesListCall) MaxResults(maxResults int64) *SavedadstylesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through saved ad styles. To retrieve the next
// page, set this parameter to the value of "nextPageToken" from the
// previous response.
func (c *SavedadstylesListCall) PageToken(pageToken string) *SavedadstylesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *SavedadstylesListCall) Do() (*SavedAdStyles, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "savedadstyles")
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
	ret := new(SavedAdStyles)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all saved ad styles in the user's account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.savedadstyles.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of saved ad styles to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through saved ad styles. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "savedadstyles",
	//   "response": {
	//     "$ref": "SavedAdStyles"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}

// method id "adsense.urlchannels.list":

type UrlchannelsListCall struct {
	s          *Service
	adClientId string
	opt_       map[string]interface{}
}

// List: List all URL channels in the specified ad client for this
// AdSense account.
func (r *UrlchannelsService) List(adClientId string) *UrlchannelsListCall {
	c := &UrlchannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.adClientId = adClientId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of URL channels to include in the response, used for paging.
func (c *UrlchannelsListCall) MaxResults(maxResults int64) *UrlchannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token, used to page through URL channels. To retrieve the next page,
// set this parameter to the value of "nextPageToken" from the previous
// response.
func (c *UrlchannelsListCall) PageToken(pageToken string) *UrlchannelsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *UrlchannelsListCall) Do() (*UrlChannels, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "adclients/{adClientId}/urlchannels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{adClientId}", url.QueryEscape(c.adClientId), 1)
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
	ret := new(UrlChannels)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all URL channels in the specified ad client for this AdSense account.",
	//   "httpMethod": "GET",
	//   "id": "adsense.urlchannels.list",
	//   "parameterOrder": [
	//     "adClientId"
	//   ],
	//   "parameters": {
	//     "adClientId": {
	//       "description": "Ad client for which to list URL channels.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of URL channels to include in the response, used for paging.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "10000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token, used to page through URL channels. To retrieve the next page, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "adclients/{adClientId}/urlchannels",
	//   "response": {
	//     "$ref": "UrlChannels"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/adsense",
	//     "https://www.googleapis.com/auth/adsense.readonly"
	//   ]
	// }

}
