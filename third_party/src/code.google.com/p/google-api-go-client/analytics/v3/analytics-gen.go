// Package analytics provides access to the Google Analytics API.
//
// See https://developers.google.com/analytics/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/analytics/v3"
//   ...
//   analyticsService, err := analytics.New(oauthHttpClient)
package analytics

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

const apiId = "analytics:v3"
const apiName = "analytics"
const apiVersion = "v3"
const basePath = "https://www.googleapis.com/analytics/v3/"

// OAuth2 scopes used by this API.
const (
	// View and manage your Google Analytics data
	AnalyticsScope = "https://www.googleapis.com/auth/analytics"

	// Edit Google Analytics management entities
	AnalyticsEditScope = "https://www.googleapis.com/auth/analytics.edit"

	// Manage Google Analytics Account users by email address
	AnalyticsManageUsersScope = "https://www.googleapis.com/auth/analytics.manage.users"

	// View your Google Analytics data
	AnalyticsReadonlyScope = "https://www.googleapis.com/auth/analytics.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Data = NewDataService(s)
	s.Management = NewManagementService(s)
	s.Metadata = NewMetadataService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Data *DataService

	Management *ManagementService

	Metadata *MetadataService
}

func NewDataService(s *Service) *DataService {
	rs := &DataService{s: s}
	rs.Ga = NewDataGaService(s)
	rs.Mcf = NewDataMcfService(s)
	rs.Realtime = NewDataRealtimeService(s)
	return rs
}

type DataService struct {
	s *Service

	Ga *DataGaService

	Mcf *DataMcfService

	Realtime *DataRealtimeService
}

func NewDataGaService(s *Service) *DataGaService {
	rs := &DataGaService{s: s}
	return rs
}

type DataGaService struct {
	s *Service
}

func NewDataMcfService(s *Service) *DataMcfService {
	rs := &DataMcfService{s: s}
	return rs
}

type DataMcfService struct {
	s *Service
}

func NewDataRealtimeService(s *Service) *DataRealtimeService {
	rs := &DataRealtimeService{s: s}
	return rs
}

type DataRealtimeService struct {
	s *Service
}

func NewManagementService(s *Service) *ManagementService {
	rs := &ManagementService{s: s}
	rs.AccountSummaries = NewManagementAccountSummariesService(s)
	rs.AccountUserLinks = NewManagementAccountUserLinksService(s)
	rs.Accounts = NewManagementAccountsService(s)
	rs.CustomDataSources = NewManagementCustomDataSourcesService(s)
	rs.DailyUploads = NewManagementDailyUploadsService(s)
	rs.Experiments = NewManagementExperimentsService(s)
	rs.Goals = NewManagementGoalsService(s)
	rs.ProfileUserLinks = NewManagementProfileUserLinksService(s)
	rs.Profiles = NewManagementProfilesService(s)
	rs.Segments = NewManagementSegmentsService(s)
	rs.Uploads = NewManagementUploadsService(s)
	rs.Webproperties = NewManagementWebpropertiesService(s)
	rs.WebpropertyUserLinks = NewManagementWebpropertyUserLinksService(s)
	return rs
}

type ManagementService struct {
	s *Service

	AccountSummaries *ManagementAccountSummariesService

	AccountUserLinks *ManagementAccountUserLinksService

	Accounts *ManagementAccountsService

	CustomDataSources *ManagementCustomDataSourcesService

	DailyUploads *ManagementDailyUploadsService

	Experiments *ManagementExperimentsService

	Goals *ManagementGoalsService

	ProfileUserLinks *ManagementProfileUserLinksService

	Profiles *ManagementProfilesService

	Segments *ManagementSegmentsService

	Uploads *ManagementUploadsService

	Webproperties *ManagementWebpropertiesService

	WebpropertyUserLinks *ManagementWebpropertyUserLinksService
}

func NewManagementAccountSummariesService(s *Service) *ManagementAccountSummariesService {
	rs := &ManagementAccountSummariesService{s: s}
	return rs
}

type ManagementAccountSummariesService struct {
	s *Service
}

func NewManagementAccountUserLinksService(s *Service) *ManagementAccountUserLinksService {
	rs := &ManagementAccountUserLinksService{s: s}
	return rs
}

type ManagementAccountUserLinksService struct {
	s *Service
}

func NewManagementAccountsService(s *Service) *ManagementAccountsService {
	rs := &ManagementAccountsService{s: s}
	return rs
}

type ManagementAccountsService struct {
	s *Service
}

func NewManagementCustomDataSourcesService(s *Service) *ManagementCustomDataSourcesService {
	rs := &ManagementCustomDataSourcesService{s: s}
	return rs
}

type ManagementCustomDataSourcesService struct {
	s *Service
}

func NewManagementDailyUploadsService(s *Service) *ManagementDailyUploadsService {
	rs := &ManagementDailyUploadsService{s: s}
	return rs
}

type ManagementDailyUploadsService struct {
	s *Service
}

func NewManagementExperimentsService(s *Service) *ManagementExperimentsService {
	rs := &ManagementExperimentsService{s: s}
	return rs
}

type ManagementExperimentsService struct {
	s *Service
}

func NewManagementGoalsService(s *Service) *ManagementGoalsService {
	rs := &ManagementGoalsService{s: s}
	return rs
}

type ManagementGoalsService struct {
	s *Service
}

func NewManagementProfileUserLinksService(s *Service) *ManagementProfileUserLinksService {
	rs := &ManagementProfileUserLinksService{s: s}
	return rs
}

type ManagementProfileUserLinksService struct {
	s *Service
}

func NewManagementProfilesService(s *Service) *ManagementProfilesService {
	rs := &ManagementProfilesService{s: s}
	return rs
}

type ManagementProfilesService struct {
	s *Service
}

func NewManagementSegmentsService(s *Service) *ManagementSegmentsService {
	rs := &ManagementSegmentsService{s: s}
	return rs
}

type ManagementSegmentsService struct {
	s *Service
}

func NewManagementUploadsService(s *Service) *ManagementUploadsService {
	rs := &ManagementUploadsService{s: s}
	return rs
}

type ManagementUploadsService struct {
	s *Service
}

func NewManagementWebpropertiesService(s *Service) *ManagementWebpropertiesService {
	rs := &ManagementWebpropertiesService{s: s}
	return rs
}

type ManagementWebpropertiesService struct {
	s *Service
}

func NewManagementWebpropertyUserLinksService(s *Service) *ManagementWebpropertyUserLinksService {
	rs := &ManagementWebpropertyUserLinksService{s: s}
	return rs
}

type ManagementWebpropertyUserLinksService struct {
	s *Service
}

func NewMetadataService(s *Service) *MetadataService {
	rs := &MetadataService{s: s}
	rs.Columns = NewMetadataColumnsService(s)
	return rs
}

type MetadataService struct {
	s *Service

	Columns *MetadataColumnsService
}

func NewMetadataColumnsService(s *Service) *MetadataColumnsService {
	rs := &MetadataColumnsService{s: s}
	return rs
}

type MetadataColumnsService struct {
	s *Service
}

type Account struct {
	// ChildLink: Child link for an account entry. Points to the list of web
	// properties for this account.
	ChildLink *AccountChildLink `json:"childLink,omitempty"`

	// Created: Time the account was created.
	Created string `json:"created,omitempty"`

	// Id: Account ID.
	Id string `json:"id,omitempty"`

	// Kind: Resource type for Analytics account.
	Kind string `json:"kind,omitempty"`

	// Name: Account name.
	Name string `json:"name,omitempty"`

	// Permissions: Permissions the user has for this account.
	Permissions *AccountPermissions `json:"permissions,omitempty"`

	// SelfLink: Link for this account.
	SelfLink string `json:"selfLink,omitempty"`

	// Updated: Time the account was last modified.
	Updated string `json:"updated,omitempty"`
}

type AccountChildLink struct {
	// Href: Link to the list of web properties for this account.
	Href string `json:"href,omitempty"`

	// Type: Type of the child link. Its value is "analytics#webproperties".
	Type string `json:"type,omitempty"`
}

type AccountPermissions struct {
	// Effective: All the permissions that the user has for this account.
	// These include any implied permissions (e.g., EDIT implies VIEW).
	Effective []string `json:"effective,omitempty"`
}

type AccountRef struct {
	// Href: Link for this account.
	Href string `json:"href,omitempty"`

	// Id: Account ID.
	Id string `json:"id,omitempty"`

	// Kind: Analytics account reference.
	Kind string `json:"kind,omitempty"`

	// Name: Account name.
	Name string `json:"name,omitempty"`
}

type AccountSummaries struct {
	// Items: A list of AccountSummaries.
	Items []*AccountSummary `json:"items,omitempty"`

	// ItemsPerPage: The maximum number of resources the response can
	// contain, regardless of the actual number of resources returned. Its
	// value ranges from 1 to 1000 with a value of 1000 by default, or
	// otherwise specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Collection type.
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to next page for this AccountSummary collection.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Link to previous page for this AccountSummary
	// collection.
	PreviousLink string `json:"previousLink,omitempty"`

	// StartIndex: The starting index of the resources, which is 1 by
	// default or otherwise specified by the start-index query parameter.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results for the query, regardless
	// of the number of results in the response.
	TotalResults int64 `json:"totalResults,omitempty"`

	// Username: Email ID of the authenticated user
	Username string `json:"username,omitempty"`
}

type AccountSummary struct {
	// Id: Account ID.
	Id string `json:"id,omitempty"`

	// Kind: Resource type for Analytics AccountSummary.
	Kind string `json:"kind,omitempty"`

	// Name: Account name.
	Name string `json:"name,omitempty"`

	// WebProperties: List of web properties under this account.
	WebProperties []*WebPropertySummary `json:"webProperties,omitempty"`
}

type Accounts struct {
	// Items: A list of accounts.
	Items []*Account `json:"items,omitempty"`

	// ItemsPerPage: The maximum number of entries the response can contain,
	// regardless of the actual number of entries returned. Its value ranges
	// from 1 to 1000 with a value of 1000 by default, or otherwise
	// specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Collection type.
	Kind string `json:"kind,omitempty"`

	// NextLink: Next link for this account collection.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Previous link for this account collection.
	PreviousLink string `json:"previousLink,omitempty"`

	// StartIndex: The starting index of the entries, which is 1 by default
	// or otherwise specified by the start-index query parameter.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results for the query, regardless
	// of the number of results in the response.
	TotalResults int64 `json:"totalResults,omitempty"`

	// Username: Email ID of the authenticated user
	Username string `json:"username,omitempty"`
}

type AnalyticsDataimportDeleteUploadDataRequest struct {
	// CustomDataImportUids: A list of upload UIDs.
	CustomDataImportUids []string `json:"customDataImportUids,omitempty"`
}

type Column struct {
	// Attributes: Map of attribute name and value for this column.
	Attributes map[string]string `json:"attributes,omitempty"`

	// Id: Column id.
	Id string `json:"id,omitempty"`

	// Kind: Resource type for Analytics column.
	Kind string `json:"kind,omitempty"`
}

type Columns struct {
	// AttributeNames: List of attributes names returned by columns.
	AttributeNames []string `json:"attributeNames,omitempty"`

	// Etag: Etag of collection. This etag can be compared with the last
	// response etag to check if response has changed.
	Etag string `json:"etag,omitempty"`

	// Items: List of columns for a report type.
	Items []*Column `json:"items,omitempty"`

	// Kind: Collection type.
	Kind string `json:"kind,omitempty"`

	// TotalResults: Total number of columns returned in the response.
	TotalResults int64 `json:"totalResults,omitempty"`
}

type CustomDataSource struct {
	// AccountId: Account ID to which this custom data source belongs.
	AccountId string `json:"accountId,omitempty"`

	ChildLink *CustomDataSourceChildLink `json:"childLink,omitempty"`

	// Created: Time this custom data source was created.
	Created string `json:"created,omitempty"`

	// Description: Description of custom data source.
	Description string `json:"description,omitempty"`

	// Id: Custom data source ID.
	Id string `json:"id,omitempty"`

	// Kind: Resource type for Analytics custom data source.
	Kind string `json:"kind,omitempty"`

	// Name: Name of this custom data source.
	Name string `json:"name,omitempty"`

	// ParentLink: Parent link for this custom data source. Points to the
	// web property to which this custom data source belongs.
	ParentLink *CustomDataSourceParentLink `json:"parentLink,omitempty"`

	// ProfilesLinked: IDs of views (profiles) linked to the custom data
	// source.
	ProfilesLinked []string `json:"profilesLinked,omitempty"`

	// SelfLink: Link for this Analytics custom data source.
	SelfLink string `json:"selfLink,omitempty"`

	// Type: Type of the custom data source.
	Type string `json:"type,omitempty"`

	// Updated: Time this custom data source was last modified.
	Updated string `json:"updated,omitempty"`

	// WebPropertyId: Web property ID of the form UA-XXXXX-YY to which this
	// custom data source belongs.
	WebPropertyId string `json:"webPropertyId,omitempty"`
}

type CustomDataSourceChildLink struct {
	// Href: Link to the list of daily uploads for this custom data source.
	// Link to the list of uploads for this custom data source.
	Href string `json:"href,omitempty"`

	// Type: Value is "analytics#dailyUploads". Value is
	// "analytics#uploads".
	Type string `json:"type,omitempty"`
}

type CustomDataSourceParentLink struct {
	// Href: Link to the web property to which this custom data source
	// belongs.
	Href string `json:"href,omitempty"`

	// Type: Value is "analytics#webproperty".
	Type string `json:"type,omitempty"`
}

type CustomDataSources struct {
	// Items: Collection of custom data sources.
	Items []*CustomDataSource `json:"items,omitempty"`

	// ItemsPerPage: The maximum number of resources the response can
	// contain, regardless of the actual number of resources returned. Its
	// value ranges from 1 to 1000 with a value of 1000 by default, or
	// otherwise specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Collection type.
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to next page for this custom data source collection.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Link to previous page for this custom data source
	// collection.
	PreviousLink string `json:"previousLink,omitempty"`

	// StartIndex: The starting index of the resources, which is 1 by
	// default or otherwise specified by the start-index query parameter.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results for the query, regardless
	// of the number of results in the response.
	TotalResults int64 `json:"totalResults,omitempty"`

	// Username: Email ID of the authenticated user
	Username string `json:"username,omitempty"`
}

type DailyUpload struct {
	// AccountId: Account ID to which this daily upload belongs.
	AccountId string `json:"accountId,omitempty"`

	// AppendCount: Number of appends for this date.
	AppendCount int64 `json:"appendCount,omitempty"`

	// CreatedTime: Time this daily upload was created.
	CreatedTime string `json:"createdTime,omitempty"`

	// CustomDataSourceId: Custom data source ID to which this daily upload
	// belongs.
	CustomDataSourceId string `json:"customDataSourceId,omitempty"`

	// Date: Date associated with daily upload.
	Date string `json:"date,omitempty"`

	// Kind: Resource type for Analytics daily upload.
	Kind string `json:"kind,omitempty"`

	// ModifiedTime: Time this daily upload was last modified.
	ModifiedTime string `json:"modifiedTime,omitempty"`

	// ParentLink: Parent link for a daily upload. Points to the custom data
	// source to which this daily upload belongs.
	ParentLink *DailyUploadParentLink `json:"parentLink,omitempty"`

	// RecentChanges: Change log for last 10 changes in chronological order.
	RecentChanges []*DailyUploadRecentChanges `json:"recentChanges,omitempty"`

	// SelfLink: Link for this daily upload.
	SelfLink string `json:"selfLink,omitempty"`

	// WebPropertyId: Web property ID of the form UA-XXXXX-YY to which this
	// daily upload belongs.
	WebPropertyId string `json:"webPropertyId,omitempty"`
}

type DailyUploadParentLink struct {
	// Href: Link to the custom data source to which this daily upload
	// belongs.
	Href string `json:"href,omitempty"`

	// Type: Value is "analytics#customDataSource".
	Type string `json:"type,omitempty"`
}

type DailyUploadRecentChanges struct {
	// Change: The type of change: APPEND, RESET, or DELETE.
	Change string `json:"change,omitempty"`

	// Time: The time when the change occurred.
	Time string `json:"time,omitempty"`
}

type DailyUploadAppend struct {
	// AccountId: Account Id to which this daily upload append belongs.
	AccountId string `json:"accountId,omitempty"`

	// AppendNumber: Append number.
	AppendNumber int64 `json:"appendNumber,omitempty"`

	// CustomDataSourceId: Custom data source Id to which this daily upload
	// append belongs.
	CustomDataSourceId string `json:"customDataSourceId,omitempty"`

	// Date: Date associated with daily upload append.
	Date string `json:"date,omitempty"`

	// Kind: Resource type for Analytics daily upload append.
	Kind string `json:"kind,omitempty"`

	NextAppendLink string `json:"nextAppendLink,omitempty"`

	// WebPropertyId: Web property Id of the form UA-XXXXX-YY to which this
	// daily upload append belongs.
	WebPropertyId string `json:"webPropertyId,omitempty"`
}

type DailyUploads struct {
	// Items: A collection of daily uploads.
	Items []*DailyUpload `json:"items,omitempty"`

	// ItemsPerPage: The maximum number of resources the response can
	// contain, regardless of the actual number of resources returned. Its
	// value ranges from 1 to 1000 with a value of 1000 by default, or
	// otherwise specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Collection type. Value is analytics#dailyUploads.
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to next page for this daily upload collection.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Link to previous page for this daily upload collection.
	PreviousLink string `json:"previousLink,omitempty"`

	// StartIndex: The starting index of the resources, which is 1 by
	// default or otherwise specified by the start-index query parameter.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results for the query, regardless
	// of the number of results in the response.
	TotalResults int64 `json:"totalResults,omitempty"`

	// Username: Email ID of the authenticated user
	Username string `json:"username,omitempty"`
}

type EntityUserLink struct {
	// Entity: Entity for this link. It can be an account, a web property,
	// or a view (profile).
	Entity *EntityUserLinkEntity `json:"entity,omitempty"`

	// Id: Entity user link ID
	Id string `json:"id,omitempty"`

	// Kind: Resource type for entity user link.
	Kind string `json:"kind,omitempty"`

	// Permissions: Permissions the user has for this entity.
	Permissions *EntityUserLinkPermissions `json:"permissions,omitempty"`

	// SelfLink: Self link for this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// UserRef: User reference.
	UserRef *UserRef `json:"userRef,omitempty"`
}

type EntityUserLinkEntity struct {
	// AccountRef: Account for this link.
	AccountRef *AccountRef `json:"accountRef,omitempty"`

	// ProfileRef: View (Profile) for this link.
	ProfileRef *ProfileRef `json:"profileRef,omitempty"`

	// WebPropertyRef: Web property for this link.
	WebPropertyRef *WebPropertyRef `json:"webPropertyRef,omitempty"`
}

type EntityUserLinkPermissions struct {
	// Effective: Effective permissions represent all the permissions that a
	// user has for this entity. These include any implied permissions
	// (e.g., EDIT implies VIEW) or inherited permissions from the parent
	// entity. Effective permissions are read-only.
	Effective []string `json:"effective,omitempty"`

	// Local: Permissions that a user has been assigned at this very level.
	// Does not include any implied or inherited permissions. Local
	// permissions are modifiable.
	Local []string `json:"local,omitempty"`
}

type EntityUserLinks struct {
	// Items: A list of entity user links.
	Items []*EntityUserLink `json:"items,omitempty"`

	// ItemsPerPage: The maximum number of entries the response can contain,
	// regardless of the actual number of entries returned. Its value ranges
	// from 1 to 1000 with a value of 1000 by default, or otherwise
	// specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Collection type.
	Kind string `json:"kind,omitempty"`

	// NextLink: Next link for this account collection.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Previous link for this account collection.
	PreviousLink string `json:"previousLink,omitempty"`

	// StartIndex: The starting index of the entries, which is 1 by default
	// or otherwise specified by the start-index query parameter.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results for the query, regardless
	// of the number of results in the response.
	TotalResults int64 `json:"totalResults,omitempty"`
}

type Experiment struct {
	// AccountId: Account ID to which this experiment belongs. This field is
	// read-only.
	AccountId string `json:"accountId,omitempty"`

	// Created: Time the experiment was created. This field is read-only.
	Created string `json:"created,omitempty"`

	// Description: Notes about this experiment.
	Description string `json:"description,omitempty"`

	// EditableInGaUi: If true, the end user will be able to edit the
	// experiment via the Google Analytics user interface.
	EditableInGaUi bool `json:"editableInGaUi,omitempty"`

	// EndTime: The ending time of the experiment (the time the status
	// changed from RUNNING to ENDED). This field is present only if the
	// experiment has ended. This field is read-only.
	EndTime string `json:"endTime,omitempty"`

	// EqualWeighting: Boolean specifying whether to distribute traffic
	// evenly across all variations. If the value is False, content
	// experiments follows the default behavior of adjusting traffic
	// dynamically based on variation performance. Optional -- defaults to
	// False. This field may not be changed for an experiment whose status
	// is ENDED.
	EqualWeighting bool `json:"equalWeighting,omitempty"`

	// Id: Experiment ID. Required for patch and update. Disallowed for
	// create.
	Id string `json:"id,omitempty"`

	// InternalWebPropertyId: Internal ID for the web property to which this
	// experiment belongs. This field is read-only.
	InternalWebPropertyId string `json:"internalWebPropertyId,omitempty"`

	// Kind: Resource type for an Analytics experiment. This field is
	// read-only.
	Kind string `json:"kind,omitempty"`

	// MinimumExperimentLengthInDays: Specifies the minimum length of the
	// experiment. Can be changed for a running experiment. This field may
	// not be changed for an experiments whose status is ENDED.
	MinimumExperimentLengthInDays int64 `json:"minimumExperimentLengthInDays,omitempty"`

	// Name: Experiment name. This field may not be changed for an
	// experiment whose status is ENDED. This field is required when
	// creating an experiment.
	Name string `json:"name,omitempty"`

	// ObjectiveMetric: The metric that the experiment is optimizing. Valid
	// values: "ga:goal(n)Completions", "ga:adsenseAdsClicks",
	// "ga:adsenseAdsViewed", "ga:adsenseRevenue", "ga:bounces",
	// "ga:pageviews", "ga:timeOnSite", "ga:transactions",
	// "ga:transactionRevenue". This field is required if status is
	// "RUNNING" and servingFramework is one of "REDIRECT" or "API".
	ObjectiveMetric string `json:"objectiveMetric,omitempty"`

	// OptimizationType: Whether the objectiveMetric should be minimized or
	// maximized. Possible values: "MAXIMUM", "MINIMUM". Optional--defaults
	// to "MAXIMUM". Cannot be specified without objectiveMetric. Cannot be
	// modified when status is "RUNNING" or "ENDED".
	OptimizationType string `json:"optimizationType,omitempty"`

	// ParentLink: Parent link for an experiment. Points to the view
	// (profile) to which this experiment belongs.
	ParentLink *ExperimentParentLink `json:"parentLink,omitempty"`

	// ProfileId: View (Profile) ID to which this experiment belongs. This
	// field is read-only.
	ProfileId string `json:"profileId,omitempty"`

	// ReasonExperimentEnded: Why the experiment ended. Possible values:
	// "STOPPED_BY_USER", "WINNER_FOUND", "EXPERIMENT_EXPIRED",
	// "ENDED_WITH_NO_WINNER", "GOAL_OBJECTIVE_CHANGED".
	// "ENDED_WITH_NO_WINNER" means that the experiment didn't expire but no
	// winner was projected to be found. If the experiment status is changed
	// via the API to ENDED this field is set to STOPPED_BY_USER. This field
	// is read-only.
	ReasonExperimentEnded string `json:"reasonExperimentEnded,omitempty"`

	// RewriteVariationUrlsAsOriginal: Boolean specifying whether variations
	// URLS are rewritten to match those of the original. This field may not
	// be changed for an experiments whose status is ENDED.
	RewriteVariationUrlsAsOriginal bool `json:"rewriteVariationUrlsAsOriginal,omitempty"`

	// SelfLink: Link for this experiment. This field is read-only.
	SelfLink string `json:"selfLink,omitempty"`

	// ServingFramework: The framework used to serve the experiment
	// variations and evaluate the results. One of:
	// - REDIRECT: Google
	// Analytics redirects traffic to different variation pages, reports the
	// chosen variation and evaluates the results.
	// - API: Google Analytics
	// chooses and reports the variation to serve and evaluates the results;
	// the caller is responsible for serving the selected variation.
	// -
	// EXTERNAL: The variations will be served externally and the chosen
	// variation reported to Google Analytics. The caller is responsible for
	// serving the selected variation and evaluating the results.
	ServingFramework string `json:"servingFramework,omitempty"`

	// Snippet: The snippet of code to include on the control page(s). This
	// field is read-only.
	Snippet string `json:"snippet,omitempty"`

	// StartTime: The starting time of the experiment (the time the status
	// changed from READY_TO_RUN to RUNNING). This field is present only if
	// the experiment has started. This field is read-only.
	StartTime string `json:"startTime,omitempty"`

	// Status: Experiment status. Possible values: "DRAFT", "READY_TO_RUN",
	// "RUNNING", "ENDED". Experiments can be created in the "DRAFT",
	// "READY_TO_RUN" or "RUNNING" state. This field is required when
	// creating an experiment.
	Status string `json:"status,omitempty"`

	// TrafficCoverage: A floating-point number between 0 and 1. Specifies
	// the fraction of the traffic that participates in the experiment. Can
	// be changed for a running experiment. This field may not be changed
	// for an experiments whose status is ENDED.
	TrafficCoverage float64 `json:"trafficCoverage,omitempty"`

	// Updated: Time the experiment was last modified. This field is
	// read-only.
	Updated string `json:"updated,omitempty"`

	// Variations: Array of variations. The first variation in the array is
	// the original. The number of variations may not change once an
	// experiment is in the RUNNING state. At least two variations are
	// required before status can be set to RUNNING.
	Variations []*ExperimentVariations `json:"variations,omitempty"`

	// WebPropertyId: Web property ID to which this experiment belongs. The
	// web property ID is of the form UA-XXXXX-YY. This field is read-only.
	WebPropertyId string `json:"webPropertyId,omitempty"`

	// WinnerConfidenceLevel: A floating-point number between 0 and 1.
	// Specifies the necessary confidence level to choose a winner. This
	// field may not be changed for an experiments whose status is ENDED.
	WinnerConfidenceLevel float64 `json:"winnerConfidenceLevel,omitempty"`

	// WinnerFound: Boolean specifying whether a winner has been found for
	// this experiment. This field is read-only.
	WinnerFound bool `json:"winnerFound,omitempty"`
}

type ExperimentParentLink struct {
	// Href: Link to the view (profile) to which this experiment belongs.
	// This field is read-only.
	Href string `json:"href,omitempty"`

	// Type: Value is "analytics#profile". This field is read-only.
	Type string `json:"type,omitempty"`
}

type ExperimentVariations struct {
	// Name: The name of the variation. This field is required when creating
	// an experiment. This field may not be changed for an experiment whose
	// status is ENDED.
	Name string `json:"name,omitempty"`

	// Status: Status of the variation. Possible values: "ACTIVE",
	// "INACTIVE". INACTIVE variations are not served. This field may not be
	// changed for an experiment whose status is ENDED.
	Status string `json:"status,omitempty"`

	// Url: The URL of the variation. This field may not be changed for an
	// experiment whose status is RUNNING or ENDED.
	Url string `json:"url,omitempty"`

	// Weight: Weight that this variation should receive. Only present if
	// the experiment is running. This field is read-only.
	Weight float64 `json:"weight,omitempty"`

	// Won: True if the experiment has ended and this variation performed
	// (statistically) significantly better than the original. This field is
	// read-only.
	Won bool `json:"won,omitempty"`
}

type Experiments struct {
	// Items: A list of experiments.
	Items []*Experiment `json:"items,omitempty"`

	// ItemsPerPage: The maximum number of resources the response can
	// contain, regardless of the actual number of resources returned. Its
	// value ranges from 1 to 1000 with a value of 1000 by default, or
	// otherwise specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Collection type.
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to next page for this experiment collection.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Link to previous page for this experiment collection.
	PreviousLink string `json:"previousLink,omitempty"`

	// StartIndex: The starting index of the resources, which is 1 by
	// default or otherwise specified by the start-index query parameter.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results for the query, regardless
	// of the number of resources in the result.
	TotalResults int64 `json:"totalResults,omitempty"`

	// Username: Email ID of the authenticated user
	Username string `json:"username,omitempty"`
}

type GaData struct {
	// ColumnHeaders: Column headers that list dimension names followed by
	// the metric names. The order of dimensions and metrics is same as
	// specified in the request.
	ColumnHeaders []*GaDataColumnHeaders `json:"columnHeaders,omitempty"`

	// ContainsSampledData: Determines if Analytics data contains samples.
	ContainsSampledData bool `json:"containsSampledData,omitempty"`

	DataTable *GaDataDataTable `json:"dataTable,omitempty"`

	// Id: Unique ID for this data response.
	Id string `json:"id,omitempty"`

	// ItemsPerPage: The maximum number of rows the response can contain,
	// regardless of the actual number of rows returned. Its value ranges
	// from 1 to 10,000 with a value of 1000 by default, or otherwise
	// specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to next page for this Analytics data query.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Link to previous page for this Analytics data query.
	PreviousLink string `json:"previousLink,omitempty"`

	// ProfileInfo: Information for the view (profile), for which the
	// Analytics data was requested.
	ProfileInfo *GaDataProfileInfo `json:"profileInfo,omitempty"`

	// Query: Analytics data request query parameters.
	Query *GaDataQuery `json:"query,omitempty"`

	// Rows: Analytics data rows, where each row contains a list of
	// dimension values followed by the metric values. The order of
	// dimensions and metrics is same as specified in the request.
	Rows [][]string `json:"rows,omitempty"`

	// SampleSize: The number of samples used to calculate the result.
	SampleSize int64 `json:"sampleSize,omitempty,string"`

	// SampleSpace: Total size of the sample space from which the samples
	// were selected.
	SampleSpace int64 `json:"sampleSpace,omitempty,string"`

	// SelfLink: Link to this page.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalResults: The total number of rows for the query, regardless of
	// the number of rows in the response.
	TotalResults int64 `json:"totalResults,omitempty"`

	// TotalsForAllResults: Total values for the requested metrics over all
	// the results, not just the results returned in this response. The
	// order of the metric totals is same as the metric order specified in
	// the request.
	TotalsForAllResults map[string]string `json:"totalsForAllResults,omitempty"`
}

type GaDataColumnHeaders struct {
	// ColumnType: Column Type. Either DIMENSION or METRIC.
	ColumnType string `json:"columnType,omitempty"`

	// DataType: Data type. Dimension column headers have only STRING as the
	// data type. Metric column headers have data types for metric values
	// such as INTEGER, DOUBLE, CURRENCY etc.
	DataType string `json:"dataType,omitempty"`

	// Name: Column name.
	Name string `json:"name,omitempty"`
}

type GaDataDataTable struct {
	Cols []*GaDataDataTableCols `json:"cols,omitempty"`

	Rows []*GaDataDataTableRows `json:"rows,omitempty"`
}

type GaDataDataTableCols struct {
	Id string `json:"id,omitempty"`

	Label string `json:"label,omitempty"`

	Type string `json:"type,omitempty"`
}

type GaDataDataTableRows struct {
	C []*GaDataDataTableRowsC `json:"c,omitempty"`
}

type GaDataDataTableRowsC struct {
	V string `json:"v,omitempty"`
}

type GaDataProfileInfo struct {
	// AccountId: Account ID to which this view (profile) belongs.
	AccountId string `json:"accountId,omitempty"`

	// InternalWebPropertyId: Internal ID for the web property to which this
	// view (profile) belongs.
	InternalWebPropertyId string `json:"internalWebPropertyId,omitempty"`

	// ProfileId: View (Profile) ID.
	ProfileId string `json:"profileId,omitempty"`

	// ProfileName: View (Profile) name.
	ProfileName string `json:"profileName,omitempty"`

	// TableId: Table ID for view (profile).
	TableId string `json:"tableId,omitempty"`

	// WebPropertyId: Web Property ID to which this view (profile) belongs.
	WebPropertyId string `json:"webPropertyId,omitempty"`
}

type GaDataQuery struct {
	// Dimensions: List of analytics dimensions.
	Dimensions string `json:"dimensions,omitempty"`

	// EndDate: End date.
	EndDate string `json:"end-date,omitempty"`

	// Filters: Comma-separated list of dimension or metric filters.
	Filters string `json:"filters,omitempty"`

	// Ids: Unique table ID.
	Ids string `json:"ids,omitempty"`

	// MaxResults: Maximum results per page.
	MaxResults int64 `json:"max-results,omitempty"`

	// Metrics: List of analytics metrics.
	Metrics []string `json:"metrics,omitempty"`

	// SamplingLevel: Desired sampling level
	SamplingLevel string `json:"samplingLevel,omitempty"`

	// Segment: Analytics advanced segment.
	Segment string `json:"segment,omitempty"`

	// Sort: List of dimensions or metrics based on which Analytics data is
	// sorted.
	Sort []string `json:"sort,omitempty"`

	// StartDate: Start date.
	StartDate string `json:"start-date,omitempty"`

	// StartIndex: Start index.
	StartIndex int64 `json:"start-index,omitempty"`
}

type Goal struct {
	// AccountId: Account ID to which this goal belongs.
	AccountId string `json:"accountId,omitempty"`

	// Active: Determines whether this goal is active.
	Active bool `json:"active,omitempty"`

	// Created: Time this goal was created.
	Created string `json:"created,omitempty"`

	// EventDetails: Details for the goal of the type EVENT.
	EventDetails *GoalEventDetails `json:"eventDetails,omitempty"`

	// Id: Goal ID.
	Id string `json:"id,omitempty"`

	// InternalWebPropertyId: Internal ID for the web property to which this
	// goal belongs.
	InternalWebPropertyId string `json:"internalWebPropertyId,omitempty"`

	// Kind: Resource type for an Analytics goal.
	Kind string `json:"kind,omitempty"`

	// Name: Goal name.
	Name string `json:"name,omitempty"`

	// ParentLink: Parent link for a goal. Points to the view (profile) to
	// which this goal belongs.
	ParentLink *GoalParentLink `json:"parentLink,omitempty"`

	// ProfileId: View (Profile) ID to which this goal belongs.
	ProfileId string `json:"profileId,omitempty"`

	// SelfLink: Link for this goal.
	SelfLink string `json:"selfLink,omitempty"`

	// Type: Goal type. Possible values are URL_DESTINATION,
	// VISIT_TIME_ON_SITE, VISIT_NUM_PAGES, AND EVENT.
	Type string `json:"type,omitempty"`

	// Updated: Time this goal was last modified.
	Updated string `json:"updated,omitempty"`

	// UrlDestinationDetails: Details for the goal of the type
	// URL_DESTINATION.
	UrlDestinationDetails *GoalUrlDestinationDetails `json:"urlDestinationDetails,omitempty"`

	// Value: Goal value.
	Value float64 `json:"value,omitempty"`

	// VisitNumPagesDetails: Details for the goal of the type
	// VISIT_NUM_PAGES.
	VisitNumPagesDetails *GoalVisitNumPagesDetails `json:"visitNumPagesDetails,omitempty"`

	// VisitTimeOnSiteDetails: Details for the goal of the type
	// VISIT_TIME_ON_SITE.
	VisitTimeOnSiteDetails *GoalVisitTimeOnSiteDetails `json:"visitTimeOnSiteDetails,omitempty"`

	// WebPropertyId: Web property ID to which this goal belongs. The web
	// property ID is of the form UA-XXXXX-YY.
	WebPropertyId string `json:"webPropertyId,omitempty"`
}

type GoalEventDetails struct {
	// EventConditions: List of event conditions.
	EventConditions []*GoalEventDetailsEventConditions `json:"eventConditions,omitempty"`

	// UseEventValue: Determines if the event value should be used as the
	// value for this goal.
	UseEventValue bool `json:"useEventValue,omitempty"`
}

type GoalEventDetailsEventConditions struct {
	// ComparisonType: Type of comparison. Possible values are LESS_THAN,
	// GREATER_THAN or EQUAL.
	ComparisonType string `json:"comparisonType,omitempty"`

	// ComparisonValue: Value used for this comparison.
	ComparisonValue int64 `json:"comparisonValue,omitempty,string"`

	// Expression: Expression used for this match.
	Expression string `json:"expression,omitempty"`

	// MatchType: Type of the match to be performed. Possible values are
	// REGEXP, BEGINS_WITH, or EXACT.
	MatchType string `json:"matchType,omitempty"`

	// Type: Type of this event condition. Possible values are CATEGORY,
	// ACTION, LABEL, or VALUE.
	Type string `json:"type,omitempty"`
}

type GoalParentLink struct {
	// Href: Link to the view (profile) to which this goal belongs.
	Href string `json:"href,omitempty"`

	// Type: Value is "analytics#profile".
	Type string `json:"type,omitempty"`
}

type GoalUrlDestinationDetails struct {
	// CaseSensitive: Determines if the goal URL must exactly match the
	// capitalization of visited URLs.
	CaseSensitive bool `json:"caseSensitive,omitempty"`

	// FirstStepRequired: Determines if the first step in this goal is
	// required.
	FirstStepRequired bool `json:"firstStepRequired,omitempty"`

	// MatchType: Match type for the goal URL. Possible values are HEAD,
	// EXACT, or REGEX.
	MatchType string `json:"matchType,omitempty"`

	// Steps: List of steps configured for this goal funnel.
	Steps []*GoalUrlDestinationDetailsSteps `json:"steps,omitempty"`

	// Url: URL for this goal.
	Url string `json:"url,omitempty"`
}

type GoalUrlDestinationDetailsSteps struct {
	// Name: Step name.
	Name string `json:"name,omitempty"`

	// Number: Step number.
	Number int64 `json:"number,omitempty"`

	// Url: URL for this step.
	Url string `json:"url,omitempty"`
}

type GoalVisitNumPagesDetails struct {
	// ComparisonType: Type of comparison. Possible values are LESS_THAN,
	// GREATER_THAN, or EQUAL.
	ComparisonType string `json:"comparisonType,omitempty"`

	// ComparisonValue: Value used for this comparison.
	ComparisonValue int64 `json:"comparisonValue,omitempty,string"`
}

type GoalVisitTimeOnSiteDetails struct {
	// ComparisonType: Type of comparison. Possible values are LESS_THAN or
	// GREATER_THAN.
	ComparisonType string `json:"comparisonType,omitempty"`

	// ComparisonValue: Value used for this comparison.
	ComparisonValue int64 `json:"comparisonValue,omitempty,string"`
}

type Goals struct {
	// Items: A list of goals.
	Items []*Goal `json:"items,omitempty"`

	// ItemsPerPage: The maximum number of resources the response can
	// contain, regardless of the actual number of resources returned. Its
	// value ranges from 1 to 1000 with a value of 1000 by default, or
	// otherwise specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Collection type.
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to next page for this goal collection.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Link to previous page for this goal collection.
	PreviousLink string `json:"previousLink,omitempty"`

	// StartIndex: The starting index of the resources, which is 1 by
	// default or otherwise specified by the start-index query parameter.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results for the query, regardless
	// of the number of resources in the result.
	TotalResults int64 `json:"totalResults,omitempty"`

	// Username: Email ID of the authenticated user
	Username string `json:"username,omitempty"`
}

type McfData struct {
	// ColumnHeaders: Column headers that list dimension names followed by
	// the metric names. The order of dimensions and metrics is same as
	// specified in the request.
	ColumnHeaders []*McfDataColumnHeaders `json:"columnHeaders,omitempty"`

	// ContainsSampledData: Determines if the Analytics data contains
	// sampled data.
	ContainsSampledData bool `json:"containsSampledData,omitempty"`

	// Id: Unique ID for this data response.
	Id string `json:"id,omitempty"`

	// ItemsPerPage: The maximum number of rows the response can contain,
	// regardless of the actual number of rows returned. Its value ranges
	// from 1 to 10,000 with a value of 1000 by default, or otherwise
	// specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to next page for this Analytics data query.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Link to previous page for this Analytics data query.
	PreviousLink string `json:"previousLink,omitempty"`

	// ProfileInfo: Information for the view (profile), for which the
	// Analytics data was requested.
	ProfileInfo *McfDataProfileInfo `json:"profileInfo,omitempty"`

	// Query: Analytics data request query parameters.
	Query *McfDataQuery `json:"query,omitempty"`

	// Rows: Analytics data rows, where each row contains a list of
	// dimension values followed by the metric values. The order of
	// dimensions and metrics is same as specified in the request.
	Rows [][]*McfDataRowsItem `json:"rows,omitempty"`

	// SampleSize: The number of samples used to calculate the result.
	SampleSize int64 `json:"sampleSize,omitempty,string"`

	// SampleSpace: Total size of the sample space from which the samples
	// were selected.
	SampleSpace int64 `json:"sampleSpace,omitempty,string"`

	// SelfLink: Link to this page.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalResults: The total number of rows for the query, regardless of
	// the number of rows in the response.
	TotalResults int64 `json:"totalResults,omitempty"`

	// TotalsForAllResults: Total values for the requested metrics over all
	// the results, not just the results returned in this response. The
	// order of the metric totals is same as the metric order specified in
	// the request.
	TotalsForAllResults map[string]string `json:"totalsForAllResults,omitempty"`
}

type McfDataColumnHeaders struct {
	// ColumnType: Column Type. Either DIMENSION or METRIC.
	ColumnType string `json:"columnType,omitempty"`

	// DataType: Data type. Dimension and metric values data types such as
	// INTEGER, DOUBLE, CURRENCY, MCF_SEQUENCE etc.
	DataType string `json:"dataType,omitempty"`

	// Name: Column name.
	Name string `json:"name,omitempty"`
}

type McfDataProfileInfo struct {
	// AccountId: Account ID to which this view (profile) belongs.
	AccountId string `json:"accountId,omitempty"`

	// InternalWebPropertyId: Internal ID for the web property to which this
	// view (profile) belongs.
	InternalWebPropertyId string `json:"internalWebPropertyId,omitempty"`

	// ProfileId: View (Profile) ID.
	ProfileId string `json:"profileId,omitempty"`

	// ProfileName: View (Profile) name.
	ProfileName string `json:"profileName,omitempty"`

	// TableId: Table ID for view (profile).
	TableId string `json:"tableId,omitempty"`

	// WebPropertyId: Web Property ID to which this view (profile) belongs.
	WebPropertyId string `json:"webPropertyId,omitempty"`
}

type McfDataQuery struct {
	// Dimensions: List of analytics dimensions.
	Dimensions string `json:"dimensions,omitempty"`

	// EndDate: End date.
	EndDate string `json:"end-date,omitempty"`

	// Filters: Comma-separated list of dimension or metric filters.
	Filters string `json:"filters,omitempty"`

	// Ids: Unique table ID.
	Ids string `json:"ids,omitempty"`

	// MaxResults: Maximum results per page.
	MaxResults int64 `json:"max-results,omitempty"`

	// Metrics: List of analytics metrics.
	Metrics []string `json:"metrics,omitempty"`

	// SamplingLevel: Desired sampling level
	SamplingLevel string `json:"samplingLevel,omitempty"`

	// Segment: Analytics advanced segment.
	Segment string `json:"segment,omitempty"`

	// Sort: List of dimensions or metrics based on which Analytics data is
	// sorted.
	Sort []string `json:"sort,omitempty"`

	// StartDate: Start date.
	StartDate string `json:"start-date,omitempty"`

	// StartIndex: Start index.
	StartIndex int64 `json:"start-index,omitempty"`
}

type McfDataRowsItem struct {
	// ConversionPathValue: A conversion path dimension value, containing a
	// list of interactions with their attributes.
	ConversionPathValue []*McfDataRowsItemConversionPathValue `json:"conversionPathValue,omitempty"`

	// PrimitiveValue: A primitive dimension value. A primitive metric
	// value.
	PrimitiveValue string `json:"primitiveValue,omitempty"`
}

type McfDataRowsItemConversionPathValue struct {
	// InteractionType: Type of an interaction on conversion path. Such as
	// CLICK, IMPRESSION etc.
	InteractionType string `json:"interactionType,omitempty"`

	// NodeValue: Node value of an interaction on conversion path. Such as
	// source, medium etc.
	NodeValue string `json:"nodeValue,omitempty"`
}

type Profile struct {
	// AccountId: Account ID to which this view (profile) belongs.
	AccountId string `json:"accountId,omitempty"`

	// ChildLink: Child link for this view (profile). Points to the list of
	// goals for this view (profile).
	ChildLink *ProfileChildLink `json:"childLink,omitempty"`

	// Created: Time this view (profile) was created.
	Created string `json:"created,omitempty"`

	// Currency: The currency type associated with this view (profile). The
	// supported values are:
	// ARS, AUD, BGN, BRL, CAD, CHF, CNY, CZK, DKK,
	// EUR, GBP, HKD, HUF, IDR, INR, JPY, KRW, LTL, MXN, NOK, NZD, PHP, PLN,
	// RUB, SEK, THB, TRY, TWD, USD, VND, ZAR
	Currency string `json:"currency,omitempty"`

	// DefaultPage: Default page for this view (profile).
	DefaultPage string `json:"defaultPage,omitempty"`

	// ECommerceTracking: Indicates whether ecommerce tracking is enabled
	// for this view (profile).
	ECommerceTracking bool `json:"eCommerceTracking,omitempty"`

	// ExcludeQueryParameters: The query parameters that are excluded from
	// this view (profile).
	ExcludeQueryParameters string `json:"excludeQueryParameters,omitempty"`

	// Id: View (Profile) ID.
	Id string `json:"id,omitempty"`

	// InternalWebPropertyId: Internal ID for the web property to which this
	// view (profile) belongs.
	InternalWebPropertyId string `json:"internalWebPropertyId,omitempty"`

	// Kind: Resource type for Analytics view (profile).
	Kind string `json:"kind,omitempty"`

	// Name: Name of this view (profile).
	Name string `json:"name,omitempty"`

	// ParentLink: Parent link for this view (profile). Points to the web
	// property to which this view (profile) belongs.
	ParentLink *ProfileParentLink `json:"parentLink,omitempty"`

	// Permissions: Permissions the user has for this view (profile).
	Permissions *ProfilePermissions `json:"permissions,omitempty"`

	// SelfLink: Link for this view (profile).
	SelfLink string `json:"selfLink,omitempty"`

	// SiteSearchCategoryParameters: Site search category parameters for
	// this view (profile).
	SiteSearchCategoryParameters string `json:"siteSearchCategoryParameters,omitempty"`

	// SiteSearchQueryParameters: The site search query parameters for this
	// view (profile).
	SiteSearchQueryParameters string `json:"siteSearchQueryParameters,omitempty"`

	// StripSiteSearchCategoryParameters: Whether or not Analytics will
	// strip search category parameters from the URLs in your reports.
	StripSiteSearchCategoryParameters bool `json:"stripSiteSearchCategoryParameters,omitempty"`

	// StripSiteSearchQueryParameters: Whether or not Analytics will strip
	// search query parameters from the URLs in your reports.
	StripSiteSearchQueryParameters bool `json:"stripSiteSearchQueryParameters,omitempty"`

	// Timezone: Time zone for which this view (profile) has been
	// configured. Time zones are identified by strings from the TZ
	// database.
	Timezone string `json:"timezone,omitempty"`

	// Type: View (Profile) type. Supported types: WEB or APP.
	Type string `json:"type,omitempty"`

	// Updated: Time this view (profile) was last modified.
	Updated string `json:"updated,omitempty"`

	// WebPropertyId: Web property ID of the form UA-XXXXX-YY to which this
	// view (profile) belongs.
	WebPropertyId string `json:"webPropertyId,omitempty"`

	// WebsiteUrl: Website URL for this view (profile).
	WebsiteUrl string `json:"websiteUrl,omitempty"`
}

type ProfileChildLink struct {
	// Href: Link to the list of goals for this view (profile).
	Href string `json:"href,omitempty"`

	// Type: Value is "analytics#goals".
	Type string `json:"type,omitempty"`
}

type ProfileParentLink struct {
	// Href: Link to the web property to which this view (profile) belongs.
	Href string `json:"href,omitempty"`

	// Type: Value is "analytics#webproperty".
	Type string `json:"type,omitempty"`
}

type ProfilePermissions struct {
	// Effective: All the permissions that the user has for this view
	// (profile). These include any implied permissions (e.g., EDIT implies
	// VIEW) or inherited permissions from the parent web property.
	Effective []string `json:"effective,omitempty"`
}

type ProfileRef struct {
	// AccountId: Account ID to which this view (profile) belongs.
	AccountId string `json:"accountId,omitempty"`

	// Href: Link for this view (profile).
	Href string `json:"href,omitempty"`

	// Id: View (Profile) ID.
	Id string `json:"id,omitempty"`

	// InternalWebPropertyId: Internal ID for the web property to which this
	// view (profile) belongs.
	InternalWebPropertyId string `json:"internalWebPropertyId,omitempty"`

	// Kind: Analytics view (profile) reference.
	Kind string `json:"kind,omitempty"`

	// Name: Name of this view (profile).
	Name string `json:"name,omitempty"`

	// WebPropertyId: Web property ID of the form UA-XXXXX-YY to which this
	// view (profile) belongs.
	WebPropertyId string `json:"webPropertyId,omitempty"`
}

type ProfileSummary struct {
	// Id: View (profile) ID.
	Id string `json:"id,omitempty"`

	// Kind: Resource type for Analytics ProfileSummary.
	Kind string `json:"kind,omitempty"`

	// Name: View (profile) name.
	Name string `json:"name,omitempty"`

	// Type: View (Profile) type. Supported types: WEB or APP.
	Type string `json:"type,omitempty"`
}

type Profiles struct {
	// Items: A list of views (profiles).
	Items []*Profile `json:"items,omitempty"`

	// ItemsPerPage: The maximum number of resources the response can
	// contain, regardless of the actual number of resources returned. Its
	// value ranges from 1 to 1000 with a value of 1000 by default, or
	// otherwise specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Collection type.
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to next page for this view (profile) collection.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Link to previous page for this view (profile)
	// collection.
	PreviousLink string `json:"previousLink,omitempty"`

	// StartIndex: The starting index of the resources, which is 1 by
	// default or otherwise specified by the start-index query parameter.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results for the query, regardless
	// of the number of results in the response.
	TotalResults int64 `json:"totalResults,omitempty"`

	// Username: Email ID of the authenticated user
	Username string `json:"username,omitempty"`
}

type RealtimeData struct {
	// ColumnHeaders: Column headers that list dimension names followed by
	// the metric names. The order of dimensions and metrics is same as
	// specified in the request.
	ColumnHeaders []*RealtimeDataColumnHeaders `json:"columnHeaders,omitempty"`

	// Id: Unique ID for this data response.
	Id string `json:"id,omitempty"`

	// Kind: Resource type.
	Kind string `json:"kind,omitempty"`

	// ProfileInfo: Information for the view (profile), for which the real
	// time data was requested.
	ProfileInfo *RealtimeDataProfileInfo `json:"profileInfo,omitempty"`

	// Query: Real time data request query parameters.
	Query *RealtimeDataQuery `json:"query,omitempty"`

	// Rows: Real time data rows, where each row contains a list of
	// dimension values followed by the metric values. The order of
	// dimensions and metrics is same as specified in the request.
	Rows [][]string `json:"rows,omitempty"`

	// SelfLink: Link to this page.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalResults: The total number of rows for the query, regardless of
	// the number of rows in the response.
	TotalResults int64 `json:"totalResults,omitempty"`

	// TotalsForAllResults: Total values for the requested metrics over all
	// the results, not just the results returned in this response. The
	// order of the metric totals is same as the metric order specified in
	// the request.
	TotalsForAllResults map[string]string `json:"totalsForAllResults,omitempty"`
}

type RealtimeDataColumnHeaders struct {
	// ColumnType: Column Type. Either DIMENSION or METRIC.
	ColumnType string `json:"columnType,omitempty"`

	// DataType: Data type. Dimension column headers have only STRING as the
	// data type. Metric column headers have data types for metric values
	// such as INTEGER, DOUBLE, CURRENCY etc.
	DataType string `json:"dataType,omitempty"`

	// Name: Column name.
	Name string `json:"name,omitempty"`
}

type RealtimeDataProfileInfo struct {
	// AccountId: Account ID to which this view (profile) belongs.
	AccountId string `json:"accountId,omitempty"`

	// InternalWebPropertyId: Internal ID for the web property to which this
	// view (profile) belongs.
	InternalWebPropertyId string `json:"internalWebPropertyId,omitempty"`

	// ProfileId: View (Profile) ID.
	ProfileId string `json:"profileId,omitempty"`

	// ProfileName: View (Profile) name.
	ProfileName string `json:"profileName,omitempty"`

	// TableId: Table ID for view (profile).
	TableId string `json:"tableId,omitempty"`

	// WebPropertyId: Web Property ID to which this view (profile) belongs.
	WebPropertyId string `json:"webPropertyId,omitempty"`
}

type RealtimeDataQuery struct {
	// Dimensions: List of real time dimensions.
	Dimensions string `json:"dimensions,omitempty"`

	// Filters: Comma-separated list of dimension or metric filters.
	Filters string `json:"filters,omitempty"`

	// Ids: Unique table ID.
	Ids string `json:"ids,omitempty"`

	// MaxResults: Maximum results per page.
	MaxResults int64 `json:"max-results,omitempty"`

	// Metrics: List of real time metrics.
	Metrics []string `json:"metrics,omitempty"`

	// Sort: List of dimensions or metrics based on which real time data is
	// sorted.
	Sort []string `json:"sort,omitempty"`
}

type Segment struct {
	// Created: Time the segment was created.
	Created string `json:"created,omitempty"`

	// Definition: Segment definition.
	Definition string `json:"definition,omitempty"`

	// Id: Segment ID.
	Id string `json:"id,omitempty"`

	// Kind: Resource type for Analytics segment.
	Kind string `json:"kind,omitempty"`

	// Name: Segment name.
	Name string `json:"name,omitempty"`

	// SegmentId: Segment ID. Can be used with the 'segment' parameter in
	// Core Reporting API.
	SegmentId string `json:"segmentId,omitempty"`

	// SelfLink: Link for this segment.
	SelfLink string `json:"selfLink,omitempty"`

	// Type: Type for a segment. Possible values are "BUILT_IN" or "CUSTOM".
	Type string `json:"type,omitempty"`

	// Updated: Time the segment was last modified.
	Updated string `json:"updated,omitempty"`
}

type Segments struct {
	// Items: A list of segments.
	Items []*Segment `json:"items,omitempty"`

	// ItemsPerPage: The maximum number of resources the response can
	// contain, regardless of the actual number of resources returned. Its
	// value ranges from 1 to 1000 with a value of 1000 by default, or
	// otherwise specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Collection type for segments.
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to next page for this segment collection.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Link to previous page for this segment collection.
	PreviousLink string `json:"previousLink,omitempty"`

	// StartIndex: The starting index of the resources, which is 1 by
	// default or otherwise specified by the start-index query parameter.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results for the query, regardless
	// of the number of results in the response.
	TotalResults int64 `json:"totalResults,omitempty"`

	// Username: Email ID of the authenticated user
	Username string `json:"username,omitempty"`
}

type Upload struct {
	// AccountId: Account Id to which this upload belongs.
	AccountId int64 `json:"accountId,omitempty,string"`

	// CustomDataSourceId: Custom data source Id to which this data import
	// belongs.
	CustomDataSourceId string `json:"customDataSourceId,omitempty"`

	// Errors: Data import errors collection.
	Errors []string `json:"errors,omitempty"`

	// Id: A unique ID for this upload.
	Id string `json:"id,omitempty"`

	// Kind: Resource type for Analytics upload.
	Kind string `json:"kind,omitempty"`

	// Status: Upload status. Possible values: PENDING, COMPLETED, FAILED,
	// DELETING, DELETED.
	Status string `json:"status,omitempty"`
}

type Uploads struct {
	// Items: A list of uploads.
	Items []*Upload `json:"items,omitempty"`

	// ItemsPerPage: The maximum number of resources the response can
	// contain, regardless of the actual number of resources returned. Its
	// value ranges from 1 to 1000 with a value of 1000 by default, or
	// otherwise specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Collection type.
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to next page for this upload collection.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Link to previous page for this upload collection.
	PreviousLink string `json:"previousLink,omitempty"`

	// StartIndex: The starting index of the resources, which is 1 by
	// default or otherwise specified by the start-index query parameter.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results for the query, regardless
	// of the number of resources in the result.
	TotalResults int64 `json:"totalResults,omitempty"`
}

type UserRef struct {
	// Email: Email ID of this user.
	Email string `json:"email,omitempty"`

	// Id: User ID.
	Id string `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`
}

type WebPropertyRef struct {
	// AccountId: Account ID to which this web property belongs.
	AccountId string `json:"accountId,omitempty"`

	// Href: Link for this web property.
	Href string `json:"href,omitempty"`

	// Id: Web property ID of the form UA-XXXXX-YY.
	Id string `json:"id,omitempty"`

	// InternalWebPropertyId: Internal ID for this web property.
	InternalWebPropertyId string `json:"internalWebPropertyId,omitempty"`

	// Kind: Analytics web property reference.
	Kind string `json:"kind,omitempty"`

	// Name: Name of this web property.
	Name string `json:"name,omitempty"`
}

type WebPropertySummary struct {
	// Id: Web property ID of the form UA-XXXXX-YY.
	Id string `json:"id,omitempty"`

	// InternalWebPropertyId: Internal ID for this web property.
	InternalWebPropertyId string `json:"internalWebPropertyId,omitempty"`

	// Kind: Resource type for Analytics WebPropertySummary.
	Kind string `json:"kind,omitempty"`

	// Level: Level for this web property. Possible values are STANDARD or
	// PREMIUM.
	Level string `json:"level,omitempty"`

	// Name: Web property name.
	Name string `json:"name,omitempty"`

	// Profiles: List of profiles under this web property.
	Profiles []*ProfileSummary `json:"profiles,omitempty"`

	// WebsiteUrl: Website url for this web property.
	WebsiteUrl string `json:"websiteUrl,omitempty"`
}

type Webproperties struct {
	// Items: A list of web properties.
	Items []*Webproperty `json:"items,omitempty"`

	// ItemsPerPage: The maximum number of resources the response can
	// contain, regardless of the actual number of resources returned. Its
	// value ranges from 1 to 1000 with a value of 1000 by default, or
	// otherwise specified by the max-results query parameter.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: Collection type.
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to next page for this web property collection.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Link to previous page for this web property collection.
	PreviousLink string `json:"previousLink,omitempty"`

	// StartIndex: The starting index of the resources, which is 1 by
	// default or otherwise specified by the start-index query parameter.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results for the query, regardless
	// of the number of results in the response.
	TotalResults int64 `json:"totalResults,omitempty"`

	// Username: Email ID of the authenticated user
	Username string `json:"username,omitempty"`
}

type Webproperty struct {
	// AccountId: Account ID to which this web property belongs.
	AccountId string `json:"accountId,omitempty"`

	// ChildLink: Child link for this web property. Points to the list of
	// views (profiles) for this web property.
	ChildLink *WebpropertyChildLink `json:"childLink,omitempty"`

	// Created: Time this web property was created.
	Created string `json:"created,omitempty"`

	// DefaultProfileId: Default view (profile) ID.
	DefaultProfileId int64 `json:"defaultProfileId,omitempty,string"`

	// Id: Web property ID of the form UA-XXXXX-YY.
	Id string `json:"id,omitempty"`

	// IndustryVertical: The industry vertical/category selected for this
	// web property.
	IndustryVertical string `json:"industryVertical,omitempty"`

	// InternalWebPropertyId: Internal ID for this web property.
	InternalWebPropertyId string `json:"internalWebPropertyId,omitempty"`

	// Kind: Resource type for Analytics WebProperty.
	Kind string `json:"kind,omitempty"`

	// Level: Level for this web property. Possible values are STANDARD or
	// PREMIUM.
	Level string `json:"level,omitempty"`

	// Name: Name of this web property.
	Name string `json:"name,omitempty"`

	// ParentLink: Parent link for this web property. Points to the account
	// to which this web property belongs.
	ParentLink *WebpropertyParentLink `json:"parentLink,omitempty"`

	// Permissions: Permissions the user has for this web property.
	Permissions *WebpropertyPermissions `json:"permissions,omitempty"`

	// ProfileCount: View (Profile) count for this web property.
	ProfileCount int64 `json:"profileCount,omitempty"`

	// SelfLink: Link for this web property.
	SelfLink string `json:"selfLink,omitempty"`

	// Updated: Time this web property was last modified.
	Updated string `json:"updated,omitempty"`

	// WebsiteUrl: Website url for this web property.
	WebsiteUrl string `json:"websiteUrl,omitempty"`
}

type WebpropertyChildLink struct {
	// Href: Link to the list of views (profiles) for this web property.
	Href string `json:"href,omitempty"`

	// Type: Type of the parent link. Its value is "analytics#profiles".
	Type string `json:"type,omitempty"`
}

type WebpropertyParentLink struct {
	// Href: Link to the account for this web property.
	Href string `json:"href,omitempty"`

	// Type: Type of the parent link. Its value is "analytics#account".
	Type string `json:"type,omitempty"`
}

type WebpropertyPermissions struct {
	// Effective: All the permissions that the user has for this web
	// property. These include any implied permissions (e.g., EDIT implies
	// VIEW) or inherited permissions from the parent account.
	Effective []string `json:"effective,omitempty"`
}

// method id "analytics.data.ga.get":

type DataGaGetCall struct {
	s         *Service
	ids       string
	startDate string
	endDate   string
	metrics   string
	opt_      map[string]interface{}
}

// Get: Returns Analytics data for a view (profile).
func (r *DataGaService) Get(ids string, startDate string, endDate string, metrics string) *DataGaGetCall {
	c := &DataGaGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.ids = ids
	c.startDate = startDate
	c.endDate = endDate
	c.metrics = metrics
	return c
}

// Dimensions sets the optional parameter "dimensions": A
// comma-separated list of Analytics dimensions. E.g.,
// 'ga:browser,ga:city'.
func (c *DataGaGetCall) Dimensions(dimensions string) *DataGaGetCall {
	c.opt_["dimensions"] = dimensions
	return c
}

// Filters sets the optional parameter "filters": A comma-separated list
// of dimension or metric filters to be applied to Analytics data.
func (c *DataGaGetCall) Filters(filters string) *DataGaGetCall {
	c.opt_["filters"] = filters
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of entries to include in this feed.
func (c *DataGaGetCall) MaxResults(maxResults int64) *DataGaGetCall {
	c.opt_["max-results"] = maxResults
	return c
}

// Output sets the optional parameter "output": The selected format for
// the response. Default format is JSON.
func (c *DataGaGetCall) Output(output string) *DataGaGetCall {
	c.opt_["output"] = output
	return c
}

// SamplingLevel sets the optional parameter "samplingLevel": The
// desired sampling level.
func (c *DataGaGetCall) SamplingLevel(samplingLevel string) *DataGaGetCall {
	c.opt_["samplingLevel"] = samplingLevel
	return c
}

// Segment sets the optional parameter "segment": An Analytics segment
// to be applied to data.
func (c *DataGaGetCall) Segment(segment string) *DataGaGetCall {
	c.opt_["segment"] = segment
	return c
}

// Sort sets the optional parameter "sort": A comma-separated list of
// dimensions or metrics that determine the sort order for Analytics
// data.
func (c *DataGaGetCall) Sort(sort string) *DataGaGetCall {
	c.opt_["sort"] = sort
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first entity to retrieve. Use this parameter as a pagination
// mechanism along with the max-results parameter.
func (c *DataGaGetCall) StartIndex(startIndex int64) *DataGaGetCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *DataGaGetCall) Do() (*GaData, error) {
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
	if v, ok := c.opt_["output"]; ok {
		params.Set("output", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["samplingLevel"]; ok {
		params.Set("samplingLevel", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["segment"]; ok {
		params.Set("segment", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sort"]; ok {
		params.Set("sort", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "data/ga")
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
	ret := new(GaData)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns Analytics data for a view (profile).",
	//   "httpMethod": "GET",
	//   "id": "analytics.data.ga.get",
	//   "parameterOrder": [
	//     "ids",
	//     "start-date",
	//     "end-date",
	//     "metrics"
	//   ],
	//   "parameters": {
	//     "dimensions": {
	//       "description": "A comma-separated list of Analytics dimensions. E.g., 'ga:browser,ga:city'.",
	//       "location": "query",
	//       "pattern": "(ga:.+)?",
	//       "type": "string"
	//     },
	//     "end-date": {
	//       "description": "End date for fetching Analytics data. Request can should specify an end date formatted as YYYY-MM-DD, or as a relative date (e.g., today, yesterday, or 7daysAgo). The default value is yesterday.",
	//       "location": "query",
	//       "pattern": "[0-9]{4}-[0-9]{2}-[0-9]{2}|today|yesterday|[0-9]+(daysAgo)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filters": {
	//       "description": "A comma-separated list of dimension or metric filters to be applied to Analytics data.",
	//       "location": "query",
	//       "pattern": "ga:.+",
	//       "type": "string"
	//     },
	//     "ids": {
	//       "description": "Unique table ID for retrieving Analytics data. Table ID is of the form ga:XXXX, where XXXX is the Analytics view (profile) ID.",
	//       "location": "query",
	//       "pattern": "ga:[0-9]+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of entries to include in this feed.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "metrics": {
	//       "description": "A comma-separated list of Analytics metrics. E.g., 'ga:visits,ga:pageviews'. At least one metric must be specified.",
	//       "location": "query",
	//       "pattern": "ga:.+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "output": {
	//       "description": "The selected format for the response. Default format is JSON.",
	//       "enum": [
	//         "dataTable",
	//         "json"
	//       ],
	//       "enumDescriptions": [
	//         "Returns the response in Google Charts Data Table format. This is useful in creating visualization using Google Charts.",
	//         "Returns the response in standard JSON format."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "samplingLevel": {
	//       "description": "The desired sampling level.",
	//       "enum": [
	//         "DEFAULT",
	//         "FASTER",
	//         "HIGHER_PRECISION"
	//       ],
	//       "enumDescriptions": [
	//         "Returns response with a sample size that balances speed and accuracy.",
	//         "Returns a fast response with a smaller sample size.",
	//         "Returns a more accurate response using a large sample size, but this may result in the response being slower."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "segment": {
	//       "description": "An Analytics segment to be applied to data.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sort": {
	//       "description": "A comma-separated list of dimensions or metrics that determine the sort order for Analytics data.",
	//       "location": "query",
	//       "pattern": "(-)?ga:.+",
	//       "type": "string"
	//     },
	//     "start-date": {
	//       "description": "Start date for fetching Analytics data. Requests can specify a start date formatted as YYYY-MM-DD, or as a relative date (e.g., today, yesterday, or 7daysAgo). The default value is 7daysAgo.",
	//       "location": "query",
	//       "pattern": "[0-9]{4}-[0-9]{2}-[0-9]{2}|today|yesterday|[0-9]+(daysAgo)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "start-index": {
	//       "description": "An index of the first entity to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "data/ga",
	//   "response": {
	//     "$ref": "GaData"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.data.mcf.get":

type DataMcfGetCall struct {
	s         *Service
	ids       string
	startDate string
	endDate   string
	metrics   string
	opt_      map[string]interface{}
}

// Get: Returns Analytics Multi-Channel Funnels data for a view
// (profile).
func (r *DataMcfService) Get(ids string, startDate string, endDate string, metrics string) *DataMcfGetCall {
	c := &DataMcfGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.ids = ids
	c.startDate = startDate
	c.endDate = endDate
	c.metrics = metrics
	return c
}

// Dimensions sets the optional parameter "dimensions": A
// comma-separated list of Multi-Channel Funnels dimensions. E.g.,
// 'mcf:source,mcf:medium'.
func (c *DataMcfGetCall) Dimensions(dimensions string) *DataMcfGetCall {
	c.opt_["dimensions"] = dimensions
	return c
}

// Filters sets the optional parameter "filters": A comma-separated list
// of dimension or metric filters to be applied to the Analytics data.
func (c *DataMcfGetCall) Filters(filters string) *DataMcfGetCall {
	c.opt_["filters"] = filters
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of entries to include in this feed.
func (c *DataMcfGetCall) MaxResults(maxResults int64) *DataMcfGetCall {
	c.opt_["max-results"] = maxResults
	return c
}

// SamplingLevel sets the optional parameter "samplingLevel": The
// desired sampling level.
func (c *DataMcfGetCall) SamplingLevel(samplingLevel string) *DataMcfGetCall {
	c.opt_["samplingLevel"] = samplingLevel
	return c
}

// Sort sets the optional parameter "sort": A comma-separated list of
// dimensions or metrics that determine the sort order for the Analytics
// data.
func (c *DataMcfGetCall) Sort(sort string) *DataMcfGetCall {
	c.opt_["sort"] = sort
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first entity to retrieve. Use this parameter as a pagination
// mechanism along with the max-results parameter.
func (c *DataMcfGetCall) StartIndex(startIndex int64) *DataMcfGetCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *DataMcfGetCall) Do() (*McfData, error) {
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
	if v, ok := c.opt_["samplingLevel"]; ok {
		params.Set("samplingLevel", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sort"]; ok {
		params.Set("sort", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "data/mcf")
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
	ret := new(McfData)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns Analytics Multi-Channel Funnels data for a view (profile).",
	//   "httpMethod": "GET",
	//   "id": "analytics.data.mcf.get",
	//   "parameterOrder": [
	//     "ids",
	//     "start-date",
	//     "end-date",
	//     "metrics"
	//   ],
	//   "parameters": {
	//     "dimensions": {
	//       "description": "A comma-separated list of Multi-Channel Funnels dimensions. E.g., 'mcf:source,mcf:medium'.",
	//       "location": "query",
	//       "pattern": "(mcf:.+)?",
	//       "type": "string"
	//     },
	//     "end-date": {
	//       "description": "End date for fetching Analytics data. Requests can specify a start date formatted as YYYY-MM-DD, or as a relative date (e.g., today, yesterday, or 7daysAgo). The default value is 7daysAgo.",
	//       "location": "query",
	//       "pattern": "[0-9]{4}-[0-9]{2}-[0-9]{2}|today|yesterday|[0-9]+(daysAgo)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "filters": {
	//       "description": "A comma-separated list of dimension or metric filters to be applied to the Analytics data.",
	//       "location": "query",
	//       "pattern": "mcf:.+",
	//       "type": "string"
	//     },
	//     "ids": {
	//       "description": "Unique table ID for retrieving Analytics data. Table ID is of the form ga:XXXX, where XXXX is the Analytics view (profile) ID.",
	//       "location": "query",
	//       "pattern": "ga:[0-9]+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of entries to include in this feed.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "metrics": {
	//       "description": "A comma-separated list of Multi-Channel Funnels metrics. E.g., 'mcf:totalConversions,mcf:totalConversionValue'. At least one metric must be specified.",
	//       "location": "query",
	//       "pattern": "mcf:.+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "samplingLevel": {
	//       "description": "The desired sampling level.",
	//       "enum": [
	//         "DEFAULT",
	//         "FASTER",
	//         "HIGHER_PRECISION"
	//       ],
	//       "enumDescriptions": [
	//         "Returns response with a sample size that balances speed and accuracy.",
	//         "Returns a fast response with a smaller sample size.",
	//         "Returns a more accurate response using a large sample size, but this may result in the response being slower."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sort": {
	//       "description": "A comma-separated list of dimensions or metrics that determine the sort order for the Analytics data.",
	//       "location": "query",
	//       "pattern": "(-)?mcf:.+",
	//       "type": "string"
	//     },
	//     "start-date": {
	//       "description": "Start date for fetching Analytics data. Requests can specify a start date formatted as YYYY-MM-DD, or as a relative date (e.g., today, yesterday, or 7daysAgo). The default value is 7daysAgo.",
	//       "location": "query",
	//       "pattern": "[0-9]{4}-[0-9]{2}-[0-9]{2}|today|yesterday|[0-9]+(daysAgo)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "start-index": {
	//       "description": "An index of the first entity to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "data/mcf",
	//   "response": {
	//     "$ref": "McfData"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.data.realtime.get":

type DataRealtimeGetCall struct {
	s       *Service
	ids     string
	metrics string
	opt_    map[string]interface{}
}

// Get: Returns real time data for a view (profile).
func (r *DataRealtimeService) Get(ids string, metrics string) *DataRealtimeGetCall {
	c := &DataRealtimeGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.ids = ids
	c.metrics = metrics
	return c
}

// Dimensions sets the optional parameter "dimensions": A
// comma-separated list of real time dimensions. E.g.,
// 'rt:medium,rt:city'.
func (c *DataRealtimeGetCall) Dimensions(dimensions string) *DataRealtimeGetCall {
	c.opt_["dimensions"] = dimensions
	return c
}

// Filters sets the optional parameter "filters": A comma-separated list
// of dimension or metric filters to be applied to real time data.
func (c *DataRealtimeGetCall) Filters(filters string) *DataRealtimeGetCall {
	c.opt_["filters"] = filters
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of entries to include in this feed.
func (c *DataRealtimeGetCall) MaxResults(maxResults int64) *DataRealtimeGetCall {
	c.opt_["max-results"] = maxResults
	return c
}

// Sort sets the optional parameter "sort": A comma-separated list of
// dimensions or metrics that determine the sort order for real time
// data.
func (c *DataRealtimeGetCall) Sort(sort string) *DataRealtimeGetCall {
	c.opt_["sort"] = sort
	return c
}

func (c *DataRealtimeGetCall) Do() (*RealtimeData, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("ids", fmt.Sprintf("%v", c.ids))
	params.Set("metrics", fmt.Sprintf("%v", c.metrics))
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "data/realtime")
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
	ret := new(RealtimeData)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns real time data for a view (profile).",
	//   "httpMethod": "GET",
	//   "id": "analytics.data.realtime.get",
	//   "parameterOrder": [
	//     "ids",
	//     "metrics"
	//   ],
	//   "parameters": {
	//     "dimensions": {
	//       "description": "A comma-separated list of real time dimensions. E.g., 'rt:medium,rt:city'.",
	//       "location": "query",
	//       "pattern": "(ga:.+)|(rt:.+)",
	//       "type": "string"
	//     },
	//     "filters": {
	//       "description": "A comma-separated list of dimension or metric filters to be applied to real time data.",
	//       "location": "query",
	//       "pattern": "(ga:.+)|(rt:.+)",
	//       "type": "string"
	//     },
	//     "ids": {
	//       "description": "Unique table ID for retrieving real time data. Table ID is of the form ga:XXXX, where XXXX is the Analytics view (profile) ID.",
	//       "location": "query",
	//       "pattern": "ga:[0-9]+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of entries to include in this feed.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "metrics": {
	//       "description": "A comma-separated list of real time metrics. E.g., 'rt:activeVisitors'. At least one metric must be specified.",
	//       "location": "query",
	//       "pattern": "(ga:.+)|(rt:.+)",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "sort": {
	//       "description": "A comma-separated list of dimensions or metrics that determine the sort order for real time data.",
	//       "location": "query",
	//       "pattern": "(-)?((ga:.+)|(rt:.+))",
	//       "type": "string"
	//     }
	//   },
	//   "path": "data/realtime",
	//   "response": {
	//     "$ref": "RealtimeData"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.accountSummaries.list":

type ManagementAccountSummariesListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Lists account summaries (lightweight tree comprised of
// accounts/properties/profiles) to which the user has access.
func (r *ManagementAccountSummariesService) List() *ManagementAccountSummariesListCall {
	c := &ManagementAccountSummariesListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of filters to include in this response.
func (c *ManagementAccountSummariesListCall) MaxResults(maxResults int64) *ManagementAccountSummariesListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first entity to retrieve. Use this parameter as a pagination
// mechanism along with the max-results parameter.
func (c *ManagementAccountSummariesListCall) StartIndex(startIndex int64) *ManagementAccountSummariesListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementAccountSummariesListCall) Do() (*AccountSummaries, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accountSummaries")
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
	ret := new(AccountSummaries)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists account summaries (lightweight tree comprised of accounts/properties/profiles) to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.accountSummaries.list",
	//   "parameters": {
	//     "max-results": {
	//       "description": "The maximum number of filters to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "start-index": {
	//       "description": "An index of the first entity to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "management/accountSummaries",
	//   "response": {
	//     "$ref": "AccountSummaries"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.accountUserLinks.delete":

type ManagementAccountUserLinksDeleteCall struct {
	s         *Service
	accountId string
	linkId    string
	opt_      map[string]interface{}
}

// Delete: Removes a user from the given account.
func (r *ManagementAccountUserLinksService) Delete(accountId string, linkId string) *ManagementAccountUserLinksDeleteCall {
	c := &ManagementAccountUserLinksDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.linkId = linkId
	return c
}

func (c *ManagementAccountUserLinksDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/entityUserLinks/{linkId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{linkId}", url.QueryEscape(c.linkId), 1)
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
	//   "description": "Removes a user from the given account.",
	//   "httpMethod": "DELETE",
	//   "id": "analytics.management.accountUserLinks.delete",
	//   "parameterOrder": [
	//     "accountId",
	//     "linkId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to delete the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "linkId": {
	//       "description": "Link ID to delete the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/entityUserLinks/{linkId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.management.accountUserLinks.insert":

type ManagementAccountUserLinksInsertCall struct {
	s              *Service
	accountId      string
	entityuserlink *EntityUserLink
	opt_           map[string]interface{}
}

// Insert: Adds a new user to the given account.
func (r *ManagementAccountUserLinksService) Insert(accountId string, entityuserlink *EntityUserLink) *ManagementAccountUserLinksInsertCall {
	c := &ManagementAccountUserLinksInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.entityuserlink = entityuserlink
	return c
}

func (c *ManagementAccountUserLinksInsertCall) Do() (*EntityUserLink, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.entityuserlink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/entityUserLinks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
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
	ret := new(EntityUserLink)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a new user to the given account.",
	//   "httpMethod": "POST",
	//   "id": "analytics.management.accountUserLinks.insert",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to create the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/entityUserLinks",
	//   "request": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "response": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.management.accountUserLinks.list":

type ManagementAccountUserLinksListCall struct {
	s         *Service
	accountId string
	opt_      map[string]interface{}
}

// List: Lists account-user links for a given account.
func (r *ManagementAccountUserLinksService) List(accountId string) *ManagementAccountUserLinksListCall {
	c := &ManagementAccountUserLinksListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of account-user links to include in this response.
func (c *ManagementAccountUserLinksListCall) MaxResults(maxResults int64) *ManagementAccountUserLinksListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first account-user link to retrieve. Use this parameter as a
// pagination mechanism along with the max-results parameter.
func (c *ManagementAccountUserLinksListCall) StartIndex(startIndex int64) *ManagementAccountUserLinksListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementAccountUserLinksListCall) Do() (*EntityUserLinks, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/entityUserLinks")
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
	ret := new(EntityUserLinks)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists account-user links for a given account.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.accountUserLinks.list",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to retrieve the user links for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of account-user links to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "start-index": {
	//       "description": "An index of the first account-user link to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/entityUserLinks",
	//   "response": {
	//     "$ref": "EntityUserLinks"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.management.accountUserLinks.update":

type ManagementAccountUserLinksUpdateCall struct {
	s              *Service
	accountId      string
	linkId         string
	entityuserlink *EntityUserLink
	opt_           map[string]interface{}
}

// Update: Updates permissions for an existing user on the given
// account.
func (r *ManagementAccountUserLinksService) Update(accountId string, linkId string, entityuserlink *EntityUserLink) *ManagementAccountUserLinksUpdateCall {
	c := &ManagementAccountUserLinksUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.linkId = linkId
	c.entityuserlink = entityuserlink
	return c
}

func (c *ManagementAccountUserLinksUpdateCall) Do() (*EntityUserLink, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.entityuserlink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/entityUserLinks/{linkId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{linkId}", url.QueryEscape(c.linkId), 1)
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
	ret := new(EntityUserLink)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates permissions for an existing user on the given account.",
	//   "httpMethod": "PUT",
	//   "id": "analytics.management.accountUserLinks.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "linkId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to update the account-user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "linkId": {
	//       "description": "Link ID to update the account-user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/entityUserLinks/{linkId}",
	//   "request": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "response": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.management.accounts.list":

type ManagementAccountsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Lists all accounts to which the user has access.
func (r *ManagementAccountsService) List() *ManagementAccountsListCall {
	c := &ManagementAccountsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of accounts to include in this response.
func (c *ManagementAccountsListCall) MaxResults(maxResults int64) *ManagementAccountsListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first account to retrieve. Use this parameter as a pagination
// mechanism along with the max-results parameter.
func (c *ManagementAccountsListCall) StartIndex(startIndex int64) *ManagementAccountsListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementAccountsListCall) Do() (*Accounts, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts")
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
	//   "description": "Lists all accounts to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.accounts.list",
	//   "parameters": {
	//     "max-results": {
	//       "description": "The maximum number of accounts to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "start-index": {
	//       "description": "An index of the first account to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "management/accounts",
	//   "response": {
	//     "$ref": "Accounts"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.customDataSources.list":

type ManagementCustomDataSourcesListCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	opt_          map[string]interface{}
}

// List: List custom data sources to which the user has access.
func (r *ManagementCustomDataSourcesService) List(accountId string, webPropertyId string) *ManagementCustomDataSourcesListCall {
	c := &ManagementCustomDataSourcesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of custom data sources to include in this response.
func (c *ManagementCustomDataSourcesListCall) MaxResults(maxResults int64) *ManagementCustomDataSourcesListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": A 1-based index
// of the first custom data source to retrieve. Use this parameter as a
// pagination mechanism along with the max-results parameter.
func (c *ManagementCustomDataSourcesListCall) StartIndex(startIndex int64) *ManagementCustomDataSourcesListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementCustomDataSourcesListCall) Do() (*CustomDataSources, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
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
	ret := new(CustomDataSources)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List custom data sources to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.customDataSources.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account Id for the custom data sources to retrieve.",
	//       "location": "path",
	//       "pattern": "\\d+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of custom data sources to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "start-index": {
	//       "description": "A 1-based index of the first custom data source to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property Id for the custom data sources to retrieve.",
	//       "location": "path",
	//       "pattern": "UA-(\\d+)-(\\d+)",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources",
	//   "response": {
	//     "$ref": "CustomDataSources"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.dailyUploads.delete":

type ManagementDailyUploadsDeleteCall struct {
	s                  *Service
	accountId          string
	webPropertyId      string
	customDataSourceId string
	date               string
	type_              string
	opt_               map[string]interface{}
}

// Delete: Delete uploaded data for the given date.
func (r *ManagementDailyUploadsService) Delete(accountId string, webPropertyId string, customDataSourceId string, date string, type_ string) *ManagementDailyUploadsDeleteCall {
	c := &ManagementDailyUploadsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.customDataSourceId = customDataSourceId
	c.date = date
	c.type_ = type_
	return c
}

func (c *ManagementDailyUploadsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("type", fmt.Sprintf("%v", c.type_))
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/dailyUploads/{date}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customDataSourceId}", url.QueryEscape(c.customDataSourceId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{date}", url.QueryEscape(c.date), 1)
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
	//   "description": "Delete uploaded data for the given date.",
	//   "httpMethod": "DELETE",
	//   "id": "analytics.management.dailyUploads.delete",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "customDataSourceId",
	//     "date",
	//     "type"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account Id associated with daily upload delete.",
	//       "location": "path",
	//       "pattern": "[0-9]+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customDataSourceId": {
	//       "description": "Custom data source Id associated with daily upload delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "date": {
	//       "description": "Date for which data is to be deleted. Date should be formatted as YYYY-MM-DD.",
	//       "location": "path",
	//       "pattern": "[0-9]{4}-[0-9]{2}-[0-9]{2}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "type": {
	//       "description": "Type of data for this delete.",
	//       "enum": [
	//         "cost"
	//       ],
	//       "enumDescriptions": [
	//         "Value for specifying cost data upload."
	//       ],
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property Id associated with daily upload delete.",
	//       "location": "path",
	//       "pattern": "UA-[0-9]+-[0-9]+",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/dailyUploads/{date}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.dailyUploads.list":

type ManagementDailyUploadsListCall struct {
	s                  *Service
	accountId          string
	webPropertyId      string
	customDataSourceId string
	startDate          string
	endDate            string
	opt_               map[string]interface{}
}

// List: List daily uploads to which the user has access.
func (r *ManagementDailyUploadsService) List(accountId string, webPropertyId string, customDataSourceId string, startDate string, endDate string) *ManagementDailyUploadsListCall {
	c := &ManagementDailyUploadsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.customDataSourceId = customDataSourceId
	c.startDate = startDate
	c.endDate = endDate
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of custom data sources to include in this response.
func (c *ManagementDailyUploadsListCall) MaxResults(maxResults int64) *ManagementDailyUploadsListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": A 1-based index
// of the first daily upload to retrieve. Use this parameter as a
// pagination mechanism along with the max-results parameter.
func (c *ManagementDailyUploadsListCall) StartIndex(startIndex int64) *ManagementDailyUploadsListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementDailyUploadsListCall) Do() (*DailyUploads, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("end-date", fmt.Sprintf("%v", c.endDate))
	params.Set("start-date", fmt.Sprintf("%v", c.startDate))
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/dailyUploads")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customDataSourceId}", url.QueryEscape(c.customDataSourceId), 1)
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
	ret := new(DailyUploads)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List daily uploads to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.dailyUploads.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "customDataSourceId",
	//     "start-date",
	//     "end-date"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account Id for the daily uploads to retrieve.",
	//       "location": "path",
	//       "pattern": "\\d+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customDataSourceId": {
	//       "description": "Custom data source Id for daily uploads to retrieve.",
	//       "location": "path",
	//       "pattern": ".{22}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "end-date": {
	//       "description": "End date of the form YYYY-MM-DD.",
	//       "location": "query",
	//       "pattern": "[0-9]{4}-[0-9]{2}-[0-9]{2}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of custom data sources to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "start-date": {
	//       "description": "Start date of the form YYYY-MM-DD.",
	//       "location": "query",
	//       "pattern": "[0-9]{4}-[0-9]{2}-[0-9]{2}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "start-index": {
	//       "description": "A 1-based index of the first daily upload to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property Id for the daily uploads to retrieve.",
	//       "location": "path",
	//       "pattern": "UA-(\\d+)-(\\d+)",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/dailyUploads",
	//   "response": {
	//     "$ref": "DailyUploads"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.dailyUploads.upload":

type ManagementDailyUploadsUploadCall struct {
	s                  *Service
	accountId          string
	webPropertyId      string
	customDataSourceId string
	date               string
	appendNumber       int64
	type_              string
	opt_               map[string]interface{}
	media_             io.Reader
}

// Upload: Update/Overwrite data for a custom data source.
func (r *ManagementDailyUploadsService) Upload(accountId string, webPropertyId string, customDataSourceId string, date string, appendNumber int64, type_ string) *ManagementDailyUploadsUploadCall {
	c := &ManagementDailyUploadsUploadCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.customDataSourceId = customDataSourceId
	c.date = date
	c.appendNumber = appendNumber
	c.type_ = type_
	return c
}

// Reset sets the optional parameter "reset": Reset/Overwrite all
// previous appends for this date and start over with this file as the
// first upload.
func (c *ManagementDailyUploadsUploadCall) Reset(reset bool) *ManagementDailyUploadsUploadCall {
	c.opt_["reset"] = reset
	return c
}
func (c *ManagementDailyUploadsUploadCall) Media(r io.Reader) *ManagementDailyUploadsUploadCall {
	c.media_ = r
	return c
}

func (c *ManagementDailyUploadsUploadCall) Do() (*DailyUploadAppend, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("appendNumber", fmt.Sprintf("%v", c.appendNumber))
	params.Set("type", fmt.Sprintf("%v", c.type_))
	if v, ok := c.opt_["reset"]; ok {
		params.Set("reset", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/dailyUploads/{date}/uploads")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	body = new(bytes.Buffer)
	ctype := "application/json"
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customDataSourceId}", url.QueryEscape(c.customDataSourceId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{date}", url.QueryEscape(c.date), 1)
	googleapi.SetOpaque(req.URL)
	if hasMedia_ {
		req.ContentLength = contentLength_
	}
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
	ret := new(DailyUploadAppend)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update/Overwrite data for a custom data source.",
	//   "httpMethod": "POST",
	//   "id": "analytics.management.dailyUploads.upload",
	//   "mediaUpload": {
	//     "accept": [
	//       "application/octet-stream"
	//     ],
	//     "maxSize": "5MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/analytics/v3/management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/dailyUploads/{date}/uploads"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/analytics/v3/management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/dailyUploads/{date}/uploads"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "customDataSourceId",
	//     "date",
	//     "appendNumber",
	//     "type"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account Id associated with daily upload.",
	//       "location": "path",
	//       "pattern": "\\d+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "appendNumber": {
	//       "description": "Append number for this upload indexed from 1.",
	//       "format": "int32",
	//       "location": "query",
	//       "maximum": "20",
	//       "minimum": "1",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "customDataSourceId": {
	//       "description": "Custom data source Id to which the data being uploaded belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "date": {
	//       "description": "Date for which data is uploaded. Date should be formatted as YYYY-MM-DD.",
	//       "location": "path",
	//       "pattern": "[0-9]{4}-[0-9]{2}-[0-9]{2}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "reset": {
	//       "default": "false",
	//       "description": "Reset/Overwrite all previous appends for this date and start over with this file as the first upload.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "type": {
	//       "description": "Type of data for this upload.",
	//       "enum": [
	//         "cost"
	//       ],
	//       "enumDescriptions": [
	//         "Value for specifying cost data upload."
	//       ],
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property Id associated with daily upload.",
	//       "location": "path",
	//       "pattern": "UA-\\d+-\\d+",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/dailyUploads/{date}/uploads",
	//   "response": {
	//     "$ref": "DailyUploadAppend"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "analytics.management.experiments.delete":

type ManagementExperimentsDeleteCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	experimentId  string
	opt_          map[string]interface{}
}

// Delete: Delete an experiment.
func (r *ManagementExperimentsService) Delete(accountId string, webPropertyId string, profileId string, experimentId string) *ManagementExperimentsDeleteCall {
	c := &ManagementExperimentsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.experimentId = experimentId
	return c
}

func (c *ManagementExperimentsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments/{experimentId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{experimentId}", url.QueryEscape(c.experimentId), 1)
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
	//   "description": "Delete an experiment.",
	//   "httpMethod": "DELETE",
	//   "id": "analytics.management.experiments.delete",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId",
	//     "experimentId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to which the experiment belongs",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "experimentId": {
	//       "description": "ID of the experiment to delete",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to which the experiment belongs",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to which the experiment belongs",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments/{experimentId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.experiments.get":

type ManagementExperimentsGetCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	experimentId  string
	opt_          map[string]interface{}
}

// Get: Returns an experiment to which the user has access.
func (r *ManagementExperimentsService) Get(accountId string, webPropertyId string, profileId string, experimentId string) *ManagementExperimentsGetCall {
	c := &ManagementExperimentsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.experimentId = experimentId
	return c
}

func (c *ManagementExperimentsGetCall) Do() (*Experiment, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments/{experimentId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{experimentId}", url.QueryEscape(c.experimentId), 1)
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
	ret := new(Experiment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns an experiment to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.experiments.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId",
	//     "experimentId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to retrieve the experiment for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "experimentId": {
	//       "description": "Experiment ID to retrieve the experiment for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to retrieve the experiment for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to retrieve the experiment for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments/{experimentId}",
	//   "response": {
	//     "$ref": "Experiment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.experiments.insert":

type ManagementExperimentsInsertCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	experiment    *Experiment
	opt_          map[string]interface{}
}

// Insert: Create a new experiment.
func (r *ManagementExperimentsService) Insert(accountId string, webPropertyId string, profileId string, experiment *Experiment) *ManagementExperimentsInsertCall {
	c := &ManagementExperimentsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.experiment = experiment
	return c
}

func (c *ManagementExperimentsInsertCall) Do() (*Experiment, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.experiment)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
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
	ret := new(Experiment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Create a new experiment.",
	//   "httpMethod": "POST",
	//   "id": "analytics.management.experiments.insert",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to create the experiment for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to create the experiment for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to create the experiment for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments",
	//   "request": {
	//     "$ref": "Experiment"
	//   },
	//   "response": {
	//     "$ref": "Experiment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.experiments.list":

type ManagementExperimentsListCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	opt_          map[string]interface{}
}

// List: Lists experiments to which the user has access.
func (r *ManagementExperimentsService) List(accountId string, webPropertyId string, profileId string) *ManagementExperimentsListCall {
	c := &ManagementExperimentsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of experiments to include in this response.
func (c *ManagementExperimentsListCall) MaxResults(maxResults int64) *ManagementExperimentsListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first experiment to retrieve. Use this parameter as a pagination
// mechanism along with the max-results parameter.
func (c *ManagementExperimentsListCall) StartIndex(startIndex int64) *ManagementExperimentsListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementExperimentsListCall) Do() (*Experiments, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
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
	ret := new(Experiments)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists experiments to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.experiments.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to retrieve experiments for.",
	//       "location": "path",
	//       "pattern": "\\d+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of experiments to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to retrieve experiments for.",
	//       "location": "path",
	//       "pattern": "\\d+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "start-index": {
	//       "description": "An index of the first experiment to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to retrieve experiments for.",
	//       "location": "path",
	//       "pattern": "UA-(\\d+)-(\\d+)",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments",
	//   "response": {
	//     "$ref": "Experiments"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.experiments.patch":

type ManagementExperimentsPatchCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	experimentId  string
	experiment    *Experiment
	opt_          map[string]interface{}
}

// Patch: Update an existing experiment. This method supports patch
// semantics.
func (r *ManagementExperimentsService) Patch(accountId string, webPropertyId string, profileId string, experimentId string, experiment *Experiment) *ManagementExperimentsPatchCall {
	c := &ManagementExperimentsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.experimentId = experimentId
	c.experiment = experiment
	return c
}

func (c *ManagementExperimentsPatchCall) Do() (*Experiment, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.experiment)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments/{experimentId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{experimentId}", url.QueryEscape(c.experimentId), 1)
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
	ret := new(Experiment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update an existing experiment. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "analytics.management.experiments.patch",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId",
	//     "experimentId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the experiment to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "experimentId": {
	//       "description": "Experiment ID of the experiment to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID of the experiment to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID of the experiment to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments/{experimentId}",
	//   "request": {
	//     "$ref": "Experiment"
	//   },
	//   "response": {
	//     "$ref": "Experiment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.experiments.update":

type ManagementExperimentsUpdateCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	experimentId  string
	experiment    *Experiment
	opt_          map[string]interface{}
}

// Update: Update an existing experiment.
func (r *ManagementExperimentsService) Update(accountId string, webPropertyId string, profileId string, experimentId string, experiment *Experiment) *ManagementExperimentsUpdateCall {
	c := &ManagementExperimentsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.experimentId = experimentId
	c.experiment = experiment
	return c
}

func (c *ManagementExperimentsUpdateCall) Do() (*Experiment, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.experiment)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments/{experimentId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{experimentId}", url.QueryEscape(c.experimentId), 1)
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
	ret := new(Experiment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update an existing experiment.",
	//   "httpMethod": "PUT",
	//   "id": "analytics.management.experiments.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId",
	//     "experimentId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID of the experiment to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "experimentId": {
	//       "description": "Experiment ID of the experiment to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID of the experiment to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID of the experiment to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/experiments/{experimentId}",
	//   "request": {
	//     "$ref": "Experiment"
	//   },
	//   "response": {
	//     "$ref": "Experiment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.goals.get":

type ManagementGoalsGetCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	goalId        string
	opt_          map[string]interface{}
}

// Get: Gets a goal to which the user has access.
func (r *ManagementGoalsService) Get(accountId string, webPropertyId string, profileId string, goalId string) *ManagementGoalsGetCall {
	c := &ManagementGoalsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.goalId = goalId
	return c
}

func (c *ManagementGoalsGetCall) Do() (*Goal, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/goals/{goalId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{goalId}", url.QueryEscape(c.goalId), 1)
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
	ret := new(Goal)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a goal to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.goals.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId",
	//     "goalId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to retrieve the goal for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "goalId": {
	//       "description": "Goal ID to retrieve the goal for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to retrieve the goal for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to retrieve the goal for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/goals/{goalId}",
	//   "response": {
	//     "$ref": "Goal"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.goals.insert":

type ManagementGoalsInsertCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	goal          *Goal
	opt_          map[string]interface{}
}

// Insert: Create a new goal.
func (r *ManagementGoalsService) Insert(accountId string, webPropertyId string, profileId string, goal *Goal) *ManagementGoalsInsertCall {
	c := &ManagementGoalsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.goal = goal
	return c
}

func (c *ManagementGoalsInsertCall) Do() (*Goal, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.goal)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/goals")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
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
	ret := new(Goal)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Create a new goal.",
	//   "httpMethod": "POST",
	//   "id": "analytics.management.goals.insert",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to create the goal for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to create the goal for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to create the goal for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/goals",
	//   "request": {
	//     "$ref": "Goal"
	//   },
	//   "response": {
	//     "$ref": "Goal"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.goals.list":

type ManagementGoalsListCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	opt_          map[string]interface{}
}

// List: Lists goals to which the user has access.
func (r *ManagementGoalsService) List(accountId string, webPropertyId string, profileId string) *ManagementGoalsListCall {
	c := &ManagementGoalsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of goals to include in this response.
func (c *ManagementGoalsListCall) MaxResults(maxResults int64) *ManagementGoalsListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first goal to retrieve. Use this parameter as a pagination mechanism
// along with the max-results parameter.
func (c *ManagementGoalsListCall) StartIndex(startIndex int64) *ManagementGoalsListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementGoalsListCall) Do() (*Goals, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/goals")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
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
	ret := new(Goals)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists goals to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.goals.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to retrieve goals for. Can either be a specific account ID or '~all', which refers to all the accounts that user has access to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of goals to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to retrieve goals for. Can either be a specific view (profile) ID or '~all', which refers to all the views (profiles) that user has access to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "start-index": {
	//       "description": "An index of the first goal to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to retrieve goals for. Can either be a specific web property ID or '~all', which refers to all the web properties that user has access to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/goals",
	//   "response": {
	//     "$ref": "Goals"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.goals.patch":

type ManagementGoalsPatchCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	goalId        string
	goal          *Goal
	opt_          map[string]interface{}
}

// Patch: Updates an existing view (profile). This method supports patch
// semantics.
func (r *ManagementGoalsService) Patch(accountId string, webPropertyId string, profileId string, goalId string, goal *Goal) *ManagementGoalsPatchCall {
	c := &ManagementGoalsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.goalId = goalId
	c.goal = goal
	return c
}

func (c *ManagementGoalsPatchCall) Do() (*Goal, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.goal)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/goals/{goalId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{goalId}", url.QueryEscape(c.goalId), 1)
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
	ret := new(Goal)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing view (profile). This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "analytics.management.goals.patch",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId",
	//     "goalId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to update the goal.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "goalId": {
	//       "description": "Index of the goal to be updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to update the goal.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to update the goal.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/goals/{goalId}",
	//   "request": {
	//     "$ref": "Goal"
	//   },
	//   "response": {
	//     "$ref": "Goal"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.goals.update":

type ManagementGoalsUpdateCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	goalId        string
	goal          *Goal
	opt_          map[string]interface{}
}

// Update: Updates an existing view (profile).
func (r *ManagementGoalsService) Update(accountId string, webPropertyId string, profileId string, goalId string, goal *Goal) *ManagementGoalsUpdateCall {
	c := &ManagementGoalsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.goalId = goalId
	c.goal = goal
	return c
}

func (c *ManagementGoalsUpdateCall) Do() (*Goal, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.goal)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/goals/{goalId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{goalId}", url.QueryEscape(c.goalId), 1)
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
	ret := new(Goal)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing view (profile).",
	//   "httpMethod": "PUT",
	//   "id": "analytics.management.goals.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId",
	//     "goalId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to update the goal.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "goalId": {
	//       "description": "Index of the goal to be updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to update the goal.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to update the goal.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/goals/{goalId}",
	//   "request": {
	//     "$ref": "Goal"
	//   },
	//   "response": {
	//     "$ref": "Goal"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.profileUserLinks.delete":

type ManagementProfileUserLinksDeleteCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	linkId        string
	opt_          map[string]interface{}
}

// Delete: Removes a user from the given view (profile).
func (r *ManagementProfileUserLinksService) Delete(accountId string, webPropertyId string, profileId string, linkId string) *ManagementProfileUserLinksDeleteCall {
	c := &ManagementProfileUserLinksDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.linkId = linkId
	return c
}

func (c *ManagementProfileUserLinksDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/entityUserLinks/{linkId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{linkId}", url.QueryEscape(c.linkId), 1)
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
	//   "description": "Removes a user from the given view (profile).",
	//   "httpMethod": "DELETE",
	//   "id": "analytics.management.profileUserLinks.delete",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId",
	//     "linkId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to delete the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "linkId": {
	//       "description": "Link ID to delete the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to delete the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web Property ID to delete the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/entityUserLinks/{linkId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.management.profileUserLinks.insert":

type ManagementProfileUserLinksInsertCall struct {
	s              *Service
	accountId      string
	webPropertyId  string
	profileId      string
	entityuserlink *EntityUserLink
	opt_           map[string]interface{}
}

// Insert: Adds a new user to the given view (profile).
func (r *ManagementProfileUserLinksService) Insert(accountId string, webPropertyId string, profileId string, entityuserlink *EntityUserLink) *ManagementProfileUserLinksInsertCall {
	c := &ManagementProfileUserLinksInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.entityuserlink = entityuserlink
	return c
}

func (c *ManagementProfileUserLinksInsertCall) Do() (*EntityUserLink, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.entityuserlink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/entityUserLinks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
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
	ret := new(EntityUserLink)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a new user to the given view (profile).",
	//   "httpMethod": "POST",
	//   "id": "analytics.management.profileUserLinks.insert",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to create the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to create the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web Property ID to create the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/entityUserLinks",
	//   "request": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "response": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.management.profileUserLinks.list":

type ManagementProfileUserLinksListCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	opt_          map[string]interface{}
}

// List: Lists profile-user links for a given view (profile).
func (r *ManagementProfileUserLinksService) List(accountId string, webPropertyId string, profileId string) *ManagementProfileUserLinksListCall {
	c := &ManagementProfileUserLinksListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of profile-user links to include in this response.
func (c *ManagementProfileUserLinksListCall) MaxResults(maxResults int64) *ManagementProfileUserLinksListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first profile-user link to retrieve. Use this parameter as a
// pagination mechanism along with the max-results parameter.
func (c *ManagementProfileUserLinksListCall) StartIndex(startIndex int64) *ManagementProfileUserLinksListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementProfileUserLinksListCall) Do() (*EntityUserLinks, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/entityUserLinks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
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
	ret := new(EntityUserLinks)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists profile-user links for a given view (profile).",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.profileUserLinks.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID which the given view (profile) belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of profile-user links to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to retrieve the profile-user links for",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "start-index": {
	//       "description": "An index of the first profile-user link to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "webPropertyId": {
	//       "description": "Web Property ID which the given view (profile) belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/entityUserLinks",
	//   "response": {
	//     "$ref": "EntityUserLinks"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.management.profileUserLinks.update":

type ManagementProfileUserLinksUpdateCall struct {
	s              *Service
	accountId      string
	webPropertyId  string
	profileId      string
	linkId         string
	entityuserlink *EntityUserLink
	opt_           map[string]interface{}
}

// Update: Updates permissions for an existing user on the given view
// (profile).
func (r *ManagementProfileUserLinksService) Update(accountId string, webPropertyId string, profileId string, linkId string, entityuserlink *EntityUserLink) *ManagementProfileUserLinksUpdateCall {
	c := &ManagementProfileUserLinksUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.linkId = linkId
	c.entityuserlink = entityuserlink
	return c
}

func (c *ManagementProfileUserLinksUpdateCall) Do() (*EntityUserLink, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.entityuserlink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/entityUserLinks/{linkId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{linkId}", url.QueryEscape(c.linkId), 1)
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
	ret := new(EntityUserLink)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates permissions for an existing user on the given view (profile).",
	//   "httpMethod": "PUT",
	//   "id": "analytics.management.profileUserLinks.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId",
	//     "linkId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to update the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "linkId": {
	//       "description": "Link ID to update the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile ID) to update the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web Property ID to update the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}/entityUserLinks/{linkId}",
	//   "request": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "response": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.management.profiles.delete":

type ManagementProfilesDeleteCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	opt_          map[string]interface{}
}

// Delete: Deletes a view (profile).
func (r *ManagementProfilesService) Delete(accountId string, webPropertyId string, profileId string) *ManagementProfilesDeleteCall {
	c := &ManagementProfilesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	return c
}

func (c *ManagementProfilesDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
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
	//   "description": "Deletes a view (profile).",
	//   "httpMethod": "DELETE",
	//   "id": "analytics.management.profiles.delete",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to delete the view (profile) for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "ID of the view (profile) to be deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to delete the view (profile) for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.profiles.get":

type ManagementProfilesGetCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	opt_          map[string]interface{}
}

// Get: Gets a view (profile) to which the user has access.
func (r *ManagementProfilesService) Get(accountId string, webPropertyId string, profileId string) *ManagementProfilesGetCall {
	c := &ManagementProfilesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	return c
}

func (c *ManagementProfilesGetCall) Do() (*Profile, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
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
	ret := new(Profile)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a view (profile) to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.profiles.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to retrieve the goal for.",
	//       "location": "path",
	//       "pattern": "[0-9]+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "View (Profile) ID to retrieve the goal for.",
	//       "location": "path",
	//       "pattern": "[0-9]+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to retrieve the goal for.",
	//       "location": "path",
	//       "pattern": "UA-[0-9]+-[0-9]+",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}",
	//   "response": {
	//     "$ref": "Profile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.profiles.insert":

type ManagementProfilesInsertCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profile       *Profile
	opt_          map[string]interface{}
}

// Insert: Create a new view (profile).
func (r *ManagementProfilesService) Insert(accountId string, webPropertyId string, profile *Profile) *ManagementProfilesInsertCall {
	c := &ManagementProfilesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profile = profile
	return c
}

func (c *ManagementProfilesInsertCall) Do() (*Profile, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.profile)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
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
	ret := new(Profile)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Create a new view (profile).",
	//   "httpMethod": "POST",
	//   "id": "analytics.management.profiles.insert",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to create the view (profile) for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to create the view (profile) for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles",
	//   "request": {
	//     "$ref": "Profile"
	//   },
	//   "response": {
	//     "$ref": "Profile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.profiles.list":

type ManagementProfilesListCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	opt_          map[string]interface{}
}

// List: Lists views (profiles) to which the user has access.
func (r *ManagementProfilesService) List(accountId string, webPropertyId string) *ManagementProfilesListCall {
	c := &ManagementProfilesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of views (profiles) to include in this response.
func (c *ManagementProfilesListCall) MaxResults(maxResults int64) *ManagementProfilesListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first entity to retrieve. Use this parameter as a pagination
// mechanism along with the max-results parameter.
func (c *ManagementProfilesListCall) StartIndex(startIndex int64) *ManagementProfilesListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementProfilesListCall) Do() (*Profiles, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
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
	ret := new(Profiles)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists views (profiles) to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.profiles.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID for the view (profiles) to retrieve. Can either be a specific account ID or '~all', which refers to all the accounts to which the user has access.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of views (profiles) to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "start-index": {
	//       "description": "An index of the first entity to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID for the views (profiles) to retrieve. Can either be a specific web property ID or '~all', which refers to all the web properties to which the user has access.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles",
	//   "response": {
	//     "$ref": "Profiles"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.profiles.patch":

type ManagementProfilesPatchCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	profile       *Profile
	opt_          map[string]interface{}
}

// Patch: Updates an existing view (profile). This method supports patch
// semantics.
func (r *ManagementProfilesService) Patch(accountId string, webPropertyId string, profileId string, profile *Profile) *ManagementProfilesPatchCall {
	c := &ManagementProfilesPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.profile = profile
	return c
}

func (c *ManagementProfilesPatchCall) Do() (*Profile, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.profile)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
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
	ret := new(Profile)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing view (profile). This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "analytics.management.profiles.patch",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to which the view (profile) belongs",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "ID of the view (profile) to be updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to which the view (profile) belongs",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}",
	//   "request": {
	//     "$ref": "Profile"
	//   },
	//   "response": {
	//     "$ref": "Profile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.profiles.update":

type ManagementProfilesUpdateCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	profileId     string
	profile       *Profile
	opt_          map[string]interface{}
}

// Update: Updates an existing view (profile).
func (r *ManagementProfilesService) Update(accountId string, webPropertyId string, profileId string, profile *Profile) *ManagementProfilesUpdateCall {
	c := &ManagementProfilesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.profileId = profileId
	c.profile = profile
	return c
}

func (c *ManagementProfilesUpdateCall) Do() (*Profile, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.profile)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{profileId}", url.QueryEscape(c.profileId), 1)
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
	ret := new(Profile)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing view (profile).",
	//   "httpMethod": "PUT",
	//   "id": "analytics.management.profiles.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "profileId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to which the view (profile) belongs",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "profileId": {
	//       "description": "ID of the view (profile) to be updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to which the view (profile) belongs",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/profiles/{profileId}",
	//   "request": {
	//     "$ref": "Profile"
	//   },
	//   "response": {
	//     "$ref": "Profile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.segments.list":

type ManagementSegmentsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Lists segments to which the user has access.
func (r *ManagementSegmentsService) List() *ManagementSegmentsListCall {
	c := &ManagementSegmentsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of segments to include in this response.
func (c *ManagementSegmentsListCall) MaxResults(maxResults int64) *ManagementSegmentsListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first segment to retrieve. Use this parameter as a pagination
// mechanism along with the max-results parameter.
func (c *ManagementSegmentsListCall) StartIndex(startIndex int64) *ManagementSegmentsListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementSegmentsListCall) Do() (*Segments, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/segments")
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
	ret := new(Segments)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists segments to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.segments.list",
	//   "parameters": {
	//     "max-results": {
	//       "description": "The maximum number of segments to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "start-index": {
	//       "description": "An index of the first segment to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "management/segments",
	//   "response": {
	//     "$ref": "Segments"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.uploads.deleteUploadData":

type ManagementUploadsDeleteUploadDataCall struct {
	s                                          *Service
	accountId                                  string
	webPropertyId                              string
	customDataSourceId                         string
	analyticsdataimportdeleteuploaddatarequest *AnalyticsDataimportDeleteUploadDataRequest
	opt_                                       map[string]interface{}
}

// DeleteUploadData: Delete data associated with a previous upload.
func (r *ManagementUploadsService) DeleteUploadData(accountId string, webPropertyId string, customDataSourceId string, analyticsdataimportdeleteuploaddatarequest *AnalyticsDataimportDeleteUploadDataRequest) *ManagementUploadsDeleteUploadDataCall {
	c := &ManagementUploadsDeleteUploadDataCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.customDataSourceId = customDataSourceId
	c.analyticsdataimportdeleteuploaddatarequest = analyticsdataimportdeleteuploaddatarequest
	return c
}

func (c *ManagementUploadsDeleteUploadDataCall) Do() error {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.analyticsdataimportdeleteuploaddatarequest)
	if err != nil {
		return err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/deleteUploadData")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customDataSourceId}", url.QueryEscape(c.customDataSourceId), 1)
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
	//   "description": "Delete data associated with a previous upload.",
	//   "httpMethod": "POST",
	//   "id": "analytics.management.uploads.deleteUploadData",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "customDataSourceId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account Id for the uploads to be deleted.",
	//       "location": "path",
	//       "pattern": "\\d+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customDataSourceId": {
	//       "description": "Custom data source Id for the uploads to be deleted.",
	//       "location": "path",
	//       "pattern": ".{22}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property Id for the uploads to be deleted.",
	//       "location": "path",
	//       "pattern": "UA-(\\d+)-(\\d+)",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/deleteUploadData",
	//   "request": {
	//     "$ref": "AnalyticsDataimportDeleteUploadDataRequest"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.uploads.get":

type ManagementUploadsGetCall struct {
	s                  *Service
	accountId          string
	webPropertyId      string
	customDataSourceId string
	uploadId           string
	opt_               map[string]interface{}
}

// Get: List uploads to which the user has access.
func (r *ManagementUploadsService) Get(accountId string, webPropertyId string, customDataSourceId string, uploadId string) *ManagementUploadsGetCall {
	c := &ManagementUploadsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.customDataSourceId = customDataSourceId
	c.uploadId = uploadId
	return c
}

func (c *ManagementUploadsGetCall) Do() (*Upload, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/uploads/{uploadId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customDataSourceId}", url.QueryEscape(c.customDataSourceId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{uploadId}", url.QueryEscape(c.uploadId), 1)
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
	ret := new(Upload)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List uploads to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.uploads.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "customDataSourceId",
	//     "uploadId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account Id for the upload to retrieve.",
	//       "location": "path",
	//       "pattern": "\\d+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customDataSourceId": {
	//       "description": "Custom data source Id for upload to retrieve.",
	//       "location": "path",
	//       "pattern": ".{22}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "uploadId": {
	//       "description": "Upload Id to retrieve.",
	//       "location": "path",
	//       "pattern": ".{22}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property Id for the upload to retrieve.",
	//       "location": "path",
	//       "pattern": "UA-(\\d+)-(\\d+)",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/uploads/{uploadId}",
	//   "response": {
	//     "$ref": "Upload"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.uploads.list":

type ManagementUploadsListCall struct {
	s                  *Service
	accountId          string
	webPropertyId      string
	customDataSourceId string
	opt_               map[string]interface{}
}

// List: List uploads to which the user has access.
func (r *ManagementUploadsService) List(accountId string, webPropertyId string, customDataSourceId string) *ManagementUploadsListCall {
	c := &ManagementUploadsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.customDataSourceId = customDataSourceId
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of uploads to include in this response.
func (c *ManagementUploadsListCall) MaxResults(maxResults int64) *ManagementUploadsListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": A 1-based index
// of the first upload to retrieve. Use this parameter as a pagination
// mechanism along with the max-results parameter.
func (c *ManagementUploadsListCall) StartIndex(startIndex int64) *ManagementUploadsListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementUploadsListCall) Do() (*Uploads, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/uploads")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customDataSourceId}", url.QueryEscape(c.customDataSourceId), 1)
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
	ret := new(Uploads)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List uploads to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.uploads.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "customDataSourceId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account Id for the uploads to retrieve.",
	//       "location": "path",
	//       "pattern": "\\d+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customDataSourceId": {
	//       "description": "Custom data source Id for uploads to retrieve.",
	//       "location": "path",
	//       "pattern": ".{22}",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of uploads to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "start-index": {
	//       "description": "A 1-based index of the first upload to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property Id for the uploads to retrieve.",
	//       "location": "path",
	//       "pattern": "UA-(\\d+)-(\\d+)",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/uploads",
	//   "response": {
	//     "$ref": "Uploads"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.uploads.uploadData":

type ManagementUploadsUploadDataCall struct {
	s                  *Service
	accountId          string
	webPropertyId      string
	customDataSourceId string
	opt_               map[string]interface{}
	media_             io.Reader
}

// UploadData: Upload data for a custom data source.
func (r *ManagementUploadsService) UploadData(accountId string, webPropertyId string, customDataSourceId string) *ManagementUploadsUploadDataCall {
	c := &ManagementUploadsUploadDataCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.customDataSourceId = customDataSourceId
	return c
}
func (c *ManagementUploadsUploadDataCall) Media(r io.Reader) *ManagementUploadsUploadDataCall {
	c.media_ = r
	return c
}

func (c *ManagementUploadsUploadDataCall) Do() (*Upload, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/uploads")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	body = new(bytes.Buffer)
	ctype := "application/json"
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{customDataSourceId}", url.QueryEscape(c.customDataSourceId), 1)
	googleapi.SetOpaque(req.URL)
	if hasMedia_ {
		req.ContentLength = contentLength_
	}
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
	ret := new(Upload)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Upload data for a custom data source.",
	//   "httpMethod": "POST",
	//   "id": "analytics.management.uploads.uploadData",
	//   "mediaUpload": {
	//     "accept": [
	//       "application/octet-stream"
	//     ],
	//     "maxSize": "1GB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/analytics/v3/management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/uploads"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/analytics/v3/management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/uploads"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "customDataSourceId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account Id associated with the upload.",
	//       "location": "path",
	//       "pattern": "\\d+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "customDataSourceId": {
	//       "description": "Custom data source Id to which the data being uploaded belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property UA-string associated with the upload.",
	//       "location": "path",
	//       "pattern": "UA-\\d+-\\d+",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/customDataSources/{customDataSourceId}/uploads",
	//   "response": {
	//     "$ref": "Upload"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "analytics.management.webproperties.get":

type ManagementWebpropertiesGetCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	opt_          map[string]interface{}
}

// Get: Gets a web property to which the user has access.
func (r *ManagementWebpropertiesService) Get(accountId string, webPropertyId string) *ManagementWebpropertiesGetCall {
	c := &ManagementWebpropertiesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	return c
}

func (c *ManagementWebpropertiesGetCall) Do() (*Webproperty, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
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
	ret := new(Webproperty)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a web property to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.webproperties.get",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to retrieve the web property for.",
	//       "location": "path",
	//       "pattern": "[0-9]+",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "ID to retrieve the web property for.",
	//       "location": "path",
	//       "pattern": "UA-[0-9]+-[0-9]+",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}",
	//   "response": {
	//     "$ref": "Webproperty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.webproperties.insert":

type ManagementWebpropertiesInsertCall struct {
	s           *Service
	accountId   string
	webproperty *Webproperty
	opt_        map[string]interface{}
}

// Insert: Create a new property if the account has fewer than 20
// properties. Web properties are visible in the Google Analytics
// interface only if they have at least one profile.
func (r *ManagementWebpropertiesService) Insert(accountId string, webproperty *Webproperty) *ManagementWebpropertiesInsertCall {
	c := &ManagementWebpropertiesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webproperty = webproperty
	return c
}

func (c *ManagementWebpropertiesInsertCall) Do() (*Webproperty, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.webproperty)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
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
	ret := new(Webproperty)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Create a new property if the account has fewer than 20 properties. Web properties are visible in the Google Analytics interface only if they have at least one profile.",
	//   "httpMethod": "POST",
	//   "id": "analytics.management.webproperties.insert",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to create the web property for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties",
	//   "request": {
	//     "$ref": "Webproperty"
	//   },
	//   "response": {
	//     "$ref": "Webproperty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.webproperties.list":

type ManagementWebpropertiesListCall struct {
	s         *Service
	accountId string
	opt_      map[string]interface{}
}

// List: Lists web properties to which the user has access.
func (r *ManagementWebpropertiesService) List(accountId string) *ManagementWebpropertiesListCall {
	c := &ManagementWebpropertiesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of web properties to include in this response.
func (c *ManagementWebpropertiesListCall) MaxResults(maxResults int64) *ManagementWebpropertiesListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first entity to retrieve. Use this parameter as a pagination
// mechanism along with the max-results parameter.
func (c *ManagementWebpropertiesListCall) StartIndex(startIndex int64) *ManagementWebpropertiesListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementWebpropertiesListCall) Do() (*Webproperties, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties")
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
	ret := new(Webproperties)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists web properties to which the user has access.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.webproperties.list",
	//   "parameterOrder": [
	//     "accountId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to retrieve web properties for. Can either be a specific account ID or '~all', which refers to all the accounts that user has access to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of web properties to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "start-index": {
	//       "description": "An index of the first entity to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties",
	//   "response": {
	//     "$ref": "Webproperties"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}

// method id "analytics.management.webproperties.patch":

type ManagementWebpropertiesPatchCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	webproperty   *Webproperty
	opt_          map[string]interface{}
}

// Patch: Updates an existing web property. This method supports patch
// semantics.
func (r *ManagementWebpropertiesService) Patch(accountId string, webPropertyId string, webproperty *Webproperty) *ManagementWebpropertiesPatchCall {
	c := &ManagementWebpropertiesPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.webproperty = webproperty
	return c
}

func (c *ManagementWebpropertiesPatchCall) Do() (*Webproperty, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.webproperty)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
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
	ret := new(Webproperty)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing web property. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "analytics.management.webproperties.patch",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to which the web property belongs",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}",
	//   "request": {
	//     "$ref": "Webproperty"
	//   },
	//   "response": {
	//     "$ref": "Webproperty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.webproperties.update":

type ManagementWebpropertiesUpdateCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	webproperty   *Webproperty
	opt_          map[string]interface{}
}

// Update: Updates an existing web property.
func (r *ManagementWebpropertiesService) Update(accountId string, webPropertyId string, webproperty *Webproperty) *ManagementWebpropertiesUpdateCall {
	c := &ManagementWebpropertiesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.webproperty = webproperty
	return c
}

func (c *ManagementWebpropertiesUpdateCall) Do() (*Webproperty, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.webproperty)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
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
	ret := new(Webproperty)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing web property.",
	//   "httpMethod": "PUT",
	//   "id": "analytics.management.webproperties.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to which the web property belongs",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}",
	//   "request": {
	//     "$ref": "Webproperty"
	//   },
	//   "response": {
	//     "$ref": "Webproperty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.edit"
	//   ]
	// }

}

// method id "analytics.management.webpropertyUserLinks.delete":

type ManagementWebpropertyUserLinksDeleteCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	linkId        string
	opt_          map[string]interface{}
}

// Delete: Removes a user from the given web property.
func (r *ManagementWebpropertyUserLinksService) Delete(accountId string, webPropertyId string, linkId string) *ManagementWebpropertyUserLinksDeleteCall {
	c := &ManagementWebpropertyUserLinksDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.linkId = linkId
	return c
}

func (c *ManagementWebpropertyUserLinksDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/entityUserLinks/{linkId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{linkId}", url.QueryEscape(c.linkId), 1)
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
	//   "description": "Removes a user from the given web property.",
	//   "httpMethod": "DELETE",
	//   "id": "analytics.management.webpropertyUserLinks.delete",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "linkId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to delete the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "linkId": {
	//       "description": "Link ID to delete the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web Property ID to delete the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/entityUserLinks/{linkId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.management.webpropertyUserLinks.insert":

type ManagementWebpropertyUserLinksInsertCall struct {
	s              *Service
	accountId      string
	webPropertyId  string
	entityuserlink *EntityUserLink
	opt_           map[string]interface{}
}

// Insert: Adds a new user to the given web property.
func (r *ManagementWebpropertyUserLinksService) Insert(accountId string, webPropertyId string, entityuserlink *EntityUserLink) *ManagementWebpropertyUserLinksInsertCall {
	c := &ManagementWebpropertyUserLinksInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.entityuserlink = entityuserlink
	return c
}

func (c *ManagementWebpropertyUserLinksInsertCall) Do() (*EntityUserLink, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.entityuserlink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/entityUserLinks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
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
	ret := new(EntityUserLink)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a new user to the given web property.",
	//   "httpMethod": "POST",
	//   "id": "analytics.management.webpropertyUserLinks.insert",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to create the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web Property ID to create the user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/entityUserLinks",
	//   "request": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "response": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.management.webpropertyUserLinks.list":

type ManagementWebpropertyUserLinksListCall struct {
	s             *Service
	accountId     string
	webPropertyId string
	opt_          map[string]interface{}
}

// List: Lists webProperty-user links for a given web property.
func (r *ManagementWebpropertyUserLinksService) List(accountId string, webPropertyId string) *ManagementWebpropertyUserLinksListCall {
	c := &ManagementWebpropertyUserLinksListCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	return c
}

// MaxResults sets the optional parameter "max-results": The maximum
// number of webProperty-user Links to include in this response.
func (c *ManagementWebpropertyUserLinksListCall) MaxResults(maxResults int64) *ManagementWebpropertyUserLinksListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": An index of the
// first webProperty-user link to retrieve. Use this parameter as a
// pagination mechanism along with the max-results parameter.
func (c *ManagementWebpropertyUserLinksListCall) StartIndex(startIndex int64) *ManagementWebpropertyUserLinksListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ManagementWebpropertyUserLinksListCall) Do() (*EntityUserLinks, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/entityUserLinks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
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
	ret := new(EntityUserLinks)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists webProperty-user links for a given web property.",
	//   "httpMethod": "GET",
	//   "id": "analytics.management.webpropertyUserLinks.list",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID which the given web property belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "max-results": {
	//       "description": "The maximum number of webProperty-user Links to include in this response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "start-index": {
	//       "description": "An index of the first webProperty-user link to retrieve. Use this parameter as a pagination mechanism along with the max-results parameter.",
	//       "format": "int32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "webPropertyId": {
	//       "description": "Web Property ID for the webProperty-user links to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/entityUserLinks",
	//   "response": {
	//     "$ref": "EntityUserLinks"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.management.webpropertyUserLinks.update":

type ManagementWebpropertyUserLinksUpdateCall struct {
	s              *Service
	accountId      string
	webPropertyId  string
	linkId         string
	entityuserlink *EntityUserLink
	opt_           map[string]interface{}
}

// Update: Updates permissions for an existing user on the given web
// property.
func (r *ManagementWebpropertyUserLinksService) Update(accountId string, webPropertyId string, linkId string, entityuserlink *EntityUserLink) *ManagementWebpropertyUserLinksUpdateCall {
	c := &ManagementWebpropertyUserLinksUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.accountId = accountId
	c.webPropertyId = webPropertyId
	c.linkId = linkId
	c.entityuserlink = entityuserlink
	return c
}

func (c *ManagementWebpropertyUserLinksUpdateCall) Do() (*EntityUserLink, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.entityuserlink)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "management/accounts/{accountId}/webproperties/{webPropertyId}/entityUserLinks/{linkId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", url.QueryEscape(c.accountId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{webPropertyId}", url.QueryEscape(c.webPropertyId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{linkId}", url.QueryEscape(c.linkId), 1)
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
	ret := new(EntityUserLink)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates permissions for an existing user on the given web property.",
	//   "httpMethod": "PUT",
	//   "id": "analytics.management.webpropertyUserLinks.update",
	//   "parameterOrder": [
	//     "accountId",
	//     "webPropertyId",
	//     "linkId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Account ID to update the account-user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "linkId": {
	//       "description": "Link ID to update the account-user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "webPropertyId": {
	//       "description": "Web property ID to update the account-user link for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "management/accounts/{accountId}/webproperties/{webPropertyId}/entityUserLinks/{linkId}",
	//   "request": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "response": {
	//     "$ref": "EntityUserLink"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics.manage.users"
	//   ]
	// }

}

// method id "analytics.metadata.columns.list":

type MetadataColumnsListCall struct {
	s          *Service
	reportType string
	opt_       map[string]interface{}
}

// List: Lists all columns for a report type
func (r *MetadataColumnsService) List(reportType string) *MetadataColumnsListCall {
	c := &MetadataColumnsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.reportType = reportType
	return c
}

func (c *MetadataColumnsListCall) Do() (*Columns, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "metadata/{reportType}/columns")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{reportType}", url.QueryEscape(c.reportType), 1)
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
	ret := new(Columns)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all columns for a report type",
	//   "httpMethod": "GET",
	//   "id": "analytics.metadata.columns.list",
	//   "parameterOrder": [
	//     "reportType"
	//   ],
	//   "parameters": {
	//     "reportType": {
	//       "description": "Report type. Allowed Values: 'ga'. Where 'ga' corresponds to the Core Reporting API",
	//       "location": "path",
	//       "pattern": "ga",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "metadata/{reportType}/columns",
	//   "response": {
	//     "$ref": "Columns"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/analytics",
	//     "https://www.googleapis.com/auth/analytics.edit",
	//     "https://www.googleapis.com/auth/analytics.readonly"
	//   ]
	// }

}
