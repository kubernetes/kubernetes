// Package sqladmin provides access to the Cloud SQL Administration API.
//
// See https://developers.google.com/cloud-sql/docs/admin-api/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/sqladmin/v1beta1"
//   ...
//   sqladminService, err := sqladmin.New(oauthHttpClient)
package sqladmin

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

const apiId = "sqladmin:v1beta1"
const apiName = "sqladmin"
const apiVersion = "v1beta1"
const basePath = "https://www.googleapis.com/sql/v1beta1/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// Manage your Google SQL Service instances
	SqlserviceAdminScope = "https://www.googleapis.com/auth/sqlservice.admin"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.BackupRuns = NewBackupRunsService(s)
	s.Instances = NewInstancesService(s)
	s.Operations = NewOperationsService(s)
	s.Tiers = NewTiersService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	BackupRuns *BackupRunsService

	Instances *InstancesService

	Operations *OperationsService

	Tiers *TiersService
}

func NewBackupRunsService(s *Service) *BackupRunsService {
	rs := &BackupRunsService{s: s}
	return rs
}

type BackupRunsService struct {
	s *Service
}

func NewInstancesService(s *Service) *InstancesService {
	rs := &InstancesService{s: s}
	return rs
}

type InstancesService struct {
	s *Service
}

func NewOperationsService(s *Service) *OperationsService {
	rs := &OperationsService{s: s}
	return rs
}

type OperationsService struct {
	s *Service
}

func NewTiersService(s *Service) *TiersService {
	rs := &TiersService{s: s}
	return rs
}

type TiersService struct {
	s *Service
}

type BackupConfiguration struct {
	// Enabled: Whether this configuration is enabled.
	Enabled bool `json:"enabled,omitempty"`

	// Id: Identifier for this configuration. This gets generated
	// automatically when a backup configuration is created.
	Id string `json:"id,omitempty"`

	// Kind: This is always sql#backupConfiguration.
	Kind string `json:"kind,omitempty"`

	// StartTime: Start time for the daily backup configuration in UTC
	// timezone in the 24 hour format - HH:MM.
	StartTime string `json:"startTime,omitempty"`
}

type BackupRun struct {
	// BackupConfiguration: Backup Configuration identifier.
	BackupConfiguration string `json:"backupConfiguration,omitempty"`

	// DueTime: The due time of this run in UTC timezone in RFC 3339 format,
	// for example 2012-11-15T16:19:00.094Z.
	DueTime string `json:"dueTime,omitempty"`

	// EndTime: The time the backup operation completed in UTC timezone in
	// RFC 3339 format, for example 2012-11-15T16:19:00.094Z.
	EndTime string `json:"endTime,omitempty"`

	// EnqueuedTime: The time the run was enqueued in UTC timezone in RFC
	// 3339 format, for example 2012-11-15T16:19:00.094Z.
	EnqueuedTime string `json:"enqueuedTime,omitempty"`

	// Error: Information about why the backup operation failed. This is
	// only present if the run has the FAILED status.
	Error *OperationError `json:"error,omitempty"`

	// Instance: Name of the database instance.
	Instance string `json:"instance,omitempty"`

	// Kind: This is always sql#backupRun.
	Kind string `json:"kind,omitempty"`

	// StartTime: The time the backup operation actually started in UTC
	// timezone in RFC 3339 format, for example 2012-11-15T16:19:00.094Z.
	StartTime string `json:"startTime,omitempty"`

	// Status: The status of this run.
	Status string `json:"status,omitempty"`
}

type BackupRunsListResponse struct {
	// Items: A list of backup runs in reverse chronological order of the
	// enqueued time.
	Items []*BackupRun `json:"items,omitempty"`

	// Kind: This is always sql#backupRunsList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, used to page through large
	// result sets. Provide this value in a subsequent request to return the
	// next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type DatabaseInstance struct {
	// CurrentDiskSize: The current disk usage of the instance in bytes.
	CurrentDiskSize int64 `json:"currentDiskSize,omitempty,string"`

	// DatabaseVersion: The database engine type and version, for example
	// MYSQL_5_5 for MySQL 5.5.
	DatabaseVersion string `json:"databaseVersion,omitempty"`

	// Etag: Etag for this resource - a version number for the settings
	// object in this resource. This field has no effect when passed as a
	// request parameter. Instead, the contents of this field should be
	// passed in an 'If-Match' http header for use in optimistic locking.
	Etag string `json:"etag,omitempty"`

	// Instance: Name of the Cloud SQL instance. This does not include the
	// project ID.
	Instance string `json:"instance,omitempty"`

	// Kind: This is always sql#instance.
	Kind string `json:"kind,omitempty"`

	// MaxDiskSize: The maximum disk size of the instance in bytes.
	MaxDiskSize int64 `json:"maxDiskSize,omitempty,string"`

	// Project: The project ID of the project containing the Cloud SQL
	// instance. The Google apps domain is prefixed if applicable.
	Project string `json:"project,omitempty"`

	// Region: The geographical region. Can be us-east1, us-central or
	// europe-west1. Defaults to us-central. The region can not be changed
	// after instance creation.
	Region string `json:"region,omitempty"`

	// Settings: The user settings.
	Settings *Settings `json:"settings,omitempty"`

	// State: The current serving state of the Cloud SQL instance. This can
	// be one of the following.
	// RUNNABLE: The instance is running, or is
	// ready to run when accessed.
	// SUSPENDED: The instance is not available,
	// for example due to problems with billing.
	// PENDING_CREATE: The
	// instance is being created.
	// MAINTENANCE: The instance is down for
	// maintenance.
	// UNKNOWN_STATE: The state of the instance is unknown.
	State string `json:"state,omitempty"`
}

type ExportContext struct {
	// Database: Databases (for example, guestbook) from which the export is
	// made. If unspecified, all databases are exported.
	Database []string `json:"database,omitempty"`

	// Kind: This is always sql#exportContext.
	Kind string `json:"kind,omitempty"`

	// Table: Tables to export, or that were exported, from the specified
	// database. If you specify tables, specify one and only one database.
	Table []string `json:"table,omitempty"`

	// Uri: The path to the file in Google Cloud Storage where the export
	// will be stored, or where it was already stored. The URI is in the
	// form gs://bucketName/fileName. If the file already exists, the
	// operation fails. If the filename ends with .gz, the contents are
	// compressed.
	Uri string `json:"uri,omitempty"`
}

type ImportContext struct {
	// Database: The database (for example, guestbook) to which the import
	// is made. If not set, it is assumed that the database is specified in
	// the file to be imported.
	Database string `json:"database,omitempty"`

	// Kind: This is always sql#importContext.
	Kind string `json:"kind,omitempty"`

	// Uri: A path to the MySQL dump file in Google Cloud Storage from which
	// the import is made. The URI is in the form gs://bucketName/fileName.
	// Compressed gzip files (.gz) are also supported.
	Uri []string `json:"uri,omitempty"`
}

type InstanceOperation struct {
	// EndTime: The time this operation finished in UTC timezone in RFC 3339
	// format, for example 2012-11-15T16:19:00.094Z.
	EndTime string `json:"endTime,omitempty"`

	// EnqueuedTime: The time this operation was enqueued in UTC timezone in
	// RFC 3339 format, for example 2012-11-15T16:19:00.094Z.
	EnqueuedTime string `json:"enqueuedTime,omitempty"`

	// Error: The error(s) encountered by this operation. Only set if the
	// operation results in an error.
	Error []*OperationError `json:"error,omitempty"`

	// ExportContext: The context for export operation, if applicable.
	ExportContext *ExportContext `json:"exportContext,omitempty"`

	// ImportContext: The context for import operation, if applicable.
	ImportContext *ImportContext `json:"importContext,omitempty"`

	// Instance: Name of the database instance.
	Instance string `json:"instance,omitempty"`

	// Kind: This is always sql#instanceOperation.
	Kind string `json:"kind,omitempty"`

	// Operation: An identifier that uniquely identifies the operation. You
	// can use this identifier to retrieve the Operations resource that has
	// information about the operation.
	Operation string `json:"operation,omitempty"`

	// OperationType: The type of the operation. Valid values are CREATE,
	// DELETE, UPDATE, RESTART, IMPORT, EXPORT, BACKUP_VOLUME,
	// RESTORE_VOLUME.
	OperationType string `json:"operationType,omitempty"`

	// StartTime: The time this operation actually started in UTC timezone
	// in RFC 3339 format, for example 2012-11-15T16:19:00.094Z.
	StartTime string `json:"startTime,omitempty"`

	// State: The state of an operation. Valid values are PENDING, RUNNING,
	// DONE, UNKNOWN.
	State string `json:"state,omitempty"`

	// UserEmailAddress: The email address of the user who initiated this
	// operation.
	UserEmailAddress string `json:"userEmailAddress,omitempty"`
}

type InstancesDeleteResponse struct {
	// Kind: This is always sql#instancesDelete.
	Kind string `json:"kind,omitempty"`

	// Operation: An identifier that uniquely identifies the operation. You
	// can use this identifier to retrieve the Operations resource that has
	// information about the operation.
	Operation string `json:"operation,omitempty"`
}

type InstancesExportRequest struct {
	// ExportContext: Contains details about the export operation.
	ExportContext *ExportContext `json:"exportContext,omitempty"`
}

type InstancesExportResponse struct {
	// Kind: This is always sql#instancesExport.
	Kind string `json:"kind,omitempty"`

	// Operation: An identifier that uniquely identifies the operation. You
	// can use this identifier to retrieve the Operations resource that has
	// information about the operation.
	Operation string `json:"operation,omitempty"`
}

type InstancesImportRequest struct {
	// ImportContext: Contains details about the import operation.
	ImportContext *ImportContext `json:"importContext,omitempty"`
}

type InstancesImportResponse struct {
	// Kind: This is always sql#instancesImport.
	Kind string `json:"kind,omitempty"`

	// Operation: An identifier that uniquely identifies the operation. You
	// can use this identifier to retrieve the Operations resource that has
	// information about the operation.
	Operation string `json:"operation,omitempty"`
}

type InstancesInsertResponse struct {
	// Kind: This is always sql#instancesInsert.
	Kind string `json:"kind,omitempty"`

	// Operation: An identifier that uniquely identifies the operation. You
	// can use this identifier to retrieve the Operations resource that has
	// information about the operation.
	Operation string `json:"operation,omitempty"`
}

type InstancesListResponse struct {
	// Items: List of database instance resources.
	Items []*DatabaseInstance `json:"items,omitempty"`

	// Kind: This is always sql#instancesList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, used to page through large
	// result sets. Provide this value in a subsequent request to return the
	// next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type InstancesRestartResponse struct {
	// Kind: This is always sql#instancesRestart.
	Kind string `json:"kind,omitempty"`

	// Operation: An identifier that uniquely identifies the operation. You
	// can use this identifier to retrieve the Operations resource that has
	// information about the operation.
	Operation string `json:"operation,omitempty"`
}

type InstancesRestoreBackupResponse struct {
	// Kind: This is always sql#instancesRestoreBackup.
	Kind string `json:"kind,omitempty"`

	// Operation: An identifier that uniquely identifies the operation. You
	// can use this identifier to retrieve the Operations resource that has
	// information about the operation.
	Operation string `json:"operation,omitempty"`
}

type InstancesUpdateResponse struct {
	// Kind: This is always sql#instancesUpdate.
	Kind string `json:"kind,omitempty"`

	// Operation: An identifier that uniquely identifies the operation. You
	// can use this identifier to retrieve information about the operation.
	Operation string `json:"operation,omitempty"`
}

type OperationError struct {
	// Code: Identifies the specific error that occurred.
	Code string `json:"code,omitempty"`

	// Kind: This is always sql#operationError.
	Kind string `json:"kind,omitempty"`
}

type OperationsListResponse struct {
	// Items: List of operation resources.
	Items []*InstanceOperation `json:"items,omitempty"`

	// Kind: This is always sql#operationsList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, used to page through large
	// result sets. Provide this value in a subsequent request to return the
	// next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type Settings struct {
	// ActivationPolicy: The activation policy for this instance. This
	// specifies when the instance should be activated and is applicable
	// only when the instance state is RUNNABLE. This can be one of the
	// following.
	// ALWAYS: The instance should always be active.
	// NEVER: The
	// instance should never be activated.
	// ON_DEMAND: The instance is
	// activated upon receiving requests.
	ActivationPolicy string `json:"activationPolicy,omitempty"`

	// AuthorizedGaeApplications: The AppEngine app ids that can access this
	// instance.
	AuthorizedGaeApplications []string `json:"authorizedGaeApplications,omitempty"`

	// BackupConfiguration: The daily backup configuration for the instance.
	BackupConfiguration []*BackupConfiguration `json:"backupConfiguration,omitempty"`

	// Kind: This is always sql#settings.
	Kind string `json:"kind,omitempty"`

	// PricingPlan: The pricing plan for this instance. This can be either
	// PER_USE or PACKAGE.
	PricingPlan string `json:"pricingPlan,omitempty"`

	// ReplicationType: The type of replication this instance uses. This can
	// be either ASYNCHRONOUS or SYNCHRONOUS.
	ReplicationType string `json:"replicationType,omitempty"`

	// Tier: The tier of service for this instance, for example D1, D2. For
	// more information, see pricing.
	Tier string `json:"tier,omitempty"`
}

type Tier struct {
	// DiskQuota: The maximum disk size of this tier in bytes.
	DiskQuota int64 `json:"DiskQuota,omitempty,string"`

	// RAM: The maximum RAM usage of this tier in bytes.
	RAM int64 `json:"RAM,omitempty,string"`

	// Kind: This is always sql#tier.
	Kind string `json:"kind,omitempty"`

	// Region: The applicable regions for this tier. Can be us-east1 and
	// europe-west1.
	Region []string `json:"region,omitempty"`

	// Tier: An identifier for the service tier, for example D1, D2 etc. For
	// related information, see Pricing.
	Tier string `json:"tier,omitempty"`
}

type TiersListResponse struct {
	// Items: List of tiers.
	Items []*Tier `json:"items,omitempty"`

	// Kind: This is always sql#tiersList.
	Kind string `json:"kind,omitempty"`
}

// method id "sql.backupRuns.get":

type BackupRunsGetCall struct {
	s                   *Service
	project             string
	instance            string
	backupConfiguration string
	dueTime             string
	opt_                map[string]interface{}
}

// Get: Retrieves a resource containing information about a backup run.
func (r *BackupRunsService) Get(project string, instance string, backupConfiguration string, dueTime string) *BackupRunsGetCall {
	c := &BackupRunsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	c.backupConfiguration = backupConfiguration
	c.dueTime = dueTime
	return c
}

func (c *BackupRunsGetCall) Do() (*BackupRun, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("dueTime", fmt.Sprintf("%v", c.dueTime))
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/backupRuns/{backupConfiguration}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{backupConfiguration}", url.QueryEscape(c.backupConfiguration), 1)
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
	ret := new(BackupRun)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a resource containing information about a backup run.",
	//   "httpMethod": "GET",
	//   "id": "sql.backupRuns.get",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "backupConfiguration",
	//     "dueTime"
	//   ],
	//   "parameters": {
	//     "backupConfiguration": {
	//       "description": "Identifier for the backup configuration. This gets generated automatically when a backup configuration is created.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dueTime": {
	//       "description": "The time when this run is due to start in RFC 3339 format, for example 2012-11-15T16:19:00.094Z.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/backupRuns/{backupConfiguration}",
	//   "response": {
	//     "$ref": "BackupRun"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.backupRuns.list":

type BackupRunsListCall struct {
	s                   *Service
	project             string
	instance            string
	backupConfiguration string
	opt_                map[string]interface{}
}

// List: Lists all backup runs associated with a given instance and
// configuration in the reverse chronological order of the enqueued
// time.
func (r *BackupRunsService) List(project string, instance string, backupConfiguration string) *BackupRunsListCall {
	c := &BackupRunsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	c.backupConfiguration = backupConfiguration
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of backup runs per response.
func (c *BackupRunsListCall) MaxResults(maxResults int64) *BackupRunsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A
// previously-returned page token representing part of the larger set of
// results to view.
func (c *BackupRunsListCall) PageToken(pageToken string) *BackupRunsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *BackupRunsListCall) Do() (*BackupRunsListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("backupConfiguration", fmt.Sprintf("%v", c.backupConfiguration))
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/backupRuns")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
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
	ret := new(BackupRunsListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all backup runs associated with a given instance and configuration in the reverse chronological order of the enqueued time.",
	//   "httpMethod": "GET",
	//   "id": "sql.backupRuns.list",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "backupConfiguration"
	//   ],
	//   "parameters": {
	//     "backupConfiguration": {
	//       "description": "Identifier for the backup configuration. This gets generated automatically when a backup configuration is created.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of backup runs per response.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A previously-returned page token representing part of the larger set of results to view.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/backupRuns",
	//   "response": {
	//     "$ref": "BackupRunsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.delete":

type InstancesDeleteCall struct {
	s        *Service
	project  string
	instance string
	opt_     map[string]interface{}
}

// Delete: Deletes a Cloud SQL instance.
func (r *InstancesService) Delete(project string, instance string) *InstancesDeleteCall {
	c := &InstancesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	return c
}

func (c *InstancesDeleteCall) Do() (*InstancesDeleteResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
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
	ret := new(InstancesDeleteResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a Cloud SQL instance.",
	//   "httpMethod": "DELETE",
	//   "id": "sql.instances.delete",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance to be deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}",
	//   "response": {
	//     "$ref": "InstancesDeleteResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.export":

type InstancesExportCall struct {
	s                      *Service
	project                string
	instance               string
	instancesexportrequest *InstancesExportRequest
	opt_                   map[string]interface{}
}

// Export: Exports data from a Cloud SQL instance to a Google Cloud
// Storage bucket as a MySQL dump file.
func (r *InstancesService) Export(project string, instance string, instancesexportrequest *InstancesExportRequest) *InstancesExportCall {
	c := &InstancesExportCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	c.instancesexportrequest = instancesexportrequest
	return c
}

func (c *InstancesExportCall) Do() (*InstancesExportResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancesexportrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/export")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
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
	ret := new(InstancesExportResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Exports data from a Cloud SQL instance to a Google Cloud Storage bucket as a MySQL dump file.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.export",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance to be exported.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/export",
	//   "request": {
	//     "$ref": "InstancesExportRequest"
	//   },
	//   "response": {
	//     "$ref": "InstancesExportResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "sql.instances.get":

type InstancesGetCall struct {
	s        *Service
	project  string
	instance string
	opt_     map[string]interface{}
}

// Get: Retrieves a resource containing information about a Cloud SQL
// instance.
func (r *InstancesService) Get(project string, instance string) *InstancesGetCall {
	c := &InstancesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	return c
}

func (c *InstancesGetCall) Do() (*DatabaseInstance, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
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
	ret := new(DatabaseInstance)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a resource containing information about a Cloud SQL instance.",
	//   "httpMethod": "GET",
	//   "id": "sql.instances.get",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}",
	//   "response": {
	//     "$ref": "DatabaseInstance"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.import":

type InstancesImportCall struct {
	s                      *Service
	project                string
	instance               string
	instancesimportrequest *InstancesImportRequest
	opt_                   map[string]interface{}
}

// Import: Imports data into a Cloud SQL instance from a MySQL dump file
// in Google Cloud Storage.
func (r *InstancesService) Import(project string, instance string, instancesimportrequest *InstancesImportRequest) *InstancesImportCall {
	c := &InstancesImportCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	c.instancesimportrequest = instancesimportrequest
	return c
}

func (c *InstancesImportCall) Do() (*InstancesImportResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancesimportrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/import")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
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
	ret := new(InstancesImportResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Imports data into a Cloud SQL instance from a MySQL dump file in Google Cloud Storage.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.import",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/import",
	//   "request": {
	//     "$ref": "InstancesImportRequest"
	//   },
	//   "response": {
	//     "$ref": "InstancesImportResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "sql.instances.insert":

type InstancesInsertCall struct {
	s                *Service
	project          string
	databaseinstance *DatabaseInstance
	opt_             map[string]interface{}
}

// Insert: Creates a new Cloud SQL instance.
func (r *InstancesService) Insert(project string, databaseinstance *DatabaseInstance) *InstancesInsertCall {
	c := &InstancesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.databaseinstance = databaseinstance
	return c
}

func (c *InstancesInsertCall) Do() (*InstancesInsertResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.databaseinstance)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
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
	ret := new(InstancesInsertResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new Cloud SQL instance.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.insert",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID of the project to which the newly created Cloud SQL instances should belong.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances",
	//   "request": {
	//     "$ref": "DatabaseInstance"
	//   },
	//   "response": {
	//     "$ref": "InstancesInsertResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.list":

type InstancesListCall struct {
	s       *Service
	project string
	opt_    map[string]interface{}
}

// List: Lists instances under a given project in the alphabetical order
// of the instance name.
func (r *InstancesService) List(project string) *InstancesListCall {
	c := &InstancesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of results to return per response.
func (c *InstancesListCall) MaxResults(maxResults int64) *InstancesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A
// previously-returned page token representing part of the larger set of
// results to view.
func (c *InstancesListCall) PageToken(pageToken string) *InstancesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *InstancesListCall) Do() (*InstancesListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
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
	ret := new(InstancesListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists instances under a given project in the alphabetical order of the instance name.",
	//   "httpMethod": "GET",
	//   "id": "sql.instances.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "The maximum number of results to return per response.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A previously-returned page token representing part of the larger set of results to view.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project for which to list Cloud SQL instances.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances",
	//   "response": {
	//     "$ref": "InstancesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.patch":

type InstancesPatchCall struct {
	s                *Service
	project          string
	instance         string
	databaseinstance *DatabaseInstance
	opt_             map[string]interface{}
}

// Patch: Updates settings of a Cloud SQL instance. Caution: This is not
// a partial update, so you must include values for all the settings
// that you want to retain. For partial updates, use patch.. This method
// supports patch semantics.
func (r *InstancesService) Patch(project string, instance string, databaseinstance *DatabaseInstance) *InstancesPatchCall {
	c := &InstancesPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	c.databaseinstance = databaseinstance
	return c
}

func (c *InstancesPatchCall) Do() (*InstancesUpdateResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.databaseinstance)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
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
	ret := new(InstancesUpdateResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates settings of a Cloud SQL instance. Caution: This is not a partial update, so you must include values for all the settings that you want to retain. For partial updates, use patch.. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "sql.instances.patch",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}",
	//   "request": {
	//     "$ref": "DatabaseInstance"
	//   },
	//   "response": {
	//     "$ref": "InstancesUpdateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.restart":

type InstancesRestartCall struct {
	s        *Service
	project  string
	instance string
	opt_     map[string]interface{}
}

// Restart: Restarts a Cloud SQL instance.
func (r *InstancesService) Restart(project string, instance string) *InstancesRestartCall {
	c := &InstancesRestartCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	return c
}

func (c *InstancesRestartCall) Do() (*InstancesRestartResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/restart")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
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
	ret := new(InstancesRestartResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Restarts a Cloud SQL instance.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.restart",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance to be restarted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/restart",
	//   "response": {
	//     "$ref": "InstancesRestartResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.restoreBackup":

type InstancesRestoreBackupCall struct {
	s                     *Service
	project               string
	instance              string
	backupConfigurationid string
	dueTime               string
	opt_                  map[string]interface{}
}

// RestoreBackup: Restores a backup of a Cloud SQL instance.
func (r *InstancesService) RestoreBackup(project string, instance string, backupConfigurationid string, dueTime string) *InstancesRestoreBackupCall {
	c := &InstancesRestoreBackupCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	c.backupConfigurationid = backupConfigurationid
	c.dueTime = dueTime
	return c
}

func (c *InstancesRestoreBackupCall) Do() (*InstancesRestoreBackupResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("backupConfiguration", fmt.Sprintf("%v", c.backupConfigurationid))
	params.Set("dueTime", fmt.Sprintf("%v", c.dueTime))
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/restoreBackup")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
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
	ret := new(InstancesRestoreBackupResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Restores a backup of a Cloud SQL instance.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.restoreBackup",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "backupConfiguration",
	//     "dueTime"
	//   ],
	//   "parameters": {
	//     "backupConfiguration": {
	//       "description": "The identifier of the backup configuration. This gets generated automatically when a backup configuration is created.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "dueTime": {
	//       "description": "The time when this run is due to start in RFC 3339 format, for example 2012-11-15T16:19:00.094Z.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/restoreBackup",
	//   "response": {
	//     "$ref": "InstancesRestoreBackupResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.update":

type InstancesUpdateCall struct {
	s                *Service
	project          string
	instance         string
	databaseinstance *DatabaseInstance
	opt_             map[string]interface{}
}

// Update: Updates settings of a Cloud SQL instance. Caution: This is
// not a partial update, so you must include values for all the settings
// that you want to retain. For partial updates, use patch.
func (r *InstancesService) Update(project string, instance string, databaseinstance *DatabaseInstance) *InstancesUpdateCall {
	c := &InstancesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	c.databaseinstance = databaseinstance
	return c
}

func (c *InstancesUpdateCall) Do() (*InstancesUpdateResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.databaseinstance)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
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
	ret := new(InstancesUpdateResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates settings of a Cloud SQL instance. Caution: This is not a partial update, so you must include values for all the settings that you want to retain. For partial updates, use patch.",
	//   "etagRequired": true,
	//   "httpMethod": "PUT",
	//   "id": "sql.instances.update",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}",
	//   "request": {
	//     "$ref": "DatabaseInstance"
	//   },
	//   "response": {
	//     "$ref": "InstancesUpdateResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.operations.get":

type OperationsGetCall struct {
	s         *Service
	project   string
	instance  string
	operation string
	opt_      map[string]interface{}
}

// Get: Retrieves an instance operation that has been performed on an
// instance.
func (r *OperationsService) Get(project string, instance string, operation string) *OperationsGetCall {
	c := &OperationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	c.operation = operation
	return c
}

func (c *OperationsGetCall) Do() (*InstanceOperation, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/operations/{operation}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{operation}", url.QueryEscape(c.operation), 1)
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
	ret := new(InstanceOperation)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves an instance operation that has been performed on an instance.",
	//   "httpMethod": "GET",
	//   "id": "sql.operations.get",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "operation"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "operation": {
	//       "description": "Instance operation ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/operations/{operation}",
	//   "response": {
	//     "$ref": "InstanceOperation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.operations.list":

type OperationsListCall struct {
	s        *Service
	project  string
	instance string
	opt_     map[string]interface{}
}

// List: Lists all instance operations that have been performed on the
// given Cloud SQL instance in the reverse chronological order of the
// start time.
func (r *OperationsService) List(project string, instance string) *OperationsListCall {
	c := &OperationsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.instance = instance
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of operations per response.
func (c *OperationsListCall) MaxResults(maxResults int64) *OperationsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A
// previously-returned page token representing part of the larger set of
// results to view.
func (c *OperationsListCall) PageToken(pageToken string) *OperationsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *OperationsListCall) Do() (*OperationsListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/operations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{instance}", url.QueryEscape(c.instance), 1)
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
	ret := new(OperationsListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all instance operations that have been performed on the given Cloud SQL instance in the reverse chronological order of the start time.",
	//   "httpMethod": "GET",
	//   "id": "sql.operations.list",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of operations per response.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A previously-returned page token representing part of the larger set of results to view.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the project that contains the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/operations",
	//   "response": {
	//     "$ref": "OperationsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.tiers.list":

type TiersListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Lists all available service tiers for Google Cloud SQL, for
// example D1, D2. For related information, see Pricing.
func (r *TiersService) List() *TiersListCall {
	c := &TiersListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *TiersListCall) Do() (*TiersListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "tiers")
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
	ret := new(TiersListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all available service tiers for Google Cloud SQL, for example D1, D2. For related information, see Pricing.",
	//   "httpMethod": "GET",
	//   "id": "sql.tiers.list",
	//   "path": "tiers",
	//   "response": {
	//     "$ref": "TiersListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}
