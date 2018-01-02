// Package sqladmin provides access to the Cloud SQL Administration API.
//
// See https://cloud.google.com/sql/docs/reference/latest
//
// Usage example:
//
//   import "google.golang.org/api/sqladmin/v1beta4"
//   ...
//   sqladminService, err := sqladmin.New(oauthHttpClient)
package sqladmin // import "google.golang.org/api/sqladmin/v1beta4"

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

const apiId = "sqladmin:v1beta4"
const apiName = "sqladmin"
const apiVersion = "v1beta4"
const basePath = "https://www.googleapis.com/sql/v1beta4/"

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
	s.Databases = NewDatabasesService(s)
	s.Flags = NewFlagsService(s)
	s.Instances = NewInstancesService(s)
	s.Operations = NewOperationsService(s)
	s.SslCerts = NewSslCertsService(s)
	s.Tiers = NewTiersService(s)
	s.Users = NewUsersService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	BackupRuns *BackupRunsService

	Databases *DatabasesService

	Flags *FlagsService

	Instances *InstancesService

	Operations *OperationsService

	SslCerts *SslCertsService

	Tiers *TiersService

	Users *UsersService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewBackupRunsService(s *Service) *BackupRunsService {
	rs := &BackupRunsService{s: s}
	return rs
}

type BackupRunsService struct {
	s *Service
}

func NewDatabasesService(s *Service) *DatabasesService {
	rs := &DatabasesService{s: s}
	return rs
}

type DatabasesService struct {
	s *Service
}

func NewFlagsService(s *Service) *FlagsService {
	rs := &FlagsService{s: s}
	return rs
}

type FlagsService struct {
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

func NewSslCertsService(s *Service) *SslCertsService {
	rs := &SslCertsService{s: s}
	return rs
}

type SslCertsService struct {
	s *Service
}

func NewTiersService(s *Service) *TiersService {
	rs := &TiersService{s: s}
	return rs
}

type TiersService struct {
	s *Service
}

func NewUsersService(s *Service) *UsersService {
	rs := &UsersService{s: s}
	return rs
}

type UsersService struct {
	s *Service
}

// AclEntry: An entry for an Access Control list.
type AclEntry struct {
	// ExpirationTime: The time when this access control entry expires in
	// RFC 3339 format, for example 2012-11-15T16:19:00.094Z.
	ExpirationTime string `json:"expirationTime,omitempty"`

	// Kind: This is always sql#aclEntry.
	Kind string `json:"kind,omitempty"`

	// Name: An optional label to identify this entry.
	Name string `json:"name,omitempty"`

	// Value: The whitelisted value for the access control list.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExpirationTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExpirationTime") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AclEntry) MarshalJSON() ([]byte, error) {
	type noMethod AclEntry
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BackupConfiguration: Database instance backup configuration.
type BackupConfiguration struct {
	// BinaryLogEnabled: Whether binary log is enabled. If backup
	// configuration is disabled, binary log must be disabled as well.
	BinaryLogEnabled bool `json:"binaryLogEnabled,omitempty"`

	// Enabled: Whether this configuration is enabled.
	Enabled bool `json:"enabled,omitempty"`

	// Kind: This is always sql#backupConfiguration.
	Kind string `json:"kind,omitempty"`

	// StartTime: Start time for the daily backup configuration in UTC
	// timezone in the 24 hour format - HH:MM.
	StartTime string `json:"startTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BinaryLogEnabled") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BinaryLogEnabled") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BackupConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod BackupConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BackupRun: A database instance backup run resource.
type BackupRun struct {
	// Description: The description of this run, only applicable to
	// on-demand backups.
	Description string `json:"description,omitempty"`

	// EndTime: The time the backup operation completed in UTC timezone in
	// RFC 3339 format, for example 2012-11-15T16:19:00.094Z.
	EndTime string `json:"endTime,omitempty"`

	// EnqueuedTime: The time the run was enqueued in UTC timezone in RFC
	// 3339 format, for example 2012-11-15T16:19:00.094Z.
	EnqueuedTime string `json:"enqueuedTime,omitempty"`

	// Error: Information about why the backup operation failed. This is
	// only present if the run has the FAILED status.
	Error *OperationError `json:"error,omitempty"`

	// Id: A unique identifier for this backup run. Note that this is unique
	// only within the scope of a particular Cloud SQL instance.
	Id int64 `json:"id,omitempty,string"`

	// Instance: Name of the database instance.
	Instance string `json:"instance,omitempty"`

	// Kind: This is always sql#backupRun.
	Kind string `json:"kind,omitempty"`

	// SelfLink: The URI of this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// StartTime: The time the backup operation actually started in UTC
	// timezone in RFC 3339 format, for example 2012-11-15T16:19:00.094Z.
	StartTime string `json:"startTime,omitempty"`

	// Status: The status of this run.
	Status string `json:"status,omitempty"`

	// Type: The type of this run; can be either "AUTOMATED" or "ON_DEMAND".
	Type string `json:"type,omitempty"`

	// WindowStartTime: The start time of the backup window during which
	// this the backup was attempted in RFC 3339 format, for example
	// 2012-11-15T16:19:00.094Z.
	WindowStartTime string `json:"windowStartTime,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *BackupRun) MarshalJSON() ([]byte, error) {
	type noMethod BackupRun
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BackupRunsListResponse: Backup run list results.
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

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BackupRunsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod BackupRunsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BinLogCoordinates: Binary log coordinates.
type BinLogCoordinates struct {
	// BinLogFileName: Name of the binary log file for a Cloud SQL instance.
	BinLogFileName string `json:"binLogFileName,omitempty"`

	// BinLogPosition: Position (offset) within the binary log file.
	BinLogPosition int64 `json:"binLogPosition,omitempty,string"`

	// Kind: This is always sql#binLogCoordinates.
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BinLogFileName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BinLogFileName") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BinLogCoordinates) MarshalJSON() ([]byte, error) {
	type noMethod BinLogCoordinates
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CloneContext: Database instance clone context.
type CloneContext struct {
	// BinLogCoordinates: Binary log coordinates, if specified, indentify
	// the the position up to which the source instance should be cloned. If
	// not specified, the source instance is cloned up to the most recent
	// binary log coordintes.
	BinLogCoordinates *BinLogCoordinates `json:"binLogCoordinates,omitempty"`

	// DestinationInstanceName: Name of the Cloud SQL instance to be created
	// as a clone.
	DestinationInstanceName string `json:"destinationInstanceName,omitempty"`

	// Kind: This is always sql#cloneContext.
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BinLogCoordinates")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BinLogCoordinates") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *CloneContext) MarshalJSON() ([]byte, error) {
	type noMethod CloneContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Database: A database resource inside a Cloud SQL instance.
type Database struct {
	// Charset: The MySQL charset value.
	Charset string `json:"charset,omitempty"`

	// Collation: The MySQL collation value.
	Collation string `json:"collation,omitempty"`

	// Etag: HTTP 1.1 Entity tag for the resource.
	Etag string `json:"etag,omitempty"`

	// Instance: The name of the Cloud SQL instance. This does not include
	// the project ID.
	Instance string `json:"instance,omitempty"`

	// Kind: This is always sql#database.
	Kind string `json:"kind,omitempty"`

	// Name: The name of the database in the Cloud SQL instance. This does
	// not include the project ID or instance name.
	Name string `json:"name,omitempty"`

	// Project: The project ID of the project containing the Cloud SQL
	// database. The Google apps domain is prefixed if applicable.
	Project string `json:"project,omitempty"`

	// SelfLink: The URI of this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Charset") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Charset") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Database) MarshalJSON() ([]byte, error) {
	type noMethod Database
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatabaseFlags: MySQL flags for Cloud SQL instances.
type DatabaseFlags struct {
	// Name: The name of the flag. These flags are passed at instance
	// startup, so include both MySQL server options and MySQL system
	// variables. Flags should be specified with underscores, not hyphens.
	// For more information, see Configuring MySQL Flags in the Google Cloud
	// SQL documentation, as well as the official MySQL documentation for
	// server options and system variables.
	Name string `json:"name,omitempty"`

	// Value: The value of the flag. Booleans should be set to on for true
	// and off for false. This field must be omitted if the flag doesn't
	// take a value.
	Value string `json:"value,omitempty"`

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

func (s *DatabaseFlags) MarshalJSON() ([]byte, error) {
	type noMethod DatabaseFlags
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatabaseInstance: A Cloud SQL instance resource.
type DatabaseInstance struct {
	// BackendType: FIRST_GEN: Basic Cloud SQL instance that runs in a
	// Google-managed container.
	// SECOND_GEN: A newer Cloud SQL backend that runs in a Compute Engine
	// VM.
	// EXTERNAL: A MySQL server that is not managed by Google.
	BackendType string `json:"backendType,omitempty"`

	// ConnectionName: Connection name of the Cloud SQL instance used in
	// connection strings.
	ConnectionName string `json:"connectionName,omitempty"`

	// CurrentDiskSize: The current disk usage of the instance in bytes.
	// This property has been deprecated. Users should use the
	// "cloudsql.googleapis.com/database/disk/bytes_used" metric in Cloud
	// Monitoring API instead. Please see
	// https://groups.google.com/d/msg/google-cloud-sql-announce/I_7-F9EBhT0/BtvFtdFeAgAJ for
	// details.
	CurrentDiskSize int64 `json:"currentDiskSize,omitempty,string"`

	// DatabaseVersion: The database engine type and version. The
	// databaseVersion field can not be changed after instance creation.
	// MySQL Second Generation instances: MYSQL_5_7 (default) or MYSQL_5_6.
	// PostgreSQL instances: POSTGRES_9_6 MySQL First Generation instances:
	// MYSQL_5_6 (default) or MYSQL_5_5
	DatabaseVersion string `json:"databaseVersion,omitempty"`

	// Etag: HTTP 1.1 Entity tag for the resource.
	Etag string `json:"etag,omitempty"`

	// FailoverReplica: The name and status of the failover replica. This
	// property is applicable only to Second Generation instances.
	FailoverReplica *DatabaseInstanceFailoverReplica `json:"failoverReplica,omitempty"`

	// GceZone: The GCE zone that the instance is serving from. In case when
	// the instance is failed over to standby zone, this value may be
	// different with what user specified in the settings.
	GceZone string `json:"gceZone,omitempty"`

	// InstanceType: The instance type. This can be one of the
	// following.
	// CLOUD_SQL_INSTANCE: A Cloud SQL instance that is not replicating from
	// a master.
	// ON_PREMISES_INSTANCE: An instance running on the customer's
	// premises.
	// READ_REPLICA_INSTANCE: A Cloud SQL instance configured as a
	// read-replica.
	InstanceType string `json:"instanceType,omitempty"`

	// IpAddresses: The assigned IP addresses for the instance.
	IpAddresses []*IpMapping `json:"ipAddresses,omitempty"`

	// Ipv6Address: The IPv6 address assigned to the instance. This property
	// is applicable only to First Generation instances.
	Ipv6Address string `json:"ipv6Address,omitempty"`

	// Kind: This is always sql#instance.
	Kind string `json:"kind,omitempty"`

	// MasterInstanceName: The name of the instance which will act as master
	// in the replication setup.
	MasterInstanceName string `json:"masterInstanceName,omitempty"`

	// MaxDiskSize: The maximum disk size of the instance in bytes.
	MaxDiskSize int64 `json:"maxDiskSize,omitempty,string"`

	// Name: Name of the Cloud SQL instance. This does not include the
	// project ID.
	Name string `json:"name,omitempty"`

	// OnPremisesConfiguration: Configuration specific to on-premises
	// instances.
	OnPremisesConfiguration *OnPremisesConfiguration `json:"onPremisesConfiguration,omitempty"`

	// Project: The project ID of the project containing the Cloud SQL
	// instance. The Google apps domain is prefixed if applicable.
	Project string `json:"project,omitempty"`

	// Region: The geographical region. Can be us-central (FIRST_GEN
	// instances only), us-central1 (SECOND_GEN instances only), asia-east1
	// or europe-west1. Defaults to us-central or us-central1 depending on
	// the instance type (First Generation or Second Generation). The region
	// can not be changed after instance creation.
	Region string `json:"region,omitempty"`

	// ReplicaConfiguration: Configuration specific to read-replicas
	// replicating from on-premises masters.
	ReplicaConfiguration *ReplicaConfiguration `json:"replicaConfiguration,omitempty"`

	// ReplicaNames: The replicas of the instance.
	ReplicaNames []string `json:"replicaNames,omitempty"`

	// SelfLink: The URI of this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// ServerCaCert: SSL configuration.
	ServerCaCert *SslCert `json:"serverCaCert,omitempty"`

	// ServiceAccountEmailAddress: The service account email address
	// assigned to the instance. This property is applicable only to Second
	// Generation instances.
	ServiceAccountEmailAddress string `json:"serviceAccountEmailAddress,omitempty"`

	// Settings: The user settings.
	Settings *Settings `json:"settings,omitempty"`

	// State: The current serving state of the Cloud SQL instance. This can
	// be one of the following.
	// RUNNABLE: The instance is running, or is ready to run when
	// accessed.
	// SUSPENDED: The instance is not available, for example due to problems
	// with billing.
	// PENDING_CREATE: The instance is being created.
	// MAINTENANCE: The instance is down for maintenance.
	// FAILED: The instance creation failed.
	// UNKNOWN_STATE: The state of the instance is unknown.
	State string `json:"state,omitempty"`

	// SuspensionReason: If the instance state is SUSPENDED, the reason for
	// the suspension.
	SuspensionReason []string `json:"suspensionReason,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "BackendType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BackendType") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatabaseInstance) MarshalJSON() ([]byte, error) {
	type noMethod DatabaseInstance
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatabaseInstanceFailoverReplica: The name and status of the failover
// replica. This property is applicable only to Second Generation
// instances.
type DatabaseInstanceFailoverReplica struct {
	// Available: The availability status of the failover replica. A false
	// status indicates that the failover replica is out of sync. The master
	// can only failover to the falover replica when the status is true.
	Available bool `json:"available,omitempty"`

	// Name: The name of the failover replica. If specified at instance
	// creation, a failover replica is created for the instance. The name
	// doesn't include the project ID. This property is applicable only to
	// Second Generation instances.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Available") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Available") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatabaseInstanceFailoverReplica) MarshalJSON() ([]byte, error) {
	type noMethod DatabaseInstanceFailoverReplica
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DatabasesListResponse: Database list response.
type DatabasesListResponse struct {
	// Items: List of database resources in the instance.
	Items []*Database `json:"items,omitempty"`

	// Kind: This is always sql#databasesList.
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

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DatabasesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod DatabasesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ExportContext: Database instance export context.
type ExportContext struct {
	// CsvExportOptions: Options for exporting data as CSV.
	CsvExportOptions *ExportContextCsvExportOptions `json:"csvExportOptions,omitempty"`

	// Databases: Databases (for example, guestbook) from which the export
	// is made. If fileType is SQL and no database is specified, all
	// databases are exported. If fileType is CSV, you can optionally
	// specify at most one database to export. If
	// csvExportOptions.selectQuery also specifies the database, this field
	// will be ignored.
	Databases []string `json:"databases,omitempty"`

	// FileType: The file type for the specified uri.
	// SQL: The file contains SQL statements.
	// CSV: The file contains CSV data.
	FileType string `json:"fileType,omitempty"`

	// Kind: This is always sql#exportContext.
	Kind string `json:"kind,omitempty"`

	// SqlExportOptions: Options for exporting data as SQL statements.
	SqlExportOptions *ExportContextSqlExportOptions `json:"sqlExportOptions,omitempty"`

	// Uri: The path to the file in Google Cloud Storage where the export
	// will be stored. The URI is in the form gs://bucketName/fileName. If
	// the file already exists, the operation fails. If fileType is SQL and
	// the filename ends with .gz, the contents are compressed.
	Uri string `json:"uri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CsvExportOptions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CsvExportOptions") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ExportContext) MarshalJSON() ([]byte, error) {
	type noMethod ExportContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ExportContextCsvExportOptions: Options for exporting data as CSV.
type ExportContextCsvExportOptions struct {
	// SelectQuery: The select query used to extract the data.
	SelectQuery string `json:"selectQuery,omitempty"`

	// ForceSendFields is a list of field names (e.g. "SelectQuery") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SelectQuery") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ExportContextCsvExportOptions) MarshalJSON() ([]byte, error) {
	type noMethod ExportContextCsvExportOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ExportContextSqlExportOptions: Options for exporting data as SQL
// statements.
type ExportContextSqlExportOptions struct {
	// SchemaOnly: Export only schemas.
	SchemaOnly bool `json:"schemaOnly,omitempty"`

	// Tables: Tables to export, or that were exported, from the specified
	// database. If you specify tables, specify one and only one database.
	Tables []string `json:"tables,omitempty"`

	// ForceSendFields is a list of field names (e.g. "SchemaOnly") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SchemaOnly") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ExportContextSqlExportOptions) MarshalJSON() ([]byte, error) {
	type noMethod ExportContextSqlExportOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FailoverContext: Database instance failover context.
type FailoverContext struct {
	// Kind: This is always sql#failoverContext.
	Kind string `json:"kind,omitempty"`

	// SettingsVersion: The current settings version of this instance.
	// Request will be rejected if this version doesn't match the current
	// settings version.
	SettingsVersion int64 `json:"settingsVersion,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FailoverContext) MarshalJSON() ([]byte, error) {
	type noMethod FailoverContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Flag: A Google Cloud SQL service flag resource.
type Flag struct {
	// AllowedStringValues: For STRING flags, a list of strings that the
	// value can be set to.
	AllowedStringValues []string `json:"allowedStringValues,omitempty"`

	// AppliesTo: The database version this flag applies to. Can be
	// MYSQL_5_5, MYSQL_5_6, or MYSQL_5_7. MYSQL_5_7 is applicable only to
	// Second Generation instances.
	AppliesTo []string `json:"appliesTo,omitempty"`

	// Kind: This is always sql#flag.
	Kind string `json:"kind,omitempty"`

	// MaxValue: For INTEGER flags, the maximum allowed value.
	MaxValue int64 `json:"maxValue,omitempty,string"`

	// MinValue: For INTEGER flags, the minimum allowed value.
	MinValue int64 `json:"minValue,omitempty,string"`

	// Name: This is the name of the flag. Flag names always use
	// underscores, not hyphens, e.g. max_allowed_packet
	Name string `json:"name,omitempty"`

	// RequiresRestart: Indicates whether changing this flag will trigger a
	// database restart. Only applicable to Second Generation instances.
	RequiresRestart bool `json:"requiresRestart,omitempty"`

	// Type: The type of the flag. Flags are typed to being BOOLEAN, STRING,
	// INTEGER or NONE. NONE is used for flags which do not take a value,
	// such as skip_grant_tables.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AllowedStringValues")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AllowedStringValues") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Flag) MarshalJSON() ([]byte, error) {
	type noMethod Flag
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FlagsListResponse: Flags list response.
type FlagsListResponse struct {
	// Items: List of flags.
	Items []*Flag `json:"items,omitempty"`

	// Kind: This is always sql#flagsList.
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

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FlagsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod FlagsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ImportContext: Database instance import context.
type ImportContext struct {
	// CsvImportOptions: Options for importing data as CSV.
	CsvImportOptions *ImportContextCsvImportOptions `json:"csvImportOptions,omitempty"`

	// Database: The database (for example, guestbook) to which the import
	// is made. If fileType is SQL and no database is specified, it is
	// assumed that the database is specified in the file to be imported. If
	// fileType is CSV, it must be specified.
	Database string `json:"database,omitempty"`

	// FileType: The file type for the specified uri.
	// SQL: The file contains SQL statements.
	// CSV: The file contains CSV data.
	FileType string `json:"fileType,omitempty"`

	// ImportUser: The PostgreSQL user for this import operation. Defaults
	// to cloudsqlsuperuser. Used only for PostgreSQL instances.
	ImportUser string `json:"importUser,omitempty"`

	// Kind: This is always sql#importContext.
	Kind string `json:"kind,omitempty"`

	// Uri: A path to the file in Google Cloud Storage from which the import
	// is made. The URI is in the form gs://bucketName/fileName. Compressed
	// gzip files (.gz) are supported when fileType is SQL.
	Uri string `json:"uri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CsvImportOptions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CsvImportOptions") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ImportContext) MarshalJSON() ([]byte, error) {
	type noMethod ImportContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ImportContextCsvImportOptions: Options for importing data as CSV.
type ImportContextCsvImportOptions struct {
	// Columns: The columns to which CSV data is imported. If not specified,
	// all columns of the database table are loaded with CSV data.
	Columns []string `json:"columns,omitempty"`

	// Table: The table to which CSV data is imported.
	Table string `json:"table,omitempty"`

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

func (s *ImportContextCsvImportOptions) MarshalJSON() ([]byte, error) {
	type noMethod ImportContextCsvImportOptions
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InstancesCloneRequest: Database instance clone request.
type InstancesCloneRequest struct {
	// CloneContext: Contains details about the clone operation.
	CloneContext *CloneContext `json:"cloneContext,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CloneContext") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CloneContext") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InstancesCloneRequest) MarshalJSON() ([]byte, error) {
	type noMethod InstancesCloneRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InstancesExportRequest: Database instance export request.
type InstancesExportRequest struct {
	// ExportContext: Contains details about the export operation.
	ExportContext *ExportContext `json:"exportContext,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExportContext") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExportContext") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InstancesExportRequest) MarshalJSON() ([]byte, error) {
	type noMethod InstancesExportRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InstancesFailoverRequest: Instance failover request.
type InstancesFailoverRequest struct {
	// FailoverContext: Failover Context.
	FailoverContext *FailoverContext `json:"failoverContext,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FailoverContext") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FailoverContext") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *InstancesFailoverRequest) MarshalJSON() ([]byte, error) {
	type noMethod InstancesFailoverRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InstancesImportRequest: Database instance import request.
type InstancesImportRequest struct {
	// ImportContext: Contains details about the import operation.
	ImportContext *ImportContext `json:"importContext,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ImportContext") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ImportContext") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InstancesImportRequest) MarshalJSON() ([]byte, error) {
	type noMethod InstancesImportRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InstancesListResponse: Database instances list response.
type InstancesListResponse struct {
	// Items: List of database instance resources.
	Items []*DatabaseInstance `json:"items,omitempty"`

	// Kind: This is always sql#instancesList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, used to page through large
	// result sets. Provide this value in a subsequent request to return the
	// next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

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

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InstancesListResponse) MarshalJSON() ([]byte, error) {
	type noMethod InstancesListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InstancesRestoreBackupRequest: Database instance restore backup
// request.
type InstancesRestoreBackupRequest struct {
	// RestoreBackupContext: Parameters required to perform the restore
	// backup operation.
	RestoreBackupContext *RestoreBackupContext `json:"restoreBackupContext,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "RestoreBackupContext") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "RestoreBackupContext") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *InstancesRestoreBackupRequest) MarshalJSON() ([]byte, error) {
	type noMethod InstancesRestoreBackupRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InstancesTruncateLogRequest: Instance truncate log request.
type InstancesTruncateLogRequest struct {
	// TruncateLogContext: Contains details about the truncate log
	// operation.
	TruncateLogContext *TruncateLogContext `json:"truncateLogContext,omitempty"`

	// ForceSendFields is a list of field names (e.g. "TruncateLogContext")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "TruncateLogContext") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *InstancesTruncateLogRequest) MarshalJSON() ([]byte, error) {
	type noMethod InstancesTruncateLogRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// IpConfiguration: IP Management configuration.
type IpConfiguration struct {
	// AuthorizedNetworks: The list of external networks that are allowed to
	// connect to the instance using the IP. In CIDR notation, also known as
	// 'slash' notation (e.g. 192.168.100.0/24).
	AuthorizedNetworks []*AclEntry `json:"authorizedNetworks,omitempty"`

	// Ipv4Enabled: Whether the instance should be assigned an IP address or
	// not.
	Ipv4Enabled bool `json:"ipv4Enabled,omitempty"`

	// RequireSsl: Whether SSL connections over IP should be enforced or
	// not.
	RequireSsl bool `json:"requireSsl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AuthorizedNetworks")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AuthorizedNetworks") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *IpConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod IpConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// IpMapping: Database instance IP Mapping.
type IpMapping struct {
	// IpAddress: The IP address assigned.
	IpAddress string `json:"ipAddress,omitempty"`

	// TimeToRetire: The due time for this IP to be retired in RFC 3339
	// format, for example 2012-11-15T16:19:00.094Z. This field is only
	// available when the IP is scheduled to be retired.
	TimeToRetire string `json:"timeToRetire,omitempty"`

	// Type: The type of this IP address. A PRIMARY address is an address
	// that can accept incoming connections. An OUTGOING address is the
	// source address of connections originating from the instance, if
	// supported.
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "IpAddress") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "IpAddress") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *IpMapping) MarshalJSON() ([]byte, error) {
	type noMethod IpMapping
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LocationPreference: Preferred location. This specifies where a Cloud
// SQL instance should preferably be located, either in a specific
// Compute Engine zone, or co-located with an App Engine application.
// Note that if the preferred location is not available, the instance
// will be located as close as possible within the region. Only one
// location may be specified.
type LocationPreference struct {
	// FollowGaeApplication: The AppEngine application to follow, it must be
	// in the same region as the Cloud SQL instance.
	FollowGaeApplication string `json:"followGaeApplication,omitempty"`

	// Kind: This is always sql#locationPreference.
	Kind string `json:"kind,omitempty"`

	// Zone: The preferred Compute Engine zone (e.g. us-centra1-a,
	// us-central1-b, etc.).
	Zone string `json:"zone,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "FollowGaeApplication") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FollowGaeApplication") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *LocationPreference) MarshalJSON() ([]byte, error) {
	type noMethod LocationPreference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MaintenanceWindow: Maintenance window. This specifies when a v2 Cloud
// SQL instance should preferably be restarted for system maintenance
// puruposes.
type MaintenanceWindow struct {
	// Day: day of week (1-7), starting on Monday.
	Day int64 `json:"day,omitempty"`

	// Hour: hour of day - 0 to 23.
	Hour int64 `json:"hour,omitempty"`

	// Kind: This is always sql#maintenanceWindow.
	Kind string `json:"kind,omitempty"`

	UpdateTrack string `json:"updateTrack,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Day") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Day") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MaintenanceWindow) MarshalJSON() ([]byte, error) {
	type noMethod MaintenanceWindow
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// MySqlReplicaConfiguration: Read-replica configuration specific to
// MySQL databases.
type MySqlReplicaConfiguration struct {
	// CaCertificate: PEM representation of the trusted CA's x509
	// certificate.
	CaCertificate string `json:"caCertificate,omitempty"`

	// ClientCertificate: PEM representation of the slave's x509
	// certificate.
	ClientCertificate string `json:"clientCertificate,omitempty"`

	// ClientKey: PEM representation of the slave's private key. The
	// corresponsing public key is encoded in the client's certificate.
	ClientKey string `json:"clientKey,omitempty"`

	// ConnectRetryInterval: Seconds to wait between connect retries.
	// MySQL's default is 60 seconds.
	ConnectRetryInterval int64 `json:"connectRetryInterval,omitempty"`

	// DumpFilePath: Path to a SQL dump file in Google Cloud Storage from
	// which the slave instance is to be created. The URI is in the form
	// gs://bucketName/fileName. Compressed gzip files (.gz) are also
	// supported. Dumps should have the binlog co-ordinates from which
	// replication should begin. This can be accomplished by setting
	// --master-data to 1 when using mysqldump.
	DumpFilePath string `json:"dumpFilePath,omitempty"`

	// Kind: This is always sql#mysqlReplicaConfiguration.
	Kind string `json:"kind,omitempty"`

	// MasterHeartbeatPeriod: Interval in milliseconds between replication
	// heartbeats.
	MasterHeartbeatPeriod int64 `json:"masterHeartbeatPeriod,omitempty,string"`

	// Password: The password for the replication connection.
	Password string `json:"password,omitempty"`

	// SslCipher: A list of permissible ciphers to use for SSL encryption.
	SslCipher string `json:"sslCipher,omitempty"`

	// Username: The username for the replication connection.
	Username string `json:"username,omitempty"`

	// VerifyServerCertificate: Whether or not to check the master's Common
	// Name value in the certificate that it sends during the SSL handshake.
	VerifyServerCertificate bool `json:"verifyServerCertificate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CaCertificate") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CaCertificate") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *MySqlReplicaConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod MySqlReplicaConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OnPremisesConfiguration: On-premises instance configuration.
type OnPremisesConfiguration struct {
	// HostPort: The host and port of the on-premises instance in host:port
	// format
	HostPort string `json:"hostPort,omitempty"`

	// Kind: This is always sql#onPremisesConfiguration.
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "HostPort") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "HostPort") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OnPremisesConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod OnPremisesConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: An Operations resource contains information about database
// instance operations such as create, delete, and restart. Operations
// resources are created in response to operations that were initiated;
// you never create them directly.
type Operation struct {
	// EndTime: The time this operation finished in UTC timezone in RFC 3339
	// format, for example 2012-11-15T16:19:00.094Z.
	EndTime string `json:"endTime,omitempty"`

	// Error: If errors occurred during processing of this operation, this
	// field will be populated.
	Error *OperationErrors `json:"error,omitempty"`

	// ExportContext: The context for export operation, if applicable.
	ExportContext *ExportContext `json:"exportContext,omitempty"`

	// ImportContext: The context for import operation, if applicable.
	ImportContext *ImportContext `json:"importContext,omitempty"`

	// InsertTime: The time this operation was enqueued in UTC timezone in
	// RFC 3339 format, for example 2012-11-15T16:19:00.094Z.
	InsertTime string `json:"insertTime,omitempty"`

	// Kind: This is always sql#operation.
	Kind string `json:"kind,omitempty"`

	// Name: An identifier that uniquely identifies the operation. You can
	// use this identifier to retrieve the Operations resource that has
	// information about the operation.
	Name string `json:"name,omitempty"`

	// OperationType: The type of the operation. Valid values are CREATE,
	// DELETE, UPDATE, RESTART, IMPORT, EXPORT, BACKUP_VOLUME,
	// RESTORE_VOLUME, CREATE_USER, DELETE_USER, CREATE_DATABASE,
	// DELETE_DATABASE .
	OperationType string `json:"operationType,omitempty"`

	// SelfLink: The URI of this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// StartTime: The time this operation actually started in UTC timezone
	// in RFC 3339 format, for example 2012-11-15T16:19:00.094Z.
	StartTime string `json:"startTime,omitempty"`

	// Status: The status of an operation. Valid values are PENDING,
	// RUNNING, DONE, UNKNOWN.
	Status string `json:"status,omitempty"`

	// TargetId: Name of the database instance related to this operation.
	TargetId string `json:"targetId,omitempty"`

	TargetLink string `json:"targetLink,omitempty"`

	// TargetProject: The project ID of the target instance related to this
	// operation.
	TargetProject string `json:"targetProject,omitempty"`

	// User: The email address of the user who initiated this operation.
	User string `json:"user,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "EndTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EndTime") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Operation) MarshalJSON() ([]byte, error) {
	type noMethod Operation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OperationError: Database instance operation error.
type OperationError struct {
	// Code: Identifies the specific error that occurred.
	Code string `json:"code,omitempty"`

	// Kind: This is always sql#operationError.
	Kind string `json:"kind,omitempty"`

	// Message: Additional information about the error encountered.
	Message string `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Code") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationError) MarshalJSON() ([]byte, error) {
	type noMethod OperationError
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OperationErrors: Database instance operation errors list wrapper.
type OperationErrors struct {
	// Errors: The list of errors encountered while processing this
	// operation.
	Errors []*OperationError `json:"errors,omitempty"`

	// Kind: This is always sql#operationErrors.
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Errors") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Errors") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationErrors) MarshalJSON() ([]byte, error) {
	type noMethod OperationErrors
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OperationsListResponse: Database instance list operations response.
type OperationsListResponse struct {
	// Items: List of operation resources.
	Items []*Operation `json:"items,omitempty"`

	// Kind: This is always sql#operationsList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, used to page through large
	// result sets. Provide this value in a subsequent request to return the
	// next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

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

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod OperationsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ReplicaConfiguration: Read-replica configuration for connecting to
// the master.
type ReplicaConfiguration struct {
	// FailoverTarget: Specifies if the replica is the failover target. If
	// the field is set to true the replica will be designated as a failover
	// replica. In case the master instance fails, the replica instance will
	// be promoted as the new master instance.
	// Only one replica can be specified as failover target, and the replica
	// has to be in different zone with the master instance.
	FailoverTarget bool `json:"failoverTarget,omitempty"`

	// Kind: This is always sql#replicaConfiguration.
	Kind string `json:"kind,omitempty"`

	// MysqlReplicaConfiguration: MySQL specific configuration when
	// replicating from a MySQL on-premises master. Replication
	// configuration information such as the username, password,
	// certificates, and keys are not stored in the instance metadata. The
	// configuration information is used only to set up the replication
	// connection and is stored by MySQL in a file named master.info in the
	// data directory.
	MysqlReplicaConfiguration *MySqlReplicaConfiguration `json:"mysqlReplicaConfiguration,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FailoverTarget") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FailoverTarget") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ReplicaConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod ReplicaConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// RestoreBackupContext: Database instance restore from backup context.
type RestoreBackupContext struct {
	// BackupRunId: The ID of the backup run to restore from.
	BackupRunId int64 `json:"backupRunId,omitempty,string"`

	// InstanceId: The ID of the instance that the backup was taken from.
	InstanceId string `json:"instanceId,omitempty"`

	// Kind: This is always sql#restoreBackupContext.
	Kind string `json:"kind,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BackupRunId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BackupRunId") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *RestoreBackupContext) MarshalJSON() ([]byte, error) {
	type noMethod RestoreBackupContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Settings: Database instance settings.
type Settings struct {
	// ActivationPolicy: The activation policy specifies when the instance
	// is activated; it is applicable only when the instance state is
	// RUNNABLE. The activation policy cannot be updated together with other
	// settings for Second Generation instances. Valid values:
	// ALWAYS: The instance is on; it is not deactivated by
	// inactivity.
	// NEVER: The instance is off; it is not activated, even if a connection
	// request arrives.
	// ON_DEMAND: The instance responds to incoming requests, and turns
	// itself off when not in use. Instances with PER_USE pricing turn off
	// after 15 minutes of inactivity. Instances with PER_PACKAGE pricing
	// turn off after 12 hours of inactivity.
	ActivationPolicy string `json:"activationPolicy,omitempty"`

	// AuthorizedGaeApplications: The App Engine app IDs that can access
	// this instance. This property is only applicable to First Generation
	// instances.
	AuthorizedGaeApplications []string `json:"authorizedGaeApplications,omitempty"`

	// AvailabilityType: Reserved for future use.
	AvailabilityType string `json:"availabilityType,omitempty"`

	// BackupConfiguration: The daily backup configuration for the instance.
	BackupConfiguration *BackupConfiguration `json:"backupConfiguration,omitempty"`

	// CrashSafeReplicationEnabled: Configuration specific to read replica
	// instances. Indicates whether database flags for crash-safe
	// replication are enabled. This property is only applicable to First
	// Generation instances.
	CrashSafeReplicationEnabled bool `json:"crashSafeReplicationEnabled,omitempty"`

	// DataDiskSizeGb: The size of data disk, in GB. The data disk size
	// minimum is 10GB. Applies only to Second Generation instances.
	DataDiskSizeGb int64 `json:"dataDiskSizeGb,omitempty,string"`

	// DataDiskType: The type of data disk. Only supported for Second
	// Generation instances. The default type is PD_SSD. Applies only to
	// Second Generation instances.
	DataDiskType string `json:"dataDiskType,omitempty"`

	// DatabaseFlags: The database flags passed to the instance at startup.
	DatabaseFlags []*DatabaseFlags `json:"databaseFlags,omitempty"`

	// DatabaseReplicationEnabled: Configuration specific to read replica
	// instances. Indicates whether replication is enabled or not.
	DatabaseReplicationEnabled bool `json:"databaseReplicationEnabled,omitempty"`

	// IpConfiguration: The settings for IP Management. This allows to
	// enable or disable the instance IP and manage which external networks
	// can connect to the instance. The IPv4 address cannot be disabled for
	// Second Generation instances.
	IpConfiguration *IpConfiguration `json:"ipConfiguration,omitempty"`

	// Kind: This is always sql#settings.
	Kind string `json:"kind,omitempty"`

	// LocationPreference: The location preference settings. This allows the
	// instance to be located as near as possible to either an App Engine
	// app or GCE zone for better performance. App Engine co-location is
	// only applicable to First Generation instances.
	LocationPreference *LocationPreference `json:"locationPreference,omitempty"`

	// MaintenanceWindow: The maintenance window for this instance. This
	// specifies when the instance may be restarted for maintenance
	// purposes. Applies only to Second Generation instances.
	MaintenanceWindow *MaintenanceWindow `json:"maintenanceWindow,omitempty"`

	// PricingPlan: The pricing plan for this instance. This can be either
	// PER_USE or PACKAGE. Only PER_USE is supported for Second Generation
	// instances.
	PricingPlan string `json:"pricingPlan,omitempty"`

	// ReplicationType: The type of replication this instance uses. This can
	// be either ASYNCHRONOUS or SYNCHRONOUS. This property is only
	// applicable to First Generation instances.
	ReplicationType string `json:"replicationType,omitempty"`

	// SettingsVersion: The version of instance settings. This is a required
	// field for update method to make sure concurrent updates are handled
	// properly. During update, use the most recent settingsVersion value
	// for this instance and do not try to update this value.
	SettingsVersion int64 `json:"settingsVersion,omitempty,string"`

	// StorageAutoResize: Configuration to increase storage size
	// automatically. The default value is true. Applies only to Second
	// Generation instances.
	StorageAutoResize *bool `json:"storageAutoResize,omitempty"`

	// StorageAutoResizeLimit: The maximum size to which storage capacity
	// can be automatically increased. The default value is 0, which
	// specifies that there is no limit. Applies only to Second Generation
	// instances.
	StorageAutoResizeLimit int64 `json:"storageAutoResizeLimit,omitempty,string"`

	// Tier: The tier of service for this instance, for example D1, D2. For
	// more information, see pricing.
	Tier string `json:"tier,omitempty"`

	// UserLabels: User-provided labels, represented as a dictionary where
	// each label is a single key value pair.
	UserLabels map[string]string `json:"userLabels,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ActivationPolicy") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ActivationPolicy") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Settings) MarshalJSON() ([]byte, error) {
	type noMethod Settings
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SslCert: SslCerts Resource
type SslCert struct {
	// Cert: PEM representation.
	Cert string `json:"cert,omitempty"`

	// CertSerialNumber: Serial number, as extracted from the certificate.
	CertSerialNumber string `json:"certSerialNumber,omitempty"`

	// CommonName: User supplied name. Constrained to [a-zA-Z.-_ ]+.
	CommonName string `json:"commonName,omitempty"`

	// CreateTime: The time when the certificate was created in RFC 3339
	// format, for example 2012-11-15T16:19:00.094Z
	CreateTime string `json:"createTime,omitempty"`

	// ExpirationTime: The time when the certificate expires in RFC 3339
	// format, for example 2012-11-15T16:19:00.094Z.
	ExpirationTime string `json:"expirationTime,omitempty"`

	// Instance: Name of the database instance.
	Instance string `json:"instance,omitempty"`

	// Kind: This is always sql#sslCert.
	Kind string `json:"kind,omitempty"`

	// SelfLink: The URI of this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Sha1Fingerprint: Sha1 Fingerprint.
	Sha1Fingerprint string `json:"sha1Fingerprint,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Cert") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Cert") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SslCert) MarshalJSON() ([]byte, error) {
	type noMethod SslCert
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SslCertDetail: SslCertDetail.
type SslCertDetail struct {
	// CertInfo: The public information about the cert.
	CertInfo *SslCert `json:"certInfo,omitempty"`

	// CertPrivateKey: The private key for the client cert, in pem format.
	// Keep private in order to protect your security.
	CertPrivateKey string `json:"certPrivateKey,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CertInfo") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CertInfo") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SslCertDetail) MarshalJSON() ([]byte, error) {
	type noMethod SslCertDetail
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SslCertsCreateEphemeralRequest: SslCerts create ephemeral certificate
// request.
type SslCertsCreateEphemeralRequest struct {
	// PublicKey: PEM encoded public key to include in the signed
	// certificate.
	PublicKey string `json:"public_key,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PublicKey") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "PublicKey") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SslCertsCreateEphemeralRequest) MarshalJSON() ([]byte, error) {
	type noMethod SslCertsCreateEphemeralRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SslCertsInsertRequest: SslCerts insert request.
type SslCertsInsertRequest struct {
	// CommonName: User supplied name. Must be a distinct name from the
	// other certificates for this instance. New certificates will not be
	// usable until the instance is restarted.
	CommonName string `json:"commonName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CommonName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CommonName") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SslCertsInsertRequest) MarshalJSON() ([]byte, error) {
	type noMethod SslCertsInsertRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SslCertsInsertResponse: SslCert insert response.
type SslCertsInsertResponse struct {
	// ClientCert: The new client certificate and private key. The new
	// certificate will not work until the instance is restarted for First
	// Generation instances.
	ClientCert *SslCertDetail `json:"clientCert,omitempty"`

	// Kind: This is always sql#sslCertsInsert.
	Kind string `json:"kind,omitempty"`

	// Operation: The operation to track the ssl certs insert request.
	Operation *Operation `json:"operation,omitempty"`

	// ServerCaCert: The server Certificate Authority's certificate. If this
	// is missing you can force a new one to be generated by calling
	// resetSslConfig method on instances resource.
	ServerCaCert *SslCert `json:"serverCaCert,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ClientCert") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClientCert") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SslCertsInsertResponse) MarshalJSON() ([]byte, error) {
	type noMethod SslCertsInsertResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SslCertsListResponse: SslCerts list response.
type SslCertsListResponse struct {
	// Items: List of client certificates for the instance.
	Items []*SslCert `json:"items,omitempty"`

	// Kind: This is always sql#sslCertsList.
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

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SslCertsListResponse) MarshalJSON() ([]byte, error) {
	type noMethod SslCertsListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Tier: A Google Cloud SQL service tier resource.
type Tier struct {
	// DiskQuota: The maximum disk size of this tier in bytes.
	DiskQuota int64 `json:"DiskQuota,omitempty,string"`

	// RAM: The maximum RAM usage of this tier in bytes.
	RAM int64 `json:"RAM,omitempty,string"`

	// Kind: This is always sql#tier.
	Kind string `json:"kind,omitempty"`

	// Region: The applicable regions for this tier.
	Region []string `json:"region,omitempty"`

	// Tier: An identifier for the service tier, for example D1, D2 etc. For
	// related information, see Pricing.
	Tier string `json:"tier,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DiskQuota") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DiskQuota") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Tier) MarshalJSON() ([]byte, error) {
	type noMethod Tier
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TiersListResponse: Tiers list response.
type TiersListResponse struct {
	// Items: List of tiers.
	Items []*Tier `json:"items,omitempty"`

	// Kind: This is always sql#tiersList.
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

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TiersListResponse) MarshalJSON() ([]byte, error) {
	type noMethod TiersListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TruncateLogContext: Database Instance truncate log context.
type TruncateLogContext struct {
	// Kind: This is always sql#truncateLogContext.
	Kind string `json:"kind,omitempty"`

	// LogType: The type of log to truncate. Valid values are
	// MYSQL_GENERAL_TABLE and MYSQL_SLOW_TABLE.
	LogType string `json:"logType,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TruncateLogContext) MarshalJSON() ([]byte, error) {
	type noMethod TruncateLogContext
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// User: A Cloud SQL user resource.
type User struct {
	// Etag: HTTP 1.1 Entity tag for the resource.
	Etag string `json:"etag,omitempty"`

	// Host: The host name from which the user can connect. For insert
	// operations, host defaults to an empty string. For update operations,
	// host is specified as part of the request URL. The host name cannot be
	// updated after insertion.
	Host string `json:"host,omitempty"`

	// Instance: The name of the Cloud SQL instance. This does not include
	// the project ID. Can be omitted for update since it is already
	// specified on the URL.
	Instance string `json:"instance,omitempty"`

	// Kind: This is always sql#user.
	Kind string `json:"kind,omitempty"`

	// Name: The name of the user in the Cloud SQL instance. Can be omitted
	// for update since it is already specified on the URL.
	Name string `json:"name,omitempty"`

	// Password: The password for the user.
	Password string `json:"password,omitempty"`

	// Project: The project ID of the project containing the Cloud SQL
	// database. The Google apps domain is prefixed if applicable. Can be
	// omitted for update since it is already specified on the URL.
	Project string `json:"project,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Etag") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Etag") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *User) MarshalJSON() ([]byte, error) {
	type noMethod User
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UsersListResponse: User list response.
type UsersListResponse struct {
	// Items: List of user resources in the instance.
	Items []*User `json:"items,omitempty"`

	// Kind: This is always sql#usersList.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: An identifier that uniquely identifies the operation.
	// You can use this identifier to retrieve the Operations resource that
	// has information about the operation.
	NextPageToken string `json:"nextPageToken,omitempty"`

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

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UsersListResponse) MarshalJSON() ([]byte, error) {
	type noMethod UsersListResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "sql.backupRuns.delete":

type BackupRunsDeleteCall struct {
	s          *Service
	project    string
	instance   string
	id         int64
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes the backup taken by a backup run.
func (r *BackupRunsService) Delete(project string, instance string, id int64) *BackupRunsDeleteCall {
	c := &BackupRunsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.id = id
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BackupRunsDeleteCall) Fields(s ...googleapi.Field) *BackupRunsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BackupRunsDeleteCall) Context(ctx context.Context) *BackupRunsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *BackupRunsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *BackupRunsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/backupRuns/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
		"id":       strconv.FormatInt(c.id, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.backupRuns.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *BackupRunsDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Deletes the backup taken by a backup run.",
	//   "httpMethod": "DELETE",
	//   "id": "sql.backupRuns.delete",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the Backup Run to delete. To find a Backup Run ID, use the list method.",
	//       "format": "int64",
	//       "location": "path",
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
	//   "path": "projects/{project}/instances/{instance}/backupRuns/{id}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.backupRuns.get":

type BackupRunsGetCall struct {
	s            *Service
	project      string
	instance     string
	id           int64
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves a resource containing information about a backup run.
func (r *BackupRunsService) Get(project string, instance string, id int64) *BackupRunsGetCall {
	c := &BackupRunsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.id = id
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BackupRunsGetCall) Fields(s ...googleapi.Field) *BackupRunsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BackupRunsGetCall) IfNoneMatch(entityTag string) *BackupRunsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BackupRunsGetCall) Context(ctx context.Context) *BackupRunsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *BackupRunsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *BackupRunsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/backupRuns/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
		"id":       strconv.FormatInt(c.id, 10),
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.backupRuns.get" call.
// Exactly one of *BackupRun or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *BackupRun.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *BackupRunsGetCall) Do(opts ...googleapi.CallOption) (*BackupRun, error) {
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
	ret := &BackupRun{
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
	//   "description": "Retrieves a resource containing information about a backup run.",
	//   "httpMethod": "GET",
	//   "id": "sql.backupRuns.get",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of this Backup Run.",
	//       "format": "int64",
	//       "location": "path",
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
	//   "path": "projects/{project}/instances/{instance}/backupRuns/{id}",
	//   "response": {
	//     "$ref": "BackupRun"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.backupRuns.insert":

type BackupRunsInsertCall struct {
	s          *Service
	project    string
	instance   string
	backuprun  *BackupRun
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Creates a new backup run on demand. This method is applicable
// only to Second Generation instances.
func (r *BackupRunsService) Insert(project string, instance string, backuprun *BackupRun) *BackupRunsInsertCall {
	c := &BackupRunsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.backuprun = backuprun
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BackupRunsInsertCall) Fields(s ...googleapi.Field) *BackupRunsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BackupRunsInsertCall) Context(ctx context.Context) *BackupRunsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *BackupRunsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *BackupRunsInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.backuprun)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/backupRuns")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.backupRuns.insert" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *BackupRunsInsertCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Creates a new backup run on demand. This method is applicable only to Second Generation instances.",
	//   "httpMethod": "POST",
	//   "id": "sql.backupRuns.insert",
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
	//   "path": "projects/{project}/instances/{instance}/backupRuns",
	//   "request": {
	//     "$ref": "BackupRun"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.backupRuns.list":

type BackupRunsListCall struct {
	s            *Service
	project      string
	instance     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all backup runs associated with a given instance and
// configuration in the reverse chronological order of the enqueued
// time.
func (r *BackupRunsService) List(project string, instance string) *BackupRunsListCall {
	c := &BackupRunsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of backup runs per response.
func (c *BackupRunsListCall) MaxResults(maxResults int64) *BackupRunsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": A
// previously-returned page token representing part of the larger set of
// results to view.
func (c *BackupRunsListCall) PageToken(pageToken string) *BackupRunsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BackupRunsListCall) Fields(s ...googleapi.Field) *BackupRunsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BackupRunsListCall) IfNoneMatch(entityTag string) *BackupRunsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BackupRunsListCall) Context(ctx context.Context) *BackupRunsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *BackupRunsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *BackupRunsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/backupRuns")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.backupRuns.list" call.
// Exactly one of *BackupRunsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *BackupRunsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *BackupRunsListCall) Do(opts ...googleapi.CallOption) (*BackupRunsListResponse, error) {
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
	ret := &BackupRunsListResponse{
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
	//   "description": "Lists all backup runs associated with a given instance and configuration in the reverse chronological order of the enqueued time.",
	//   "httpMethod": "GET",
	//   "id": "sql.backupRuns.list",
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
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *BackupRunsListCall) Pages(ctx context.Context, f func(*BackupRunsListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "sql.databases.delete":

type DatabasesDeleteCall struct {
	s          *Service
	project    string
	instance   string
	database   string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a database from a Cloud SQL instance.
func (r *DatabasesService) Delete(project string, instance string, database string) *DatabasesDeleteCall {
	c := &DatabasesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.database = database
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatabasesDeleteCall) Fields(s ...googleapi.Field) *DatabasesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatabasesDeleteCall) Context(ctx context.Context) *DatabasesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatabasesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatabasesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/databases/{database}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
		"database": c.database,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.databases.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatabasesDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Deletes a database from a Cloud SQL instance.",
	//   "httpMethod": "DELETE",
	//   "id": "sql.databases.delete",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "database"
	//   ],
	//   "parameters": {
	//     "database": {
	//       "description": "Name of the database to be deleted in the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "Database instance ID. This does not include the project ID.",
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
	//   "path": "projects/{project}/instances/{instance}/databases/{database}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.databases.get":

type DatabasesGetCall struct {
	s            *Service
	project      string
	instance     string
	database     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves a resource containing information about a database
// inside a Cloud SQL instance.
func (r *DatabasesService) Get(project string, instance string, database string) *DatabasesGetCall {
	c := &DatabasesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.database = database
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatabasesGetCall) Fields(s ...googleapi.Field) *DatabasesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DatabasesGetCall) IfNoneMatch(entityTag string) *DatabasesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatabasesGetCall) Context(ctx context.Context) *DatabasesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatabasesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatabasesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/databases/{database}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
		"database": c.database,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.databases.get" call.
// Exactly one of *Database or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Database.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatabasesGetCall) Do(opts ...googleapi.CallOption) (*Database, error) {
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
	ret := &Database{
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
	//   "description": "Retrieves a resource containing information about a database inside a Cloud SQL instance.",
	//   "httpMethod": "GET",
	//   "id": "sql.databases.get",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "database"
	//   ],
	//   "parameters": {
	//     "database": {
	//       "description": "Name of the database in the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "Database instance ID. This does not include the project ID.",
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
	//   "path": "projects/{project}/instances/{instance}/databases/{database}",
	//   "response": {
	//     "$ref": "Database"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.databases.insert":

type DatabasesInsertCall struct {
	s          *Service
	project    string
	instance   string
	database   *Database
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Inserts a resource containing information about a database
// inside a Cloud SQL instance.
func (r *DatabasesService) Insert(project string, instance string, database *Database) *DatabasesInsertCall {
	c := &DatabasesInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.database = database
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatabasesInsertCall) Fields(s ...googleapi.Field) *DatabasesInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatabasesInsertCall) Context(ctx context.Context) *DatabasesInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatabasesInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatabasesInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.database)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/databases")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.databases.insert" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatabasesInsertCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Inserts a resource containing information about a database inside a Cloud SQL instance.",
	//   "httpMethod": "POST",
	//   "id": "sql.databases.insert",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Database instance ID. This does not include the project ID.",
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
	//   "path": "projects/{project}/instances/{instance}/databases",
	//   "request": {
	//     "$ref": "Database"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.databases.list":

type DatabasesListCall struct {
	s            *Service
	project      string
	instance     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists databases in the specified Cloud SQL instance.
func (r *DatabasesService) List(project string, instance string) *DatabasesListCall {
	c := &DatabasesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatabasesListCall) Fields(s ...googleapi.Field) *DatabasesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *DatabasesListCall) IfNoneMatch(entityTag string) *DatabasesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatabasesListCall) Context(ctx context.Context) *DatabasesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatabasesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatabasesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/databases")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.databases.list" call.
// Exactly one of *DatabasesListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *DatabasesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *DatabasesListCall) Do(opts ...googleapi.CallOption) (*DatabasesListResponse, error) {
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
	ret := &DatabasesListResponse{
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
	//   "description": "Lists databases in the specified Cloud SQL instance.",
	//   "httpMethod": "GET",
	//   "id": "sql.databases.list",
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
	//       "description": "Project ID of the project for which to list Cloud SQL instances.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/databases",
	//   "response": {
	//     "$ref": "DatabasesListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.databases.patch":

type DatabasesPatchCall struct {
	s          *Service
	project    string
	instance   string
	database   string
	database2  *Database
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a resource containing information about a database
// inside a Cloud SQL instance. This method supports patch semantics.
func (r *DatabasesService) Patch(project string, instance string, database string, database2 *Database) *DatabasesPatchCall {
	c := &DatabasesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.database = database
	c.database2 = database2
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatabasesPatchCall) Fields(s ...googleapi.Field) *DatabasesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatabasesPatchCall) Context(ctx context.Context) *DatabasesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatabasesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatabasesPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.database2)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/databases/{database}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
		"database": c.database,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.databases.patch" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatabasesPatchCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Updates a resource containing information about a database inside a Cloud SQL instance. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "sql.databases.patch",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "database"
	//   ],
	//   "parameters": {
	//     "database": {
	//       "description": "Name of the database to be updated in the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "Database instance ID. This does not include the project ID.",
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
	//   "path": "projects/{project}/instances/{instance}/databases/{database}",
	//   "request": {
	//     "$ref": "Database"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.databases.update":

type DatabasesUpdateCall struct {
	s          *Service
	project    string
	instance   string
	database   string
	database2  *Database
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates a resource containing information about a database
// inside a Cloud SQL instance.
func (r *DatabasesService) Update(project string, instance string, database string, database2 *Database) *DatabasesUpdateCall {
	c := &DatabasesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.database = database
	c.database2 = database2
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *DatabasesUpdateCall) Fields(s ...googleapi.Field) *DatabasesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *DatabasesUpdateCall) Context(ctx context.Context) *DatabasesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *DatabasesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *DatabasesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.database2)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/databases/{database}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
		"database": c.database,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.databases.update" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *DatabasesUpdateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Updates a resource containing information about a database inside a Cloud SQL instance.",
	//   "httpMethod": "PUT",
	//   "id": "sql.databases.update",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "database"
	//   ],
	//   "parameters": {
	//     "database": {
	//       "description": "Name of the database to be updated in the instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "Database instance ID. This does not include the project ID.",
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
	//   "path": "projects/{project}/instances/{instance}/databases/{database}",
	//   "request": {
	//     "$ref": "Database"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.flags.list":

type FlagsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List all available database flags for Google Cloud SQL
// instances.
func (r *FlagsService) List() *FlagsListCall {
	c := &FlagsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// DatabaseVersion sets the optional parameter "databaseVersion":
// Database version for flag retrieval. Flags are specific to the
// database version.
func (c *FlagsListCall) DatabaseVersion(databaseVersion string) *FlagsListCall {
	c.urlParams_.Set("databaseVersion", databaseVersion)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *FlagsListCall) Fields(s ...googleapi.Field) *FlagsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *FlagsListCall) IfNoneMatch(entityTag string) *FlagsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *FlagsListCall) Context(ctx context.Context) *FlagsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *FlagsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *FlagsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "flags")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.flags.list" call.
// Exactly one of *FlagsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *FlagsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *FlagsListCall) Do(opts ...googleapi.CallOption) (*FlagsListResponse, error) {
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
	ret := &FlagsListResponse{
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
	//   "description": "List all available database flags for Google Cloud SQL instances.",
	//   "httpMethod": "GET",
	//   "id": "sql.flags.list",
	//   "parameters": {
	//     "databaseVersion": {
	//       "description": "Database version for flag retrieval. Flags are specific to the database version.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "flags",
	//   "response": {
	//     "$ref": "FlagsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.clone":

type InstancesCloneCall struct {
	s                     *Service
	project               string
	instance              string
	instancesclonerequest *InstancesCloneRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Clone: Creates a Cloud SQL instance as a clone of the source
// instance. The API is not ready for Second Generation instances yet.
func (r *InstancesService) Clone(project string, instance string, instancesclonerequest *InstancesCloneRequest) *InstancesCloneCall {
	c := &InstancesCloneCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.instancesclonerequest = instancesclonerequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesCloneCall) Fields(s ...googleapi.Field) *InstancesCloneCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesCloneCall) Context(ctx context.Context) *InstancesCloneCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesCloneCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesCloneCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancesclonerequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/clone")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.clone" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesCloneCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Creates a Cloud SQL instance as a clone of the source instance. The API is not ready for Second Generation instances yet.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.clone",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "The ID of the Cloud SQL instance to be cloned (source). This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "Project ID of the source as well as the clone Cloud SQL instance.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/clone",
	//   "request": {
	//     "$ref": "InstancesCloneRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.delete":

type InstancesDeleteCall struct {
	s          *Service
	project    string
	instance   string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a Cloud SQL instance.
func (r *InstancesService) Delete(project string, instance string) *InstancesDeleteCall {
	c := &InstancesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesDeleteCall) Fields(s ...googleapi.Field) *InstancesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesDeleteCall) Context(ctx context.Context) *InstancesDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
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
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// Export: Exports data from a Cloud SQL instance to a Google Cloud
// Storage bucket as a MySQL dump file.
func (r *InstancesService) Export(project string, instance string, instancesexportrequest *InstancesExportRequest) *InstancesExportCall {
	c := &InstancesExportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.instancesexportrequest = instancesexportrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesExportCall) Fields(s ...googleapi.Field) *InstancesExportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesExportCall) Context(ctx context.Context) *InstancesExportCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesExportCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesExportCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancesexportrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/export")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.export" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesExportCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "sql.instances.failover":

type InstancesFailoverCall struct {
	s                        *Service
	project                  string
	instance                 string
	instancesfailoverrequest *InstancesFailoverRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
	header_                  http.Header
}

// Failover: Failover the instance to its failover replica instance.
func (r *InstancesService) Failover(project string, instance string, instancesfailoverrequest *InstancesFailoverRequest) *InstancesFailoverCall {
	c := &InstancesFailoverCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.instancesfailoverrequest = instancesfailoverrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesFailoverCall) Fields(s ...googleapi.Field) *InstancesFailoverCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesFailoverCall) Context(ctx context.Context) *InstancesFailoverCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesFailoverCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesFailoverCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancesfailoverrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/failover")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.failover" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesFailoverCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Failover the instance to its failover replica instance.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.failover",
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
	//       "description": "ID of the project that contains the read replica.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/failover",
	//   "request": {
	//     "$ref": "InstancesFailoverRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.get":

type InstancesGetCall struct {
	s            *Service
	project      string
	instance     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves a resource containing information about a Cloud SQL
// instance.
func (r *InstancesService) Get(project string, instance string) *InstancesGetCall {
	c := &InstancesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesGetCall) Fields(s ...googleapi.Field) *InstancesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InstancesGetCall) IfNoneMatch(entityTag string) *InstancesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesGetCall) Context(ctx context.Context) *InstancesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.get" call.
// Exactly one of *DatabaseInstance or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *DatabaseInstance.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InstancesGetCall) Do(opts ...googleapi.CallOption) (*DatabaseInstance, error) {
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
	ret := &DatabaseInstance{
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
	//   "description": "Retrieves a resource containing information about a Cloud SQL instance.",
	//   "httpMethod": "GET",
	//   "id": "sql.instances.get",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Database instance ID. This does not include the project ID.",
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
	//     "https://www.googleapis.com/auth/cloud-platform",
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
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// Import: Imports data into a Cloud SQL instance from a MySQL dump file
// in Google Cloud Storage.
func (r *InstancesService) Import(project string, instance string, instancesimportrequest *InstancesImportRequest) *InstancesImportCall {
	c := &InstancesImportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.instancesimportrequest = instancesimportrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesImportCall) Fields(s ...googleapi.Field) *InstancesImportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesImportCall) Context(ctx context.Context) *InstancesImportCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesImportCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesImportCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancesimportrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/import")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.import" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesImportCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//     "$ref": "Operation"
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
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Insert: Creates a new Cloud SQL instance.
func (r *InstancesService) Insert(project string, databaseinstance *DatabaseInstance) *InstancesInsertCall {
	c := &InstancesInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.databaseinstance = databaseinstance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesInsertCall) Fields(s ...googleapi.Field) *InstancesInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesInsertCall) Context(ctx context.Context) *InstancesInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.databaseinstance)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.insert" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesInsertCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.list":

type InstancesListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists instances under a given project in the alphabetical order
// of the instance name.
func (r *InstancesService) List(project string) *InstancesListCall {
	c := &InstancesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// Filter sets the optional parameter "filter": An expression for
// filtering the results of the request, such as by name or label.
func (c *InstancesListCall) Filter(filter string) *InstancesListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of results to return per response.
func (c *InstancesListCall) MaxResults(maxResults int64) *InstancesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": A
// previously-returned page token representing part of the larger set of
// results to view.
func (c *InstancesListCall) PageToken(pageToken string) *InstancesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesListCall) Fields(s ...googleapi.Field) *InstancesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InstancesListCall) IfNoneMatch(entityTag string) *InstancesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesListCall) Context(ctx context.Context) *InstancesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.list" call.
// Exactly one of *InstancesListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *InstancesListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InstancesListCall) Do(opts ...googleapi.CallOption) (*InstancesListResponse, error) {
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
	ret := &InstancesListResponse{
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
	//   "description": "Lists instances under a given project in the alphabetical order of the instance name.",
	//   "httpMethod": "GET",
	//   "id": "sql.instances.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "An expression for filtering the results of the request, such as by name or label.",
	//       "location": "query",
	//       "type": "string"
	//     },
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
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *InstancesListCall) Pages(ctx context.Context, f func(*InstancesListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "sql.instances.patch":

type InstancesPatchCall struct {
	s                *Service
	project          string
	instance         string
	databaseinstance *DatabaseInstance
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Patch: Updates settings of a Cloud SQL instance. Caution: This is not
// a partial update, so you must include values for all the settings
// that you want to retain. For partial updates, use patch.. This method
// supports patch semantics.
func (r *InstancesService) Patch(project string, instance string, databaseinstance *DatabaseInstance) *InstancesPatchCall {
	c := &InstancesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.databaseinstance = databaseinstance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesPatchCall) Fields(s ...googleapi.Field) *InstancesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesPatchCall) Context(ctx context.Context) *InstancesPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.databaseinstance)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.patch" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesPatchCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.promoteReplica":

type InstancesPromoteReplicaCall struct {
	s          *Service
	project    string
	instance   string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// PromoteReplica: Promotes the read replica instance to be a
// stand-alone Cloud SQL instance.
func (r *InstancesService) PromoteReplica(project string, instance string) *InstancesPromoteReplicaCall {
	c := &InstancesPromoteReplicaCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesPromoteReplicaCall) Fields(s ...googleapi.Field) *InstancesPromoteReplicaCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesPromoteReplicaCall) Context(ctx context.Context) *InstancesPromoteReplicaCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesPromoteReplicaCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesPromoteReplicaCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/promoteReplica")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.promoteReplica" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesPromoteReplicaCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Promotes the read replica instance to be a stand-alone Cloud SQL instance.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.promoteReplica",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL read replica instance name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "ID of the project that contains the read replica.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/promoteReplica",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.resetSslConfig":

type InstancesResetSslConfigCall struct {
	s          *Service
	project    string
	instance   string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// ResetSslConfig: Deletes all client certificates and generates a new
// server SSL certificate for the instance. The changes will not take
// effect until the instance is restarted. Existing instances without a
// server certificate will need to call this once to set a server
// certificate.
func (r *InstancesService) ResetSslConfig(project string, instance string) *InstancesResetSslConfigCall {
	c := &InstancesResetSslConfigCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesResetSslConfigCall) Fields(s ...googleapi.Field) *InstancesResetSslConfigCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesResetSslConfigCall) Context(ctx context.Context) *InstancesResetSslConfigCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesResetSslConfigCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesResetSslConfigCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/resetSslConfig")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.resetSslConfig" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesResetSslConfigCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Deletes all client certificates and generates a new server SSL certificate for the instance. The changes will not take effect until the instance is restarted. Existing instances without a server certificate will need to call this once to set a server certificate.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.resetSslConfig",
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
	//   "path": "projects/{project}/instances/{instance}/resetSslConfig",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.restart":

type InstancesRestartCall struct {
	s          *Service
	project    string
	instance   string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Restart: Restarts a Cloud SQL instance.
func (r *InstancesService) Restart(project string, instance string) *InstancesRestartCall {
	c := &InstancesRestartCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesRestartCall) Fields(s ...googleapi.Field) *InstancesRestartCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesRestartCall) Context(ctx context.Context) *InstancesRestartCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesRestartCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesRestartCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/restart")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.restart" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesRestartCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.restoreBackup":

type InstancesRestoreBackupCall struct {
	s                             *Service
	project                       string
	instance                      string
	instancesrestorebackuprequest *InstancesRestoreBackupRequest
	urlParams_                    gensupport.URLParams
	ctx_                          context.Context
	header_                       http.Header
}

// RestoreBackup: Restores a backup of a Cloud SQL instance.
func (r *InstancesService) RestoreBackup(project string, instance string, instancesrestorebackuprequest *InstancesRestoreBackupRequest) *InstancesRestoreBackupCall {
	c := &InstancesRestoreBackupCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.instancesrestorebackuprequest = instancesrestorebackuprequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesRestoreBackupCall) Fields(s ...googleapi.Field) *InstancesRestoreBackupCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesRestoreBackupCall) Context(ctx context.Context) *InstancesRestoreBackupCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesRestoreBackupCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesRestoreBackupCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancesrestorebackuprequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/restoreBackup")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.restoreBackup" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesRestoreBackupCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Restores a backup of a Cloud SQL instance.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.restoreBackup",
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
	//   "path": "projects/{project}/instances/{instance}/restoreBackup",
	//   "request": {
	//     "$ref": "InstancesRestoreBackupRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.startReplica":

type InstancesStartReplicaCall struct {
	s          *Service
	project    string
	instance   string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// StartReplica: Starts the replication in the read replica instance.
func (r *InstancesService) StartReplica(project string, instance string) *InstancesStartReplicaCall {
	c := &InstancesStartReplicaCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesStartReplicaCall) Fields(s ...googleapi.Field) *InstancesStartReplicaCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesStartReplicaCall) Context(ctx context.Context) *InstancesStartReplicaCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesStartReplicaCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesStartReplicaCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/startReplica")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.startReplica" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesStartReplicaCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Starts the replication in the read replica instance.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.startReplica",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL read replica instance name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "ID of the project that contains the read replica.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/startReplica",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.stopReplica":

type InstancesStopReplicaCall struct {
	s          *Service
	project    string
	instance   string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// StopReplica: Stops the replication in the read replica instance.
func (r *InstancesService) StopReplica(project string, instance string) *InstancesStopReplicaCall {
	c := &InstancesStopReplicaCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesStopReplicaCall) Fields(s ...googleapi.Field) *InstancesStopReplicaCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesStopReplicaCall) Context(ctx context.Context) *InstancesStopReplicaCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesStopReplicaCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesStopReplicaCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/stopReplica")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.stopReplica" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesStopReplicaCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Stops the replication in the read replica instance.",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.stopReplica",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Cloud SQL read replica instance name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "ID of the project that contains the read replica.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/stopReplica",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.instances.truncateLog":

type InstancesTruncateLogCall struct {
	s                           *Service
	project                     string
	instance                    string
	instancestruncatelogrequest *InstancesTruncateLogRequest
	urlParams_                  gensupport.URLParams
	ctx_                        context.Context
	header_                     http.Header
}

// TruncateLog: Truncate MySQL general and slow query log tables
func (r *InstancesService) TruncateLog(project string, instance string, instancestruncatelogrequest *InstancesTruncateLogRequest) *InstancesTruncateLogCall {
	c := &InstancesTruncateLogCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.instancestruncatelogrequest = instancestruncatelogrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesTruncateLogCall) Fields(s ...googleapi.Field) *InstancesTruncateLogCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesTruncateLogCall) Context(ctx context.Context) *InstancesTruncateLogCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesTruncateLogCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesTruncateLogCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.instancestruncatelogrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/truncateLog")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.truncateLog" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesTruncateLogCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Truncate MySQL general and slow query log tables",
	//   "httpMethod": "POST",
	//   "id": "sql.instances.truncateLog",
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
	//       "description": "Project ID of the Cloud SQL project.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/truncateLog",
	//   "request": {
	//     "$ref": "InstancesTruncateLogRequest"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
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
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Update: Updates settings of a Cloud SQL instance. Caution: This is
// not a partial update, so you must include values for all the settings
// that you want to retain. For partial updates, use patch.
func (r *InstancesService) Update(project string, instance string, databaseinstance *DatabaseInstance) *InstancesUpdateCall {
	c := &InstancesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.databaseinstance = databaseinstance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InstancesUpdateCall) Fields(s ...googleapi.Field) *InstancesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InstancesUpdateCall) Context(ctx context.Context) *InstancesUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *InstancesUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *InstancesUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.databaseinstance)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.instances.update" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InstancesUpdateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.operations.get":

type OperationsGetCall struct {
	s            *Service
	project      string
	operation    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Retrieves an instance operation that has been performed on an
// instance.
func (r *OperationsService) Get(project string, operation string) *OperationsGetCall {
	c := &OperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.operation = operation
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OperationsGetCall) Fields(s ...googleapi.Field) *OperationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OperationsGetCall) IfNoneMatch(entityTag string) *OperationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OperationsGetCall) Context(ctx context.Context) *OperationsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OperationsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OperationsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/operations/{operation}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"operation": c.operation,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.operations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *OperationsGetCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Retrieves an instance operation that has been performed on an instance.",
	//   "httpMethod": "GET",
	//   "id": "sql.operations.get",
	//   "parameterOrder": [
	//     "project",
	//     "operation"
	//   ],
	//   "parameters": {
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
	//   "path": "projects/{project}/operations/{operation}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.operations.list":

type OperationsListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all instance operations that have been performed on the
// given Cloud SQL instance in the reverse chronological order of the
// start time.
func (r *OperationsService) List(project string, instance string) *OperationsListCall {
	c := &OperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.urlParams_.Set("instance", instance)
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of operations per response.
func (c *OperationsListCall) MaxResults(maxResults int64) *OperationsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": A
// previously-returned page token representing part of the larger set of
// results to view.
func (c *OperationsListCall) PageToken(pageToken string) *OperationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OperationsListCall) Fields(s ...googleapi.Field) *OperationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *OperationsListCall) IfNoneMatch(entityTag string) *OperationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OperationsListCall) Context(ctx context.Context) *OperationsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OperationsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OperationsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/operations")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.operations.list" call.
// Exactly one of *OperationsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *OperationsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OperationsListCall) Do(opts ...googleapi.CallOption) (*OperationsListResponse, error) {
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
	ret := &OperationsListResponse{
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
	//       "location": "query",
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
	//   "path": "projects/{project}/operations",
	//   "response": {
	//     "$ref": "OperationsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *OperationsListCall) Pages(ctx context.Context, f func(*OperationsListResponse) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "sql.sslCerts.createEphemeral":

type SslCertsCreateEphemeralCall struct {
	s                              *Service
	project                        string
	instance                       string
	sslcertscreateephemeralrequest *SslCertsCreateEphemeralRequest
	urlParams_                     gensupport.URLParams
	ctx_                           context.Context
	header_                        http.Header
}

// CreateEphemeral: Generates a short-lived X509 certificate containing
// the provided public key and signed by a private key specific to the
// target instance. Users may use the certificate to authenticate as
// themselves when connecting to the database.
func (r *SslCertsService) CreateEphemeral(project string, instance string, sslcertscreateephemeralrequest *SslCertsCreateEphemeralRequest) *SslCertsCreateEphemeralCall {
	c := &SslCertsCreateEphemeralCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.sslcertscreateephemeralrequest = sslcertscreateephemeralrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SslCertsCreateEphemeralCall) Fields(s ...googleapi.Field) *SslCertsCreateEphemeralCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SslCertsCreateEphemeralCall) Context(ctx context.Context) *SslCertsCreateEphemeralCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SslCertsCreateEphemeralCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SslCertsCreateEphemeralCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.sslcertscreateephemeralrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/createEphemeral")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.sslCerts.createEphemeral" call.
// Exactly one of *SslCert or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *SslCert.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *SslCertsCreateEphemeralCall) Do(opts ...googleapi.CallOption) (*SslCert, error) {
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
	ret := &SslCert{
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
	//   "description": "Generates a short-lived X509 certificate containing the provided public key and signed by a private key specific to the target instance. Users may use the certificate to authenticate as themselves when connecting to the database.",
	//   "httpMethod": "POST",
	//   "id": "sql.sslCerts.createEphemeral",
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
	//       "description": "Project ID of the Cloud SQL project.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/createEphemeral",
	//   "request": {
	//     "$ref": "SslCertsCreateEphemeralRequest"
	//   },
	//   "response": {
	//     "$ref": "SslCert"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.sslCerts.delete":

type SslCertsDeleteCall struct {
	s               *Service
	project         string
	instance        string
	sha1Fingerprint string
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// Delete: Deletes the SSL certificate. The change will not take effect
// until the instance is restarted.
func (r *SslCertsService) Delete(project string, instance string, sha1Fingerprint string) *SslCertsDeleteCall {
	c := &SslCertsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.sha1Fingerprint = sha1Fingerprint
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SslCertsDeleteCall) Fields(s ...googleapi.Field) *SslCertsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SslCertsDeleteCall) Context(ctx context.Context) *SslCertsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SslCertsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SslCertsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/sslCerts/{sha1Fingerprint}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":         c.project,
		"instance":        c.instance,
		"sha1Fingerprint": c.sha1Fingerprint,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.sslCerts.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SslCertsDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Deletes the SSL certificate. The change will not take effect until the instance is restarted.",
	//   "httpMethod": "DELETE",
	//   "id": "sql.sslCerts.delete",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "sha1Fingerprint"
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
	//     },
	//     "sha1Fingerprint": {
	//       "description": "Sha1 FingerPrint.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/sslCerts/{sha1Fingerprint}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.sslCerts.get":

type SslCertsGetCall struct {
	s               *Service
	project         string
	instance        string
	sha1Fingerprint string
	urlParams_      gensupport.URLParams
	ifNoneMatch_    string
	ctx_            context.Context
	header_         http.Header
}

// Get: Retrieves a particular SSL certificate. Does not include the
// private key (required for usage). The private key must be saved from
// the response to initial creation.
func (r *SslCertsService) Get(project string, instance string, sha1Fingerprint string) *SslCertsGetCall {
	c := &SslCertsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.sha1Fingerprint = sha1Fingerprint
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SslCertsGetCall) Fields(s ...googleapi.Field) *SslCertsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *SslCertsGetCall) IfNoneMatch(entityTag string) *SslCertsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SslCertsGetCall) Context(ctx context.Context) *SslCertsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SslCertsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SslCertsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/sslCerts/{sha1Fingerprint}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":         c.project,
		"instance":        c.instance,
		"sha1Fingerprint": c.sha1Fingerprint,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.sslCerts.get" call.
// Exactly one of *SslCert or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *SslCert.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *SslCertsGetCall) Do(opts ...googleapi.CallOption) (*SslCert, error) {
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
	ret := &SslCert{
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
	//   "description": "Retrieves a particular SSL certificate. Does not include the private key (required for usage). The private key must be saved from the response to initial creation.",
	//   "httpMethod": "GET",
	//   "id": "sql.sslCerts.get",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "sha1Fingerprint"
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
	//     },
	//     "sha1Fingerprint": {
	//       "description": "Sha1 FingerPrint.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/sslCerts/{sha1Fingerprint}",
	//   "response": {
	//     "$ref": "SslCert"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.sslCerts.insert":

type SslCertsInsertCall struct {
	s                     *Service
	project               string
	instance              string
	sslcertsinsertrequest *SslCertsInsertRequest
	urlParams_            gensupport.URLParams
	ctx_                  context.Context
	header_               http.Header
}

// Insert: Creates an SSL certificate and returns it along with the
// private key and server certificate authority. The new certificate
// will not be usable until the instance is restarted.
func (r *SslCertsService) Insert(project string, instance string, sslcertsinsertrequest *SslCertsInsertRequest) *SslCertsInsertCall {
	c := &SslCertsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.sslcertsinsertrequest = sslcertsinsertrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SslCertsInsertCall) Fields(s ...googleapi.Field) *SslCertsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SslCertsInsertCall) Context(ctx context.Context) *SslCertsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SslCertsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SslCertsInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.sslcertsinsertrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/sslCerts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.sslCerts.insert" call.
// Exactly one of *SslCertsInsertResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *SslCertsInsertResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SslCertsInsertCall) Do(opts ...googleapi.CallOption) (*SslCertsInsertResponse, error) {
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
	ret := &SslCertsInsertResponse{
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
	//   "description": "Creates an SSL certificate and returns it along with the private key and server certificate authority. The new certificate will not be usable until the instance is restarted.",
	//   "httpMethod": "POST",
	//   "id": "sql.sslCerts.insert",
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
	//       "description": "Project ID of the project to which the newly created Cloud SQL instances should belong.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/sslCerts",
	//   "request": {
	//     "$ref": "SslCertsInsertRequest"
	//   },
	//   "response": {
	//     "$ref": "SslCertsInsertResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.sslCerts.list":

type SslCertsListCall struct {
	s            *Service
	project      string
	instance     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all of the current SSL certificates for the instance.
func (r *SslCertsService) List(project string, instance string) *SslCertsListCall {
	c := &SslCertsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SslCertsListCall) Fields(s ...googleapi.Field) *SslCertsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *SslCertsListCall) IfNoneMatch(entityTag string) *SslCertsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SslCertsListCall) Context(ctx context.Context) *SslCertsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *SslCertsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *SslCertsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/sslCerts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.sslCerts.list" call.
// Exactly one of *SslCertsListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *SslCertsListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SslCertsListCall) Do(opts ...googleapi.CallOption) (*SslCertsListResponse, error) {
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
	ret := &SslCertsListResponse{
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
	//   "description": "Lists all of the current SSL certificates for the instance.",
	//   "httpMethod": "GET",
	//   "id": "sql.sslCerts.list",
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
	//       "description": "Project ID of the project for which to list Cloud SQL instances.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/instances/{instance}/sslCerts",
	//   "response": {
	//     "$ref": "SslCertsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.tiers.list":

type TiersListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists all available service tiers for Google Cloud SQL, for
// example D1, D2. For related information, see Pricing.
func (r *TiersService) List(project string) *TiersListCall {
	c := &TiersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TiersListCall) Fields(s ...googleapi.Field) *TiersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TiersListCall) IfNoneMatch(entityTag string) *TiersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TiersListCall) Context(ctx context.Context) *TiersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TiersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TiersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/tiers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.tiers.list" call.
// Exactly one of *TiersListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *TiersListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TiersListCall) Do(opts ...googleapi.CallOption) (*TiersListResponse, error) {
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
	ret := &TiersListResponse{
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
	//   "description": "Lists all available service tiers for Google Cloud SQL, for example D1, D2. For related information, see Pricing.",
	//   "httpMethod": "GET",
	//   "id": "sql.tiers.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "Project ID of the project for which to list tiers.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{project}/tiers",
	//   "response": {
	//     "$ref": "TiersListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.users.delete":

type UsersDeleteCall struct {
	s          *Service
	project    string
	instance   string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a user from a Cloud SQL instance.
func (r *UsersService) Delete(project string, instance string, host string, name string) *UsersDeleteCall {
	c := &UsersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.urlParams_.Set("host", host)
	c.urlParams_.Set("name", name)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDeleteCall) Fields(s ...googleapi.Field) *UsersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDeleteCall) Context(ctx context.Context) *UsersDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.users.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Deletes a user from a Cloud SQL instance.",
	//   "httpMethod": "DELETE",
	//   "id": "sql.users.delete",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "host",
	//     "name"
	//   ],
	//   "parameters": {
	//     "host": {
	//       "description": "Host of the user in the instance.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "Database instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "Name of the user in the instance.",
	//       "location": "query",
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
	//   "path": "projects/{project}/instances/{instance}/users",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.users.insert":

type UsersInsertCall struct {
	s          *Service
	project    string
	instance   string
	user       *User
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Creates a new user in a Cloud SQL instance.
func (r *UsersService) Insert(project string, instance string, user *User) *UsersInsertCall {
	c := &UsersInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.user = user
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersInsertCall) Fields(s ...googleapi.Field) *UsersInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersInsertCall) Context(ctx context.Context) *UsersInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.user)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.users.insert" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersInsertCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Creates a new user in a Cloud SQL instance.",
	//   "httpMethod": "POST",
	//   "id": "sql.users.insert",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Database instance ID. This does not include the project ID.",
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
	//   "path": "projects/{project}/instances/{instance}/users",
	//   "request": {
	//     "$ref": "User"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.users.list":

type UsersListCall struct {
	s            *Service
	project      string
	instance     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists users in the specified Cloud SQL instance.
func (r *UsersService) List(project string, instance string) *UsersListCall {
	c := &UsersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersListCall) Fields(s ...googleapi.Field) *UsersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersListCall) IfNoneMatch(entityTag string) *UsersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersListCall) Context(ctx context.Context) *UsersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.users.list" call.
// Exactly one of *UsersListResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *UsersListResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersListCall) Do(opts ...googleapi.CallOption) (*UsersListResponse, error) {
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
	ret := &UsersListResponse{
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
	//   "description": "Lists users in the specified Cloud SQL instance.",
	//   "httpMethod": "GET",
	//   "id": "sql.users.list",
	//   "parameterOrder": [
	//     "project",
	//     "instance"
	//   ],
	//   "parameters": {
	//     "instance": {
	//       "description": "Database instance ID. This does not include the project ID.",
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
	//   "path": "projects/{project}/instances/{instance}/users",
	//   "response": {
	//     "$ref": "UsersListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}

// method id "sql.users.update":

type UsersUpdateCall struct {
	s          *Service
	project    string
	instance   string
	user       *User
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates an existing user in a Cloud SQL instance.
func (r *UsersService) Update(project string, instance string, host string, name string, user *User) *UsersUpdateCall {
	c := &UsersUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.instance = instance
	c.urlParams_.Set("host", host)
	c.urlParams_.Set("name", name)
	c.user = user
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersUpdateCall) Fields(s ...googleapi.Field) *UsersUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersUpdateCall) Context(ctx context.Context) *UsersUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.user)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "projects/{project}/instances/{instance}/users")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":  c.project,
		"instance": c.instance,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "sql.users.update" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersUpdateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	ret := &Operation{
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
	//   "description": "Updates an existing user in a Cloud SQL instance.",
	//   "httpMethod": "PUT",
	//   "id": "sql.users.update",
	//   "parameterOrder": [
	//     "project",
	//     "instance",
	//     "host",
	//     "name"
	//   ],
	//   "parameters": {
	//     "host": {
	//       "description": "Host of the user in the instance.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "instance": {
	//       "description": "Database instance ID. This does not include the project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "Name of the user in the instance.",
	//       "location": "query",
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
	//   "path": "projects/{project}/instances/{instance}/users",
	//   "request": {
	//     "$ref": "User"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/sqlservice.admin"
	//   ]
	// }

}
