// Package dataproc provides access to the Google Cloud Dataproc API.
//
// See https://cloud.google.com/dataproc/
//
// Usage example:
//
//   import "google.golang.org/api/dataproc/v1alpha1"
//   ...
//   dataprocService, err := dataproc.New(oauthHttpClient)
package dataproc // import "google.golang.org/api/dataproc/v1alpha1"

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

const apiId = "dataproc:v1alpha1"
const apiName = "dataproc"
const apiVersion = "v1alpha1"
const basePath = "https://dataproc.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Operations = NewOperationsService(s)
	s.Projects = NewProjectsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Operations *OperationsService

	Projects *ProjectsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewOperationsService(s *Service) *OperationsService {
	rs := &OperationsService{s: s}
	return rs
}

type OperationsService struct {
	s *Service
}

func NewProjectsService(s *Service) *ProjectsService {
	rs := &ProjectsService{s: s}
	rs.Regions = NewProjectsRegionsService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	Regions *ProjectsRegionsService
}

func NewProjectsRegionsService(s *Service) *ProjectsRegionsService {
	rs := &ProjectsRegionsService{s: s}
	rs.Clusters = NewProjectsRegionsClustersService(s)
	rs.Jobs = NewProjectsRegionsJobsService(s)
	return rs
}

type ProjectsRegionsService struct {
	s *Service

	Clusters *ProjectsRegionsClustersService

	Jobs *ProjectsRegionsJobsService
}

func NewProjectsRegionsClustersService(s *Service) *ProjectsRegionsClustersService {
	rs := &ProjectsRegionsClustersService{s: s}
	return rs
}

type ProjectsRegionsClustersService struct {
	s *Service
}

func NewProjectsRegionsJobsService(s *Service) *ProjectsRegionsJobsService {
	rs := &ProjectsRegionsJobsService{s: s}
	return rs
}

type ProjectsRegionsJobsService struct {
	s *Service
}

// AcceleratorConfiguration: Specifies the type and number of
// accelerator cards attached to the instances of an instance group (see
// GPUs on Compute Engine).
type AcceleratorConfiguration struct {
	// AcceleratorCount: The number of the accelerator cards of this type
	// exposed to this instance.
	AcceleratorCount int64 `json:"acceleratorCount,omitempty"`

	// AcceleratorTypeUri: Full or partial URI of the accelerator type
	// resource to expose to this instance. See Google Compute Engine
	// AcceleratorTypes( /compute/docs/reference/beta/acceleratorTypes)
	AcceleratorTypeUri string `json:"acceleratorTypeUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AcceleratorCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AcceleratorCount") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *AcceleratorConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod AcceleratorConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CancelJobRequest: A request to cancel a job.
type CancelJobRequest struct {
}

// CancelOperationRequest: The request message for
// Operations.CancelOperation.
type CancelOperationRequest struct {
}

// Cluster: Describes the identifying information, configuration, and
// status of a cluster of Google Compute Engine instances.
type Cluster struct {
	// ClusterName: Required The cluster name. Cluster names within a
	// project must be unique. Names from deleted clusters can be reused.
	ClusterName string `json:"clusterName,omitempty"`

	// ClusterUuid: Output-only A cluster UUID (Unique Universal
	// Identifier). Cloud Dataproc generates this value when it creates the
	// cluster.
	ClusterUuid string `json:"clusterUuid,omitempty"`

	// Configuration: Required The cluster configuration. It may differ from
	// a user's initial configuration due to Cloud Dataproc setting of
	// default values and updating clusters.
	Configuration *ClusterConfiguration `json:"configuration,omitempty"`

	// CreateTime: Output-only The timestamp of cluster creation.
	CreateTime string `json:"createTime,omitempty"`

	// Labels: Optional The labels to associate with this cluster.Label keys
	// must be between 1 and 63 characters long, and must conform to the
	// following PCRE regular expression: \p{Ll}\p{Lo}{0,62}Label values
	// must be between 1 and 63 characters long, and must conform to the
	// following PCRE regular expression: \p{Ll}\p{Lo}\p{N}_-{0,63}No more
	// than 64 labels can be associated with a given cluster.
	Labels map[string]string `json:"labels,omitempty"`

	// Metrics: Contains cluster daemon metrics such as HDFS and YARN stats.
	Metrics *ClusterMetrics `json:"metrics,omitempty"`

	// ProjectId: Required The Google Cloud Platform project ID that the
	// cluster belongs to.
	ProjectId string `json:"projectId,omitempty"`

	// Status: Output-only Cluster status.
	Status *ClusterStatus `json:"status,omitempty"`

	// StatusHistory: Output-only Previous cluster statuses.
	StatusHistory []*ClusterStatus `json:"statusHistory,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ClusterName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClusterName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Cluster) MarshalJSON() ([]byte, error) {
	type noMethod Cluster
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClusterConfiguration: The cluster configuration.
type ClusterConfiguration struct {
	// ConfigurationBucket: Optional A Google Cloud Storage staging bucket
	// used for sharing generated SSH keys and configuration. If you do not
	// specify a staging bucket, Cloud Dataproc will determine an
	// appropriate Cloud Storage location (US, ASIA, or EU) for your
	// cluster's staging bucket according to the Google Compute Engine zone
	// where your cluster is deployed, then it will create and manage this
	// project-level, per-location bucket for you.
	ConfigurationBucket string `json:"configurationBucket,omitempty"`

	// GceClusterConfiguration: Optional The shared Google Compute Engine
	// configuration settings for all instances in a cluster.
	GceClusterConfiguration *GceClusterConfiguration `json:"gceClusterConfiguration,omitempty"`

	// GceConfiguration: Deprecated The Google Compute Engine configuration
	// settings for cluster resources.
	GceConfiguration *GceConfiguration `json:"gceConfiguration,omitempty"`

	// InitializationActions: Optional Commands to execute on each node
	// after configuration is completed. By default, executables are run on
	// master and all worker nodes. You can test a node's <code>role</code>
	// metadata to run an executable on a master or worker node, as shown
	// below:
	// ROLE=$(/usr/share/google/get_metadata_value attributes/role)
	// if [[ "${ROLE}" == 'Master' ]]; then
	//   ... master specific actions ...
	// else
	//   ... worker specific actions ...
	// fi
	//
	InitializationActions []*NodeInitializationAction `json:"initializationActions,omitempty"`

	// MasterConfiguration: Optional The Google Compute Engine configuration
	// settings for the master instance in a cluster.
	MasterConfiguration *InstanceGroupConfiguration `json:"masterConfiguration,omitempty"`

	// MasterDiskConfiguration: Deprecated The configuration settings of
	// master node disk options.
	MasterDiskConfiguration *DiskConfiguration `json:"masterDiskConfiguration,omitempty"`

	// MasterName: Deprecated The Master's hostname. Dataproc derives the
	// name from cluster_name if not set by user (recommended practice is to
	// let Dataproc derive the name). Derived master name example: hadoop-m.
	MasterName string `json:"masterName,omitempty"`

	// NumWorkers: Deprecated The number of worker nodes in the cluster.
	NumWorkers int64 `json:"numWorkers,omitempty"`

	// SecondaryWorkerConfiguration: Optional The Google Compute Engine
	// configuration settings for additional worker instances in a cluster.
	SecondaryWorkerConfiguration *InstanceGroupConfiguration `json:"secondaryWorkerConfiguration,omitempty"`

	// SoftwareConfiguration: Optional The configuration settings for
	// software inside the cluster.
	SoftwareConfiguration *SoftwareConfiguration `json:"softwareConfiguration,omitempty"`

	// WorkerConfiguration: Optional The Google Compute Engine configuration
	// settings for worker instances in a cluster.
	WorkerConfiguration *InstanceGroupConfiguration `json:"workerConfiguration,omitempty"`

	// WorkerDiskConfiguration: Deprecated The configuration settings of
	// worker node disk options.
	WorkerDiskConfiguration *DiskConfiguration `json:"workerDiskConfiguration,omitempty"`

	// Workers: Deprecated The list of worker node names. Dataproc derives
	// the names from cluster_name and num_workers if not set by user
	// (recommended practice is to let Dataproc derive the name). Derived
	// worker node name example: hadoop-w-0.
	Workers []string `json:"workers,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ConfigurationBucket")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ConfigurationBucket") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ClusterConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod ClusterConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClusterMetrics: Contains cluster daemon metrics, such as HDFS and
// YARN stats.
type ClusterMetrics struct {
	// HdfsMetrics: The HDFS metrics.
	HdfsMetrics map[string]string `json:"hdfsMetrics,omitempty"`

	// YarnMetrics: The YARN metrics.
	YarnMetrics map[string]string `json:"yarnMetrics,omitempty"`

	// ForceSendFields is a list of field names (e.g. "HdfsMetrics") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "HdfsMetrics") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ClusterMetrics) MarshalJSON() ([]byte, error) {
	type noMethod ClusterMetrics
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClusterOperationMetadata: Metadata describing the operation.
type ClusterOperationMetadata struct {
	// ClusterName: Output-only. Name of the cluster for the operation.
	ClusterName string `json:"clusterName,omitempty"`

	// ClusterUuid: Output-only. Cluster UUID for the operation.
	ClusterUuid string `json:"clusterUuid,omitempty"`

	// Description: Output-only. Short description of operation.
	Description string `json:"description,omitempty"`

	// Labels: Output-only. Labels associated with the operation
	Labels map[string]string `json:"labels,omitempty"`

	// OperationType: Output-only. The operation type.
	OperationType string `json:"operationType,omitempty"`

	// Status: Output-only. Current operation status.
	Status *ClusterOperationStatus `json:"status,omitempty"`

	// StatusHistory: Output-only. The previous operation status.
	StatusHistory []*ClusterOperationStatus `json:"statusHistory,omitempty"`

	// Warnings: Output-only. Errors encountered during operation execution.
	Warnings []string `json:"warnings,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClusterName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClusterName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ClusterOperationMetadata) MarshalJSON() ([]byte, error) {
	type noMethod ClusterOperationMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClusterOperationStatus: The status of the operation.
type ClusterOperationStatus struct {
	// Details: Output-only.A message containing any operation metadata
	// details.
	Details string `json:"details,omitempty"`

	// InnerState: Output-only. A message containing the detailed operation
	// state.
	InnerState string `json:"innerState,omitempty"`

	// State: Output-only. A message containing the operation state.
	//
	// Possible values:
	//   "UNKNOWN" - Unused.
	//   "PENDING" - The operation has been created.
	//   "RUNNING" - The operation is running.
	//   "DONE" - The operation is done; either cancelled or completed.
	State string `json:"state,omitempty"`

	// StateStartTime: Output-only. The time this state was entered.
	StateStartTime string `json:"stateStartTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Details") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Details") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ClusterOperationStatus) MarshalJSON() ([]byte, error) {
	type noMethod ClusterOperationStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ClusterStatus: The status of a cluster and its instances.
type ClusterStatus struct {
	// Detail: Optional details of cluster's state.
	Detail string `json:"detail,omitempty"`

	// State: The cluster's state.
	//
	// Possible values:
	//   "UNKNOWN" - The cluster state is unknown.
	//   "CREATING" - The cluster is being created and set up. It is not
	// ready for use.
	//   "RUNNING" - The cluster is currently running and healthy. It is
	// ready for use.
	//   "ERROR" - The cluster encountered an error. It is not ready for
	// use.
	//   "DELETING" - The cluster is being deleted. It cannot be used.
	//   "UPDATING" - The cluster is being updated. It continues to accept
	// and process jobs.
	State string `json:"state,omitempty"`

	// StateStartTime: Time when this state was entered.
	StateStartTime string `json:"stateStartTime,omitempty"`

	// Substate: Output-only Additional state information that includes
	// status reported by the agent.
	//
	// Possible values:
	//   "UNSPECIFIED"
	//   "UNHEALTHY" - The cluster is known to be in an unhealthy state (for
	// example, critical daemons are not running or HDFS capacity is
	// exhausted).Applies to RUNNING state.
	//   "STALE_STATUS" - The agent-reported status is out of date (may
	// occur if Cloud Dataproc loses communication with Agent).Applies to
	// RUNNING state.
	Substate string `json:"substate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Detail") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Detail") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ClusterStatus) MarshalJSON() ([]byte, error) {
	type noMethod ClusterStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DiagnoseClusterOutputLocation: The location where output from
// diagnostic command can be found.
type DiagnoseClusterOutputLocation struct {
	// OutputUri: Output-only The Google Cloud Storage URI of the diagnostic
	// output. This will be a plain text file with summary of collected
	// diagnostics.
	OutputUri string `json:"outputUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OutputUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OutputUri") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DiagnoseClusterOutputLocation) MarshalJSON() ([]byte, error) {
	type noMethod DiagnoseClusterOutputLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DiagnoseClusterResults: The location of diagnostic output.
type DiagnoseClusterResults struct {
	// OutputUri: Output-only. The Google Cloud Storage URI of the
	// diagnostic output. The output report is a plain text file with a
	// summary of collected diagnostics.
	OutputUri string `json:"outputUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OutputUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "OutputUri") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *DiagnoseClusterResults) MarshalJSON() ([]byte, error) {
	type noMethod DiagnoseClusterResults
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// DiskConfiguration: Specifies the configuration of disk options for a
// group of VM instances.
type DiskConfiguration struct {
	// BootDiskSizeGb: Optional Size in GB of the boot disk (default is
	// 500GB).
	BootDiskSizeGb int64 `json:"bootDiskSizeGb,omitempty"`

	// NumLocalSsds: Optional Number of attached SSDs, from 0 to 4 (default
	// is 0). If SSDs are not attached, the boot disk is used to store
	// runtime logs, and HDFS data. If one or more SSDs are attached, this
	// runtime bulk data is spread across them, and the boot disk contains
	// only basic configuration and installed binaries.
	NumLocalSsds int64 `json:"numLocalSsds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BootDiskSizeGb") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BootDiskSizeGb") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *DiskConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod DiskConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Empty: A generic empty message that you can re-use to avoid defining
// duplicated empty messages in your APIs. A typical example is to use
// it as the request or the response type of an API method. For
// instance:
// service Foo {
//   rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty);
// }
// The JSON representation for Empty is empty JSON object {}.
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// GceClusterConfiguration: Common configuration settings for resources
// of Google Compute Engine cluster instances, applicable to all
// instances in the cluster.
type GceClusterConfiguration struct {
	// InternalIpOnly: If true, all instances in the cluser will only have
	// internal IP addresses. By default, clusters are not restricted to
	// internal IP addresses, and will have ephemeral external IP addresses
	// assigned to each instance. This restriction can only be enabled for
	// subnetwork enabled networks, and all off-cluster dependencies must be
	// configured to be accessible without external IP addresses.
	InternalIpOnly bool `json:"internalIpOnly,omitempty"`

	// Metadata: The Google Compute Engine metadata entries to add to all
	// instances.
	Metadata map[string]string `json:"metadata,omitempty"`

	// NetworkUri: The Google Compute Engine network to be used for machine
	// communications. Cannot be specified with subnetwork_uri. If neither
	// network_uri nor subnetwork_uri is specified, the "default" network of
	// the project is used, if it exists. Cannot be a "Custom Subnet
	// Network" (see https://cloud.google.com/compute/docs/subnetworks for
	// more information). Example:
	// compute.googleapis.com/projects/[project_id]/regions/global/default.
	NetworkUri string `json:"networkUri,omitempty"`

	// ServiceAccount: Optional The service account of the instances.
	// Defaults to the default Google Compute Engine service account. Custom
	// service accounts need permissions equivalent to the folloing IAM
	// roles:
	// roles/logging.logWriter
	// roles/storage.objectAdmin(see
	// https://cloud.google.com/compute/docs/access/service-accounts#custom_service_accounts for more information). Example:
	// [account_id]@[project_id].iam.gserviceaccount.com
	ServiceAccount string `json:"serviceAccount,omitempty"`

	// ServiceAccountScopes: The service account scopes included in Google
	// Compute Engine instances. Must include devstorage.full_control to
	// enable the Google Cloud Storage connector. Example
	// "auth.googleapis.com/compute" and
	// "auth.googleapis.com/devstorage.full_control".
	ServiceAccountScopes []string `json:"serviceAccountScopes,omitempty"`

	// SubnetworkUri: The Google Compute Engine subnetwork to be used for
	// machine communications. Cannot be specified with network_uri.
	// Example:
	// compute.googleapis.com/projects/[project_id]/regions/us-east1/sub0.
	SubnetworkUri string `json:"subnetworkUri,omitempty"`

	// Tags: The Google Compute Engine tags to add to all instances.
	Tags []string `json:"tags,omitempty"`

	// ZoneUri: Required The zone where the Google Compute Engine cluster
	// will be located. Example: "compute.googleapis.com/projects/project_id
	// /zones/us-east1-a".
	ZoneUri string `json:"zoneUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "InternalIpOnly") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InternalIpOnly") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *GceClusterConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod GceClusterConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// GceConfiguration: Deprecated Common configuration settings for
// resources of Google Compute Engine cluster instances, applicable to
// all instances in the cluster.
type GceConfiguration struct {
	// ImageUri: Deprecated The Google Compute Engine image resource used
	// for cluster instances. Example:
	// "compute.googleapis.com/projects/debian-cloud
	// /global/images/backports-debian-7-wheezy-v20140904".
	ImageUri string `json:"imageUri,omitempty"`

	// MachineTypeUri: Deprecated The Google Compute Engine machine type
	// used for cluster instances. Example:
	// "compute.googleapis.com/projects/project_id
	// /zones/us-east1-a/machineTypes/n1-standard-2".
	MachineTypeUri string `json:"machineTypeUri,omitempty"`

	// NetworkUri: Deprecated The Google Compute Engine network to be used
	// for machine communications. Inbound SSH connections are necessary to
	// complete cluster configuration. Example
	// "compute.googleapis.com/projects/project_id
	// /zones/us-east1-a/default".
	NetworkUri string `json:"networkUri,omitempty"`

	// ServiceAccountScopes: Deprecated The service account scopes included
	// in Google Compute Engine instances. Must include
	// devstorage.full_control to enable the Google Cloud Storage connector.
	// Example "auth.googleapis.com/compute" and
	// "auth.googleapis.com/devstorage.full_control".
	ServiceAccountScopes []string `json:"serviceAccountScopes,omitempty"`

	// ZoneUri: Deprecated The zone where the Google Compute Engine cluster
	// will be located. Example: "compute.googleapis.com/projects/project_id
	// /zones/us-east1-a".
	ZoneUri string `json:"zoneUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ImageUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ImageUri") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *GceConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod GceConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// HadoopJob: A Cloud Dataproc job for running Hadoop MapReduce jobs on
// YARN.
type HadoopJob struct {
	// ArchiveUris: Optional HCFS URIs of archives to be extracted in the
	// working directory of Hadoop drivers and tasks. Supported file types:
	// .jar, .tar, .tar.gz, .tgz, or .zip.
	ArchiveUris []string `json:"archiveUris,omitempty"`

	// Args: Optional The arguments to pass to the driver. Do not include
	// arguments, such as -libjars or -Dfoo=bar, that can be set as job
	// properties, since a collision may occur that causes an incorrect job
	// submission.
	Args []string `json:"args,omitempty"`

	// FileUris: Optional HCFS URIs of files to be copied to the working
	// directory of Hadoop drivers and distributed tasks. Useful for naively
	// parallel tasks.
	FileUris []string `json:"fileUris,omitempty"`

	// JarFileUris: Optional Jar file URIs to add to the CLASSPATHs of the
	// Hadoop driver and tasks.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// LoggingConfiguration: Optional The runtime log configuration for job
	// execution.
	LoggingConfiguration *LoggingConfiguration `json:"loggingConfiguration,omitempty"`

	// MainClass: The name of the driver's main class. The jar file
	// containing the class must be in the default CLASSPATH or specified in
	// jar_file_uris.
	MainClass string `json:"mainClass,omitempty"`

	// MainJarFileUri: The Hadoop Compatible Filesystem (HCFS) URI of the
	// jar file containing the main class. Examples:
	// gs://foo-bucket/analytics-binaries/extract-useful-metrics-mr.jar
	// hdfs:/tmp/test-samples/custom-wordcount.jar
	// file:///home/usr/lib/hadoop-mapreduce/hadoop-mapreduce-examples.jar
	MainJarFileUri string `json:"mainJarFileUri,omitempty"`

	// Properties: Optional A mapping of property names to values, used to
	// configure Hadoop. Properties that conflict with values set by the
	// Cloud Dataproc API may be overwritten. Can include properties set in
	// /etc/hadoop/conf/*-site and classes in user code.
	Properties map[string]string `json:"properties,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ArchiveUris") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ArchiveUris") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *HadoopJob) MarshalJSON() ([]byte, error) {
	type noMethod HadoopJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// HiveJob: A Cloud Dataproc job for running Hive queries on YARN.
type HiveJob struct {
	// ContinueOnFailure: Optional Whether to continue executing queries if
	// a query fails. The default value is false. Setting to true can be
	// useful when executing independent parallel queries.
	ContinueOnFailure bool `json:"continueOnFailure,omitempty"`

	// JarFileUris: Optional HCFS URIs of jar files to add to the CLASSPATH
	// of the Hive server and Hadoop MapReduce (MR) tasks. Can contain Hive
	// SerDes and UDFs.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// Properties: Optional A mapping of property names and values, used to
	// configure Hive. Properties that conflict with values set by the Cloud
	// Dataproc API may be overwritten. Can include properties set in
	// /etc/hadoop/conf/*-site.xml, /etc/hive/conf/hive-site.xml, and
	// classes in user code.
	Properties map[string]string `json:"properties,omitempty"`

	// QueryFileUri: The HCFS URI of the script that contains Hive queries.
	QueryFileUri string `json:"queryFileUri,omitempty"`

	// QueryList: A list of queries.
	QueryList *QueryList `json:"queryList,omitempty"`

	// ScriptVariables: Optional Mapping of query variable names to values
	// (equivalent to the Hive command: 'SET name="value";').
	ScriptVariables map[string]string `json:"scriptVariables,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ContinueOnFailure")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ContinueOnFailure") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *HiveJob) MarshalJSON() ([]byte, error) {
	type noMethod HiveJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// InstanceGroupConfiguration: The configuration settings for Google
// Compute Engine resources in an instance group, such as a master or
// worker group.
type InstanceGroupConfiguration struct {
	// Accelerators: Optional The Google Compute Engine accelerator
	// configuration for these instances.
	Accelerators []*AcceleratorConfiguration `json:"accelerators,omitempty"`

	// DiskConfiguration: Disk option configuration settings.
	DiskConfiguration *DiskConfiguration `json:"diskConfiguration,omitempty"`

	// ImageUri: Output-only The Google Compute Engine image resource used
	// for cluster instances. Inferred from
	// SoftwareConfiguration.image_version. Example:
	// "compute.googleapis.com/projects/debian-cloud
	// /global/images/backports-debian-7-wheezy-v20140904".
	ImageUri string `json:"imageUri,omitempty"`

	// InstanceNames: The list of instance names. Dataproc derives the names
	// from cluster_name, num_instances, and the instance group if not set
	// by user (recommended practice is to let Dataproc derive the name).
	InstanceNames []string `json:"instanceNames,omitempty"`

	// IsPreemptible: Specifies that this instance group contains
	// Preemptible Instances.
	IsPreemptible bool `json:"isPreemptible,omitempty"`

	// MachineTypeUri: The Google Compute Engine machine type used for
	// cluster instances. Example:
	// "compute.googleapis.com/projects/project_id
	// /zones/us-east1-a/machineTypes/n1-standard-2".
	MachineTypeUri string `json:"machineTypeUri,omitempty"`

	// ManagedGroupConfiguration: Output-only The configuration for Google
	// Compute Engine Instance Group Manager that manages this group. This
	// is only used for preemptible instance groups.
	ManagedGroupConfiguration *ManagedGroupConfiguration `json:"managedGroupConfiguration,omitempty"`

	// NumInstances: The number of VM instances in the instance group. For
	// master instance groups, must be set to 1.
	NumInstances int64 `json:"numInstances,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Accelerators") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Accelerators") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *InstanceGroupConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod InstanceGroupConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Job: A Cloud Dataproc job resource.
type Job struct {
	// DriverControlFilesUri: Output-only If present, the location of
	// miscellaneous control files which may be used as part of job setup
	// and handling. If not present, control files may be placed in the same
	// location as driver_output_uri.
	DriverControlFilesUri string `json:"driverControlFilesUri,omitempty"`

	// DriverInputResourceUri: Output-only A URI pointing to the location of
	// the stdin of the job's driver program, only set if the job is
	// interactive.
	DriverInputResourceUri string `json:"driverInputResourceUri,omitempty"`

	// DriverOutputResourceUri: Output-only A URI pointing to the location
	// of the stdout of the job's driver program.
	DriverOutputResourceUri string `json:"driverOutputResourceUri,omitempty"`

	// DriverOutputUri: Output-only A URI pointing to the location of the
	// mixed stdout/stderr of the job's driver program&mdash;for example,
	// <code>gs://sysbucket123/foo-cluster/jobid-123/driver/output</code>.
	DriverOutputUri string `json:"driverOutputUri,omitempty"`

	// HadoopJob: Job is a Hadoop job.
	HadoopJob *HadoopJob `json:"hadoopJob,omitempty"`

	// HiveJob: Job is a Hive job.
	HiveJob *HiveJob `json:"hiveJob,omitempty"`

	// Interactive: Optional If set to true, then the driver's stdin will be
	// kept open and driver_input_uri will be set to provide a path at which
	// additional input can be sent to the driver.
	Interactive bool `json:"interactive,omitempty"`

	// Labels: Optional The labels to associate with this job.Label keys
	// must be between 1 and 63 characters long, and must conform to the
	// following regular expression: \p{Ll}\p{Lo}{0,62}Label values must be
	// between 1 and 63 characters long, and must conform to the following
	// regular expression: \p{Ll}\p{Lo}\p{N}_-{0,63}No more than 64 labels
	// can be associated with a given job.
	Labels map[string]string `json:"labels,omitempty"`

	// PigJob: Job is a Pig job.
	PigJob *PigJob `json:"pigJob,omitempty"`

	// Placement: Required Job information, including how, when, and where
	// to run the job.
	Placement *JobPlacement `json:"placement,omitempty"`

	// PysparkJob: Job is a Pyspark job.
	PysparkJob *PySparkJob `json:"pysparkJob,omitempty"`

	// Reference: Optional The fully-qualified reference to the job, which
	// can be used to obtain the equivalent REST path of the job resource.
	// If this property is not specified when a job is created, the server
	// generates a <code>job_id</code>.
	Reference *JobReference `json:"reference,omitempty"`

	// Scheduling: Optional Job scheduling configuration.
	Scheduling *JobScheduling `json:"scheduling,omitempty"`

	// SparkJob: Job is a Spark job.
	SparkJob *SparkJob `json:"sparkJob,omitempty"`

	// SparkSqlJob: Job is a SparkSql job.
	SparkSqlJob *SparkSqlJob `json:"sparkSqlJob,omitempty"`

	// Status: Output-only The job status. Additional application-specific
	// status information may be contained in the <code>type_job</code> and
	// <code>yarn_applications</code> fields.
	Status *JobStatus `json:"status,omitempty"`

	// StatusHistory: Output-only The previous job status.
	StatusHistory []*JobStatus `json:"statusHistory,omitempty"`

	// SubmittedBy: Output-only The email address of the user submitting the
	// job. For jobs submitted on the cluster, the address is
	// <code>username@hostname</code>.
	SubmittedBy string `json:"submittedBy,omitempty"`

	// YarnApplications: Output-only The collection of Yarn applications
	// spun up by this job.
	YarnApplications []*YarnApplication `json:"yarnApplications,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "DriverControlFilesUri") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DriverControlFilesUri") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Job) MarshalJSON() ([]byte, error) {
	type noMethod Job
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// JobPlacement: Cloud Dataproc job configuration.
type JobPlacement struct {
	// ClusterName: Required The name of the cluster where the job will be
	// submitted.
	ClusterName string `json:"clusterName,omitempty"`

	// ClusterUuid: Output-only A cluster UUID generated by the Dataproc
	// service when the job is submitted.
	ClusterUuid string `json:"clusterUuid,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClusterName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClusterName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobPlacement) MarshalJSON() ([]byte, error) {
	type noMethod JobPlacement
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// JobReference: Encapsulates the full scoping used to reference a job.
type JobReference struct {
	// JobId: Required The job ID, which must be unique within the project.
	// The job ID is generated by the server upon job submission or provided
	// by the user as a means to perform retries without creating duplicate
	// jobs. The ID must contain only letters (a-z, A-Z), numbers (0-9),
	// underscores (_), or dashes (-). The maximum length is 100 characters.
	JobId string `json:"jobId,omitempty"`

	// ProjectId: Required The ID of the Google Cloud Platform project that
	// the job belongs to.
	ProjectId string `json:"projectId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "JobId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "JobId") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobReference) MarshalJSON() ([]byte, error) {
	type noMethod JobReference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// JobScheduling: Job scheduling options.Beta Feature: These options are
// available for testing purposes only. They may be changed before final
// release.
type JobScheduling struct {
	// MaxFailuresPerHour: Optional Maximum number of times per hour a
	// driver may be restarted as a result of driver terminating with
	// non-zero code before job is reported failed.A job may be reported as
	// thrashing if driver exits with non-zero code 4 times within 10 minute
	// window.Maximum value is 10.
	MaxFailuresPerHour int64 `json:"maxFailuresPerHour,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxFailuresPerHour")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MaxFailuresPerHour") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *JobScheduling) MarshalJSON() ([]byte, error) {
	type noMethod JobScheduling
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// JobStatus: Cloud Dataproc job status.
type JobStatus struct {
	// Details: Optional Job state details, such as an error description if
	// the state is <code>ERROR</code>.
	Details string `json:"details,omitempty"`

	// EndTime: The time when the job completed.
	EndTime string `json:"endTime,omitempty"`

	// InsertTime: The time of the job request.
	InsertTime string `json:"insertTime,omitempty"`

	// StartTime: The time when the server started the job.
	StartTime string `json:"startTime,omitempty"`

	// State: Required A state message specifying the overall job state.
	//
	// Possible values:
	//   "STATE_UNSPECIFIED" - The job state is unknown.
	//   "PENDING" - The job is pending; it has been submitted, but is not
	// yet running.
	//   "SETUP_DONE" - Job has been received by the service and completed
	// initial setup; it will shortly be submitted to the cluster.
	//   "RUNNING" - The job is running on the cluster.
	//   "CANCEL_PENDING" - A CancelJob request has been received, but is
	// pending.
	//   "CANCEL_STARTED" - Transient in-flight resources have been
	// canceled, and the request to cancel the running job has been issued
	// to the cluster.
	//   "CANCELLED" - The job cancelation was successful.
	//   "DONE" - The job has completed successfully.
	//   "ERROR" - The job has completed, but encountered an error.
	//   "ATTEMPT_FAILURE" - Job attempt has failed. The detail field
	// contains failure details for this attempt.Applies to restartable jobs
	// only.
	State string `json:"state,omitempty"`

	// StateStartTime: Output-only The time when this state was entered.
	StateStartTime string `json:"stateStartTime,omitempty"`

	// Substate: Output-only Additional state information, which includes
	// status reported by the agent.
	//
	// Possible values:
	//   "UNSPECIFIED"
	//   "SUBMITTED" - The Job is submitted to the agent.Applies to RUNNING
	// state.
	//   "QUEUED" - The Job has been received and is awaiting execution (it
	// may be waiting for a condition to be met). See the "details" field
	// for the reason for the delay.Applies to RUNNING state.
	//   "STALE_STATUS" - The agent-reported status is out of date, which
	// may be caused by a loss of communication between the agent and Cloud
	// Dataproc. If the agent does not send a timely update, the job will
	// fail.Applies to RUNNING state.
	Substate string `json:"substate,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Details") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Details") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *JobStatus) MarshalJSON() ([]byte, error) {
	type noMethod JobStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListClustersResponse: The list of all clusters in a project.
type ListClustersResponse struct {
	// Clusters: Output-only The clusters in the project.
	Clusters []*Cluster `json:"clusters,omitempty"`

	// NextPageToken: The standard List next-page token.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Clusters") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Clusters") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListClustersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListClustersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListJobsRequest: A request to list jobs in a project.
type ListJobsRequest struct {
	// ClusterName: Optional If set, the returned jobs list includes only
	// jobs that were submitted to the named cluster.
	ClusterName string `json:"clusterName,omitempty"`

	// Filter: Optional A filter constraining which jobs to list. Valid
	// filters contain job state and label terms such as: labels.key1 = val1
	// AND (labels.k2 = val2 OR labels.k3 = val3)
	Filter string `json:"filter,omitempty"`

	// JobStateMatcher: Optional Specifies enumerated categories of jobs to
	// list.
	//
	// Possible values:
	//   "ALL" - Match all jobs, regardless of state.
	//   "ACTIVE" - Only match jobs in non-terminal states: PENDING,
	// RUNNING, CANCEL_PENDING
	//   "NON_ACTIVE" - Only match jobs in terminal states: CANCELLED, DONE,
	// ERROR
	JobStateMatcher string `json:"jobStateMatcher,omitempty"`

	// PageSize: Optional The number of results to return in each response.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: Optional The page token, returned by a previous call, to
	// request the next page of results.
	PageToken string `json:"pageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClusterName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClusterName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListJobsRequest) MarshalJSON() ([]byte, error) {
	type noMethod ListJobsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListJobsResponse: A response to a request to list jobs in a project.
type ListJobsResponse struct {
	// Jobs: Output-only Jobs list.
	Jobs []*Job `json:"jobs,omitempty"`

	// NextPageToken: Optional This token is included in the response if
	// there are more results to fetch. To fetch additional results, provide
	// this value as the page_token in a subsequent
	// <code>ListJobsRequest</code>.
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

	// NullFields is a list of field names (e.g. "Jobs") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListJobsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListJobsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ListOperationsResponse: The response message for
// Operations.ListOperations.
type ListOperationsResponse struct {
	// NextPageToken: The standard List next-page token.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Operations: A list of operations that match the specified filter in
	// the request.
	Operations []*Operation `json:"operations,omitempty"`

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

	// NullFields is a list of field names (e.g. "NextPageToken") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *ListOperationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListOperationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LoggingConfiguration: The runtime logging configuration of the job.
type LoggingConfiguration struct {
	// DriverLogLevels: The per-package log levels for the driver. This may
	// include 'root' package name to configure rootLogger. Examples:
	// com.google = FATAL, root = INFO, org.apache = DEBUG
	DriverLogLevels map[string]string `json:"driverLogLevels,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DriverLogLevels") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DriverLogLevels") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *LoggingConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod LoggingConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// ManagedGroupConfiguration: Specifies the resources used to actively
// manage an instance group.
type ManagedGroupConfiguration struct {
	// InstanceGroupManagerName: Output-only The name of Instance Group
	// Manager managing this group.
	InstanceGroupManagerName string `json:"instanceGroupManagerName,omitempty"`

	// InstanceTemplateName: Output-only The name of Instance Template used
	// for Managed Instance Group.
	InstanceTemplateName string `json:"instanceTemplateName,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "InstanceGroupManagerName") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "InstanceGroupManagerName")
	// to include in API requests with the JSON null value. By default,
	// fields with empty values are omitted from API requests. However, any
	// field with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *ManagedGroupConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod ManagedGroupConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// NodeInitializationAction: Specifies an executable to run on a fully
// configured node and a timeout period for executable completion.
type NodeInitializationAction struct {
	// ExecutableFile: Required Google Cloud Storage URI of executable file.
	ExecutableFile string `json:"executableFile,omitempty"`

	// ExecutionTimeout: Optional Amount of time executable has to complete.
	// Default is 10 minutes. Cluster creation fails with an explanatory
	// error message (the name of the executable that caused the error and
	// the exceeded timeout period) if the executable is not completed at
	// end of the timeout period.
	ExecutionTimeout string `json:"executionTimeout,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExecutableFile") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ExecutableFile") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *NodeInitializationAction) MarshalJSON() ([]byte, error) {
	type noMethod NodeInitializationAction
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Operation: An asynchronous operation in a project that runs over a
// given cluster. Used to track the progress of a user request that is
// running asynchronously. Examples include creating a cluster, updating
// a cluster, and deleting a cluster.
type Operation struct {
	// Done: Indicates if the operation is done. If true, the operation is
	// complete and the result is available. If false, the operation is
	// still in progress.
	Done bool `json:"done,omitempty"`

	// Error: The error result of the operation in case of failure.
	Error *Status `json:"error,omitempty"`

	// Metadata: Service-specific metadata associated with the operation.
	Metadata googleapi.RawMessage `json:"metadata,omitempty"`

	// Name: The name of the operation resource, in the format
	// projects/project_id/operations/operation_id
	Name string `json:"name,omitempty"`

	// Response: The operation response. If the called method returns no
	// data on success, the response is google.protobuf.Empty. If the called
	// method is Get,Create or Update, the response is the resource. For all
	// other methods, the response type is a concatenation of the method
	// name and "Response". For example, if the called method is
	// TakeSnapshot(), the response type is TakeSnapshotResponse.
	Response googleapi.RawMessage `json:"response,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Done") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Done") to include in API
	// requests with the JSON null value. By default, fields with empty
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

// OperationMetadata: Metadata describing the operation.
type OperationMetadata struct {
	// ClusterName: Name of the cluster for the operation.
	ClusterName string `json:"clusterName,omitempty"`

	// ClusterUuid: Cluster UUId for the operation.
	ClusterUuid string `json:"clusterUuid,omitempty"`

	// Description: Output-only Short description of operation.
	Description string `json:"description,omitempty"`

	// Details: A message containing any operation metadata details.
	Details string `json:"details,omitempty"`

	// EndTime: The time that the operation completed.
	EndTime string `json:"endTime,omitempty"`

	// InnerState: A message containing the detailed operation state.
	InnerState string `json:"innerState,omitempty"`

	// InsertTime: The time that the operation was requested.
	InsertTime string `json:"insertTime,omitempty"`

	// OperationType: Output-only The operation type.
	OperationType string `json:"operationType,omitempty"`

	// StartTime: The time that the operation was started by the server.
	StartTime string `json:"startTime,omitempty"`

	// State: A message containing the operation state.
	//
	// Possible values:
	//   "UNKNOWN" - Unused.
	//   "PENDING" - The operation has been created.
	//   "RUNNING" - The operation is currently running.
	//   "DONE" - The operation is done, either cancelled or completed.
	State string `json:"state,omitempty"`

	// Status: Output-only Current operation status.
	Status *OperationStatus `json:"status,omitempty"`

	// StatusHistory: Output-only Previous operation status.
	StatusHistory []*OperationStatus `json:"statusHistory,omitempty"`

	// Warnings: Output-only Errors encountered during operation execution.
	Warnings []string `json:"warnings,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClusterName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ClusterName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationMetadata) MarshalJSON() ([]byte, error) {
	type noMethod OperationMetadata
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// OperationStatus: The status of the operation.
type OperationStatus struct {
	// Details: A message containing any operation metadata details.
	Details string `json:"details,omitempty"`

	// InnerState: A message containing the detailed operation state.
	InnerState string `json:"innerState,omitempty"`

	// State: A message containing the operation state.
	//
	// Possible values:
	//   "UNKNOWN" - Unused.
	//   "PENDING" - The operation has been created.
	//   "RUNNING" - The operation is running.
	//   "DONE" - The operation is done; either cancelled or completed.
	State string `json:"state,omitempty"`

	// StateStartTime: The time this state was entered.
	StateStartTime string `json:"stateStartTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Details") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Details") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *OperationStatus) MarshalJSON() ([]byte, error) {
	type noMethod OperationStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PigJob: A Cloud Dataproc job for running Pig queries on YARN.
type PigJob struct {
	// ContinueOnFailure: Optional Whether to continue executing queries if
	// a query fails. The default value is false. Setting to true can be
	// useful when executing independent parallel queries.
	ContinueOnFailure bool `json:"continueOnFailure,omitempty"`

	// JarFileUris: Optional HCFS URIs of jar files to add to the CLASSPATH
	// of the Pig Client and Hadoop MapReduce (MR) tasks. Can contain Pig
	// UDFs.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// LoggingConfiguration: Optional The runtime log configuration for job
	// execution.
	LoggingConfiguration *LoggingConfiguration `json:"loggingConfiguration,omitempty"`

	// Properties: Optional A mapping of property names to values, used to
	// configure Pig. Properties that conflict with values set by the Cloud
	// Dataproc API may be overwritten. Can include properties set in
	// /etc/hadoop/conf/*-site.xml, /etc/pig/conf/pig.properties, and
	// classes in user code.
	Properties map[string]string `json:"properties,omitempty"`

	// QueryFileUri: The HCFS URI of the script that contains the Pig
	// queries.
	QueryFileUri string `json:"queryFileUri,omitempty"`

	// QueryList: A list of queries.
	QueryList *QueryList `json:"queryList,omitempty"`

	// ScriptVariables: Optional Mapping of query variable names to values
	// (equivalent to the Pig command: "name=value").
	ScriptVariables map[string]string `json:"scriptVariables,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ContinueOnFailure")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ContinueOnFailure") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *PigJob) MarshalJSON() ([]byte, error) {
	type noMethod PigJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PySparkJob: A Cloud Dataproc job for running PySpark applications on
// YARN.
type PySparkJob struct {
	// ArchiveUris: Optional HCFS URIs of archives to be extracted in the
	// working directory of .jar, .tar, .tar.gz, .tgz, and .zip.
	ArchiveUris []string `json:"archiveUris,omitempty"`

	// Args: Optional The arguments to pass to the driver. Do not include
	// arguments, such as --conf, that can be set as job properties, since a
	// collision may occur that causes an incorrect job submission.
	Args []string `json:"args,omitempty"`

	// FileUris: Optional HCFS URIs of files to be copied to the working
	// directory of Python drivers and distributed tasks. Useful for naively
	// parallel tasks.
	FileUris []string `json:"fileUris,omitempty"`

	// JarFileUris: Optional HCFS URIs of jar files to add to the CLASSPATHs
	// of the Python driver and tasks.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// LoggingConfiguration: Optional The runtime log configuration for job
	// execution.
	LoggingConfiguration *LoggingConfiguration `json:"loggingConfiguration,omitempty"`

	// MainPythonFileUri: Required The Hadoop Compatible Filesystem (HCFS)
	// URI of the main Python file to use as the driver. Must be a .py file.
	MainPythonFileUri string `json:"mainPythonFileUri,omitempty"`

	// Properties: Optional A mapping of property names to values, used to
	// configure PySpark. Properties that conflict with values set by the
	// Cloud Dataproc API may be overwritten. Can include properties set in
	// /etc/spark/conf/spark-defaults.conf and classes in user code.
	Properties map[string]string `json:"properties,omitempty"`

	// PythonFileUris: Optional HCFS file URIs of Python files to pass to
	// the PySpark framework. Supported file types: .py, .egg, and .zip.
	PythonFileUris []string `json:"pythonFileUris,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ArchiveUris") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ArchiveUris") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PySparkJob) MarshalJSON() ([]byte, error) {
	type noMethod PySparkJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// QueryList: A list of queries to run on a cluster.
type QueryList struct {
	// Queries: Required The queries to execute. You do not need to
	// terminate a query with a semicolon. Multiple queries can be specified
	// in one string by separating each with a semicolon. Here is an example
	// of an Cloud Dataproc API snippet that uses a QueryList to specify a
	// HiveJob:
	// "hiveJob": {
	//   "queryList": {
	//     "queries": [
	//       "query1",
	//       "query2",
	//       "query3;query4",
	//     ]
	//   }
	// }
	//
	Queries []string `json:"queries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Queries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Queries") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *QueryList) MarshalJSON() ([]byte, error) {
	type noMethod QueryList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SoftwareConfiguration: Specifies the selection and configuration of
// software inside the cluster.
type SoftwareConfiguration struct {
	// ImageVersion: Optional The version of software inside the cluster. It
	// must match the regular expression 0-9+.0-9+. If unspecified it will
	// default to latest version.
	ImageVersion string `json:"imageVersion,omitempty"`

	// Properties: Optional The properties to set on daemon configuration
	// files.Property keys are specified in "prefix:property" format, such
	// as "core:fs.defaultFS". The following are supported prefixes and
	// their mappings:  core - core-site.xml  hdfs - hdfs-site.xml  mapred -
	// mapred-site.xml  yarn - yarn-site.xml  hive - hive-site.xml  pig -
	// pig.properties  spark - spark-defaults.conf
	Properties map[string]string `json:"properties,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ImageVersion") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ImageVersion") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SoftwareConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod SoftwareConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SparkJob: A Cloud Dataproc job for running Spark applications on
// YARN.
type SparkJob struct {
	// ArchiveUris: Optional HCFS URIs of archives to be extracted in the
	// working directory of Spark drivers and tasks. Supported file types:
	// .jar, .tar, .tar.gz, .tgz, and .zip.
	ArchiveUris []string `json:"archiveUris,omitempty"`

	// Args: Optional The arguments to pass to the driver. Do not include
	// arguments, such as --conf, that can be set as job properties, since a
	// collision may occur that causes an incorrect job submission.
	Args []string `json:"args,omitempty"`

	// FileUris: Optional HCFS URIs of files to be copied to the working
	// directory of Spark drivers and distributed tasks. Useful for naively
	// parallel tasks.
	FileUris []string `json:"fileUris,omitempty"`

	// JarFileUris: Optional HCFS URIs of jar files to add to the CLASSPATHs
	// of the Spark driver and tasks.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// LoggingConfiguration: Optional The runtime log configuration for job
	// execution.
	LoggingConfiguration *LoggingConfiguration `json:"loggingConfiguration,omitempty"`

	// MainClass: The name of the driver's main class. The jar file that
	// contains the class must be in the default CLASSPATH or specified in
	// jar_file_uris.
	MainClass string `json:"mainClass,omitempty"`

	// MainJarFileUri: The Hadoop Compatible Filesystem (HCFS) URI of the
	// jar file that contains the main class.
	MainJarFileUri string `json:"mainJarFileUri,omitempty"`

	// Properties: Optional A mapping of property names to values, used to
	// configure Spark. Properties that conflict with values set by the
	// Cloud Dataproc API may be overwritten. Can include properties set in
	// /etc/spark/conf/spark-defaults.conf and classes in user code.
	Properties map[string]string `json:"properties,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ArchiveUris") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ArchiveUris") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SparkJob) MarshalJSON() ([]byte, error) {
	type noMethod SparkJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SparkSqlJob: A Cloud Dataproc job for running Spark SQL queries.
type SparkSqlJob struct {
	// JarFileUris: Optional HCFS URIs of jar files to be added to the Spark
	// CLASSPATH.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// LoggingConfiguration: Optional The runtime log configuration for job
	// execution.
	LoggingConfiguration *LoggingConfiguration `json:"loggingConfiguration,omitempty"`

	// Properties: Optional A mapping of property names to values, used to
	// configure Spark SQL's SparkConf. Properties that conflict with values
	// set by the Cloud Dataproc API may be overwritten.
	Properties map[string]string `json:"properties,omitempty"`

	// QueryFileUri: The HCFS URI of the script that contains SQL queries.
	QueryFileUri string `json:"queryFileUri,omitempty"`

	// QueryList: A list of queries.
	QueryList *QueryList `json:"queryList,omitempty"`

	// ScriptVariables: Optional Mapping of query variable names to values
	// (equivalent to the Spark SQL command: SET name="value";).
	ScriptVariables map[string]string `json:"scriptVariables,omitempty"`

	// ForceSendFields is a list of field names (e.g. "JarFileUris") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "JarFileUris") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SparkSqlJob) MarshalJSON() ([]byte, error) {
	type noMethod SparkSqlJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Status: The Status type defines a logical error model that is
// suitable for different programming environments, including REST APIs
// and RPC APIs. It is used by gRPC (https://github.com/grpc). The error
// model is designed to be:
// Simple to use and understand for most users
// Flexible enough to meet unexpected needsOverviewThe Status message
// contains three pieces of data: error code, error message, and error
// details. The error code should be an enum value of google.rpc.Code,
// but it may accept additional error codes if needed. The error message
// should be a developer-facing English message that helps developers
// understand and resolve the error. If a localized user-facing error
// message is needed, put the localized message in the error details or
// localize it in the client. The optional error details may contain
// arbitrary information about the error. There is a predefined set of
// error detail types in the package google.rpc that can be used for
// common error conditions.Language mappingThe Status message is the
// logical representation of the error model, but it is not necessarily
// the actual wire format. When the Status message is exposed in
// different client libraries and different wire protocols, it can be
// mapped differently. For example, it will likely be mapped to some
// exceptions in Java, but more likely mapped to some error codes in
// C.Other usesThe error model and the Status message can be used in a
// variety of environments, either with or without APIs, to provide a
// consistent developer experience across different environments.Example
// uses of this error model include:
// Partial errors. If a service needs to return partial errors to the
// client, it may embed the Status in the normal response to indicate
// the partial errors.
// Workflow errors. A typical workflow has multiple steps. Each step may
// have a Status message for error reporting.
// Batch operations. If a client uses batch request and batch response,
// the Status message should be used directly inside batch response, one
// for each error sub-response.
// Asynchronous operations. If an API call embeds asynchronous operation
// results in its response, the status of those operations should be
// represented directly using the Status message.
// Logging. If some API errors are stored in logs, the message Status
// could be used directly after any stripping needed for
// security/privacy reasons.
type Status struct {
	// Code: The status code, which should be an enum value of
	// google.rpc.Code.
	Code int64 `json:"code,omitempty"`

	// Details: A list of messages that carry the error details. There will
	// be a common set of message types for APIs to use.
	Details []googleapi.RawMessage `json:"details,omitempty"`

	// Message: A developer-facing error message, which should be in
	// English. Any user-facing error message should be localized and sent
	// in the google.rpc.Status.details field, or localized by the client.
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

func (s *Status) MarshalJSON() ([]byte, error) {
	type noMethod Status
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SubmitJobRequest: A job submission request.
type SubmitJobRequest struct {
	// Job: Required The job resource.
	Job *Job `json:"job,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Job") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Job") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SubmitJobRequest) MarshalJSON() ([]byte, error) {
	type noMethod SubmitJobRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// YarnApplication: A YARN application created by a job. Application
// information is a subset of
// <code>org.apache.hadoop.yarn.proto.YarnProtos.ApplicationReportProto</
// code>.
type YarnApplication struct {
	// Name: Required The application name.
	Name string `json:"name,omitempty"`

	// Progress: Required The numerical progress of the application, from 1
	// to 100.
	Progress float64 `json:"progress,omitempty"`

	// State: Required The application state.
	//
	// Possible values:
	//   "STATE_UNSPECIFIED" - Status is unspecified.
	//   "NEW" - Status is NEW.
	//   "NEW_SAVING" - Status is NEW_SAVING.
	//   "SUBMITTED" - Status is SUBMITTED.
	//   "ACCEPTED" - Status is ACCEPTED.
	//   "RUNNING" - Status is RUNNING.
	//   "FINISHED" - Status is FINISHED.
	//   "FAILED" - Status is FAILED.
	//   "KILLED" - Status is KILLED.
	State string `json:"state,omitempty"`

	// TrackingUrl: Optional The HTTP URL of the ApplicationMaster,
	// HistoryServer, or TimelineServer that provides application-specific
	// information. The URL uses the internal hostname, and requires a proxy
	// server for resolution and, possibly, access.
	TrackingUrl string `json:"trackingUrl,omitempty"`

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

func (s *YarnApplication) MarshalJSON() ([]byte, error) {
	type noMethod YarnApplication
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

func (s *YarnApplication) UnmarshalJSON(data []byte) error {
	type noMethod YarnApplication
	var s1 struct {
		Progress gensupport.JSONFloat64 `json:"progress"`
		*noMethod
	}
	s1.noMethod = (*noMethod)(s)
	if err := json.Unmarshal(data, &s1); err != nil {
		return err
	}
	s.Progress = float64(s1.Progress)
	return nil
}

// method id "dataproc.operations.cancel":

type OperationsCancelCall struct {
	s                      *Service
	name                   string
	canceloperationrequest *CancelOperationRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
	header_                http.Header
}

// Cancel: Starts asynchronous cancellation on a long-running operation.
// The server makes a best effort to cancel the operation, but success
// is not guaranteed. If the server doesn't support this method, it
// returns google.rpc.Code.UNIMPLEMENTED. Clients may use
// Operations.GetOperation or other methods to check whether the
// cancellation succeeded or the operation completed despite
// cancellation.
func (r *OperationsService) Cancel(name string, canceloperationrequest *CancelOperationRequest) *OperationsCancelCall {
	c := &OperationsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.canceloperationrequest = canceloperationrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OperationsCancelCall) Fields(s ...googleapi.Field) *OperationsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OperationsCancelCall) Context(ctx context.Context) *OperationsCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OperationsCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OperationsCancelCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.canceloperationrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/{+name}:cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.operations.cancel" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *OperationsCancelCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns google.rpc.Code.UNIMPLEMENTED. Clients may use Operations.GetOperation or other methods to check whether the cancellation succeeded or the operation completed despite cancellation.",
	//   "flatPath": "v1alpha1/operations/{operationsId}:cancel",
	//   "httpMethod": "POST",
	//   "id": "dataproc.operations.cancel",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be cancelled.",
	//       "location": "path",
	//       "pattern": "^operations/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/{+name}:cancel",
	//   "request": {
	//     "$ref": "CancelOperationRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.operations.delete":

type OperationsDeleteCall struct {
	s          *Service
	name       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes a long-running operation. It indicates the client is
// no longer interested in the operation result. It does not cancel the
// operation.
func (r *OperationsService) Delete(name string) *OperationsDeleteCall {
	c := &OperationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *OperationsDeleteCall) Fields(s ...googleapi.Field) *OperationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *OperationsDeleteCall) Context(ctx context.Context) *OperationsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *OperationsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *OperationsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.operations.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *OperationsDeleteCall) Do(opts ...googleapi.CallOption) (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Deletes a long-running operation. It indicates the client is no longer interested in the operation result. It does not cancel the operation.",
	//   "flatPath": "v1alpha1/operations/{operationsId}",
	//   "httpMethod": "DELETE",
	//   "id": "dataproc.operations.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be deleted.",
	//       "location": "path",
	//       "pattern": "^operations/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/{+name}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.operations.get":

type OperationsGetCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the latest state of a long-running operation. Clients may
// use this method to poll the operation result at intervals as
// recommended by the API service.
func (r *OperationsService) Get(name string) *OperationsGetCall {
	c := &OperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.operations.get" call.
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
	//   "description": "Gets the latest state of a long-running operation. Clients may use this method to poll the operation result at intervals as recommended by the API service.",
	//   "flatPath": "v1alpha1/operations/{operationsId}",
	//   "httpMethod": "GET",
	//   "id": "dataproc.operations.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The operation resource name.",
	//       "location": "path",
	//       "pattern": "^operations/.+$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/{+name}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.operations.list":

type OperationsListCall struct {
	s            *Service
	name         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Lists operations that match the specified filter in the
// request. If the server doesn't support this method, it returns
// google.rpc.Code.UNIMPLEMENTED.
func (r *OperationsService) List(name string) *OperationsListCall {
	c := &OperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": Required A JSON object
// that contains filters for the list operation, in the format
// {"key1":"value1","key2":"value2", ..., }. Possible keys include
// project_id, cluster_name, and operation_state_matcher.If project_id
// is set, requests the list of operations that belong to the specified
// Google Cloud Platform project ID. This key is required.If
// cluster_name is set, requests the list of operations that were
// submitted to the specified cluster name. This key is optional.If
// operation_state_matcher is set, requests the list of operations that
// match one of the following status options: ALL, ACTIVE, or
// NON_ACTIVE.
func (c *OperationsListCall) Filter(filter string) *OperationsListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The standard List
// page size.
func (c *OperationsListCall) PageSize(pageSize int64) *OperationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard List
// page token.
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.operations.list" call.
// Exactly one of *ListOperationsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListOperationsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OperationsListCall) Do(opts ...googleapi.CallOption) (*ListOperationsResponse, error) {
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
	ret := &ListOperationsResponse{
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
	//   "description": "Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns google.rpc.Code.UNIMPLEMENTED.",
	//   "flatPath": "v1alpha1/operations",
	//   "httpMethod": "GET",
	//   "id": "dataproc.operations.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Required A JSON object that contains filters for the list operation, in the format {\"key1\":\"value1\",\"key2\":\"value2\", ..., }. Possible keys include project_id, cluster_name, and operation_state_matcher.If project_id is set, requests the list of operations that belong to the specified Google Cloud Platform project ID. This key is required.If cluster_name is set, requests the list of operations that were submitted to the specified cluster name. This key is optional.If operation_state_matcher is set, requests the list of operations that match one of the following status options: ALL, ACTIVE, or NON_ACTIVE.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "name": {
	//       "description": "The operation collection name.",
	//       "location": "path",
	//       "pattern": "^operations$",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The standard List page size.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The standard List page token.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/{+name}",
	//   "response": {
	//     "$ref": "ListOperationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *OperationsListCall) Pages(ctx context.Context, f func(*ListOperationsResponse) error) error {
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

// method id "dataproc.projects.regions.clusters.create":

type ProjectsRegionsClustersCreateCall struct {
	s          *Service
	projectId  string
	region     string
	cluster    *Cluster
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Create: Request to create a cluster in a project.
func (r *ProjectsRegionsClustersService) Create(projectId string, region string, cluster *Cluster) *ProjectsRegionsClustersCreateCall {
	c := &ProjectsRegionsClustersCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.cluster = cluster
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRegionsClustersCreateCall) Fields(s ...googleapi.Field) *ProjectsRegionsClustersCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRegionsClustersCreateCall) Context(ctx context.Context) *ProjectsRegionsClustersCreateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRegionsClustersCreateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRegionsClustersCreateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.cluster)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/regions/{region}/clusters")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"region":    c.region,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.projects.regions.clusters.create" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsRegionsClustersCreateCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Request to create a cluster in a project.",
	//   "flatPath": "v1alpha1/projects/{projectId}/regions/{region}/clusters",
	//   "httpMethod": "POST",
	//   "id": "dataproc.projects.regions.clusters.create",
	//   "parameterOrder": [
	//     "projectId",
	//     "region"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "Required The ID of the Google Cloud Platform project that the cluster belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Required The Dataproc region in which to handle the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/regions/{region}/clusters",
	//   "request": {
	//     "$ref": "Cluster"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.regions.clusters.delete":

type ProjectsRegionsClustersDeleteCall struct {
	s           *Service
	projectId   string
	region      string
	clusterName string
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Delete: Request to delete a cluster in a project.
func (r *ProjectsRegionsClustersService) Delete(projectId string, region string, clusterName string) *ProjectsRegionsClustersDeleteCall {
	c := &ProjectsRegionsClustersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.clusterName = clusterName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRegionsClustersDeleteCall) Fields(s ...googleapi.Field) *ProjectsRegionsClustersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRegionsClustersDeleteCall) Context(ctx context.Context) *ProjectsRegionsClustersDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRegionsClustersDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRegionsClustersDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/regions/{region}/clusters/{clusterName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"region":      c.region,
		"clusterName": c.clusterName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.projects.regions.clusters.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsRegionsClustersDeleteCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Request to delete a cluster in a project.",
	//   "flatPath": "v1alpha1/projects/{projectId}/regions/{region}/clusters/{clusterName}",
	//   "httpMethod": "DELETE",
	//   "id": "dataproc.projects.regions.clusters.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "region",
	//     "clusterName"
	//   ],
	//   "parameters": {
	//     "clusterName": {
	//       "description": "Required The cluster name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Required The ID of the Google Cloud Platform project that the cluster belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Required The Dataproc region in which to handle the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/regions/{region}/clusters/{clusterName}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.regions.clusters.get":

type ProjectsRegionsClustersGetCall struct {
	s            *Service
	projectId    string
	region       string
	clusterName  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Request to get the resource representation for a cluster in a
// project.
func (r *ProjectsRegionsClustersService) Get(projectId string, region string, clusterName string) *ProjectsRegionsClustersGetCall {
	c := &ProjectsRegionsClustersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.clusterName = clusterName
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRegionsClustersGetCall) Fields(s ...googleapi.Field) *ProjectsRegionsClustersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsRegionsClustersGetCall) IfNoneMatch(entityTag string) *ProjectsRegionsClustersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRegionsClustersGetCall) Context(ctx context.Context) *ProjectsRegionsClustersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRegionsClustersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRegionsClustersGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/regions/{region}/clusters/{clusterName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"region":      c.region,
		"clusterName": c.clusterName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.projects.regions.clusters.get" call.
// Exactly one of *Cluster or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Cluster.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsRegionsClustersGetCall) Do(opts ...googleapi.CallOption) (*Cluster, error) {
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
	ret := &Cluster{
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
	//   "description": "Request to get the resource representation for a cluster in a project.",
	//   "flatPath": "v1alpha1/projects/{projectId}/regions/{region}/clusters/{clusterName}",
	//   "httpMethod": "GET",
	//   "id": "dataproc.projects.regions.clusters.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "region",
	//     "clusterName"
	//   ],
	//   "parameters": {
	//     "clusterName": {
	//       "description": "Required The cluster name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Required The ID of the Google Cloud Platform project that the cluster belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Required The Dataproc region in which to handle the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/regions/{region}/clusters/{clusterName}",
	//   "response": {
	//     "$ref": "Cluster"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.regions.clusters.list":

type ProjectsRegionsClustersListCall struct {
	s            *Service
	projectId    string
	region       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Request a list of all regions/{region}/clusters in a project.
func (r *ProjectsRegionsClustersService) List(projectId string, region string) *ProjectsRegionsClustersListCall {
	c := &ProjectsRegionsClustersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	return c
}

// Filter sets the optional parameter "filter": Optional A filter
// constraining which clusters to list. Valid filters contain label
// terms such as: labels.key1 = val1 AND (-labels.k2 = val2 OR labels.k3
// = val3)
func (c *ProjectsRegionsClustersListCall) Filter(filter string) *ProjectsRegionsClustersListCall {
	c.urlParams_.Set("filter", filter)
	return c
}

// PageSize sets the optional parameter "pageSize": The standard List
// page size.
func (c *ProjectsRegionsClustersListCall) PageSize(pageSize int64) *ProjectsRegionsClustersListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard List
// page token.
func (c *ProjectsRegionsClustersListCall) PageToken(pageToken string) *ProjectsRegionsClustersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRegionsClustersListCall) Fields(s ...googleapi.Field) *ProjectsRegionsClustersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsRegionsClustersListCall) IfNoneMatch(entityTag string) *ProjectsRegionsClustersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRegionsClustersListCall) Context(ctx context.Context) *ProjectsRegionsClustersListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRegionsClustersListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRegionsClustersListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/regions/{region}/clusters")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"region":    c.region,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.projects.regions.clusters.list" call.
// Exactly one of *ListClustersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListClustersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsRegionsClustersListCall) Do(opts ...googleapi.CallOption) (*ListClustersResponse, error) {
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
	ret := &ListClustersResponse{
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
	//   "description": "Request a list of all regions/{region}/clusters in a project.",
	//   "flatPath": "v1alpha1/projects/{projectId}/regions/{region}/clusters",
	//   "httpMethod": "GET",
	//   "id": "dataproc.projects.regions.clusters.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "region"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "Optional A filter constraining which clusters to list. Valid filters contain label terms such as: labels.key1 = val1 AND (-labels.k2 = val2 OR labels.k3 = val3)",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "The standard List page size.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The standard List page token.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Required The ID of the Google Cloud Platform project that the cluster belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Required The Dataproc region in which to handle the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/regions/{region}/clusters",
	//   "response": {
	//     "$ref": "ListClustersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsRegionsClustersListCall) Pages(ctx context.Context, f func(*ListClustersResponse) error) error {
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

// method id "dataproc.projects.regions.clusters.patch":

type ProjectsRegionsClustersPatchCall struct {
	s           *Service
	projectId   string
	region      string
	clusterName string
	cluster     *Cluster
	urlParams_  gensupport.URLParams
	ctx_        context.Context
	header_     http.Header
}

// Patch: Request to update a cluster in a project.
func (r *ProjectsRegionsClustersService) Patch(projectId string, region string, clusterName string, cluster *Cluster) *ProjectsRegionsClustersPatchCall {
	c := &ProjectsRegionsClustersPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.clusterName = clusterName
	c.cluster = cluster
	return c
}

// UpdateMask sets the optional parameter "updateMask": Required
// Specifies the path, relative to <code>Cluster</code>, of the field to
// update. For example, to change the number of workers in a cluster to
// 5, the <code>update_mask</code> parameter would be specified as
// <code>"configuration.worker_configuration.num_instances,"</code> and
// the PATCH request body would specify the new value, as follows:
// {
//   "configuration":{
//     "workerConfiguration":{
//       "numInstances":"5"
//     }
//   }
// }
// <strong>Note:</strong> Currently,
// <code>configuration.worker_configuration.num_instances</code> is the
// only field that can be updated.
func (c *ProjectsRegionsClustersPatchCall) UpdateMask(updateMask string) *ProjectsRegionsClustersPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRegionsClustersPatchCall) Fields(s ...googleapi.Field) *ProjectsRegionsClustersPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRegionsClustersPatchCall) Context(ctx context.Context) *ProjectsRegionsClustersPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRegionsClustersPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRegionsClustersPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.cluster)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/regions/{region}/clusters/{clusterName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"region":      c.region,
		"clusterName": c.clusterName,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.projects.regions.clusters.patch" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsRegionsClustersPatchCall) Do(opts ...googleapi.CallOption) (*Operation, error) {
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
	//   "description": "Request to update a cluster in a project.",
	//   "flatPath": "v1alpha1/projects/{projectId}/regions/{region}/clusters/{clusterName}",
	//   "httpMethod": "PATCH",
	//   "id": "dataproc.projects.regions.clusters.patch",
	//   "parameterOrder": [
	//     "projectId",
	//     "region",
	//     "clusterName"
	//   ],
	//   "parameters": {
	//     "clusterName": {
	//       "description": "Required The cluster name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Required The ID of the Google Cloud Platform project the cluster belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Required The Dataproc region in which to handle the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "Required Specifies the path, relative to \u003ccode\u003eCluster\u003c/code\u003e, of the field to update. For example, to change the number of workers in a cluster to 5, the \u003ccode\u003eupdate_mask\u003c/code\u003e parameter would be specified as \u003ccode\u003e\"configuration.worker_configuration.num_instances,\"\u003c/code\u003e and the PATCH request body would specify the new value, as follows:\n{\n  \"configuration\":{\n    \"workerConfiguration\":{\n      \"numInstances\":\"5\"\n    }\n  }\n}\n\u003cstrong\u003eNote:\u003c/strong\u003e Currently, \u003ccode\u003econfiguration.worker_configuration.num_instances\u003c/code\u003e is the only field that can be updated.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/regions/{region}/clusters/{clusterName}",
	//   "request": {
	//     "$ref": "Cluster"
	//   },
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.regions.jobs.cancel":

type ProjectsRegionsJobsCancelCall struct {
	s                *Service
	projectId        string
	region           string
	jobId            string
	canceljobrequest *CancelJobRequest
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Cancel: Starts a job cancellation request. To access the job resource
// after cancellation, call regions/{region}/jobs:list or
// regions/{region}/jobs:get.
func (r *ProjectsRegionsJobsService) Cancel(projectId string, region string, jobId string, canceljobrequest *CancelJobRequest) *ProjectsRegionsJobsCancelCall {
	c := &ProjectsRegionsJobsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.jobId = jobId
	c.canceljobrequest = canceljobrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRegionsJobsCancelCall) Fields(s ...googleapi.Field) *ProjectsRegionsJobsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRegionsJobsCancelCall) Context(ctx context.Context) *ProjectsRegionsJobsCancelCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRegionsJobsCancelCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRegionsJobsCancelCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.canceljobrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}:cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"region":    c.region,
		"jobId":     c.jobId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.projects.regions.jobs.cancel" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsRegionsJobsCancelCall) Do(opts ...googleapi.CallOption) (*Job, error) {
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
	ret := &Job{
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
	//   "description": "Starts a job cancellation request. To access the job resource after cancellation, call regions/{region}/jobs:list or regions/{region}/jobs:get.",
	//   "flatPath": "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}:cancel",
	//   "httpMethod": "POST",
	//   "id": "dataproc.projects.regions.jobs.cancel",
	//   "parameterOrder": [
	//     "projectId",
	//     "region",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "Required The job ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Required The ID of the Google Cloud Platform project that the job belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Required The Dataproc region in which to handle the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}:cancel",
	//   "request": {
	//     "$ref": "CancelJobRequest"
	//   },
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.regions.jobs.delete":

type ProjectsRegionsJobsDeleteCall struct {
	s          *Service
	projectId  string
	region     string
	jobId      string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes the job from the project. If the job is active, the
// delete fails, and the response returns FAILED_PRECONDITION.
func (r *ProjectsRegionsJobsService) Delete(projectId string, region string, jobId string) *ProjectsRegionsJobsDeleteCall {
	c := &ProjectsRegionsJobsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.jobId = jobId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRegionsJobsDeleteCall) Fields(s ...googleapi.Field) *ProjectsRegionsJobsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRegionsJobsDeleteCall) Context(ctx context.Context) *ProjectsRegionsJobsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRegionsJobsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRegionsJobsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"region":    c.region,
		"jobId":     c.jobId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.projects.regions.jobs.delete" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsRegionsJobsDeleteCall) Do(opts ...googleapi.CallOption) (*Job, error) {
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
	ret := &Job{
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
	//   "description": "Deletes the job from the project. If the job is active, the delete fails, and the response returns FAILED_PRECONDITION.",
	//   "flatPath": "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}",
	//   "httpMethod": "DELETE",
	//   "id": "dataproc.projects.regions.jobs.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "region",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "Required The job ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Required The ID of the Google Cloud Platform project that the job belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Required The Dataproc region in which to handle the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}",
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.regions.jobs.get":

type ProjectsRegionsJobsGetCall struct {
	s            *Service
	projectId    string
	region       string
	jobId        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets the resource representation for a job in a project.
func (r *ProjectsRegionsJobsService) Get(projectId string, region string, jobId string) *ProjectsRegionsJobsGetCall {
	c := &ProjectsRegionsJobsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.jobId = jobId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRegionsJobsGetCall) Fields(s ...googleapi.Field) *ProjectsRegionsJobsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsRegionsJobsGetCall) IfNoneMatch(entityTag string) *ProjectsRegionsJobsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRegionsJobsGetCall) Context(ctx context.Context) *ProjectsRegionsJobsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRegionsJobsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRegionsJobsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"region":    c.region,
		"jobId":     c.jobId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.projects.regions.jobs.get" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsRegionsJobsGetCall) Do(opts ...googleapi.CallOption) (*Job, error) {
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
	ret := &Job{
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
	//   "description": "Gets the resource representation for a job in a project.",
	//   "flatPath": "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}",
	//   "httpMethod": "GET",
	//   "id": "dataproc.projects.regions.jobs.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "region",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "Required The job ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Required The ID of the Google Cloud Platform project that the job belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Required The Dataproc region in which to handle the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}",
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.regions.jobs.list":

type ProjectsRegionsJobsListCall struct {
	s               *Service
	projectId       string
	region          string
	listjobsrequest *ListJobsRequest
	urlParams_      gensupport.URLParams
	ctx_            context.Context
	header_         http.Header
}

// List: Lists regions/{region}/jobs in a project.
func (r *ProjectsRegionsJobsService) List(projectId string, region string, listjobsrequest *ListJobsRequest) *ProjectsRegionsJobsListCall {
	c := &ProjectsRegionsJobsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.listjobsrequest = listjobsrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRegionsJobsListCall) Fields(s ...googleapi.Field) *ProjectsRegionsJobsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRegionsJobsListCall) Context(ctx context.Context) *ProjectsRegionsJobsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRegionsJobsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRegionsJobsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.listjobsrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/regions/{region}/jobs:list")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"region":    c.region,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.projects.regions.jobs.list" call.
// Exactly one of *ListJobsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListJobsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsRegionsJobsListCall) Do(opts ...googleapi.CallOption) (*ListJobsResponse, error) {
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
	ret := &ListJobsResponse{
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
	//   "description": "Lists regions/{region}/jobs in a project.",
	//   "flatPath": "v1alpha1/projects/{projectId}/regions/{region}/jobs:list",
	//   "httpMethod": "POST",
	//   "id": "dataproc.projects.regions.jobs.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "region"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "Required The ID of the Google Cloud Platform project that the job belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Required The Dataproc region in which to handle the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/regions/{region}/jobs:list",
	//   "request": {
	//     "$ref": "ListJobsRequest"
	//   },
	//   "response": {
	//     "$ref": "ListJobsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *ProjectsRegionsJobsListCall) Pages(ctx context.Context, f func(*ListJobsResponse) error) error {
	c.ctx_ = ctx
	defer func(pt string) { c.listjobsrequest.PageToken = pt }(c.listjobsrequest.PageToken) // reset paging to original point
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
		c.listjobsrequest.PageToken = x.NextPageToken
	}
}

// method id "dataproc.projects.regions.jobs.patch":

type ProjectsRegionsJobsPatchCall struct {
	s          *Service
	projectId  string
	region     string
	jobId      string
	job        *Job
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates a job in a project.
func (r *ProjectsRegionsJobsService) Patch(projectId string, region string, jobId string, job *Job) *ProjectsRegionsJobsPatchCall {
	c := &ProjectsRegionsJobsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.jobId = jobId
	c.job = job
	return c
}

// UpdateMask sets the optional parameter "updateMask": Required
// Specifies the path, relative to <code>Job</code>, of the field to
// update. For example, to update the labels of a Job the
// <code>update_mask</code> parameter would be specified as
// <code>labels</code>, and the PATCH request body would specify the new
// value. <strong>Note:</strong> Currently, <code>labels</code> is the
// only field that can be updated.
func (c *ProjectsRegionsJobsPatchCall) UpdateMask(updateMask string) *ProjectsRegionsJobsPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRegionsJobsPatchCall) Fields(s ...googleapi.Field) *ProjectsRegionsJobsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRegionsJobsPatchCall) Context(ctx context.Context) *ProjectsRegionsJobsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRegionsJobsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRegionsJobsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.job)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"region":    c.region,
		"jobId":     c.jobId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.projects.regions.jobs.patch" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsRegionsJobsPatchCall) Do(opts ...googleapi.CallOption) (*Job, error) {
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
	ret := &Job{
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
	//   "description": "Updates a job in a project.",
	//   "flatPath": "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}",
	//   "httpMethod": "PATCH",
	//   "id": "dataproc.projects.regions.jobs.patch",
	//   "parameterOrder": [
	//     "projectId",
	//     "region",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "Required The job ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Required The ID of the Google Cloud Platform project that the job belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Required The Dataproc region in which to handle the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "Required Specifies the path, relative to \u003ccode\u003eJob\u003c/code\u003e, of the field to update. For example, to update the labels of a Job the \u003ccode\u003eupdate_mask\u003c/code\u003e parameter would be specified as \u003ccode\u003elabels\u003c/code\u003e, and the PATCH request body would specify the new value. \u003cstrong\u003eNote:\u003c/strong\u003e Currently, \u003ccode\u003elabels\u003c/code\u003e is the only field that can be updated.",
	//       "format": "google-fieldmask",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/regions/{region}/jobs/{jobId}",
	//   "request": {
	//     "$ref": "Job"
	//   },
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.regions.jobs.submit":

type ProjectsRegionsJobsSubmitCall struct {
	s                *Service
	projectId        string
	region           string
	submitjobrequest *SubmitJobRequest
	urlParams_       gensupport.URLParams
	ctx_             context.Context
	header_          http.Header
}

// Submit: Submits a job to a cluster.
func (r *ProjectsRegionsJobsService) Submit(projectId string, region string, submitjobrequest *SubmitJobRequest) *ProjectsRegionsJobsSubmitCall {
	c := &ProjectsRegionsJobsSubmitCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.region = region
	c.submitjobrequest = submitjobrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsRegionsJobsSubmitCall) Fields(s ...googleapi.Field) *ProjectsRegionsJobsSubmitCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsRegionsJobsSubmitCall) Context(ctx context.Context) *ProjectsRegionsJobsSubmitCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *ProjectsRegionsJobsSubmitCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *ProjectsRegionsJobsSubmitCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.submitjobrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/regions/{region}/jobs:submit")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"region":    c.region,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "dataproc.projects.regions.jobs.submit" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsRegionsJobsSubmitCall) Do(opts ...googleapi.CallOption) (*Job, error) {
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
	ret := &Job{
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
	//   "description": "Submits a job to a cluster.",
	//   "flatPath": "v1alpha1/projects/{projectId}/regions/{region}/jobs:submit",
	//   "httpMethod": "POST",
	//   "id": "dataproc.projects.regions.jobs.submit",
	//   "parameterOrder": [
	//     "projectId",
	//     "region"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "Required The ID of the Google Cloud Platform project that the job belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "region": {
	//       "description": "Required The Dataproc region in which to handle the request.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/regions/{region}/jobs:submit",
	//   "request": {
	//     "$ref": "SubmitJobRequest"
	//   },
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}
