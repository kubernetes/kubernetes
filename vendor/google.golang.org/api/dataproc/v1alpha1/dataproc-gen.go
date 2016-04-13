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

	// Administrate log data for your projects
	LoggingAdminScope = "https://www.googleapis.com/auth/logging.admin"

	// View log data for your projects
	LoggingReadScope = "https://www.googleapis.com/auth/logging.read"

	// Submit log data for your projects
	LoggingWriteScope = "https://www.googleapis.com/auth/logging.write"
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
	rs.Clusters = NewProjectsClustersService(s)
	rs.Jobs = NewProjectsJobsService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	Clusters *ProjectsClustersService

	Jobs *ProjectsJobsService
}

func NewProjectsClustersService(s *Service) *ProjectsClustersService {
	rs := &ProjectsClustersService{s: s}
	rs.Agents = NewProjectsClustersAgentsService(s)
	rs.Tasks = NewProjectsClustersTasksService(s)
	return rs
}

type ProjectsClustersService struct {
	s *Service

	Agents *ProjectsClustersAgentsService

	Tasks *ProjectsClustersTasksService
}

func NewProjectsClustersAgentsService(s *Service) *ProjectsClustersAgentsService {
	rs := &ProjectsClustersAgentsService{s: s}
	return rs
}

type ProjectsClustersAgentsService struct {
	s *Service
}

func NewProjectsClustersTasksService(s *Service) *ProjectsClustersTasksService {
	rs := &ProjectsClustersTasksService{s: s}
	return rs
}

type ProjectsClustersTasksService struct {
	s *Service
}

func NewProjectsJobsService(s *Service) *ProjectsJobsService {
	rs := &ProjectsJobsService{s: s}
	return rs
}

type ProjectsJobsService struct {
	s *Service
}

// Agent: A record for a single agent within Dataproc.
type Agent struct {
	// AgentId: [Required] 64 characters matching the regular expression:
	// [a-z0-9.-]{1,64} An agent chosen ID. This should typically be the
	// hostname of the GCE virtual machine on which the agent is currently
	// running.
	AgentId string `json:"agentId,omitempty"`

	// AgentVersion: The version of this agent in HTTP User-Agent Header
	// value format (RFC 2616 section 14.43), e.g., "Dataproc-Agent/1.2".
	AgentVersion string `json:"agentVersion,omitempty"`

	// LastAgentUpdateTime: [Out] the last time this agent checked-in with
	// Dataproc.
	LastAgentUpdateTime string `json:"lastAgentUpdateTime,omitempty"`

	// Status: Agent status.
	//
	// Possible values:
	//   "STATUS_UNKNOWN"
	//   "INITIALIZING"
	//   "PERFORMING_CUSTOM_INITIALIZATION_ACTIONS"
	//   "RUNNING"
	//   "SETUP_FAILED"
	Status string `json:"status,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AgentId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Agent) MarshalJSON() ([]byte, error) {
	type noMethod Agent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
	// ClusterName: [Required] The cluster name. Cluster names within a
	// project must be unique. Names from deleted clusters can be reused.
	ClusterName string `json:"clusterName,omitempty"`

	// ClusterUuid: [Output-only] A cluster UUID (Unique Universal
	// Identifier). Cloud Dataproc generates this value when it creates the
	// cluster.
	ClusterUuid string `json:"clusterUuid,omitempty"`

	// Configuration: [Required] The cluster configuration. It may differ
	// from a user's initial configuration due to Cloud Dataproc setting of
	// default values and updating clusters.
	Configuration *ClusterConfiguration `json:"configuration,omitempty"`

	// CreateTime: [Output-only] The timestamp of cluster creation.
	CreateTime string `json:"createTime,omitempty"`

	// ProjectId: [Required] The Google Cloud Platform project ID that the
	// cluster belongs to.
	ProjectId string `json:"projectId,omitempty"`

	// Status: [Output-only] Cluster status.
	Status *ClusterStatus `json:"status,omitempty"`

	// StatusHistory: [Output-only] Previous cluster statuses.
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
}

func (s *Cluster) MarshalJSON() ([]byte, error) {
	type noMethod Cluster
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ClusterConfiguration: The cluster configuration.
type ClusterConfiguration struct {
	// ConfigurationBucket: [Optional] A Google Cloud Storage staging bucket
	// used for sharing generated SSH keys and configuration. If you do not
	// specify a staging bucket, Cloud Dataproc will determine an
	// appropriate Cloud Storage location (US, ASIA, or EU) for your
	// cluster's staging bucket according to the Google Compute Engine zone
	// where your cluster is deployed, then it will create and manage this
	// project-level, per-location bucket for you.
	ConfigurationBucket string `json:"configurationBucket,omitempty"`

	// GceClusterConfiguration: [Optional] The shared Google Compute Engine
	// configuration settings for all instances in a cluster.
	GceClusterConfiguration *GceClusterConfiguration `json:"gceClusterConfiguration,omitempty"`

	// GceConfiguration: [Deprecated] The Google Compute Engine
	// configuration settings for cluster resources.
	GceConfiguration *GceConfiguration `json:"gceConfiguration,omitempty"`

	// InitializationActions: [Optional] Commands to execute on each node
	// after configuration is completed. By default, executables are run on
	// master and all worker nodes. You can test a node's role metadata to
	// run an executable on a master or worker node, as shown below:
	// ROLE=$(/usr/share/google/get_metadata_value attributes/role) if [[
	// "${ROLE}" == 'Master' ]]; then ... master specific actions ... else
	// ... worker specific actions ... fi
	InitializationActions []*NodeInitializationAction `json:"initializationActions,omitempty"`

	// MasterConfiguration: [Optional] The Google Compute Engine
	// configuration settings for the master instance in a cluster.
	MasterConfiguration *InstanceGroupConfiguration `json:"masterConfiguration,omitempty"`

	// MasterDiskConfiguration: [Deprecated] The configuration settings of
	// master node disk options.
	MasterDiskConfiguration *DiskConfiguration `json:"masterDiskConfiguration,omitempty"`

	// MasterName: [Deprecated] The Master's hostname. Dataproc derives the
	// name from cluster_name if not set by user (recommended practice is to
	// let Dataproc derive the name). Derived master name example: hadoop-m.
	MasterName string `json:"masterName,omitempty"`

	// NumWorkers: [Deprecated] The number of worker nodes in the cluster.
	NumWorkers int64 `json:"numWorkers,omitempty"`

	// SecondaryWorkerConfiguration: [Optional] The Google Compute Engine
	// configuration settings for additional worker instances in a cluster.
	SecondaryWorkerConfiguration *InstanceGroupConfiguration `json:"secondaryWorkerConfiguration,omitempty"`

	// SoftwareConfiguration: [Optional] The configuration settings for
	// software inside the cluster.
	SoftwareConfiguration *SoftwareConfiguration `json:"softwareConfiguration,omitempty"`

	// WorkerConfiguration: [Optional] The Google Compute Engine
	// configuration settings for worker instances in a cluster.
	WorkerConfiguration *InstanceGroupConfiguration `json:"workerConfiguration,omitempty"`

	// WorkerDiskConfiguration: [Deprecated] The configuration settings of
	// worker node disk options.
	WorkerDiskConfiguration *DiskConfiguration `json:"workerDiskConfiguration,omitempty"`

	// Workers: [Deprecated] The list of worker node names. Dataproc derives
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
}

func (s *ClusterConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod ClusterConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ClusterStatus: The status of a cluster and its instances.
type ClusterStatus struct {
	// Detail: Optional details of cluster's state.
	Detail string `json:"detail,omitempty"`

	// State: The cluster's state.
	//
	// Possible values:
	//   "UNKNOWN"
	//   "CREATING"
	//   "RUNNING"
	//   "ERROR"
	//   "DELETING"
	//   "UPDATING"
	State string `json:"state,omitempty"`

	// StateStartTime: Time when this state was entered.
	StateStartTime string `json:"stateStartTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Detail") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ClusterStatus) MarshalJSON() ([]byte, error) {
	type noMethod ClusterStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DiagnoseClusterOutputLocation: The location where output from
// diagnostic command can be found.
type DiagnoseClusterOutputLocation struct {
	// OutputUri: [Output-only] The Google Cloud Storage URI of the
	// diagnostic output. This will be a plain text file with summary of
	// collected diagnostics.
	OutputUri string `json:"outputUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "OutputUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DiagnoseClusterOutputLocation) MarshalJSON() ([]byte, error) {
	type noMethod DiagnoseClusterOutputLocation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// DiskConfiguration: Specifies the configuration of disk options for a
// group of VM instances.
type DiskConfiguration struct {
	// BootDiskSizeGb: [Optional] Size in GB of the boot disk (default is
	// 500GB).
	BootDiskSizeGb int64 `json:"bootDiskSizeGb,omitempty"`

	// NumLocalSsds: [Optional] Number of attached SSDs, from 0 to 4
	// (default is 0). If SSDs are not attached, the boot disk is used to
	// store runtime logs, and HDFS data. If one or more SSDs are attached,
	// this runtime bulk data is spread across them, and the boot disk
	// contains only basic configuration and installed binaries.
	NumLocalSsds int64 `json:"numLocalSsds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BootDiskSizeGb") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DiskConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod DiskConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Empty: A generic empty message that you can re-use to avoid defining
// duplicated empty messages in your APIs. A typical example is to use
// it as the request or the response type of an API method. For
// instance: service Foo { rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty); } The JSON representation for `Empty` is
// empty JSON object `{}`.
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// GceClusterConfiguration: Common configuration settings for resources
// of Google Compute Engine cluster instances, applicable to all
// instances in the cluster.
type GceClusterConfiguration struct {
	// NetworkUri: The Google Compute Engine network to be used for machine
	// communications. Inbound SSH connections are necessary to complete
	// cluster configuration. Example
	// "compute.googleapis.com/projects/[project_id]
	// /zones/us-east1-a/default".
	NetworkUri string `json:"networkUri,omitempty"`

	// ServiceAccountScopes: The service account scopes included in Google
	// Compute Engine instances. Must include devstorage.full_control to
	// enable the Google Cloud Storage connector. Example
	// "auth.googleapis.com/compute" and
	// "auth.googleapis.com/devstorage.full_control".
	ServiceAccountScopes []string `json:"serviceAccountScopes,omitempty"`

	// ZoneUri: [Required] The zone where the Google Compute Engine cluster
	// will be located. Example:
	// "compute.googleapis.com/projects/[project_id] /zones/us-east1-a".
	ZoneUri string `json:"zoneUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "NetworkUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GceClusterConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod GceClusterConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// GceConfiguration: [Deprecated] Common configuration settings for
// resources of Google Compute Engine cluster instances, applicable to
// all instances in the cluster.
type GceConfiguration struct {
	// ImageUri: [Deprecated] The Google Compute Engine image resource used
	// for cluster instances. Example:
	// "compute.googleapis.com/projects/debian-cloud
	// /global/images/backports-debian-7-wheezy-v20140904".
	ImageUri string `json:"imageUri,omitempty"`

	// MachineTypeUri: [Deprecated] The Google Compute Engine machine type
	// used for cluster instances. Example:
	// "compute.googleapis.com/projects/[project_id]
	// /zones/us-east1-a/machineTypes/n1-standard-2".
	MachineTypeUri string `json:"machineTypeUri,omitempty"`

	// NetworkUri: [Deprecated] The Google Compute Engine network to be used
	// for machine communications. Inbound SSH connections are necessary to
	// complete cluster configuration. Example
	// "compute.googleapis.com/projects/[project_id]
	// /zones/us-east1-a/default".
	NetworkUri string `json:"networkUri,omitempty"`

	// ServiceAccountScopes: [Deprecated] The service account scopes
	// included in Google Compute Engine instances. Must include
	// devstorage.full_control to enable the Google Cloud Storage connector.
	// Example "auth.googleapis.com/compute" and
	// "auth.googleapis.com/devstorage.full_control".
	ServiceAccountScopes []string `json:"serviceAccountScopes,omitempty"`

	// ZoneUri: [Deprecated] The zone where the Google Compute Engine
	// cluster will be located. Example:
	// "compute.googleapis.com/projects/[project_id] /zones/us-east1-a".
	ZoneUri string `json:"zoneUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ImageUri") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GceConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod GceConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// HadoopJob: A Cloud Dataproc job for running Hadoop MapReduce jobs on
// YARN.
type HadoopJob struct {
	// ArchiveUris: [Optional] HCFS URIs of archives to be extracted in the
	// working directory of Hadoop drivers and tasks. Supported file types:
	// .jar, .tar, .tar.gz, .tgz, or .zip.
	ArchiveUris []string `json:"archiveUris,omitempty"`

	// Args: [Optional] The arguments to pass to the driver. Do not include
	// arguments, such as -libjars or -Dfoo=bar, that can be set as job
	// properties, since a collision may occur that causes an incorrect job
	// submission.
	Args []string `json:"args,omitempty"`

	// FileUris: [Optional] HCFS URIs of files to be copied to the working
	// directory of Hadoop drivers and distributed tasks. Useful for naively
	// parallel tasks.
	FileUris []string `json:"fileUris,omitempty"`

	// JarFileUris: [Optional] Jar file URIs to add to the CLASSPATHs of the
	// Hadoop driver and tasks.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// LoggingConfiguration: [Optional] The runtime log configuration for
	// job execution.
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

	// Properties: [Optional] A mapping of property names to values, used to
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
}

func (s *HadoopJob) MarshalJSON() ([]byte, error) {
	type noMethod HadoopJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// HiveJob: A Cloud Dataproc job for running Hive queries on YARN.
type HiveJob struct {
	// ContinueOnFailure: [Optional] Whether to continue executing queries
	// if a query fails. The default value is false. Setting to true can be
	// useful when executing independent parallel queries.
	ContinueOnFailure bool `json:"continueOnFailure,omitempty"`

	// JarFileUris: [Optional] HCFS URIs of jar files to add to the
	// CLASSPATH of the Hive server and Hadoop MapReduce (MR) tasks. Can
	// contain Hive SerDes and UDFs.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// Properties: [Optional] A mapping of property names and values, used
	// to configure Hive. Properties that conflict with values set by the
	// Cloud Dataproc API may be overwritten. Can include properties set in
	// /etc/hadoop/conf/*-site.xml, /etc/hive/conf/hive-site.xml, and
	// classes in user code.
	Properties map[string]string `json:"properties,omitempty"`

	// QueryFileUri: The HCFS URI of the script that contains Hive queries.
	QueryFileUri string `json:"queryFileUri,omitempty"`

	// QueryList: A list of queries.
	QueryList *QueryList `json:"queryList,omitempty"`

	// ScriptVariables: [Optional] Mapping of query variable names to values
	// (equivalent to the Hive command: 'SET name="value";').
	ScriptVariables map[string]string `json:"scriptVariables,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ContinueOnFailure")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *HiveJob) MarshalJSON() ([]byte, error) {
	type noMethod HiveJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// InstanceGroupConfiguration: The configuration settings for Google
// Compute Engine resources in an instance group, such as a master or
// worker group.
type InstanceGroupConfiguration struct {
	// DiskConfiguration: Disk option configuration settings.
	DiskConfiguration *DiskConfiguration `json:"diskConfiguration,omitempty"`

	// ImageUri: [Output-only] The Google Compute Engine image resource used
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
	// "compute.googleapis.com/projects/[project_id]
	// /zones/us-east1-a/machineTypes/n1-standard-2".
	MachineTypeUri string `json:"machineTypeUri,omitempty"`

	// ManagedGroupConfiguration: [Output-only] The configuration for Google
	// Compute Engine Instance Group Manager that manages this group. This
	// is only used for preemptible instance groups.
	ManagedGroupConfiguration *ManagedGroupConfiguration `json:"managedGroupConfiguration,omitempty"`

	// NumInstances: The number of VM instances in the instance group. For
	// master instance groups, must be set to 1.
	NumInstances int64 `json:"numInstances,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DiskConfiguration")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *InstanceGroupConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod InstanceGroupConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Job: A Cloud Dataproc job resource.
type Job struct {
	// DriverControlFilesUri: [Output-only] If present, the location of
	// miscellaneous control files which may be used as part of job setup
	// and handling. If not present, control files may be placed in the same
	// location as driver_output_uri.
	DriverControlFilesUri string `json:"driverControlFilesUri,omitempty"`

	// DriverInputResourceUri: [Output-only] A URI pointing to the location
	// of the stdin of the job's driver program, only set if the job is
	// interactive.
	DriverInputResourceUri string `json:"driverInputResourceUri,omitempty"`

	// DriverOutputResourceUri: [Output-only] A URI pointing to the location
	// of the stdout of the job's driver program.
	DriverOutputResourceUri string `json:"driverOutputResourceUri,omitempty"`

	// DriverOutputUri: [Output-only] A URI pointing to the location of the
	// mixed stdout/stderr of the job's driver program—for example,
	// gs://sysbucket123/foo-cluster/jobid-123/driver/output.
	DriverOutputUri string `json:"driverOutputUri,omitempty"`

	// HadoopJob: Job is a Hadoop job.
	HadoopJob *HadoopJob `json:"hadoopJob,omitempty"`

	// HiveJob: Job is a Hive job.
	HiveJob *HiveJob `json:"hiveJob,omitempty"`

	// Interactive: [Optional] If set to true, then the driver's stdin will
	// be kept open and driver_input_uri will be set to provide a path at
	// which additional input can be sent to the driver.
	Interactive bool `json:"interactive,omitempty"`

	// PigJob: Job is a Pig job.
	PigJob *PigJob `json:"pigJob,omitempty"`

	// Placement: [Required] Job information, including how, when, and where
	// to run the job.
	Placement *JobPlacement `json:"placement,omitempty"`

	// PysparkJob: Job is a Pyspark job.
	PysparkJob *PySparkJob `json:"pysparkJob,omitempty"`

	// Reference: [Optional] The fully-qualified reference to the job, which
	// can be used to obtain the equivalent REST path of the job resource.
	// If this property is not specified when a job is created, the server
	// generates a job_id.
	Reference *JobReference `json:"reference,omitempty"`

	// SparkJob: Job is a Spark job.
	SparkJob *SparkJob `json:"sparkJob,omitempty"`

	// SparkSqlJob: Job is a SparkSql job.
	SparkSqlJob *SparkSqlJob `json:"sparkSqlJob,omitempty"`

	// Status: [Output-only] The job status. Additional application-specific
	// status information may be contained in the type_job and
	// yarn_applications fields.
	Status *JobStatus `json:"status,omitempty"`

	// StatusHistory: [Output-only] The previous job status.
	StatusHistory []*JobStatus `json:"statusHistory,omitempty"`

	// SubmittedBy: [Output-only] The email address of the user submitting
	// the job. For jobs submitted on the cluster, the address is
	// username@hostname.
	SubmittedBy string `json:"submittedBy,omitempty"`

	// YarnApplications: [Output-only] The collection of Yarn applications
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
}

func (s *Job) MarshalJSON() ([]byte, error) {
	type noMethod Job
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// JobPlacement: Cloud Dataproc job configuration.
type JobPlacement struct {
	// ClusterName: [Required] The name of the cluster where the job will be
	// submitted.
	ClusterName string `json:"clusterName,omitempty"`

	// ClusterUuid: [Output-only] A cluster UUID generated by the Dataproc
	// service when the job is submitted.
	ClusterUuid string `json:"clusterUuid,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClusterName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *JobPlacement) MarshalJSON() ([]byte, error) {
	type noMethod JobPlacement
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// JobReference: Encapsulates the full scoping used to reference a job.
type JobReference struct {
	// JobId: [Required] The job ID, which must be unique within the
	// project. The job ID is generated by the server upon job submission or
	// provided by the user as a means to perform retries without creating
	// duplicate jobs. The ID must contain only letters (a-z, A-Z), numbers
	// (0-9), underscores (_), or dashes (-). The maximum length is 512
	// characters.
	JobId string `json:"jobId,omitempty"`

	// ProjectId: [Required] The ID of the Google Cloud Platform project
	// that the job belongs to.
	ProjectId string `json:"projectId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "JobId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *JobReference) MarshalJSON() ([]byte, error) {
	type noMethod JobReference
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// JobStatus: Cloud Dataproc job status.
type JobStatus struct {
	// Details: [Optional] Job state details, such as an error description
	// if the state is ERROR.
	Details string `json:"details,omitempty"`

	// EndTime: The time when the job completed.
	EndTime string `json:"endTime,omitempty"`

	// InsertTime: The time of the job request.
	InsertTime string `json:"insertTime,omitempty"`

	// StartTime: The time when the server started the job.
	StartTime string `json:"startTime,omitempty"`

	// State: [Required] A state message specifying the overall job state.
	//
	// Possible values:
	//   "STATE_UNSPECIFIED"
	//   "PENDING"
	//   "SETUP_DONE"
	//   "RUNNING"
	//   "CANCEL_PENDING"
	//   "CANCEL_STARTED"
	//   "CANCELLED"
	//   "DONE"
	//   "ERROR"
	State string `json:"state,omitempty"`

	// StateStartTime: [Output-only] The time when this state was entered.
	StateStartTime string `json:"stateStartTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Details") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *JobStatus) MarshalJSON() ([]byte, error) {
	type noMethod JobStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LeaseTasksRequest: A request to lease tasks for execution by an
// agent.
type LeaseTasksRequest struct {
	// AgentId: The agent's id.
	AgentId string `json:"agentId,omitempty"`

	// CurrentAgentTime: The current timestamp at the worker.
	CurrentAgentTime string `json:"currentAgentTime,omitempty"`

	// RequestedLeaseDuration: The requested initial lease period.
	RequestedLeaseDuration string `json:"requestedLeaseDuration,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AgentId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LeaseTasksRequest) MarshalJSON() ([]byte, error) {
	type noMethod LeaseTasksRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// LeaseTasksResponse: A Response for task leasing.
type LeaseTasksResponse struct {
	// LeaseExpirationTime: The worker-local lease expiration time.
	LeaseExpirationTime string `json:"leaseExpirationTime,omitempty"`

	// ReportStatusInterval: The interval at which status should be
	// reported.
	ReportStatusInterval string `json:"reportStatusInterval,omitempty"`

	// Tasks: A list of tasks that have been leased.
	Tasks []*Task `json:"tasks,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "LeaseExpirationTime")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *LeaseTasksResponse) MarshalJSON() ([]byte, error) {
	type noMethod LeaseTasksResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListAgentsResponse: Response to ListAgents
type ListAgentsResponse struct {
	// Agents: A list of agents.
	Agents []*Agent `json:"agents,omitempty"`

	// NextPageToken: The token to send to ListAgents to acquire any
	// following pages. Will be empty for last page.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Agents") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListAgentsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListAgentsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListClustersResponse: The list of all clusters in a project.
type ListClustersResponse struct {
	// Clusters: [Output-only] The clusters in the project.
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
}

func (s *ListClustersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListClustersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListJobsRequest: A request to list jobs in a project.
type ListJobsRequest struct {
	// ClusterName: [Optional] If set, the returned jobs list includes only
	// jobs that were submitted to the named cluster.
	ClusterName string `json:"clusterName,omitempty"`

	// JobStateMatcher: [Optional] Specifies enumerated categories of jobs
	// to list.
	//
	// Possible values:
	//   "ALL"
	//   "ACTIVE"
	//   "NON_ACTIVE"
	JobStateMatcher string `json:"jobStateMatcher,omitempty"`

	// PageSize: [Optional] The number of results to return in each
	// response.
	PageSize int64 `json:"pageSize,omitempty"`

	// PageToken: [Optional] The page token, returned by a previous call, to
	// request the next page of results.
	PageToken string `json:"pageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClusterName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListJobsRequest) MarshalJSON() ([]byte, error) {
	type noMethod ListJobsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListJobsResponse: A response to a request to list jobs in a project.
type ListJobsResponse struct {
	// Jobs: [Output-only] Jobs list.
	Jobs []*Job `json:"jobs,omitempty"`

	// NextPageToken: [Optional] This token is included in the response if
	// there are more results to fetch. To fetch additional results, provide
	// this value as the page_token in a subsequent ListJobsRequest.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Jobs") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListJobsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListJobsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *ListOperationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListOperationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
}

func (s *LoggingConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod LoggingConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ManagedGroupConfiguration: Specifies the resources used to actively
// manage an instance group.
type ManagedGroupConfiguration struct {
	// InstanceGroupManagerName: [Output-only] The name of Instance Group
	// Manager managing this group.
	InstanceGroupManagerName string `json:"instanceGroupManagerName,omitempty"`

	// InstanceTemplateName: [Output-only] The name of Instance Template
	// used for Managed Instance Group.
	InstanceTemplateName string `json:"instanceTemplateName,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "InstanceGroupManagerName") to unconditionally include in API
	// requests. By default, fields with empty values are omitted from API
	// requests. However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ManagedGroupConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod ManagedGroupConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// NodeInitializationAction: Specifies an executable to run on a fully
// configured node and a timeout period for executable completion.
type NodeInitializationAction struct {
	// ExecutableFile: [Required] Google Cloud Storage URI of executable
	// file.
	ExecutableFile string `json:"executableFile,omitempty"`

	// ExecutionTimeout: [Optional] Amount of time executable has to
	// complete. Default is 10 minutes. Cluster creation fails with an
	// explanatory error message (the name of the executable that caused the
	// error and the exceeded timeout period) if the executable is not
	// completed at end of the timeout period.
	ExecutionTimeout string `json:"executionTimeout,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ExecutableFile") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *NodeInitializationAction) MarshalJSON() ([]byte, error) {
	type noMethod NodeInitializationAction
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Operation: An asynchronous operation in a project that runs over a
// given cluster. Used to track the progress of a user request that is
// running asynchronously. Examples include creating a cluster, updating
// a cluster, and deleting a cluster.
type Operation struct {
	// Done: Indicates if the operation is done. If true, the operation is
	// complete and the `result` is available. If false, the operation is
	// still in progress.
	Done bool `json:"done,omitempty"`

	// Error: The error result of the operation in case of failure.
	Error *Status `json:"error,omitempty"`

	// Metadata: Service-specific metadata associated with the operation.
	Metadata OperationMetadata `json:"metadata,omitempty"`

	// Name: The name of the operation resource, in the format
	// projects/[project_id]/operations/[operation_id]
	Name string `json:"name,omitempty"`

	// Response: The operation response. If the called method returns no
	// data on success, the response is `google.protobuf.Empty`. If the
	// called method is `Get`,`Create` or `Update`, the response is the
	// resource. For all other methods, the response type is a concatenation
	// of the method name and "Response". For example, if the called method
	// is `TakeSnapshot()`, the response type is `TakeSnapshotResponse`.
	Response OperationResponse `json:"response,omitempty"`

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
}

func (s *Operation) MarshalJSON() ([]byte, error) {
	type noMethod Operation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type OperationMetadata interface{}

type OperationResponse interface{}

// OperationMetadata1: Metadata describing the operation.
type OperationMetadata1 struct {
	// ClusterName: Name of the cluster for the operation.
	ClusterName string `json:"clusterName,omitempty"`

	// ClusterUuid: Cluster UUId for the operation.
	ClusterUuid string `json:"clusterUuid,omitempty"`

	// Details: A message containing any operation metadata details.
	Details string `json:"details,omitempty"`

	// EndTime: The time that the operation completed.
	EndTime string `json:"endTime,omitempty"`

	// InnerState: A message containing the detailed operation state.
	InnerState string `json:"innerState,omitempty"`

	// InsertTime: The time that the operation was requested.
	InsertTime string `json:"insertTime,omitempty"`

	// StartTime: The time that the operation was started by the server.
	StartTime string `json:"startTime,omitempty"`

	// State: A message containing the operation state.
	//
	// Possible values:
	//   "UNKNOWN"
	//   "PENDING"
	//   "RUNNING"
	//   "DONE"
	State string `json:"state,omitempty"`

	// Status: [Output-only] Current operation status.
	Status *OperationStatus `json:"status,omitempty"`

	// StatusHistory: [Output-only] Previous operation status.
	StatusHistory []*OperationStatus `json:"statusHistory,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ClusterName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *OperationMetadata1) MarshalJSON() ([]byte, error) {
	type noMethod OperationMetadata1
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
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
	//   "UNKNOWN"
	//   "PENDING"
	//   "RUNNING"
	//   "DONE"
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
}

func (s *OperationStatus) MarshalJSON() ([]byte, error) {
	type noMethod OperationStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PigJob: A Cloud Dataproc job for running Pig queries on YARN.
type PigJob struct {
	// ContinueOnFailure: [Optional] Whether to continue executing queries
	// if a query fails. The default value is false. Setting to true can be
	// useful when executing independent parallel queries.
	ContinueOnFailure bool `json:"continueOnFailure,omitempty"`

	// JarFileUris: [Optional] HCFS URIs of jar files to add to the
	// CLASSPATH of the Pig Client and Hadoop MapReduce (MR) tasks. Can
	// contain Pig UDFs.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// LoggingConfiguration: [Optional] The runtime log configuration for
	// job execution.
	LoggingConfiguration *LoggingConfiguration `json:"loggingConfiguration,omitempty"`

	// Properties: [Optional] A mapping of property names to values, used to
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

	// ScriptVariables: [Optional] Mapping of query variable names to values
	// (equivalent to the Pig command: "name=[value]").
	ScriptVariables map[string]string `json:"scriptVariables,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ContinueOnFailure")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PigJob) MarshalJSON() ([]byte, error) {
	type noMethod PigJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PySparkJob: A Cloud Dataproc job for running PySpark applications on
// YARN.
type PySparkJob struct {
	// ArchiveUris: [Optional] HCFS URIs of archives to be extracted in the
	// working directory of .jar, .tar, .tar.gz, .tgz, and .zip.
	ArchiveUris []string `json:"archiveUris,omitempty"`

	// Args: [Optional] The arguments to pass to the driver. Do not include
	// arguments, such as --conf, that can be set as job properties, since a
	// collision may occur that causes an incorrect job submission.
	Args []string `json:"args,omitempty"`

	// FileUris: [Optional] HCFS URIs of files to be copied to the working
	// directory of Python drivers and distributed tasks. Useful for naively
	// parallel tasks.
	FileUris []string `json:"fileUris,omitempty"`

	// JarFileUris: [Optional] HCFS URIs of jar files to add to the
	// CLASSPATHs of the Python driver and tasks.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// LoggingConfiguration: [Optional] The runtime log configuration for
	// job execution.
	LoggingConfiguration *LoggingConfiguration `json:"loggingConfiguration,omitempty"`

	// MainPythonFileUri: [Required] The Hadoop Compatible Filesystem (HCFS)
	// URI of the main Python file to use as the driver. Must be a .py file.
	MainPythonFileUri string `json:"mainPythonFileUri,omitempty"`

	// Properties: [Optional] A mapping of property names to values, used to
	// configure PySpark. Properties that conflict with values set by the
	// Cloud Dataproc API may be overwritten. Can include properties set in
	// /etc/spark/conf/spark-defaults.conf and classes in user code.
	Properties map[string]string `json:"properties,omitempty"`

	// PythonFileUris: [Optional] HCFS file URIs of Python files to pass to
	// the PySpark framework. Supported file types: .py, .egg, and .zip.
	PythonFileUris []string `json:"pythonFileUris,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ArchiveUris") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PySparkJob) MarshalJSON() ([]byte, error) {
	type noMethod PySparkJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// QueryList: A list of queries to run on a cluster.
type QueryList struct {
	// Queries: [Required] The queries to execute. You do not need to
	// terminate a query with a semicolon. Multiple queries can be specified
	// in one string by separating each with a semicolon. Here is an example
	// of an Cloud Dataproc API snippet that uses a QueryList to specify a
	// HiveJob: "hiveJob": { "queryList": { "queries": [ "query1", "query2",
	// "query3;query4", ] } }
	Queries []string `json:"queries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Queries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *QueryList) MarshalJSON() ([]byte, error) {
	type noMethod QueryList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReportTaskStatusRequest: A request to report task status, which may
// result in the corresponding lease being extended.
type ReportTaskStatusRequest struct {
	// AgentId: The id of the agent reporting task status.
	AgentId string `json:"agentId,omitempty"`

	// CurrentWorkerTime: The current timestamp at the worker.
	CurrentWorkerTime string `json:"currentWorkerTime,omitempty"`

	// Status: Status for a single task.
	Status *TaskStatus `json:"status,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AgentId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReportTaskStatusRequest) MarshalJSON() ([]byte, error) {
	type noMethod ReportTaskStatusRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReportTaskStatusResponse: A response to a task status report.
type ReportTaskStatusResponse struct {
	// LeaseExpirationTime: New task lease expiration timestamp in
	// worker-local time.
	LeaseExpirationTime string `json:"leaseExpirationTime,omitempty"`

	// ReportStatusInterval: The interval at which status should be
	// reported.
	ReportStatusInterval string `json:"reportStatusInterval,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "LeaseExpirationTime")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReportTaskStatusResponse) MarshalJSON() ([]byte, error) {
	type noMethod ReportTaskStatusResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SoftwareConfiguration: Specifies the selection and configuration of
// software inside the cluster.
type SoftwareConfiguration struct {
	// ImageVersion: [Optional] The version of software inside the cluster.
	// It must match the regular expression [0-9]+\.[0-9]+. If unspecified
	// it will default to latest version.
	ImageVersion string `json:"imageVersion,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ImageVersion") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SoftwareConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod SoftwareConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SparkJob: A Cloud Dataproc job for running Spark applications on
// YARN.
type SparkJob struct {
	// ArchiveUris: [Optional] HCFS URIs of archives to be extracted in the
	// working directory of Spark drivers and tasks. Supported file types:
	// .jar, .tar, .tar.gz, .tgz, and .zip.
	ArchiveUris []string `json:"archiveUris,omitempty"`

	// Args: [Optional] The arguments to pass to the driver. Do not include
	// arguments, such as --conf, that can be set as job properties, since a
	// collision may occur that causes an incorrect job submission.
	Args []string `json:"args,omitempty"`

	// FileUris: [Optional] HCFS URIs of files to be copied to the working
	// directory of Spark drivers and distributed tasks. Useful for naively
	// parallel tasks.
	FileUris []string `json:"fileUris,omitempty"`

	// JarFileUris: [Optional] HCFS URIs of jar files to add to the
	// CLASSPATHs of the Spark driver and tasks.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// LoggingConfiguration: [Optional] The runtime log configuration for
	// job execution.
	LoggingConfiguration *LoggingConfiguration `json:"loggingConfiguration,omitempty"`

	// MainClass: The name of the driver's main class. The jar file that
	// contains the class must be in the default CLASSPATH or specified in
	// jar_file_uris.
	MainClass string `json:"mainClass,omitempty"`

	// MainJarFileUri: The Hadoop Compatible Filesystem (HCFS) URI of the
	// jar file that contains the main class.
	MainJarFileUri string `json:"mainJarFileUri,omitempty"`

	// Properties: [Optional] A mapping of property names to values, used to
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
}

func (s *SparkJob) MarshalJSON() ([]byte, error) {
	type noMethod SparkJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SparkSqlJob: A Cloud Dataproc job for running Spark SQL queries.
type SparkSqlJob struct {
	// JarFileUris: [Optional] HCFS URIs of jar files to be added to the
	// Spark CLASSPATH.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// LoggingConfiguration: [Optional] The runtime log configuration for
	// job execution.
	LoggingConfiguration *LoggingConfiguration `json:"loggingConfiguration,omitempty"`

	// Properties: [Optional] A mapping of property names to values, used to
	// configure Spark SQL's SparkConf. Properties that conflict with values
	// set by the Cloud Dataproc API may be overwritten.
	Properties map[string]string `json:"properties,omitempty"`

	// QueryFileUri: The HCFS URI of the script that contains SQL queries.
	QueryFileUri string `json:"queryFileUri,omitempty"`

	// QueryList: A list of queries.
	QueryList *QueryList `json:"queryList,omitempty"`

	// ScriptVariables: [Optional] Mapping of query variable names to values
	// (equivalent to the Spark SQL command: SET name="value";).
	ScriptVariables map[string]string `json:"scriptVariables,omitempty"`

	// ForceSendFields is a list of field names (e.g. "JarFileUris") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SparkSqlJob) MarshalJSON() ([]byte, error) {
	type noMethod SparkSqlJob
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Status: The `Status` type defines a logical error model that is
// suitable for different programming environments, including REST APIs
// and RPC APIs. It is used by [gRPC](https://github.com/grpc). The
// error model is designed to be: - Simple to use and understand for
// most users - Flexible enough to meet unexpected needs # Overview The
// `Status` message contains three pieces of data: error code, error
// message, and error details. The error code should be an enum value of
// google.rpc.Code, but it may accept additional error codes if needed.
// The error message should be a developer-facing English message that
// helps developers *understand* and *resolve* the error. If a localized
// user-facing error message is needed, put the localized message in the
// error details or localize it in the client. The optional error
// details may contain arbitrary information about the error. There is a
// predefined set of error detail types in the package `google.rpc`
// which can be used for common error conditions. # Language mapping The
// `Status` message is the logical representation of the error model,
// but it is not necessarily the actual wire format. When the `Status`
// message is exposed in different client libraries and different wire
// protocols, it can be mapped differently. For example, it will likely
// be mapped to some exceptions in Java, but more likely mapped to some
// error codes in C. # Other uses The error model and the `Status`
// message can be used in a variety of environments, either with or
// without APIs, to provide a consistent developer experience across
// different environments. Example uses of this error model include: -
// Partial errors. If a service needs to return partial errors to the
// client, it may embed the `Status` in the normal response to indicate
// the partial errors. - Workflow errors. A typical workflow has
// multiple steps. Each step may have a `Status` message for error
// reporting purpose. - Batch operations. If a client uses batch request
// and batch response, the `Status` message should be used directly
// inside batch response, one for each error sub-response. -
// Asynchronous operations. If an API call embeds asynchronous operation
// results in its response, the status of those operations should be
// represented directly using the `Status` message. - Logging. If some
// API errors are stored in logs, the message `Status` could be used
// directly after any stripping needed for security/privacy reasons.
type Status struct {
	// Code: The status code, which should be an enum value of
	// google.rpc.Code.
	Code int64 `json:"code,omitempty"`

	// Details: A list of messages that carry the error details. There will
	// be a common set of message types for APIs to use.
	Details []StatusDetails `json:"details,omitempty"`

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
}

func (s *Status) MarshalJSON() ([]byte, error) {
	type noMethod Status
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type StatusDetails interface{}

// SubmitJobRequest: A job submission request.
type SubmitJobRequest struct {
	// Job: [Required] The job resource.
	Job *Job `json:"job,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Job") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SubmitJobRequest) MarshalJSON() ([]byte, error) {
	type noMethod SubmitJobRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SystemTaskStatus: Status of cluster configuration task.
type SystemTaskStatus struct {
	// State: The outcome of reconfiguration.
	//
	// Possible values:
	//   "STATE_UNSPECIFIED"
	//   "PENDING"
	//   "RUNNING"
	//   "FINISHED"
	//   "FAILED"
	State string `json:"state,omitempty"`

	// ForceSendFields is a list of field names (e.g. "State") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SystemTaskStatus) MarshalJSON() ([]byte, error) {
	type noMethod SystemTaskStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Task: A single Task for agent execution. A task in Cloud Dataproc is
// a unit of work originating with the Cloud Dataproc service and
// intended for execution on a Cloud Dataproc agent. The most
// straightforward example of a Task in Cloud Dataproc is one
// corresponding to an end-user originated Job. This task is then
// allocated to an available agent for execution. Other examples of
// Cloud Dataproc Tasks could include executing system-maintenance
// scripts, periodic agent upgrades, etc. Using the example of a Task
// corresponding to a Job, the flow through Cloud Dataproc will appear
// as: - The end-user submits a Job to Cloud Dataproc for execution. -
// The Cloud Dataproc service determines the best Cluster to execute the
// Job on and creates a Task for the job. - An agent polls Cloud
// Dataproc for outstanding Tasks via the LeaseTasks method and the Task
// created in step 2 is provided to the agent. - During execution of the
// Task, the agent updates the Task status via the ReportTaskStatus
// method. - Upon completion of a Task, the agent will finally
// ReportTaskStatus with a TaskStatus indicating the status of the
// driver application's exit status. - The Cloud Dataproc service
// updates the status of the user-submitted Job using the Task status
// reported by the agent.
type Task struct {
	// Configuration: Configuration for this task.
	Configuration *TaskConfiguration `json:"configuration,omitempty"`

	// Status: The status of a task.
	Status *TaskStatus `json:"status,omitempty"`

	// TaskId: System defined task id.
	TaskId string `json:"taskId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Configuration") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Task) MarshalJSON() ([]byte, error) {
	type noMethod Task
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TaskClusterConfiguration: A system task corresponding to a request
// for Master node to update configuration based on provided values.
type TaskClusterConfiguration struct {
	// AddMembers: New nodes to register with cluster.
	AddMembers []string `json:"addMembers,omitempty"`

	// RemoveMembers: Existing nodes to decommission.
	RemoveMembers []string `json:"removeMembers,omitempty"`

	// Type: Type of configuration change.
	//
	// Possible values:
	//   "TYPE_UNSPECIFIED"
	//   "TYPE_CLUSTER_MEMBERSHIP_CHANGE"
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AddMembers") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TaskClusterConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod TaskClusterConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TaskConfiguration: Configuration for a Dataproc Task.
type TaskConfiguration struct {
	// ClusterConfiguration: Cluster reconfiguration task.
	ClusterConfiguration *TaskClusterConfiguration `json:"clusterConfiguration,omitempty"`

	// JobConfiguration: Configuration of a Job-based task.
	JobConfiguration *TaskJobConfiguration `json:"jobConfiguration,omitempty"`

	// MaintenanceCommand: Execute cluster maintenance command.
	MaintenanceCommand *TaskMaintenanceCommand `json:"maintenanceCommand,omitempty"`

	// ForceSendFields is a list of field names (e.g.
	// "ClusterConfiguration") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TaskConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod TaskConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TaskJobConfiguration: A task corresponding to a single job execution
// request.
type TaskJobConfiguration struct {
	// ArchiveUris: Required archives for the driver program or distributed
	// program. Used by Hadoop, Spark, and PySpark jobs.
	ArchiveUris []string `json:"archiveUris,omitempty"`

	// Args: Arguments for the driver program. Used by Hadoop, Spark, and
	// PySpark jobs.
	Args []string `json:"args,omitempty"`

	// ContinueOnFailure: True to continue processing pig or hive queries if
	// an earlier query fails.
	ContinueOnFailure bool `json:"continueOnFailure,omitempty"`

	// DriverControlFilesUri: [Output-only] If present, the location of
	// miscellaneous control files which may be used as part of job setup
	// and handling. If not present, control files may be placed in the same
	// location as driver_output_uri.
	DriverControlFilesUri string `json:"driverControlFilesUri,omitempty"`

	// DriverInputUri: [Output-only] A URI pointing to the location of the
	// stdin of the job's driver program, only set if the job is
	// interactive.
	DriverInputUri string `json:"driverInputUri,omitempty"`

	// DriverOutputUri: Output URI for driver output.
	DriverOutputUri string `json:"driverOutputUri,omitempty"`

	// FileUris: Required files for the driver program or distributed
	// program. Used by Hadoop, Spark, and PySpark jobs.
	FileUris []string `json:"fileUris,omitempty"`

	// Interactive: [Optional] If set to true, then the driver's stdin will
	// be kept open and driver_input_uri will be set to provide a path at
	// which additional input can be sent to the driver.
	Interactive bool `json:"interactive,omitempty"`

	// JarFileUris: JAR files that are required by the job.
	JarFileUris []string `json:"jarFileUris,omitempty"`

	// LoggingConfiguration: Logging configuration for the job.
	LoggingConfiguration *TaskLoggingConfiguration `json:"loggingConfiguration,omitempty"`

	// MainClass: A class name that is contained either in core Hadoop or
	// Spark libraries or within a JAR specified within jar_file_uris.
	MainClass string `json:"mainClass,omitempty"`

	// MainJarFileUri: A JAR containing the main driver and containing a
	// METADATA entry for a main class contained within the jar.
	MainJarFileUri string `json:"mainJarFileUri,omitempty"`

	// MainPythonFileUri: The main Python file for a PySpark application.
	MainPythonFileUri string `json:"mainPythonFileUri,omitempty"`

	// Properties: Properties for the submitted job.
	Properties map[string]string `json:"properties,omitempty"`

	// PythonFileUris: URIs of files required by the PySpark application
	PythonFileUris []string `json:"pythonFileUris,omitempty"`

	// QueryFileUri: A URI of a file containing queries
	QueryFileUri string `json:"queryFileUri,omitempty"`

	// QueryList: A list of queries specified within the API.
	QueryList *TaskQueryList `json:"queryList,omitempty"`

	// ScriptVariables: Variables to be substituted in Pig and Hive scripts.
	ScriptVariables map[string]string `json:"scriptVariables,omitempty"`

	// SubmittedBy: The user that the job should be attributed to in Hadoop
	// as a posix-style username. If the user is not a member of the
	// system's user-database, the task will be started as a system account.
	SubmittedBy string `json:"submittedBy,omitempty"`

	// Type: The type of the job.
	//
	// Possible values:
	//   "TYPE_UNSPECIFIED"
	//   "HADOOP"
	//   "SPARK"
	//   "PYSPARK"
	//   "HIVE"
	//   "PIG"
	//   "SPARK_SQL"
	Type string `json:"type,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ArchiveUris") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TaskJobConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod TaskJobConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TaskJobStatus: Status of a single job-based task.
type TaskJobStatus struct {
	// DriverExitCode: If the driver has exited, its exit code.
	DriverExitCode int64 `json:"driverExitCode,omitempty"`

	// DriverState: The state of the driver.
	//
	// Possible values:
	//   "DRIVER_STATE_UNSPECIFIED"
	//   "PENDING"
	//   "RUNNING"
	//   "KILL_PENDING"
	//   "KILLED"
	//   "FINISHED"
	//   "FAILED"
	DriverState string `json:"driverState,omitempty"`

	// YarnApplications: A list of YARN applications that have been launched
	// for this task.
	YarnApplications []*TaskYarnApplication `json:"yarnApplications,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DriverExitCode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TaskJobStatus) MarshalJSON() ([]byte, error) {
	type noMethod TaskJobStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TaskLoggingConfiguration: Logging configuration for the task.
type TaskLoggingConfiguration struct {
	// LogLevels: Map of logger name to log4j log level.
	LogLevels map[string]string `json:"logLevels,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LogLevels") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TaskLoggingConfiguration) MarshalJSON() ([]byte, error) {
	type noMethod TaskLoggingConfiguration
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TaskMaintenanceCommand: A system task corresponding to a request to
// run a maintenance command on targeted agent.
type TaskMaintenanceCommand struct {
	// Args: Arguments to pass to the script.
	Args []string `json:"args,omitempty"`

	// Environment: The environment variables.
	Environment map[string]string `json:"environment,omitempty"`

	// GcsUri: The executable is stored on GCS.
	GcsUri string `json:"gcsUri,omitempty"`

	// LocalPath: The executable is a file on agent.
	LocalPath string `json:"localPath,omitempty"`

	// ScriptOutputUri: The GCS URI where executable output will be stored.
	ScriptOutputUri string `json:"scriptOutputUri,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Args") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TaskMaintenanceCommand) MarshalJSON() ([]byte, error) {
	type noMethod TaskMaintenanceCommand
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TaskQueryList: A list of queries to execute as part of the task.
type TaskQueryList struct {
	// Queries: The queries to execute. The format of the queries is
	// task-type dependent, but in each case each query should be executed
	// within its own invocation of the interpreter for that task type.
	Queries []string `json:"queries,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Queries") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TaskQueryList) MarshalJSON() ([]byte, error) {
	type noMethod TaskQueryList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TaskStatus: Status for a single Task.
type TaskStatus struct {
	// JobStatus: The status of the Job.
	JobStatus *TaskJobStatus `json:"jobStatus,omitempty"`

	// SystemTaskStatus: The status of the SystemTask.
	SystemTaskStatus *SystemTaskStatus `json:"systemTaskStatus,omitempty"`

	// ForceSendFields is a list of field names (e.g. "JobStatus") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TaskStatus) MarshalJSON() ([]byte, error) {
	type noMethod TaskStatus
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TaskYarnApplication: YARN applications that have been associated with
// a task.
type TaskYarnApplication struct {
	// Id: YARN application id.
	Id int64 `json:"id,omitempty"`

	// Name: YARN application name.
	Name string `json:"name,omitempty"`

	// Progress: The progress of the YARN application.
	Progress float64 `json:"progress,omitempty"`

	// State: The state of the YARN application.
	//
	// Possible values:
	//   "STATE_UNSPECIFIED"
	//   "NEW"
	//   "NEW_SAVING"
	//   "SUBMITTED"
	//   "ACCEPTED"
	//   "RUNNING"
	//   "FINISHED"
	//   "FAILED"
	//   "KILLED"
	State string `json:"state,omitempty"`

	// TrackingUrl: The tracking URL for the YARN application. This URL may
	// or may not be accessible from outside the cluster.
	TrackingUrl string `json:"trackingUrl,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TaskYarnApplication) MarshalJSON() ([]byte, error) {
	type noMethod TaskYarnApplication
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// YarnApplication: A YARN application created by a job. Application
// information is a subset of
// org.apache.hadoop.yarn.proto.YarnProtos.ApplicationReportProto.
type YarnApplication struct {
	// Name: [Required] The application name.
	Name string `json:"name,omitempty"`

	// Progress: [Required] The numerical progress of the application, from
	// 1 to 100.
	Progress float64 `json:"progress,omitempty"`

	// State: [Required] The application state.
	//
	// Possible values:
	//   "STATE_UNSPECIFIED"
	//   "NEW"
	//   "NEW_SAVING"
	//   "SUBMITTED"
	//   "ACCEPTED"
	//   "RUNNING"
	//   "FINISHED"
	//   "FAILED"
	//   "KILLED"
	State string `json:"state,omitempty"`

	// TrackingUrl: [Optional] The HTTP URL of the ApplicationMaster,
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
}

func (s *YarnApplication) MarshalJSON() ([]byte, error) {
	type noMethod YarnApplication
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "dataproc.operations.cancel":

type OperationsCancelCall struct {
	s                      *Service
	name                   string
	canceloperationrequest *CancelOperationRequest
	urlParams_             gensupport.URLParams
	ctx_                   context.Context
}

// Cancel: Starts asynchronous cancellation on a long-running operation.
// The server makes a best effort to cancel the operation, but success
// is not guaranteed. If the server doesn't support this method, it
// returns `google.rpc.Code.UNIMPLEMENTED`. Clients may use
// [Operations.GetOperation] or other methods to check whether the
// cancellation succeeded or the operation completed despite
// cancellation.
func (r *OperationsService) Cancel(name string, canceloperationrequest *CancelOperationRequest) *OperationsCancelCall {
	c := &OperationsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.canceloperationrequest = canceloperationrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *OperationsCancelCall) QuotaUser(quotaUser string) *OperationsCancelCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *OperationsCancelCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.canceloperationrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/{+name}:cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.operations.cancel" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *OperationsCancelCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Clients may use [Operations.GetOperation] or other methods to check whether the cancellation succeeded or the operation completed despite cancellation.",
	//   "httpMethod": "POST",
	//   "id": "dataproc.operations.cancel",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be cancelled.",
	//       "location": "path",
	//       "pattern": "^operations/.*$",
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
}

// Delete: Deletes a long-running operation. It indicates the client is
// no longer interested in the operation result. It does not cancel the
// operation.
func (r *OperationsService) Delete(name string) *OperationsDeleteCall {
	c := &OperationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *OperationsDeleteCall) QuotaUser(quotaUser string) *OperationsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *OperationsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.operations.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *OperationsDeleteCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes a long-running operation. It indicates the client is no longer interested in the operation result. It does not cancel the operation.",
	//   "httpMethod": "DELETE",
	//   "id": "dataproc.operations.delete",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the operation resource to be deleted.",
	//       "location": "path",
	//       "pattern": "^operations/.*$",
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
}

// Get: Gets the latest state of a long-running operation. Clients may
// use this method to poll the operation result at intervals as
// recommended by the API service.
func (r *OperationsService) Get(name string) *OperationsGetCall {
	c := &OperationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *OperationsGetCall) QuotaUser(quotaUser string) *OperationsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *OperationsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.operations.get" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *OperationsGetCall) Do() (*Operation, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the latest state of a long-running operation. Clients may use this method to poll the operation result at intervals as recommended by the API service.",
	//   "httpMethod": "GET",
	//   "id": "dataproc.operations.get",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The operation resource name.",
	//       "location": "path",
	//       "pattern": "^operations/.*$",
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
}

// List: Lists operations that match the specified filter in the
// request. If the server doesn't support this method, it returns
// `google.rpc.Code.UNIMPLEMENTED`.
func (r *OperationsService) List(name string) *OperationsListCall {
	c := &OperationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	return c
}

// Filter sets the optional parameter "filter": [Required] A JSON object
// that contains filters for the list operation, in the format
// {"key1":"value1","key2":"value2", ..., }. Possible keys include
// project_id, cluster_name, and operation_state_matcher. If project_id
// is set, requests the list of operations that belong to the specified
// Google Cloud Platform project ID. This key is required. If
// cluster_name is set, requests the list of operations that were
// submitted to the specified cluster name. This key is optional. If
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

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *OperationsListCall) QuotaUser(quotaUser string) *OperationsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
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

func (c *OperationsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.operations.list" call.
// Exactly one of *ListOperationsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListOperationsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *OperationsListCall) Do() (*ListOperationsResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`.",
	//   "httpMethod": "GET",
	//   "id": "dataproc.operations.list",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "filter": {
	//       "description": "[Required] A JSON object that contains filters for the list operation, in the format {\"key1\":\"value1\",\"key2\":\"value2\", ..., }. Possible keys include project_id, cluster_name, and operation_state_matcher. If project_id is set, requests the list of operations that belong to the specified Google Cloud Platform project ID. This key is required. If cluster_name is set, requests the list of operations that were submitted to the specified cluster name. This key is optional. If operation_state_matcher is set, requests the list of operations that match one of the following status options: ALL, ACTIVE, or NON_ACTIVE.",
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

// method id "dataproc.projects.clusters.create":

type ProjectsClustersCreateCall struct {
	s          *Service
	projectId  string
	cluster    *Cluster
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Request to create a cluster in a project.
func (r *ProjectsClustersService) Create(projectId string, cluster *Cluster) *ProjectsClustersCreateCall {
	c := &ProjectsClustersCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.cluster = cluster
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersCreateCall) QuotaUser(quotaUser string) *ProjectsClustersCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersCreateCall) Fields(s ...googleapi.Field) *ProjectsClustersCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersCreateCall) Context(ctx context.Context) *ProjectsClustersCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.cluster)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.create" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsClustersCreateCall) Do() (*Operation, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Request to create a cluster in a project.",
	//   "httpMethod": "POST",
	//   "id": "dataproc.projects.clusters.create",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "[Required] The ID of the Google Cloud Platform project that the cluster belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters",
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

// method id "dataproc.projects.clusters.delete":

type ProjectsClustersDeleteCall struct {
	s           *Service
	projectId   string
	clusterName string
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Delete: Request to delete a cluster in a project.
func (r *ProjectsClustersService) Delete(projectId string, clusterName string) *ProjectsClustersDeleteCall {
	c := &ProjectsClustersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.clusterName = clusterName
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersDeleteCall) QuotaUser(quotaUser string) *ProjectsClustersDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersDeleteCall) Fields(s ...googleapi.Field) *ProjectsClustersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersDeleteCall) Context(ctx context.Context) *ProjectsClustersDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters/{clusterName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"clusterName": c.clusterName,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.delete" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsClustersDeleteCall) Do() (*Operation, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Request to delete a cluster in a project.",
	//   "httpMethod": "DELETE",
	//   "id": "dataproc.projects.clusters.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "clusterName"
	//   ],
	//   "parameters": {
	//     "clusterName": {
	//       "description": "[Required] The cluster name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "[Required] The ID of the Google Cloud Platform project that the cluster belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters/{clusterName}",
	//   "response": {
	//     "$ref": "Operation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.clusters.get":

type ProjectsClustersGetCall struct {
	s            *Service
	projectId    string
	clusterName  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Request to get the resource representation for a cluster in a
// project.
func (r *ProjectsClustersService) Get(projectId string, clusterName string) *ProjectsClustersGetCall {
	c := &ProjectsClustersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.clusterName = clusterName
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersGetCall) QuotaUser(quotaUser string) *ProjectsClustersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersGetCall) Fields(s ...googleapi.Field) *ProjectsClustersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsClustersGetCall) IfNoneMatch(entityTag string) *ProjectsClustersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersGetCall) Context(ctx context.Context) *ProjectsClustersGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters/{clusterName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"clusterName": c.clusterName,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.get" call.
// Exactly one of *Cluster or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Cluster.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsClustersGetCall) Do() (*Cluster, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Request to get the resource representation for a cluster in a project.",
	//   "httpMethod": "GET",
	//   "id": "dataproc.projects.clusters.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "clusterName"
	//   ],
	//   "parameters": {
	//     "clusterName": {
	//       "description": "[Required] The cluster name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "[Required] The ID of the Google Cloud Platform project that the cluster belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters/{clusterName}",
	//   "response": {
	//     "$ref": "Cluster"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.clusters.list":

type ProjectsClustersListCall struct {
	s            *Service
	projectId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Request a list of all clusters in a project.
func (r *ProjectsClustersService) List(projectId string) *ProjectsClustersListCall {
	c := &ProjectsClustersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	return c
}

// PageSize sets the optional parameter "pageSize": The standard List
// page size.
func (c *ProjectsClustersListCall) PageSize(pageSize int64) *ProjectsClustersListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The standard List
// page token.
func (c *ProjectsClustersListCall) PageToken(pageToken string) *ProjectsClustersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersListCall) QuotaUser(quotaUser string) *ProjectsClustersListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersListCall) Fields(s ...googleapi.Field) *ProjectsClustersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsClustersListCall) IfNoneMatch(entityTag string) *ProjectsClustersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersListCall) Context(ctx context.Context) *ProjectsClustersListCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.list" call.
// Exactly one of *ListClustersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListClustersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsClustersListCall) Do() (*ListClustersResponse, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Request a list of all clusters in a project.",
	//   "httpMethod": "GET",
	//   "id": "dataproc.projects.clusters.list",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
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
	//       "description": "[Required] The ID of the Google Cloud Platform project that the cluster belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters",
	//   "response": {
	//     "$ref": "ListClustersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.clusters.patch":

type ProjectsClustersPatchCall struct {
	s           *Service
	projectId   string
	clusterName string
	cluster     *Cluster
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Patch: Request to update a cluster in a project.
func (r *ProjectsClustersService) Patch(projectId string, clusterName string, cluster *Cluster) *ProjectsClustersPatchCall {
	c := &ProjectsClustersPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.clusterName = clusterName
	c.cluster = cluster
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersPatchCall) QuotaUser(quotaUser string) *ProjectsClustersPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UpdateMask sets the optional parameter "updateMask": [Required]
// Specifies the path, relative to Cluster, of the field to update. For
// example, to change the number of workers in a cluster to 5, the
// update_mask parameter would be specified as
// "configuration.worker_configuration.num_instances," and the PATCH
// request body would specify the new value, as follows: {
// "configuration":{ "workerConfiguration":{ "numInstances":"5" } } }
// Note: Currently, configuration.worker_configuration.num_instances is
// the only field that can be updated.
func (c *ProjectsClustersPatchCall) UpdateMask(updateMask string) *ProjectsClustersPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersPatchCall) Fields(s ...googleapi.Field) *ProjectsClustersPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersPatchCall) Context(ctx context.Context) *ProjectsClustersPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.cluster)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters/{clusterName}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"clusterName": c.clusterName,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.patch" call.
// Exactly one of *Operation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Operation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsClustersPatchCall) Do() (*Operation, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Request to update a cluster in a project.",
	//   "httpMethod": "PATCH",
	//   "id": "dataproc.projects.clusters.patch",
	//   "parameterOrder": [
	//     "projectId",
	//     "clusterName"
	//   ],
	//   "parameters": {
	//     "clusterName": {
	//       "description": "[Required] The cluster name.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "[Required] The ID of the Google Cloud Platform project the cluster belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "[Required] Specifies the path, relative to Cluster, of the field to update. For example, to change the number of workers in a cluster to 5, the update_mask parameter would be specified as \"configuration.worker_configuration.num_instances,\" and the PATCH request body would specify the new value, as follows: { \"configuration\":{ \"workerConfiguration\":{ \"numInstances\":\"5\" } } } Note: Currently, configuration.worker_configuration.num_instances is the only field that can be updated.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters/{clusterName}",
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

// method id "dataproc.projects.clusters.agents.create":

type ProjectsClustersAgentsCreateCall struct {
	s           *Service
	projectId   string
	clusterUuid string
	agentId     string
	agent       *Agent
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Create: Add a new agent to Dataproc's view of a cluster. This is the
// first Dataproc method that an agent should invoke after starting. If
// an agent has already been created with the given agent_id within the
// same cluster, this method will return a Conflict status code and the
// agent is expected to call GetAgent to retrieve the last registration
// and subsequently call UpdateAgent, if required.
func (r *ProjectsClustersAgentsService) Create(projectId string, clusterUuid string, agentId string, agent *Agent) *ProjectsClustersAgentsCreateCall {
	c := &ProjectsClustersAgentsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.clusterUuid = clusterUuid
	c.agentId = agentId
	c.agent = agent
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersAgentsCreateCall) QuotaUser(quotaUser string) *ProjectsClustersAgentsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersAgentsCreateCall) Fields(s ...googleapi.Field) *ProjectsClustersAgentsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersAgentsCreateCall) Context(ctx context.Context) *ProjectsClustersAgentsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersAgentsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.agent)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/agents/{agentId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"clusterUuid": c.clusterUuid,
		"agentId":     c.agentId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.agents.create" call.
// Exactly one of *Agent or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Agent.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsClustersAgentsCreateCall) Do() (*Agent, error) {
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
	ret := &Agent{
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
	//   "description": "Add a new agent to Dataproc's view of a cluster. This is the first Dataproc method that an agent should invoke after starting. If an agent has already been created with the given agent_id within the same cluster, this method will return a Conflict status code and the agent is expected to call GetAgent to retrieve the last registration and subsequently call UpdateAgent, if required.",
	//   "httpMethod": "PUT",
	//   "id": "dataproc.projects.clusters.agents.create",
	//   "parameterOrder": [
	//     "projectId",
	//     "clusterUuid",
	//     "agentId"
	//   ],
	//   "parameters": {
	//     "agentId": {
	//       "description": "[Required] Agent ID being registered.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clusterUuid": {
	//       "description": "Cluster that this agent is associated with",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project ID that this agent is associated with",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/agents/{agentId}",
	//   "request": {
	//     "$ref": "Agent"
	//   },
	//   "response": {
	//     "$ref": "Agent"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ]
	// }

}

// method id "dataproc.projects.clusters.agents.delete":

type ProjectsClustersAgentsDeleteCall struct {
	s           *Service
	projectId   string
	clusterUuid string
	agentId     string
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Delete: Delete an agent from Dataproc's view of a cluster. Deleting
// an agent is not required, but could be used in a shutdown sequence to
// indicate to Dataproc that the agent is to be considered dead and all
// agent-owned resources and tasks are free to be re-distributed.
func (r *ProjectsClustersAgentsService) Delete(projectId string, clusterUuid string, agentId string) *ProjectsClustersAgentsDeleteCall {
	c := &ProjectsClustersAgentsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.clusterUuid = clusterUuid
	c.agentId = agentId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersAgentsDeleteCall) QuotaUser(quotaUser string) *ProjectsClustersAgentsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersAgentsDeleteCall) Fields(s ...googleapi.Field) *ProjectsClustersAgentsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersAgentsDeleteCall) Context(ctx context.Context) *ProjectsClustersAgentsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersAgentsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/agents/{agentId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"clusterUuid": c.clusterUuid,
		"agentId":     c.agentId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.agents.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsClustersAgentsDeleteCall) Do() (*Empty, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Delete an agent from Dataproc's view of a cluster. Deleting an agent is not required, but could be used in a shutdown sequence to indicate to Dataproc that the agent is to be considered dead and all agent-owned resources and tasks are free to be re-distributed.",
	//   "httpMethod": "DELETE",
	//   "id": "dataproc.projects.clusters.agents.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "clusterUuid",
	//     "agentId"
	//   ],
	//   "parameters": {
	//     "agentId": {
	//       "description": "The agent.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clusterUuid": {
	//       "description": "The agent's cluster.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The agent's project.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/agents/{agentId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ]
	// }

}

// method id "dataproc.projects.clusters.agents.get":

type ProjectsClustersAgentsGetCall struct {
	s            *Service
	projectId    string
	clusterUuid  string
	agentId      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Retrieve an agent.
func (r *ProjectsClustersAgentsService) Get(projectId string, clusterUuid string, agentId string) *ProjectsClustersAgentsGetCall {
	c := &ProjectsClustersAgentsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.clusterUuid = clusterUuid
	c.agentId = agentId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersAgentsGetCall) QuotaUser(quotaUser string) *ProjectsClustersAgentsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersAgentsGetCall) Fields(s ...googleapi.Field) *ProjectsClustersAgentsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsClustersAgentsGetCall) IfNoneMatch(entityTag string) *ProjectsClustersAgentsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersAgentsGetCall) Context(ctx context.Context) *ProjectsClustersAgentsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersAgentsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/agents/{agentId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"clusterUuid": c.clusterUuid,
		"agentId":     c.agentId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.agents.get" call.
// Exactly one of *Agent or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Agent.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsClustersAgentsGetCall) Do() (*Agent, error) {
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
	ret := &Agent{
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
	//   "description": "Retrieve an agent.",
	//   "httpMethod": "GET",
	//   "id": "dataproc.projects.clusters.agents.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "clusterUuid",
	//     "agentId"
	//   ],
	//   "parameters": {
	//     "agentId": {
	//       "description": "The agent's ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clusterUuid": {
	//       "description": "The agent's cluster.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The agent's project ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/agents/{agentId}",
	//   "response": {
	//     "$ref": "Agent"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ]
	// }

}

// method id "dataproc.projects.clusters.agents.list":

type ProjectsClustersAgentsListCall struct {
	s            *Service
	projectId    string
	clusterUuid  string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: List all agents Dataproc is aware of within a cluster.
func (r *ProjectsClustersAgentsService) List(projectId string, clusterUuid string) *ProjectsClustersAgentsListCall {
	c := &ProjectsClustersAgentsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.clusterUuid = clusterUuid
	return c
}

// PageSize sets the optional parameter "pageSize": Requested page size
// for listing.
func (c *ProjectsClustersAgentsListCall) PageSize(pageSize int64) *ProjectsClustersAgentsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": Page continuation
// token.
func (c *ProjectsClustersAgentsListCall) PageToken(pageToken string) *ProjectsClustersAgentsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersAgentsListCall) QuotaUser(quotaUser string) *ProjectsClustersAgentsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersAgentsListCall) Fields(s ...googleapi.Field) *ProjectsClustersAgentsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsClustersAgentsListCall) IfNoneMatch(entityTag string) *ProjectsClustersAgentsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersAgentsListCall) Context(ctx context.Context) *ProjectsClustersAgentsListCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersAgentsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/agents")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"clusterUuid": c.clusterUuid,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.agents.list" call.
// Exactly one of *ListAgentsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListAgentsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsClustersAgentsListCall) Do() (*ListAgentsResponse, error) {
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
	ret := &ListAgentsResponse{
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
	//   "description": "List all agents Dataproc is aware of within a cluster.",
	//   "httpMethod": "GET",
	//   "id": "dataproc.projects.clusters.agents.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "clusterUuid"
	//   ],
	//   "parameters": {
	//     "clusterUuid": {
	//       "description": "The cluster from which to list agents.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Requested page size for listing.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page continuation token.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The project from which to list agents.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/agents",
	//   "response": {
	//     "$ref": "ListAgentsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ]
	// }

}

// method id "dataproc.projects.clusters.agents.update":

type ProjectsClustersAgentsUpdateCall struct {
	s           *Service
	projectId   string
	clusterUuid string
	agentId     string
	agent       *Agent
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Update: Update Dataproc's view of an agent. This is currently used to
// provide a is_healthy bit, but is expected to be extended to include
// daemon information and VM metrics for inclusion in cloud metrics.
func (r *ProjectsClustersAgentsService) Update(projectId string, clusterUuid string, agentId string, agent *Agent) *ProjectsClustersAgentsUpdateCall {
	c := &ProjectsClustersAgentsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.clusterUuid = clusterUuid
	c.agentId = agentId
	c.agent = agent
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersAgentsUpdateCall) QuotaUser(quotaUser string) *ProjectsClustersAgentsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersAgentsUpdateCall) Fields(s ...googleapi.Field) *ProjectsClustersAgentsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersAgentsUpdateCall) Context(ctx context.Context) *ProjectsClustersAgentsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersAgentsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.agent)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/agents/{agentId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"clusterUuid": c.clusterUuid,
		"agentId":     c.agentId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.agents.update" call.
// Exactly one of *Agent or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Agent.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsClustersAgentsUpdateCall) Do() (*Agent, error) {
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
	ret := &Agent{
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
	//   "description": "Update Dataproc's view of an agent. This is currently used to provide a is_healthy bit, but is expected to be extended to include daemon information and VM metrics for inclusion in cloud metrics.",
	//   "httpMethod": "PUT",
	//   "id": "dataproc.projects.clusters.agents.update",
	//   "parameterOrder": [
	//     "projectId",
	//     "clusterUuid",
	//     "agentId"
	//   ],
	//   "parameters": {
	//     "agentId": {
	//       "description": "[Required] ID of agent sending the update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "clusterUuid": {
	//       "description": "The cluster on which the agent is running.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The agent's project.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/agents/{agentId}",
	//   "request": {
	//     "$ref": "Agent"
	//   },
	//   "response": {
	//     "$ref": "Agent"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ]
	// }

}

// method id "dataproc.projects.clusters.tasks.lease":

type ProjectsClustersTasksLeaseCall struct {
	s                 *Service
	projectId         string
	clusterUuid       string
	leasetasksrequest *LeaseTasksRequest
	urlParams_        gensupport.URLParams
	ctx_              context.Context
}

// Lease: Obtain a lease on one or more tasks. Any given task may be in
// any state and each agent is expected to start any non-started tasks
// and to monitor any YarnApplications spawned by any already running
// tasks. It's expected that monitoring previously launched tasks will
// be more prevalent when drivers are run entirely within YARN
// containers. While there's a single lease expiration time, in the
// event of multiple tasks being leased to the agent in a single call,
// each task has a unique lease and status must be reported before the
// lease times out or the task can be considered orphaned. The service
// will determine how many tasks to lease to agents in a single call.
func (r *ProjectsClustersTasksService) Lease(projectId string, clusterUuid string, leasetasksrequest *LeaseTasksRequest) *ProjectsClustersTasksLeaseCall {
	c := &ProjectsClustersTasksLeaseCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.clusterUuid = clusterUuid
	c.leasetasksrequest = leasetasksrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersTasksLeaseCall) QuotaUser(quotaUser string) *ProjectsClustersTasksLeaseCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersTasksLeaseCall) Fields(s ...googleapi.Field) *ProjectsClustersTasksLeaseCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersTasksLeaseCall) Context(ctx context.Context) *ProjectsClustersTasksLeaseCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersTasksLeaseCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.leasetasksrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/tasks:lease")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"clusterUuid": c.clusterUuid,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.tasks.lease" call.
// Exactly one of *LeaseTasksResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *LeaseTasksResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsClustersTasksLeaseCall) Do() (*LeaseTasksResponse, error) {
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
	ret := &LeaseTasksResponse{
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
	//   "description": "Obtain a lease on one or more tasks. Any given task may be in any state and each agent is expected to start any non-started tasks and to monitor any YarnApplications spawned by any already running tasks. It's expected that monitoring previously launched tasks will be more prevalent when drivers are run entirely within YARN containers. While there's a single lease expiration time, in the event of multiple tasks being leased to the agent in a single call, each task has a unique lease and status must be reported before the lease times out or the task can be considered orphaned. The service will determine how many tasks to lease to agents in a single call.",
	//   "httpMethod": "POST",
	//   "id": "dataproc.projects.clusters.tasks.lease",
	//   "parameterOrder": [
	//     "projectId",
	//     "clusterUuid"
	//   ],
	//   "parameters": {
	//     "clusterUuid": {
	//       "description": "The cluster id of the agent.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The project id of the agent.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/tasks:lease",
	//   "request": {
	//     "$ref": "LeaseTasksRequest"
	//   },
	//   "response": {
	//     "$ref": "LeaseTasksResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ]
	// }

}

// method id "dataproc.projects.clusters.tasks.report":

type ProjectsClustersTasksReportCall struct {
	s                       *Service
	projectId               string
	clusterUuid             string
	taskId                  string
	reporttaskstatusrequest *ReportTaskStatusRequest
	urlParams_              gensupport.URLParams
	ctx_                    context.Context
}

// Report: Report status for a task and extend the lease provided for
// the task.
func (r *ProjectsClustersTasksService) Report(projectId string, clusterUuid string, taskId string, reporttaskstatusrequest *ReportTaskStatusRequest) *ProjectsClustersTasksReportCall {
	c := &ProjectsClustersTasksReportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.clusterUuid = clusterUuid
	c.taskId = taskId
	c.reporttaskstatusrequest = reporttaskstatusrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsClustersTasksReportCall) QuotaUser(quotaUser string) *ProjectsClustersTasksReportCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsClustersTasksReportCall) Fields(s ...googleapi.Field) *ProjectsClustersTasksReportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsClustersTasksReportCall) Context(ctx context.Context) *ProjectsClustersTasksReportCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsClustersTasksReportCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.reporttaskstatusrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/tasks/{taskId}:report")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId":   c.projectId,
		"clusterUuid": c.clusterUuid,
		"taskId":      c.taskId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.clusters.tasks.report" call.
// Exactly one of *ReportTaskStatusResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ReportTaskStatusResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsClustersTasksReportCall) Do() (*ReportTaskStatusResponse, error) {
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
	ret := &ReportTaskStatusResponse{
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
	//   "description": "Report status for a task and extend the lease provided for the task.",
	//   "httpMethod": "POST",
	//   "id": "dataproc.projects.clusters.tasks.report",
	//   "parameterOrder": [
	//     "projectId",
	//     "clusterUuid",
	//     "taskId"
	//   ],
	//   "parameters": {
	//     "clusterUuid": {
	//       "description": "The cluster id of the agent.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "The project id of the agent.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "taskId": {
	//       "description": "The task that is being reported on.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/clusters/{clusterUuid}/tasks/{taskId}:report",
	//   "request": {
	//     "$ref": "ReportTaskStatusRequest"
	//   },
	//   "response": {
	//     "$ref": "ReportTaskStatusResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/logging.admin",
	//     "https://www.googleapis.com/auth/logging.read",
	//     "https://www.googleapis.com/auth/logging.write"
	//   ]
	// }

}

// method id "dataproc.projects.jobs.cancel":

type ProjectsJobsCancelCall struct {
	s                *Service
	projectId        string
	jobId            string
	canceljobrequest *CancelJobRequest
	urlParams_       gensupport.URLParams
	ctx_             context.Context
}

// Cancel: Starts a job cancellation request. To access the job resource
// after cancellation, call jobs:list or jobs:get.
func (r *ProjectsJobsService) Cancel(projectId string, jobId string, canceljobrequest *CancelJobRequest) *ProjectsJobsCancelCall {
	c := &ProjectsJobsCancelCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.jobId = jobId
	c.canceljobrequest = canceljobrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsJobsCancelCall) QuotaUser(quotaUser string) *ProjectsJobsCancelCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsJobsCancelCall) Fields(s ...googleapi.Field) *ProjectsJobsCancelCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsJobsCancelCall) Context(ctx context.Context) *ProjectsJobsCancelCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsJobsCancelCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.canceljobrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/jobs/{jobId}:cancel")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"jobId":     c.jobId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.jobs.cancel" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsJobsCancelCall) Do() (*Job, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Starts a job cancellation request. To access the job resource after cancellation, call jobs:list or jobs:get.",
	//   "httpMethod": "POST",
	//   "id": "dataproc.projects.jobs.cancel",
	//   "parameterOrder": [
	//     "projectId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "[Required] The job ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "[Required] The ID of the Google Cloud Platform project that the job belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/jobs/{jobId}:cancel",
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

// method id "dataproc.projects.jobs.delete":

type ProjectsJobsDeleteCall struct {
	s          *Service
	projectId  string
	jobId      string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes the job from the project. If the job is active, the
// delete fails, and the response returns `FAILED_PRECONDITION`.
func (r *ProjectsJobsService) Delete(projectId string, jobId string) *ProjectsJobsDeleteCall {
	c := &ProjectsJobsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.jobId = jobId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsJobsDeleteCall) QuotaUser(quotaUser string) *ProjectsJobsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsJobsDeleteCall) Fields(s ...googleapi.Field) *ProjectsJobsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsJobsDeleteCall) Context(ctx context.Context) *ProjectsJobsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsJobsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/jobs/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"jobId":     c.jobId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.jobs.delete" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsJobsDeleteCall) Do() (*Job, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Deletes the job from the project. If the job is active, the delete fails, and the response returns `FAILED_PRECONDITION`.",
	//   "httpMethod": "DELETE",
	//   "id": "dataproc.projects.jobs.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "[Required] The job ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "[Required] The ID of the Google Cloud Platform project that the job belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/jobs/{jobId}",
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.jobs.get":

type ProjectsJobsGetCall struct {
	s            *Service
	projectId    string
	jobId        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the resource representation for a job in a project.
func (r *ProjectsJobsService) Get(projectId string, jobId string) *ProjectsJobsGetCall {
	c := &ProjectsJobsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.jobId = jobId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsJobsGetCall) QuotaUser(quotaUser string) *ProjectsJobsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsJobsGetCall) Fields(s ...googleapi.Field) *ProjectsJobsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsJobsGetCall) IfNoneMatch(entityTag string) *ProjectsJobsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsJobsGetCall) Context(ctx context.Context) *ProjectsJobsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsJobsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/jobs/{jobId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
		"jobId":     c.jobId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.jobs.get" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsJobsGetCall) Do() (*Job, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the resource representation for a job in a project.",
	//   "httpMethod": "GET",
	//   "id": "dataproc.projects.jobs.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "description": "[Required] The job ID.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "[Required] The ID of the Google Cloud Platform project that the job belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/jobs/{jobId}",
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform"
	//   ]
	// }

}

// method id "dataproc.projects.jobs.submit":

type ProjectsJobsSubmitCall struct {
	s                *Service
	projectId        string
	submitjobrequest *SubmitJobRequest
	urlParams_       gensupport.URLParams
	ctx_             context.Context
}

// Submit: Submits a job to a cluster.
func (r *ProjectsJobsService) Submit(projectId string, submitjobrequest *SubmitJobRequest) *ProjectsJobsSubmitCall {
	c := &ProjectsJobsSubmitCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.projectId = projectId
	c.submitjobrequest = submitjobrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsJobsSubmitCall) QuotaUser(quotaUser string) *ProjectsJobsSubmitCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsJobsSubmitCall) Fields(s ...googleapi.Field) *ProjectsJobsSubmitCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsJobsSubmitCall) Context(ctx context.Context) *ProjectsJobsSubmitCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsJobsSubmitCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.submitjobrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1alpha1/projects/{projectId}/jobs:submit")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"projectId": c.projectId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "dataproc.projects.jobs.submit" call.
// Exactly one of *Job or error will be non-nil. Any non-2xx status code
// is an error. Response headers are in either
// *Job.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *ProjectsJobsSubmitCall) Do() (*Job, error) {
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
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Submits a job to a cluster.",
	//   "httpMethod": "POST",
	//   "id": "dataproc.projects.jobs.submit",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "[Required] The ID of the Google Cloud Platform project that the job belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1alpha1/projects/{projectId}/jobs:submit",
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
