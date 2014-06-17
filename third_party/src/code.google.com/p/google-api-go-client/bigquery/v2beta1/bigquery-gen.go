// Package bigquery provides access to the BigQuery API.
//
// See https://code.google.com/apis/bigquery/docs/v2/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/bigquery/v2beta1"
//   ...
//   bigqueryService, err := bigquery.New(oauthHttpClient)
package bigquery

import (
	"bytes"
	"fmt"
	"net/http"
	"io"
	"encoding/json"
	"errors"
	"strings"
	"strconv"
	"net/url"
	"code.google.com/p/google-api-go-client/googleapi"
)

var _ = bytes.NewBuffer
var _ = strconv.Itoa
var _ = fmt.Sprintf
var _ = json.NewDecoder
var _ = io.Copy
var _ = url.Parse
var _ = googleapi.Version
var _ = errors.New

const apiId = "bigquery:v2beta1"
const apiName = "bigquery"
const apiVersion = "v2beta1"
const basePath = "https://www.googleapis.com/bigquery/v2beta1/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data in Google BigQuery
	BigqueryScope = "https://www.googleapis.com/auth/bigquery"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client}
	s.Datasets = &DatasetsService{s: s}
	s.Jobs = &JobsService{s: s}
	s.Projects = &ProjectsService{s: s}
	s.Tabledata = &TabledataService{s: s}
	s.Tables = &TablesService{s: s}
	return s, nil
}

type Service struct {
	client *http.Client

	Datasets *DatasetsService

	Jobs *JobsService

	Projects *ProjectsService

	Tabledata *TabledataService

	Tables *TablesService
}

type DatasetsService struct {
	s *Service
}

type JobsService struct {
	s *Service
}

type ProjectsService struct {
	s *Service
}

type TabledataService struct {
	s *Service
}

type TablesService struct {
	s *Service
}

type Bigqueryfield struct {
	Fields []*Bigqueryfield `json:"fields,omitempty"`

	Mode string `json:"mode,omitempty"`

	Name string `json:"name,omitempty"`

	Type string `json:"type,omitempty"`
}

type Bigqueryschema struct {
	Fields []*Bigqueryfield `json:"fields,omitempty"`
}

type Dataset struct {
	// Access: [Optional] Describes users' rights on the dataset. You can
	// assign the same role to multiple users, and assign multiple roles to
	// the same user.
	// Default values assigned to a new dataset are as
	// follows: OWNER - Project owners, dataset creator READ - Project
	// readers WRITE - Project writers
	// See ACLs and Rights for a description
	// of these rights. If you specify any of these roles when creating a
	// dataset, the assigned roles will overwrite the defaults listed
	// above.
	// To revoke rights to a dataset, call datasets.update() and omit
	// the names of anyone whose rights you wish to revoke. However, every
	// dataset must have at least one entity granted OWNER role.
	// Each access
	// object can have only one of the following members: userByEmail,
	// groupByEmail, domain, or allAuthenticatedUsers.
	Access []*DatasetAccess `json:"access,omitempty"`

	// CreationTime: [Output only] The date when this dataset was created,
	// in milliseconds since the epoch.
	CreationTime int64 `json:"creationTime,omitempty,string"`

	// DatasetId: [Deprecated -- overlaps with datasetRef] A unique ID for
	// this dataset. Must a string of 1-1024 characters satisfying the
	// regular expression [A-Za-z0-9_].
	DatasetId string `json:"datasetId,omitempty"`

	// DatasetReference: [Required] Reference identifying dataset.
	DatasetReference *Datasetreference `json:"datasetReference,omitempty"`

	// Description: [Optional] An arbitrary string description for the
	// dataset. This might be shown in BigQuery UI for browsing the dataset.
	Description string `json:"description,omitempty"`

	// FriendlyName: [Optional] A descriptive name for this dataset, which
	// might be shown in any BigQuery user interfaces for browsing the
	// dataset. Use datasetId for making API calls.
	FriendlyName string `json:"friendlyName,omitempty"`

	// Id: [Output only] The fully-qualified unique name of this dataset in
	// the format projectId:datasetId. The dataset name without the project
	// name is given in the datasetId field. When creating a new dataset,
	// leave this field blank, and instead specify the datasetId field.
	Id string `json:"id,omitempty"`

	// Kind: [Output only] The resource type.
	Kind string `json:"kind,omitempty"`

	// LastModifiedTime: [Output only] The date when this dataset or any of
	// its tables was last modified, in milliseconds since the epoch.
	LastModifiedTime int64 `json:"lastModifiedTime,omitempty,string"`

	// ProjectId: [Deprecated -- overlaps with datasetRef].
	ProjectId string `json:"projectId,omitempty"`

	// SelfLink: [Output only] An URL that can be used to access this
	// resource again. You can use this URL in Get or Update requests to
	// this resource. Not used as an input to helix.
	SelfLink string `json:"selfLink,omitempty"`
}

type DatasetAccess struct {
	// AllAuthenticatedUsers: [Pick one] If True, any authenticated user is
	// granted the assigned role.
	AllAuthenticatedUsers string `json:"allAuthenticatedUsers,omitempty"`

	// Domain: [Pick one] A domain to grant access to. Any users signed in
	// with the domain specified will be granted the specified access.
	// Example: "example.com".
	Domain string `json:"domain,omitempty"`

	// GroupByEmail: [Pick one] A fully-qualified email address of a mailing
	// list to grant access to. This must be either a Google Groups mailing
	// list (ends in @googlegroups.com) or a group managed by an enterprise
	// version of Google Groups.
	GroupByEmail string `json:"groupByEmail,omitempty"`

	// Role: [Required] Describes the rights granted to the user specified
	// by the other member of the access object. The following string values
	// are supported: READ - User can call any list() or get() method on any
	// collection or resource. WRITE - User can call any method on any
	// collection except for datasets, on which they can call list() and
	// get(). OWNER - User can call any method. The dataset creator is
	// granted this role by default.
	Role string `json:"role,omitempty"`

	// SpecialGroup: [Pick one] A special group to grant access to. The
	// valid values are: projectOwners: Owners of the enclosing project.
	// projectReaders: Readers of the enclosing project. projectWriters:
	// Writers of the enclosing project.
	SpecialGroup string `json:"specialGroup,omitempty"`

	// UserByEmail: [Pick one] A fully qualified email address of a user to
	// grant access to. For example: fred@example.com.
	UserByEmail string `json:"userByEmail,omitempty"`
}

type DatasetList struct {
	// Datasets: An array of one or more summarized dataset resources.
	// Absent when there are no datasets in the specified project.
	Datasets []*DatasetListDatasets `json:"datasets,omitempty"`

	// Etag: A hash of this page of results. See Paging Through Results in
	// the developer's guide.
	Etag string `json:"etag,omitempty"`

	// Kind: The resource type.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token to request the next page of results. Present
	// only when there is more than one page of results.* See Paging Through
	// Results in the developer's guide.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type DatasetListDatasets struct {
	// DatasetId: [Deprecated] A unique ID for this dataset; this is the id
	// values without the project name.
	DatasetId string `json:"datasetId,omitempty"`

	// DatasetReference: Reference identifying dataset.
	DatasetReference *Datasetreference `json:"datasetReference,omitempty"`

	// FriendlyName: A descriptive name for this dataset, if one exists.
	FriendlyName string `json:"friendlyName,omitempty"`

	// Id: The fully-qualified unique name of this dataset in the format
	// projectId:datasetId.
	Id string `json:"id,omitempty"`

	// ProjectId: [Deprecated] The ID of the container project.
	ProjectId string `json:"projectId,omitempty"`
}

type Datasetreference struct {
	// DatasetId: [Required] A unique ID for this dataset, without the
	// project name.
	DatasetId string `json:"datasetId,omitempty"`

	// ProjectId: [Optional] The ID of the container project.
	ProjectId string `json:"projectId,omitempty"`
}

type ErrorProto struct {
	Arguments []string `json:"arguments,omitempty"`

	Code string `json:"code,omitempty"`

	DebugInfo string `json:"debugInfo,omitempty"`

	Domain string `json:"domain,omitempty"`

	ErrorMessage string `json:"errorMessage,omitempty"`

	Location string `json:"location,omitempty"`

	LocationType string `json:"locationType,omitempty"`
}

type Job struct {
	Configuration *Jobconfiguration `json:"configuration,omitempty"`

	Id string `json:"id,omitempty"`

	JobId string `json:"jobId,omitempty"`

	JobReference *Jobreference `json:"jobReference,omitempty"`

	Kind string `json:"kind,omitempty"`

	ProjectId string `json:"projectId,omitempty"`

	SelfLink string `json:"selfLink,omitempty"`

	Statistics *Jobstatistics `json:"statistics,omitempty"`

	Status *Jobstatus `json:"status,omitempty"`
}

type JobList struct {
	Etag string `json:"etag,omitempty"`

	Jobs []*JobListJobs `json:"jobs,omitempty"`

	Kind string `json:"kind,omitempty"`

	NextPageToken string `json:"nextPageToken,omitempty"`

	TotalItems int64 `json:"totalItems,omitempty"`
}

type JobListJobs struct {
	Configuration *Jobconfiguration `json:"configuration,omitempty"`

	EndTime int64 `json:"endTime,omitempty,string"`

	ErrorResult *ErrorProto `json:"errorResult,omitempty"`

	Id string `json:"id,omitempty"`

	JobId string `json:"jobId,omitempty"`

	JobReference *Jobreference `json:"jobReference,omitempty"`

	ProjectId string `json:"projectId,omitempty"`

	StartTime int64 `json:"startTime,omitempty,string"`

	State string `json:"state,omitempty"`

	Statistics *Jobstatistics `json:"statistics,omitempty"`

	Status *Jobstatus `json:"status,omitempty"`
}

type JobQueryRequest struct {
	// DefaultDataset: [Optional] Specifies the default datasetId and
	// projectId to assume for any unqualified table names in the query. If
	// not set, all table names in the query string must be fully-qualified
	// in the format projectId:datasetId.tableid.
	DefaultDataset *Datasetreference `json:"defaultDataset,omitempty"`

	// DestinationTable: [Optional] Specifies the table the query results
	// should be written to. The table will be created if it does not exist.
	DestinationTable *Tablereference `json:"destinationTable,omitempty"`

	Kind string `json:"kind,omitempty"`

	// MaxResults: [Optional] The maximum number of results to return per
	// page of results. If the response list exceeds the maximum response
	// size for a single response, you will have to page through the
	// results. Default is to return the maximum response size.
	MaxResults int64 `json:"maxResults,omitempty"`

	// Query: [Required] A query string, following the BigQuery query syntax
	// of the query to execute. Table names should be qualified by dataset
	// name in the format projectId:datasetId.tableId unless you specify the
	// defaultDataset value. If the table is in the same project as the job,
	// you can omit the project ID. Example: SELECT f1 FROM
	// myProjectId:myDatasetId.myTableId.
	Query string `json:"query,omitempty"`
}

type JobStopResponse struct {
	Job interface{} `json:"job,omitempty"`

	Kind string `json:"kind,omitempty"`
}

type Jobconfiguration struct {
	Extract *Jobconfigurationextract `json:"extract,omitempty"`

	Link *Jobconfigurationlink `json:"link,omitempty"`

	Load *Jobconfigurationload `json:"load,omitempty"`

	Properties *Jobconfigurationproperties `json:"properties,omitempty"`

	Query *Jobconfigurationquery `json:"query,omitempty"`
}

type Jobconfigurationextract struct {
	DestinationUri string `json:"destinationUri,omitempty"`

	SourceTable *Tablereference `json:"sourceTable,omitempty"`
}

type Jobconfigurationlink struct {
	CreateDisposition string `json:"createDisposition,omitempty"`

	DestinationTable *Tablereference `json:"destinationTable,omitempty"`

	SourceUri []string `json:"sourceUri,omitempty"`
}

type Jobconfigurationload struct {
	CreateDisposition string `json:"createDisposition,omitempty"`

	DestinationTable *Tablereference `json:"destinationTable,omitempty"`

	FieldDelimiter string `json:"fieldDelimiter,omitempty"`

	Schema *Bigqueryschema `json:"schema,omitempty"`

	SkipLeadingRows int64 `json:"skipLeadingRows,omitempty"`

	SourceUris []string `json:"sourceUris,omitempty"`

	WriteDisposition string `json:"writeDisposition,omitempty"`
}

type Jobconfigurationproperties struct {
}

type Jobconfigurationquery struct {
	CreateDisposition string `json:"createDisposition,omitempty"`

	DefaultDataset *Datasetreference `json:"defaultDataset,omitempty"`

	DestinationTable *Tablereference `json:"destinationTable,omitempty"`

	Query string `json:"query,omitempty"`

	WriteDisposition string `json:"writeDisposition,omitempty"`
}

type Jobreference struct {
	JobId string `json:"jobId,omitempty"`

	ProjectId string `json:"projectId,omitempty"`
}

type Jobstatistics struct {
	EndTime int64 `json:"endTime,omitempty,string"`

	StartTime int64 `json:"startTime,omitempty,string"`
}

type Jobstatus struct {
	ErrorResult *ErrorProto `json:"errorResult,omitempty"`

	Errors []*ErrorProto `json:"errors,omitempty"`

	State string `json:"state,omitempty"`
}

type ProjectList struct {
	Etag string `json:"etag,omitempty"`

	Kind string `json:"kind,omitempty"`

	NextPageToken string `json:"nextPageToken,omitempty"`

	Projects []*ProjectListProjects `json:"projects,omitempty"`

	TotalItems int64 `json:"totalItems,omitempty"`
}

type ProjectListProjects struct {
	FriendlyName string `json:"friendlyName,omitempty"`

	Id string `json:"id,omitempty"`

	ProjectReference *Projectreference `json:"projectReference,omitempty"`
}

type Projectreference struct {
	ProjectId string `json:"projectId,omitempty"`
}

type QueryResults struct {
	Job *Job `json:"job,omitempty"`

	Kind string `json:"kind,omitempty"`

	Rows []*QueryResultsRows `json:"rows,omitempty"`

	Schema *Bigqueryschema `json:"schema,omitempty"`

	TotalRows uint64 `json:"totalRows,omitempty,string"`
}

type QueryResultsRows struct {
	F []*QueryResultsRowsF `json:"f,omitempty"`
}

type QueryResultsRowsF struct {
	V interface{} `json:"v,omitempty"`
}

type Table struct {
	CreationTime int64 `json:"creationTime,omitempty,string"`

	DatasetId string `json:"datasetId,omitempty"`

	Description string `json:"description,omitempty"`

	FriendlyName string `json:"friendlyName,omitempty"`

	Id string `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`

	LastModifiedTime int64 `json:"lastModifiedTime,omitempty,string"`

	ProjectId string `json:"projectId,omitempty"`

	Schema *Bigqueryschema `json:"schema,omitempty"`

	SelfLink string `json:"selfLink,omitempty"`

	TableId string `json:"tableId,omitempty"`

	TableReference *Tablereference `json:"tableReference,omitempty"`
}

type TableDataList struct {
	Kind string `json:"kind,omitempty"`

	Rows []*TableDataListRows `json:"rows,omitempty"`

	TotalRows int64 `json:"totalRows,omitempty,string"`
}

type TableDataListRows struct {
	F []*TableDataListRowsF `json:"f,omitempty"`
}

type TableDataListRowsF struct {
	V interface{} `json:"v,omitempty"`
}

type TableList struct {
	Etag string `json:"etag,omitempty"`

	Kind string `json:"kind,omitempty"`

	NextPageToken string `json:"nextPageToken,omitempty"`

	Tables []*TableListTables `json:"tables,omitempty"`

	TotalItems int64 `json:"totalItems,omitempty"`
}

type TableListTables struct {
	DatasetId string `json:"datasetId,omitempty"`

	FriendlyName string `json:"friendlyName,omitempty"`

	Id string `json:"id,omitempty"`

	ProjectId string `json:"projectId,omitempty"`

	TableId string `json:"tableId,omitempty"`

	TableReference *Tablereference `json:"tableReference,omitempty"`
}

type Tablereference struct {
	DatasetId string `json:"datasetId,omitempty"`

	ProjectId string `json:"projectId,omitempty"`

	TableId string `json:"tableId,omitempty"`
}

// method id "bigquery.datasets.delete":

type DatasetsDeleteCall struct {
	s         *Service
	projectId string
	datasetId string
	opt_      map[string]interface{}
}

// Delete: Deletes the dataset specified by datasetId value. Before you
// can delete a dataset, you must delete all its tables, either manually
// or by specifying deleteContents. Immediately after deletion, you can
// create another dataset with the same name.
func (r *DatasetsService) Delete(projectId string, datasetId string) *DatasetsDeleteCall {
	c := &DatasetsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.datasetId = datasetId
	return c
}

// DeleteContents sets the optional parameter "deleteContents":
// [Optional] If True, delete all the tables in the dataset. If False
// and the dataset contains tables, the request will fail. Default is
// False.
func (c *DatasetsDeleteCall) DeleteContents(deleteContents bool) *DatasetsDeleteCall {
	c.opt_["deleteContents"] = deleteContents
	return c
}

func (c *DatasetsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["deleteContents"]; ok {
		params.Set("deleteContents", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets/{datasetId}")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{datasetId}", cleanPathString(c.datasetId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the dataset specified by datasetId value. Before you can delete a dataset, you must delete all its tables, either manually or by specifying deleteContents. Immediately after deletion, you can create another dataset with the same name.",
	//   "httpMethod": "DELETE",
	//   "id": "bigquery.datasets.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset identifier of dataset being deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "deleteContents": {
	//       "description": "[Optional] If True, delete all the tables in the dataset. If False and the dataset contains tables, the request will fail. Default is False.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "projectId": {
	//       "description": "Project identifier of dataset being deleted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.datasets.get":

type DatasetsGetCall struct {
	s         *Service
	projectId string
	datasetId string
	opt_      map[string]interface{}
}

// Get: Returns the dataset specified by datasetID.
func (r *DatasetsService) Get(projectId string, datasetId string) *DatasetsGetCall {
	c := &DatasetsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.datasetId = datasetId
	return c
}

func (c *DatasetsGetCall) Do() (*Dataset, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets/{datasetId}")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{datasetId}", cleanPathString(c.datasetId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Dataset)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the dataset specified by datasetID.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.datasets.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset identifier of the dataset requested.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project identifier containing dataset requested.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}",
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.datasets.insert":

type DatasetsInsertCall struct {
	s         *Service
	projectId string
	dataset   *Dataset
	opt_      map[string]interface{}
}

// Insert: Creates a new empty dataset.
func (r *DatasetsService) Insert(projectId string, dataset *Dataset) *DatasetsInsertCall {
	c := &DatasetsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.dataset = dataset
	return c
}

func (c *DatasetsInsertCall) Do() (*Dataset, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.dataset)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Dataset)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new empty dataset.",
	//   "httpMethod": "POST",
	//   "id": "bigquery.datasets.insert",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "Project identifier that will contain dataset being created.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets",
	//   "request": {
	//     "$ref": "Dataset"
	//   },
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.datasets.list":

type DatasetsListCall struct {
	s         *Service
	projectId string
	opt_      map[string]interface{}
}

// List: Lists all the datasets in the specified project to which the
// caller has read access; however, a project owner can list (but not
// necessarily get) all datasets in his project.
func (r *DatasetsService) List(projectId string) *DatasetsListCall {
	c := &DatasetsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	return c
}

// MaxResults sets the optional parameter "maxResults": [Optional] The
// maximum number of rows to return. If not specified, it will return up
// to the maximum amount of data that will fit in a reply.
func (c *DatasetsListCall) MaxResults(maxResults int64) *DatasetsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": [Optional] A page
// token used when requesting a specific page in a set of paged results.
func (c *DatasetsListCall) PageToken(pageToken string) *DatasetsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *DatasetsListCall) Do() (*DatasetList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(DatasetList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all the datasets in the specified project to which the caller has read access; however, a project owner can list (but not necessarily get) all datasets in his project.",
	//   "httpMethod": "GET",
	//   "id": "bigquery.datasets.list",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "description": "[Optional] The maximum number of rows to return. If not specified, it will return up to the maximum amount of data that will fit in a reply.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "[Optional] A page token used when requesting a specific page in a set of paged results.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project identifier containing datasets to be listed.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets",
	//   "response": {
	//     "$ref": "DatasetList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.datasets.patch":

type DatasetsPatchCall struct {
	s         *Service
	projectId string
	datasetId string
	dataset   *Dataset
	opt_      map[string]interface{}
}

// Patch: Updates information in an existing dataset, specified by
// datasetId. Properties not included in the submitted resource will not
// be changed. If you include the access property without any values
// assigned, the request will fail as you must specify at least one
// owner for a dataset. This method supports patch semantics.
func (r *DatasetsService) Patch(projectId string, datasetId string, dataset *Dataset) *DatasetsPatchCall {
	c := &DatasetsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.datasetId = datasetId
	c.dataset = dataset
	return c
}

func (c *DatasetsPatchCall) Do() (*Dataset, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.dataset)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets/{datasetId}")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{datasetId}", cleanPathString(c.datasetId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Dataset)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates information in an existing dataset, specified by datasetId. Properties not included in the submitted resource will not be changed. If you include the access property without any values assigned, the request will fail as you must specify at least one owner for a dataset. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "bigquery.datasets.patch",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset identifier containing dataset being updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project identifier containing dataset being updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}",
	//   "request": {
	//     "$ref": "Dataset"
	//   },
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.datasets.update":

type DatasetsUpdateCall struct {
	s         *Service
	projectId string
	datasetId string
	dataset   *Dataset
	opt_      map[string]interface{}
}

// Update: Updates information in an existing dataset, specified by
// datasetId. Properties not included in the submitted resource will not
// be changed. If you include the access property without any values
// assigned, the request will fail as you must specify at least one
// owner for a dataset.
func (r *DatasetsService) Update(projectId string, datasetId string, dataset *Dataset) *DatasetsUpdateCall {
	c := &DatasetsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.datasetId = datasetId
	c.dataset = dataset
	return c
}

func (c *DatasetsUpdateCall) Do() (*Dataset, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.dataset)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets/{datasetId}")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{datasetId}", cleanPathString(c.datasetId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Dataset)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates information in an existing dataset, specified by datasetId. Properties not included in the submitted resource will not be changed. If you include the access property without any values assigned, the request will fail as you must specify at least one owner for a dataset.",
	//   "httpMethod": "PUT",
	//   "id": "bigquery.datasets.update",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "description": "Dataset identifier containing dataset being updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "description": "Project identifier containing dataset being updated.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}",
	//   "request": {
	//     "$ref": "Dataset"
	//   },
	//   "response": {
	//     "$ref": "Dataset"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.jobs.get":

type JobsGetCall struct {
	s         *Service
	projectId string
	jobId     string
	opt_      map[string]interface{}
}

// Get: 
func (r *JobsService) Get(projectId string, jobId string) *JobsGetCall {
	c := &JobsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.jobId = jobId
	return c
}

func (c *JobsGetCall) Do() (*Job, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/jobs/{jobId}")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{jobId}", cleanPathString(c.jobId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Job)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "GET",
	//   "id": "bigquery.jobs.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/jobs/{jobId}",
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.jobs.insert":

type JobsInsertCall struct {
	s         *Service
	projectId string
	job       *Job
	opt_      map[string]interface{}
	media_    io.Reader
}

// Insert: 
func (r *JobsService) Insert(projectId string, job *Job) *JobsInsertCall {
	c := &JobsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.job = job
	return c
}
func (c *JobsInsertCall) Media(r io.Reader) *JobsInsertCall {
	c.media_ = r
	return c
}

func (c *JobsInsertCall) Do() (*Job, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.job)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/jobs")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls += "?" + params.Encode()
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("POST", urls, body)
	if hasMedia_ {
		req.ContentLength = contentLength_
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Job)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "POST",
	//   "id": "bigquery.jobs.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "application/octet-stream"
	//     ],
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/bigquery/v2beta1/projects/{projectId}/jobs"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/bigquery/v2beta1/projects/{projectId}/jobs"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/jobs",
	//   "request": {
	//     "$ref": "Job"
	//   },
	//   "response": {
	//     "$ref": "Job"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.jobs.list":

type JobsListCall struct {
	s         *Service
	projectId string
	opt_      map[string]interface{}
}

// List: 
func (r *JobsService) List(projectId string) *JobsListCall {
	c := &JobsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	return c
}

// AllUsers sets the optional parameter "allUsers": Whether to display
// jobs owned by all users in the project
func (c *JobsListCall) AllUsers(allUsers bool) *JobsListCall {
	c.opt_["allUsers"] = allUsers
	return c
}

// MaxResults sets the optional parameter "maxResults": maximum number
// of results to return
func (c *JobsListCall) MaxResults(maxResults int64) *JobsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": 
func (c *JobsListCall) PageToken(pageToken string) *JobsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// Projection sets the optional parameter "projection": Restrict
// information returned to a set of selected fields.
func (c *JobsListCall) Projection(projection string) *JobsListCall {
	c.opt_["projection"] = projection
	return c
}

// StartIndex sets the optional parameter "startIndex": start index for
// paginated results
func (c *JobsListCall) StartIndex(startIndex int64) *JobsListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

// StateFilter sets the optional parameter "stateFilter": filter for job
// state
func (c *JobsListCall) StateFilter(stateFilter string) *JobsListCall {
	c.opt_["stateFilter"] = stateFilter
	return c
}

func (c *JobsListCall) Do() (*JobList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["allUsers"]; ok {
		params.Set("allUsers", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["stateFilter"]; ok {
		params.Set("stateFilter", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/jobs")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(JobList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "GET",
	//   "id": "bigquery.jobs.list",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "allUsers": {
	//       "description": "Whether to display jobs owned by all users in the project",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projection": {
	//       "description": "Restrict information returned to a set of selected fields.",
	//       "enum": [
	//         "full",
	//         "minimal"
	//       ],
	//       "enumDescriptions": [
	//         "Includes all job data.",
	//         "Does not include the job configuration."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "start index for paginated results",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "stateFilter": {
	//       "description": "filter for job state",
	//       "enum": [
	//         "done",
	//         "pending",
	//         "running"
	//       ],
	//       "enumDescriptions": [
	//         "finished jobs",
	//         "pending jobs",
	//         "running jobs"
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/jobs",
	//   "response": {
	//     "$ref": "JobList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.jobs.query":

type JobsQueryCall struct {
	s               *Service
	projectId       string
	jobqueryrequest *JobQueryRequest
	opt_            map[string]interface{}
}

// Query: 
func (r *JobsService) Query(projectId string, jobqueryrequest *JobQueryRequest) *JobsQueryCall {
	c := &JobsQueryCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.jobqueryrequest = jobqueryrequest
	return c
}

func (c *JobsQueryCall) Do() (*QueryResults, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.jobqueryrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/queries")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(QueryResults)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "POST",
	//   "id": "bigquery.jobs.query",
	//   "parameterOrder": [
	//     "projectId"
	//   ],
	//   "parameters": {
	//     "projectId": {
	//       "description": "project name billed for the query",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/queries",
	//   "request": {
	//     "$ref": "JobQueryRequest"
	//   },
	//   "response": {
	//     "$ref": "QueryResults"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.jobs.stop":

type JobsStopCall struct {
	s         *Service
	projectId string
	jobId     string
	opt_      map[string]interface{}
}

// Stop: 
func (r *JobsService) Stop(projectId string, jobId string) *JobsStopCall {
	c := &JobsStopCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.jobId = jobId
	return c
}

func (c *JobsStopCall) Do() (*JobStopResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "project/{projectId}/jobs/{jobId}/stop")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{jobId}", cleanPathString(c.jobId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(JobStopResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "POST",
	//   "id": "bigquery.jobs.stop",
	//   "parameterOrder": [
	//     "projectId",
	//     "jobId"
	//   ],
	//   "parameters": {
	//     "jobId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "project/{projectId}/jobs/{jobId}/stop",
	//   "response": {
	//     "$ref": "JobStopResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.projects.list":

type ProjectsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: 
func (r *ProjectsService) List() *ProjectsListCall {
	c := &ProjectsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "maxResults": 
func (c *ProjectsListCall) MaxResults(maxResults int64) *ProjectsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": 
func (c *ProjectsListCall) PageToken(pageToken string) *ProjectsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *ProjectsListCall) Do() (*ProjectList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ProjectList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "GET",
	//   "id": "bigquery.projects.list",
	//   "parameters": {
	//     "maxResults": {
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects",
	//   "response": {
	//     "$ref": "ProjectList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.tabledata.list":

type TabledataListCall struct {
	s         *Service
	projectId string
	datasetId string
	tableId   string
	opt_      map[string]interface{}
}

// List: 
func (r *TabledataService) List(projectId string, datasetId string, tableId string) *TabledataListCall {
	c := &TabledataListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.datasetId = datasetId
	c.tableId = tableId
	return c
}

// MaxResults sets the optional parameter "maxResults": 
func (c *TabledataListCall) MaxResults(maxResults int64) *TabledataListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// StartIndex sets the optional parameter "startIndex": 
func (c *TabledataListCall) StartIndex(startIndex uint64) *TabledataListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *TabledataListCall) Do() (*TableDataList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets/{datasetId}/tables/{tableId}/data")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{datasetId}", cleanPathString(c.datasetId), 1)
	urls = strings.Replace(urls, "{tableId}", cleanPathString(c.tableId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(TableDataList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "GET",
	//   "id": "bigquery.tabledata.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId",
	//     "tableId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "tableId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables/{tableId}/data",
	//   "response": {
	//     "$ref": "TableDataList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.tables.delete":

type TablesDeleteCall struct {
	s         *Service
	projectId string
	datasetId string
	tableId   string
	opt_      map[string]interface{}
}

// Delete: 
func (r *TablesService) Delete(projectId string, datasetId string, tableId string) *TablesDeleteCall {
	c := &TablesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.datasetId = datasetId
	c.tableId = tableId
	return c
}

func (c *TablesDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets/{datasetId}/tables/{tableId}")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{datasetId}", cleanPathString(c.datasetId), 1)
	urls = strings.Replace(urls, "{tableId}", cleanPathString(c.tableId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "httpMethod": "DELETE",
	//   "id": "bigquery.tables.delete",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId",
	//     "tableId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tableId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables/{tableId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.tables.get":

type TablesGetCall struct {
	s         *Service
	projectId string
	datasetId string
	tableId   string
	opt_      map[string]interface{}
}

// Get: 
func (r *TablesService) Get(projectId string, datasetId string, tableId string) *TablesGetCall {
	c := &TablesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.datasetId = datasetId
	c.tableId = tableId
	return c
}

func (c *TablesGetCall) Do() (*Table, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets/{datasetId}/tables/{tableId}")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{datasetId}", cleanPathString(c.datasetId), 1)
	urls = strings.Replace(urls, "{tableId}", cleanPathString(c.tableId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Table)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "GET",
	//   "id": "bigquery.tables.get",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId",
	//     "tableId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tableId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables/{tableId}",
	//   "response": {
	//     "$ref": "Table"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.tables.insert":

type TablesInsertCall struct {
	s         *Service
	projectId string
	datasetId string
	table     *Table
	opt_      map[string]interface{}
}

// Insert: 
func (r *TablesService) Insert(projectId string, datasetId string, table *Table) *TablesInsertCall {
	c := &TablesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.datasetId = datasetId
	c.table = table
	return c
}

func (c *TablesInsertCall) Do() (*Table, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.table)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets/{datasetId}/tables")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{datasetId}", cleanPathString(c.datasetId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Table)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "POST",
	//   "id": "bigquery.tables.insert",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables",
	//   "request": {
	//     "$ref": "Table"
	//   },
	//   "response": {
	//     "$ref": "Table"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.tables.list":

type TablesListCall struct {
	s         *Service
	projectId string
	datasetId string
	opt_      map[string]interface{}
}

// List: 
func (r *TablesService) List(projectId string, datasetId string) *TablesListCall {
	c := &TablesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.datasetId = datasetId
	return c
}

// MaxResults sets the optional parameter "maxResults": 
func (c *TablesListCall) MaxResults(maxResults int64) *TablesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": 
func (c *TablesListCall) PageToken(pageToken string) *TablesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *TablesListCall) Do() (*TableList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets/{datasetId}/tables")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{datasetId}", cleanPathString(c.datasetId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(TableList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "GET",
	//   "id": "bigquery.tables.list",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables",
	//   "response": {
	//     "$ref": "TableList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.tables.patch":

type TablesPatchCall struct {
	s         *Service
	projectId string
	datasetId string
	tableId   string
	table     *Table
	opt_      map[string]interface{}
}

// Patch: 
func (r *TablesService) Patch(projectId string, datasetId string, tableId string, table *Table) *TablesPatchCall {
	c := &TablesPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.datasetId = datasetId
	c.tableId = tableId
	c.table = table
	return c
}

func (c *TablesPatchCall) Do() (*Table, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.table)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets/{datasetId}/tables/{tableId}")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{datasetId}", cleanPathString(c.datasetId), 1)
	urls = strings.Replace(urls, "{tableId}", cleanPathString(c.tableId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Table)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "PATCH",
	//   "id": "bigquery.tables.patch",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId",
	//     "tableId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tableId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables/{tableId}",
	//   "request": {
	//     "$ref": "Table"
	//   },
	//   "response": {
	//     "$ref": "Table"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

// method id "bigquery.tables.update":

type TablesUpdateCall struct {
	s         *Service
	projectId string
	datasetId string
	tableId   string
	table     *Table
	opt_      map[string]interface{}
}

// Update: 
func (r *TablesService) Update(projectId string, datasetId string, tableId string, table *Table) *TablesUpdateCall {
	c := &TablesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.projectId = projectId
	c.datasetId = datasetId
	c.tableId = tableId
	c.table = table
	return c
}

func (c *TablesUpdateCall) Do() (*Table, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.table)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/bigquery/v2beta1/", "projects/{projectId}/datasets/{datasetId}/tables/{tableId}")
	urls = strings.Replace(urls, "{projectId}", cleanPathString(c.projectId), 1)
	urls = strings.Replace(urls, "{datasetId}", cleanPathString(c.datasetId), 1)
	urls = strings.Replace(urls, "{tableId}", cleanPathString(c.tableId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Table)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "httpMethod": "PUT",
	//   "id": "bigquery.tables.update",
	//   "parameterOrder": [
	//     "projectId",
	//     "datasetId",
	//     "tableId"
	//   ],
	//   "parameters": {
	//     "datasetId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "projectId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tableId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "projects/{projectId}/datasets/{datasetId}/tables/{tableId}",
	//   "request": {
	//     "$ref": "Table"
	//   },
	//   "response": {
	//     "$ref": "Table"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/bigquery"
	//   ]
	// }

}

func cleanPathString(s string) string {
	return strings.Map(func(r rune) rune {
		if r >= 0x30 && r <= 0x7a {
			return r
		}
		return -1
	}, s)
}
