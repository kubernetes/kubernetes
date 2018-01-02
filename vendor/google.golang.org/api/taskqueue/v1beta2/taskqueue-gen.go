// Package taskqueue provides access to the TaskQueue API.
//
// See https://developers.google.com/appengine/docs/python/taskqueue/rest
//
// Usage example:
//
//   import "google.golang.org/api/taskqueue/v1beta2"
//   ...
//   taskqueueService, err := taskqueue.New(oauthHttpClient)
package taskqueue // import "google.golang.org/api/taskqueue/v1beta2"

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

const apiId = "taskqueue:v1beta2"
const apiName = "taskqueue"
const apiVersion = "v1beta2"
const basePath = "https://www.googleapis.com/taskqueue/v1beta2/projects/"

// OAuth2 scopes used by this API.
const (
	// Manage your Tasks and Taskqueues
	TaskqueueScope = "https://www.googleapis.com/auth/taskqueue"

	// Consume Tasks from your Taskqueues
	TaskqueueConsumerScope = "https://www.googleapis.com/auth/taskqueue.consumer"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Taskqueues = NewTaskqueuesService(s)
	s.Tasks = NewTasksService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Taskqueues *TaskqueuesService

	Tasks *TasksService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewTaskqueuesService(s *Service) *TaskqueuesService {
	rs := &TaskqueuesService{s: s}
	return rs
}

type TaskqueuesService struct {
	s *Service
}

func NewTasksService(s *Service) *TasksService {
	rs := &TasksService{s: s}
	return rs
}

type TasksService struct {
	s *Service
}

type Task struct {
	// EnqueueTimestamp: Time (in seconds since the epoch) at which the task
	// was enqueued.
	EnqueueTimestamp int64 `json:"enqueueTimestamp,omitempty,string"`

	// Id: Name of the task.
	Id string `json:"id,omitempty"`

	// Kind: The kind of object returned, in this case set to task.
	Kind string `json:"kind,omitempty"`

	// LeaseTimestamp: Time (in seconds since the epoch) at which the task
	// lease will expire. This value is 0 if the task isnt currently leased
	// out to a worker.
	LeaseTimestamp int64 `json:"leaseTimestamp,omitempty,string"`

	// PayloadBase64: A bag of bytes which is the task payload. The payload
	// on the JSON side is always Base64 encoded.
	PayloadBase64 string `json:"payloadBase64,omitempty"`

	// QueueName: Name of the queue that the task is in.
	QueueName string `json:"queueName,omitempty"`

	// RetryCount: The number of leases applied to this task.
	RetryCount int64 `json:"retry_count,omitempty"`

	// Tag: Tag for the task, could be used later to lease tasks grouped by
	// a specific tag.
	Tag string `json:"tag,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "EnqueueTimestamp") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EnqueueTimestamp") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *Task) MarshalJSON() ([]byte, error) {
	type noMethod Task
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TaskQueue struct {
	// Acl: ACLs that are applicable to this TaskQueue object.
	Acl *TaskQueueAcl `json:"acl,omitempty"`

	// Id: Name of the taskqueue.
	Id string `json:"id,omitempty"`

	// Kind: The kind of REST object returned, in this case taskqueue.
	Kind string `json:"kind,omitempty"`

	// MaxLeases: The number of times we should lease out tasks before
	// giving up on them. If unset we lease them out forever until a worker
	// deletes the task.
	MaxLeases int64 `json:"maxLeases,omitempty"`

	// Stats: Statistics for the TaskQueue object in question.
	Stats *TaskQueueStats `json:"stats,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Acl") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Acl") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TaskQueue) MarshalJSON() ([]byte, error) {
	type noMethod TaskQueue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TaskQueueAcl: ACLs that are applicable to this TaskQueue object.
type TaskQueueAcl struct {
	// AdminEmails: Email addresses of users who are "admins" of the
	// TaskQueue. This means they can control the queue, eg set ACLs for the
	// queue.
	AdminEmails []string `json:"adminEmails,omitempty"`

	// ConsumerEmails: Email addresses of users who can "consume" tasks from
	// the TaskQueue. This means they can Dequeue and Delete tasks from the
	// queue.
	ConsumerEmails []string `json:"consumerEmails,omitempty"`

	// ProducerEmails: Email addresses of users who can "produce" tasks into
	// the TaskQueue. This means they can Insert tasks into the queue.
	ProducerEmails []string `json:"producerEmails,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AdminEmails") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AdminEmails") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TaskQueueAcl) MarshalJSON() ([]byte, error) {
	type noMethod TaskQueueAcl
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TaskQueueStats: Statistics for the TaskQueue object in question.
type TaskQueueStats struct {
	// LeasedLastHour: Number of tasks leased in the last hour.
	LeasedLastHour int64 `json:"leasedLastHour,omitempty,string"`

	// LeasedLastMinute: Number of tasks leased in the last minute.
	LeasedLastMinute int64 `json:"leasedLastMinute,omitempty,string"`

	// OldestTask: The timestamp (in seconds since the epoch) of the oldest
	// unfinished task.
	OldestTask int64 `json:"oldestTask,omitempty,string"`

	// TotalTasks: Number of tasks in the queue.
	TotalTasks int64 `json:"totalTasks,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LeasedLastHour") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "LeasedLastHour") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *TaskQueueStats) MarshalJSON() ([]byte, error) {
	type noMethod TaskQueueStats
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Tasks struct {
	// Items: The actual list of tasks returned as a result of the lease
	// operation.
	Items []*Task `json:"items,omitempty"`

	// Kind: The kind of object returned, a list of tasks.
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

func (s *Tasks) MarshalJSON() ([]byte, error) {
	type noMethod Tasks
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type Tasks2 struct {
	// Items: The actual list of tasks currently active in the TaskQueue.
	Items []*Task `json:"items,omitempty"`

	// Kind: The kind of object returned, a list of tasks.
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

func (s *Tasks2) MarshalJSON() ([]byte, error) {
	type noMethod Tasks2
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "taskqueue.taskqueues.get":

type TaskqueuesGetCall struct {
	s            *Service
	project      string
	taskqueue    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Get detailed information about a TaskQueue.
func (r *TaskqueuesService) Get(project string, taskqueue string) *TaskqueuesGetCall {
	c := &TaskqueuesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.taskqueue = taskqueue
	return c
}

// GetStats sets the optional parameter "getStats": Whether to get
// stats.
func (c *TaskqueuesGetCall) GetStats(getStats bool) *TaskqueuesGetCall {
	c.urlParams_.Set("getStats", fmt.Sprint(getStats))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TaskqueuesGetCall) Fields(s ...googleapi.Field) *TaskqueuesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TaskqueuesGetCall) IfNoneMatch(entityTag string) *TaskqueuesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TaskqueuesGetCall) Context(ctx context.Context) *TaskqueuesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TaskqueuesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TaskqueuesGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"taskqueue": c.taskqueue,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "taskqueue.taskqueues.get" call.
// Exactly one of *TaskQueue or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *TaskQueue.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TaskqueuesGetCall) Do(opts ...googleapi.CallOption) (*TaskQueue, error) {
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
	ret := &TaskQueue{
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
	//   "description": "Get detailed information about a TaskQueue.",
	//   "httpMethod": "GET",
	//   "id": "taskqueue.taskqueues.get",
	//   "parameterOrder": [
	//     "project",
	//     "taskqueue"
	//   ],
	//   "parameters": {
	//     "getStats": {
	//       "description": "Whether to get stats. Optional.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "project": {
	//       "description": "The project under which the queue lies.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "taskqueue": {
	//       "description": "The id of the taskqueue to get the properties of.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/taskqueues/{taskqueue}",
	//   "response": {
	//     "$ref": "TaskQueue"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/taskqueue",
	//     "https://www.googleapis.com/auth/taskqueue.consumer"
	//   ]
	// }

}

// method id "taskqueue.tasks.delete":

type TasksDeleteCall struct {
	s          *Service
	project    string
	taskqueue  string
	task       string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Delete a task from a TaskQueue.
func (r *TasksService) Delete(project string, taskqueue string, task string) *TasksDeleteCall {
	c := &TasksDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.taskqueue = taskqueue
	c.task = task
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasksDeleteCall) Fields(s ...googleapi.Field) *TasksDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasksDeleteCall) Context(ctx context.Context) *TasksDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasksDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasksDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}/tasks/{task}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"taskqueue": c.taskqueue,
		"task":      c.task,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "taskqueue.tasks.delete" call.
func (c *TasksDeleteCall) Do(opts ...googleapi.CallOption) error {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Delete a task from a TaskQueue.",
	//   "httpMethod": "DELETE",
	//   "id": "taskqueue.tasks.delete",
	//   "parameterOrder": [
	//     "project",
	//     "taskqueue",
	//     "task"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "The project under which the queue lies.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "task": {
	//       "description": "The id of the task to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "taskqueue": {
	//       "description": "The taskqueue to delete a task from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/taskqueues/{taskqueue}/tasks/{task}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/taskqueue",
	//     "https://www.googleapis.com/auth/taskqueue.consumer"
	//   ]
	// }

}

// method id "taskqueue.tasks.get":

type TasksGetCall struct {
	s            *Service
	project      string
	taskqueue    string
	task         string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Get a particular task from a TaskQueue.
func (r *TasksService) Get(project string, taskqueue string, task string) *TasksGetCall {
	c := &TasksGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.taskqueue = taskqueue
	c.task = task
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasksGetCall) Fields(s ...googleapi.Field) *TasksGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TasksGetCall) IfNoneMatch(entityTag string) *TasksGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasksGetCall) Context(ctx context.Context) *TasksGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasksGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasksGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}/tasks/{task}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"taskqueue": c.taskqueue,
		"task":      c.task,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "taskqueue.tasks.get" call.
// Exactly one of *Task or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Task.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *TasksGetCall) Do(opts ...googleapi.CallOption) (*Task, error) {
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
	ret := &Task{
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
	//   "description": "Get a particular task from a TaskQueue.",
	//   "httpMethod": "GET",
	//   "id": "taskqueue.tasks.get",
	//   "parameterOrder": [
	//     "project",
	//     "taskqueue",
	//     "task"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "The project under which the queue lies.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "task": {
	//       "description": "The task to get properties of.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "taskqueue": {
	//       "description": "The taskqueue in which the task belongs.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/taskqueues/{taskqueue}/tasks/{task}",
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/taskqueue",
	//     "https://www.googleapis.com/auth/taskqueue.consumer"
	//   ]
	// }

}

// method id "taskqueue.tasks.insert":

type TasksInsertCall struct {
	s          *Service
	project    string
	taskqueue  string
	task       *Task
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Insert a new task in a TaskQueue
func (r *TasksService) Insert(project string, taskqueue string, task *Task) *TasksInsertCall {
	c := &TasksInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.taskqueue = taskqueue
	c.task = task
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasksInsertCall) Fields(s ...googleapi.Field) *TasksInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasksInsertCall) Context(ctx context.Context) *TasksInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasksInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasksInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.task)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}/tasks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"taskqueue": c.taskqueue,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "taskqueue.tasks.insert" call.
// Exactly one of *Task or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Task.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *TasksInsertCall) Do(opts ...googleapi.CallOption) (*Task, error) {
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
	ret := &Task{
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
	//   "description": "Insert a new task in a TaskQueue",
	//   "httpMethod": "POST",
	//   "id": "taskqueue.tasks.insert",
	//   "parameterOrder": [
	//     "project",
	//     "taskqueue"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "The project under which the queue lies",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "taskqueue": {
	//       "description": "The taskqueue to insert the task into",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/taskqueues/{taskqueue}/tasks",
	//   "request": {
	//     "$ref": "Task"
	//   },
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/taskqueue",
	//     "https://www.googleapis.com/auth/taskqueue.consumer"
	//   ]
	// }

}

// method id "taskqueue.tasks.lease":

type TasksLeaseCall struct {
	s          *Service
	project    string
	taskqueue  string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Lease: Lease 1 or more tasks from a TaskQueue.
func (r *TasksService) Lease(project string, taskqueue string, numTasks int64, leaseSecs int64) *TasksLeaseCall {
	c := &TasksLeaseCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.taskqueue = taskqueue
	c.urlParams_.Set("numTasks", fmt.Sprint(numTasks))
	c.urlParams_.Set("leaseSecs", fmt.Sprint(leaseSecs))
	return c
}

// GroupByTag sets the optional parameter "groupByTag": When true, all
// returned tasks will have the same tag
func (c *TasksLeaseCall) GroupByTag(groupByTag bool) *TasksLeaseCall {
	c.urlParams_.Set("groupByTag", fmt.Sprint(groupByTag))
	return c
}

// Tag sets the optional parameter "tag": The tag allowed for tasks in
// the response. Must only be specified if group_by_tag is true. If
// group_by_tag is true and tag is not specified the tag will be that of
// the oldest task by eta, i.e. the first available tag
func (c *TasksLeaseCall) Tag(tag string) *TasksLeaseCall {
	c.urlParams_.Set("tag", tag)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasksLeaseCall) Fields(s ...googleapi.Field) *TasksLeaseCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasksLeaseCall) Context(ctx context.Context) *TasksLeaseCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasksLeaseCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasksLeaseCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}/tasks/lease")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"taskqueue": c.taskqueue,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "taskqueue.tasks.lease" call.
// Exactly one of *Tasks or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Tasks.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TasksLeaseCall) Do(opts ...googleapi.CallOption) (*Tasks, error) {
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
	ret := &Tasks{
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
	//   "description": "Lease 1 or more tasks from a TaskQueue.",
	//   "httpMethod": "POST",
	//   "id": "taskqueue.tasks.lease",
	//   "parameterOrder": [
	//     "project",
	//     "taskqueue",
	//     "numTasks",
	//     "leaseSecs"
	//   ],
	//   "parameters": {
	//     "groupByTag": {
	//       "description": "When true, all returned tasks will have the same tag",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "leaseSecs": {
	//       "description": "The lease in seconds.",
	//       "format": "int32",
	//       "location": "query",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "numTasks": {
	//       "description": "The number of tasks to lease.",
	//       "format": "int32",
	//       "location": "query",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "project": {
	//       "description": "The project under which the queue lies.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tag": {
	//       "description": "The tag allowed for tasks in the response. Must only be specified if group_by_tag is true. If group_by_tag is true and tag is not specified the tag will be that of the oldest task by eta, i.e. the first available tag",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "taskqueue": {
	//       "description": "The taskqueue to lease a task from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/taskqueues/{taskqueue}/tasks/lease",
	//   "response": {
	//     "$ref": "Tasks"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/taskqueue",
	//     "https://www.googleapis.com/auth/taskqueue.consumer"
	//   ]
	// }

}

// method id "taskqueue.tasks.list":

type TasksListCall struct {
	s            *Service
	project      string
	taskqueue    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: List Tasks in a TaskQueue
func (r *TasksService) List(project string, taskqueue string) *TasksListCall {
	c := &TasksListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.taskqueue = taskqueue
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasksListCall) Fields(s ...googleapi.Field) *TasksListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TasksListCall) IfNoneMatch(entityTag string) *TasksListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasksListCall) Context(ctx context.Context) *TasksListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasksListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasksListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}/tasks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"taskqueue": c.taskqueue,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "taskqueue.tasks.list" call.
// Exactly one of *Tasks2 or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Tasks2.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TasksListCall) Do(opts ...googleapi.CallOption) (*Tasks2, error) {
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
	ret := &Tasks2{
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
	//   "description": "List Tasks in a TaskQueue",
	//   "httpMethod": "GET",
	//   "id": "taskqueue.tasks.list",
	//   "parameterOrder": [
	//     "project",
	//     "taskqueue"
	//   ],
	//   "parameters": {
	//     "project": {
	//       "description": "The project under which the queue lies.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "taskqueue": {
	//       "description": "The id of the taskqueue to list tasks from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/taskqueues/{taskqueue}/tasks",
	//   "response": {
	//     "$ref": "Tasks2"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/taskqueue",
	//     "https://www.googleapis.com/auth/taskqueue.consumer"
	//   ]
	// }

}

// method id "taskqueue.tasks.patch":

type TasksPatchCall struct {
	s          *Service
	project    string
	taskqueue  string
	task       string
	task2      *Task
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Update tasks that are leased out of a TaskQueue. This method
// supports patch semantics.
func (r *TasksService) Patch(project string, taskqueue string, task string, newLeaseSeconds int64, task2 *Task) *TasksPatchCall {
	c := &TasksPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.taskqueue = taskqueue
	c.task = task
	c.urlParams_.Set("newLeaseSeconds", fmt.Sprint(newLeaseSeconds))
	c.task2 = task2
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasksPatchCall) Fields(s ...googleapi.Field) *TasksPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasksPatchCall) Context(ctx context.Context) *TasksPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasksPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasksPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.task2)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}/tasks/{task}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"taskqueue": c.taskqueue,
		"task":      c.task,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "taskqueue.tasks.patch" call.
// Exactly one of *Task or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Task.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *TasksPatchCall) Do(opts ...googleapi.CallOption) (*Task, error) {
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
	ret := &Task{
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
	//   "description": "Update tasks that are leased out of a TaskQueue. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "taskqueue.tasks.patch",
	//   "parameterOrder": [
	//     "project",
	//     "taskqueue",
	//     "task",
	//     "newLeaseSeconds"
	//   ],
	//   "parameters": {
	//     "newLeaseSeconds": {
	//       "description": "The new lease in seconds.",
	//       "format": "int32",
	//       "location": "query",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "project": {
	//       "description": "The project under which the queue lies.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "task": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "taskqueue": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/taskqueues/{taskqueue}/tasks/{task}",
	//   "request": {
	//     "$ref": "Task"
	//   },
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/taskqueue",
	//     "https://www.googleapis.com/auth/taskqueue.consumer"
	//   ]
	// }

}

// method id "taskqueue.tasks.update":

type TasksUpdateCall struct {
	s          *Service
	project    string
	taskqueue  string
	task       string
	task2      *Task
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Update tasks that are leased out of a TaskQueue.
func (r *TasksService) Update(project string, taskqueue string, task string, newLeaseSeconds int64, task2 *Task) *TasksUpdateCall {
	c := &TasksUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	c.taskqueue = taskqueue
	c.task = task
	c.urlParams_.Set("newLeaseSeconds", fmt.Sprint(newLeaseSeconds))
	c.task2 = task2
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasksUpdateCall) Fields(s ...googleapi.Field) *TasksUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasksUpdateCall) Context(ctx context.Context) *TasksUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasksUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasksUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.task2)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}/tasks/{task}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"project":   c.project,
		"taskqueue": c.taskqueue,
		"task":      c.task,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "taskqueue.tasks.update" call.
// Exactly one of *Task or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Task.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *TasksUpdateCall) Do(opts ...googleapi.CallOption) (*Task, error) {
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
	ret := &Task{
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
	//   "description": "Update tasks that are leased out of a TaskQueue.",
	//   "httpMethod": "POST",
	//   "id": "taskqueue.tasks.update",
	//   "parameterOrder": [
	//     "project",
	//     "taskqueue",
	//     "task",
	//     "newLeaseSeconds"
	//   ],
	//   "parameters": {
	//     "newLeaseSeconds": {
	//       "description": "The new lease in seconds.",
	//       "format": "int32",
	//       "location": "query",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "project": {
	//       "description": "The project under which the queue lies.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "task": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "taskqueue": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{project}/taskqueues/{taskqueue}/tasks/{task}",
	//   "request": {
	//     "$ref": "Task"
	//   },
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/taskqueue",
	//     "https://www.googleapis.com/auth/taskqueue.consumer"
	//   ]
	// }

}
