// Package taskqueue provides access to the TaskQueue API.
//
// See https://developers.google.com/appengine/docs/python/taskqueue/rest
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/taskqueue/v1beta1"
//   ...
//   taskqueueService, err := taskqueue.New(oauthHttpClient)
package taskqueue

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

const apiId = "taskqueue:v1beta1"
const apiName = "taskqueue"
const apiVersion = "v1beta1"
const basePath = "https://www.googleapis.com/taskqueue/v1beta1/projects/"

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
	client   *http.Client
	BasePath string // API endpoint base URL

	Taskqueues *TaskqueuesService

	Tasks *TasksService
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
}

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
}

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
}

type Tasks struct {
	// Items: The actual list of tasks returned as a result of the lease
	// operation.
	Items []*Task `json:"items,omitempty"`

	// Kind: The kind of object returned, a list of tasks.
	Kind string `json:"kind,omitempty"`
}

type Tasks2 struct {
	// Items: The actual list of tasks currently active in the TaskQueue.
	Items []*Task `json:"items,omitempty"`

	// Kind: The kind of object returned, a list of tasks.
	Kind string `json:"kind,omitempty"`
}

// method id "taskqueue.taskqueues.get":

type TaskqueuesGetCall struct {
	s         *Service
	project   string
	taskqueue string
	opt_      map[string]interface{}
}

// Get: Get detailed information about a TaskQueue.
func (r *TaskqueuesService) Get(project string, taskqueue string) *TaskqueuesGetCall {
	c := &TaskqueuesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.taskqueue = taskqueue
	return c
}

// GetStats sets the optional parameter "getStats": Whether to get
// stats.
func (c *TaskqueuesGetCall) GetStats(getStats bool) *TaskqueuesGetCall {
	c.opt_["getStats"] = getStats
	return c
}

func (c *TaskqueuesGetCall) Do() (*TaskQueue, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["getStats"]; ok {
		params.Set("getStats", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{taskqueue}", url.QueryEscape(c.taskqueue), 1)
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
	ret := new(TaskQueue)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	s         *Service
	project   string
	taskqueue string
	task      string
	opt_      map[string]interface{}
}

// Delete: Delete a task from a TaskQueue.
func (r *TasksService) Delete(project string, taskqueue string, task string) *TasksDeleteCall {
	c := &TasksDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.taskqueue = taskqueue
	c.task = task
	return c
}

func (c *TasksDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}/tasks/{task}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{taskqueue}", url.QueryEscape(c.taskqueue), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{task}", url.QueryEscape(c.task), 1)
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
	s         *Service
	project   string
	taskqueue string
	task      string
	opt_      map[string]interface{}
}

// Get: Get a particular task from a TaskQueue.
func (r *TasksService) Get(project string, taskqueue string, task string) *TasksGetCall {
	c := &TasksGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.taskqueue = taskqueue
	c.task = task
	return c
}

func (c *TasksGetCall) Do() (*Task, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}/tasks/{task}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{taskqueue}", url.QueryEscape(c.taskqueue), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{task}", url.QueryEscape(c.task), 1)
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
	ret := new(Task)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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

// method id "taskqueue.tasks.lease":

type TasksLeaseCall struct {
	s         *Service
	project   string
	taskqueue string
	numTasks  int64
	leaseSecs int64
	opt_      map[string]interface{}
}

// Lease: Lease 1 or more tasks from a TaskQueue.
func (r *TasksService) Lease(project string, taskqueue string, numTasks int64, leaseSecs int64) *TasksLeaseCall {
	c := &TasksLeaseCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.taskqueue = taskqueue
	c.numTasks = numTasks
	c.leaseSecs = leaseSecs
	return c
}

func (c *TasksLeaseCall) Do() (*Tasks, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("leaseSecs", fmt.Sprintf("%v", c.leaseSecs))
	params.Set("numTasks", fmt.Sprintf("%v", c.numTasks))
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}/tasks/lease")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{taskqueue}", url.QueryEscape(c.taskqueue), 1)
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
	ret := new(Tasks)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	s         *Service
	project   string
	taskqueue string
	opt_      map[string]interface{}
}

// List: List Tasks in a TaskQueue
func (r *TasksService) List(project string, taskqueue string) *TasksListCall {
	c := &TasksListCall{s: r.s, opt_: make(map[string]interface{})}
	c.project = project
	c.taskqueue = taskqueue
	return c
}

func (c *TasksListCall) Do() (*Tasks2, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{project}/taskqueues/{taskqueue}/tasks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{project}", url.QueryEscape(c.project), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{taskqueue}", url.QueryEscape(c.taskqueue), 1)
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
	ret := new(Tasks2)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
