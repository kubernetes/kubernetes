// Package tasks provides access to the Tasks API.
//
// See https://developers.google.com/google-apps/tasks/firstapp
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/tasks/v1"
//   ...
//   tasksService, err := tasks.New(oauthHttpClient)
package tasks

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

const apiId = "tasks:v1"
const apiName = "tasks"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/tasks/v1/"

// OAuth2 scopes used by this API.
const (
	// Manage your tasks
	TasksScope = "https://www.googleapis.com/auth/tasks"

	// View your tasks
	TasksReadonlyScope = "https://www.googleapis.com/auth/tasks.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Tasklists = NewTasklistsService(s)
	s.Tasks = NewTasksService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Tasklists *TasklistsService

	Tasks *TasksService
}

func NewTasklistsService(s *Service) *TasklistsService {
	rs := &TasklistsService{s: s}
	return rs
}

type TasklistsService struct {
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
	// Completed: Completion date of the task (as a RFC 3339 timestamp).
	// This field is omitted if the task has not been completed.
	Completed string `json:"completed,omitempty"`

	// Deleted: Flag indicating whether the task has been deleted. The
	// default if False.
	Deleted bool `json:"deleted,omitempty"`

	// Due: Due date of the task (as a RFC 3339 timestamp). Optional.
	Due string `json:"due,omitempty"`

	// Etag: ETag of the resource.
	Etag string `json:"etag,omitempty"`

	// Hidden: Flag indicating whether the task is hidden. This is the case
	// if the task had been marked completed when the task list was last
	// cleared. The default is False. This field is read-only.
	Hidden bool `json:"hidden,omitempty"`

	// Id: Task identifier.
	Id string `json:"id,omitempty"`

	// Kind: Type of the resource. This is always "tasks#task".
	Kind string `json:"kind,omitempty"`

	// Links: Collection of links. This collection is read-only.
	Links []*TaskLinks `json:"links,omitempty"`

	// Notes: Notes describing the task. Optional.
	Notes string `json:"notes,omitempty"`

	// Parent: Parent task identifier. This field is omitted if it is a
	// top-level task. This field is read-only. Use the "move" method to
	// move the task under a different parent or to the top level.
	Parent string `json:"parent,omitempty"`

	// Position: String indicating the position of the task among its
	// sibling tasks under the same parent task or at the top level. If this
	// string is greater than another task's corresponding position string
	// according to lexicographical ordering, the task is positioned after
	// the other task under the same parent task (or at the top level). This
	// field is read-only. Use the "move" method to move the task to another
	// position.
	Position string `json:"position,omitempty"`

	// SelfLink: URL pointing to this task. Used to retrieve, update, or
	// delete this task.
	SelfLink string `json:"selfLink,omitempty"`

	// Status: Status of the task. This is either "needsAction" or
	// "completed".
	Status string `json:"status,omitempty"`

	// Title: Title of the task.
	Title string `json:"title,omitempty"`

	// Updated: Last modification time of the task (as a RFC 3339
	// timestamp).
	Updated string `json:"updated,omitempty"`
}

type TaskLinks struct {
	// Description: The description. In HTML speak: Everything between <a>
	// and </a>.
	Description string `json:"description,omitempty"`

	// Link: The URL.
	Link string `json:"link,omitempty"`

	// Type: Type of the link, e.g. "email".
	Type string `json:"type,omitempty"`
}

type TaskList struct {
	// Etag: ETag of the resource.
	Etag string `json:"etag,omitempty"`

	// Id: Task list identifier.
	Id string `json:"id,omitempty"`

	// Kind: Type of the resource. This is always "tasks#taskList".
	Kind string `json:"kind,omitempty"`

	// SelfLink: URL pointing to this task list. Used to retrieve, update,
	// or delete this task list.
	SelfLink string `json:"selfLink,omitempty"`

	// Title: Title of the task list.
	Title string `json:"title,omitempty"`

	// Updated: Last modification time of the task list (as a RFC 3339
	// timestamp).
	Updated string `json:"updated,omitempty"`
}

type TaskLists struct {
	// Etag: ETag of the resource.
	Etag string `json:"etag,omitempty"`

	// Items: Collection of task lists.
	Items []*TaskList `json:"items,omitempty"`

	// Kind: Type of the resource. This is always "tasks#taskLists".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Token that can be used to request the next page of
	// this result.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type Tasks struct {
	// Etag: ETag of the resource.
	Etag string `json:"etag,omitempty"`

	// Items: Collection of tasks.
	Items []*Task `json:"items,omitempty"`

	// Kind: Type of the resource. This is always "tasks#tasks".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Token used to access the next page of this result.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

// method id "tasks.tasklists.delete":

type TasklistsDeleteCall struct {
	s          *Service
	tasklistid string
	opt_       map[string]interface{}
}

// Delete: Deletes the authenticated user's specified task list.
func (r *TasklistsService) Delete(tasklistid string) *TasklistsDeleteCall {
	c := &TasklistsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	return c
}

func (c *TasklistsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists/{tasklist}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
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
	//   "description": "Deletes the authenticated user's specified task list.",
	//   "httpMethod": "DELETE",
	//   "id": "tasks.tasklists.delete",
	//   "parameterOrder": [
	//     "tasklist"
	//   ],
	//   "parameters": {
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/@me/lists/{tasklist}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks"
	//   ]
	// }

}

// method id "tasks.tasklists.get":

type TasklistsGetCall struct {
	s          *Service
	tasklistid string
	opt_       map[string]interface{}
}

// Get: Returns the authenticated user's specified task list.
func (r *TasklistsService) Get(tasklistid string) *TasklistsGetCall {
	c := &TasklistsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	return c
}

func (c *TasklistsGetCall) Do() (*TaskList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists/{tasklist}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
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
	ret := new(TaskList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the authenticated user's specified task list.",
	//   "httpMethod": "GET",
	//   "id": "tasks.tasklists.get",
	//   "parameterOrder": [
	//     "tasklist"
	//   ],
	//   "parameters": {
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/@me/lists/{tasklist}",
	//   "response": {
	//     "$ref": "TaskList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks",
	//     "https://www.googleapis.com/auth/tasks.readonly"
	//   ]
	// }

}

// method id "tasks.tasklists.insert":

type TasklistsInsertCall struct {
	s        *Service
	tasklist *TaskList
	opt_     map[string]interface{}
}

// Insert: Creates a new task list and adds it to the authenticated
// user's task lists.
func (r *TasklistsService) Insert(tasklist *TaskList) *TasklistsInsertCall {
	c := &TasklistsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklist = tasklist
	return c
}

func (c *TasklistsInsertCall) Do() (*TaskList, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tasklist)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(TaskList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new task list and adds it to the authenticated user's task lists.",
	//   "httpMethod": "POST",
	//   "id": "tasks.tasklists.insert",
	//   "path": "users/@me/lists",
	//   "request": {
	//     "$ref": "TaskList"
	//   },
	//   "response": {
	//     "$ref": "TaskList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks"
	//   ]
	// }

}

// method id "tasks.tasklists.list":

type TasklistsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Returns all the authenticated user's task lists.
func (r *TasklistsService) List() *TasklistsListCall {
	c := &TasklistsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of task lists returned on one page.  The default is 100.
func (c *TasklistsListCall) MaxResults(maxResults int64) *TasklistsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Token specifying
// the result page to return.
func (c *TasklistsListCall) PageToken(pageToken string) *TasklistsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *TasklistsListCall) Do() (*TaskLists, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists")
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
	ret := new(TaskLists)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns all the authenticated user's task lists.",
	//   "httpMethod": "GET",
	//   "id": "tasks.tasklists.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of task lists returned on one page. Optional. The default is 100.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Token specifying the result page to return. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/@me/lists",
	//   "response": {
	//     "$ref": "TaskLists"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks",
	//     "https://www.googleapis.com/auth/tasks.readonly"
	//   ]
	// }

}

// method id "tasks.tasklists.patch":

type TasklistsPatchCall struct {
	s          *Service
	tasklistid string
	tasklist   *TaskList
	opt_       map[string]interface{}
}

// Patch: Updates the authenticated user's specified task list. This
// method supports patch semantics.
func (r *TasklistsService) Patch(tasklistid string, tasklist *TaskList) *TasklistsPatchCall {
	c := &TasklistsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	c.tasklist = tasklist
	return c
}

func (c *TasklistsPatchCall) Do() (*TaskList, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tasklist)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists/{tasklist}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
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
	ret := new(TaskList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the authenticated user's specified task list. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "tasks.tasklists.patch",
	//   "parameterOrder": [
	//     "tasklist"
	//   ],
	//   "parameters": {
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/@me/lists/{tasklist}",
	//   "request": {
	//     "$ref": "TaskList"
	//   },
	//   "response": {
	//     "$ref": "TaskList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks"
	//   ]
	// }

}

// method id "tasks.tasklists.update":

type TasklistsUpdateCall struct {
	s          *Service
	tasklistid string
	tasklist   *TaskList
	opt_       map[string]interface{}
}

// Update: Updates the authenticated user's specified task list.
func (r *TasklistsService) Update(tasklistid string, tasklist *TaskList) *TasklistsUpdateCall {
	c := &TasklistsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	c.tasklist = tasklist
	return c
}

func (c *TasklistsUpdateCall) Do() (*TaskList, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tasklist)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists/{tasklist}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
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
	ret := new(TaskList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the authenticated user's specified task list.",
	//   "httpMethod": "PUT",
	//   "id": "tasks.tasklists.update",
	//   "parameterOrder": [
	//     "tasklist"
	//   ],
	//   "parameters": {
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/@me/lists/{tasklist}",
	//   "request": {
	//     "$ref": "TaskList"
	//   },
	//   "response": {
	//     "$ref": "TaskList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks"
	//   ]
	// }

}

// method id "tasks.tasks.clear":

type TasksClearCall struct {
	s          *Service
	tasklistid string
	opt_       map[string]interface{}
}

// Clear: Clears all completed tasks from the specified task list. The
// affected tasks will be marked as 'hidden' and no longer be returned
// by default when retrieving all tasks for a task list.
func (r *TasksService) Clear(tasklistid string) *TasksClearCall {
	c := &TasksClearCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	return c
}

func (c *TasksClearCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/clear")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
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
	//   "description": "Clears all completed tasks from the specified task list. The affected tasks will be marked as 'hidden' and no longer be returned by default when retrieving all tasks for a task list.",
	//   "httpMethod": "POST",
	//   "id": "tasks.tasks.clear",
	//   "parameterOrder": [
	//     "tasklist"
	//   ],
	//   "parameters": {
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "lists/{tasklist}/clear",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks"
	//   ]
	// }

}

// method id "tasks.tasks.delete":

type TasksDeleteCall struct {
	s          *Service
	tasklistid string
	taskid     string
	opt_       map[string]interface{}
}

// Delete: Deletes the specified task from the task list.
func (r *TasksService) Delete(tasklistid string, taskid string) *TasksDeleteCall {
	c := &TasksDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	c.taskid = taskid
	return c
}

func (c *TasksDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks/{task}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{task}", url.QueryEscape(c.taskid), 1)
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
	//   "description": "Deletes the specified task from the task list.",
	//   "httpMethod": "DELETE",
	//   "id": "tasks.tasks.delete",
	//   "parameterOrder": [
	//     "tasklist",
	//     "task"
	//   ],
	//   "parameters": {
	//     "task": {
	//       "description": "Task identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "lists/{tasklist}/tasks/{task}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks"
	//   ]
	// }

}

// method id "tasks.tasks.get":

type TasksGetCall struct {
	s          *Service
	tasklistid string
	taskid     string
	opt_       map[string]interface{}
}

// Get: Returns the specified task.
func (r *TasksService) Get(tasklistid string, taskid string) *TasksGetCall {
	c := &TasksGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	c.taskid = taskid
	return c
}

func (c *TasksGetCall) Do() (*Task, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks/{task}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{task}", url.QueryEscape(c.taskid), 1)
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
	//   "description": "Returns the specified task.",
	//   "httpMethod": "GET",
	//   "id": "tasks.tasks.get",
	//   "parameterOrder": [
	//     "tasklist",
	//     "task"
	//   ],
	//   "parameters": {
	//     "task": {
	//       "description": "Task identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "lists/{tasklist}/tasks/{task}",
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks",
	//     "https://www.googleapis.com/auth/tasks.readonly"
	//   ]
	// }

}

// method id "tasks.tasks.insert":

type TasksInsertCall struct {
	s          *Service
	tasklistid string
	task       *Task
	opt_       map[string]interface{}
}

// Insert: Creates a new task on the specified task list.
func (r *TasksService) Insert(tasklistid string, task *Task) *TasksInsertCall {
	c := &TasksInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	c.task = task
	return c
}

// Parent sets the optional parameter "parent": Parent task identifier.
// If the task is created at the top level, this parameter is omitted.
func (c *TasksInsertCall) Parent(parent string) *TasksInsertCall {
	c.opt_["parent"] = parent
	return c
}

// Previous sets the optional parameter "previous": Previous sibling
// task identifier. If the task is created at the first position among
// its siblings, this parameter is omitted.
func (c *TasksInsertCall) Previous(previous string) *TasksInsertCall {
	c.opt_["previous"] = previous
	return c
}

func (c *TasksInsertCall) Do() (*Task, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.task)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["parent"]; ok {
		params.Set("parent", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["previous"]; ok {
		params.Set("previous", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
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
	ret := new(Task)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new task on the specified task list.",
	//   "httpMethod": "POST",
	//   "id": "tasks.tasks.insert",
	//   "parameterOrder": [
	//     "tasklist"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "Parent task identifier. If the task is created at the top level, this parameter is omitted. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "previous": {
	//       "description": "Previous sibling task identifier. If the task is created at the first position among its siblings, this parameter is omitted. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "lists/{tasklist}/tasks",
	//   "request": {
	//     "$ref": "Task"
	//   },
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks"
	//   ]
	// }

}

// method id "tasks.tasks.list":

type TasksListCall struct {
	s          *Service
	tasklistid string
	opt_       map[string]interface{}
}

// List: Returns all tasks in the specified task list.
func (r *TasksService) List(tasklistid string) *TasksListCall {
	c := &TasksListCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	return c
}

// CompletedMax sets the optional parameter "completedMax": Upper bound
// for a task's completion date (as a RFC 3339 timestamp) to filter by.
// The default is not to filter by completion date.
func (c *TasksListCall) CompletedMax(completedMax string) *TasksListCall {
	c.opt_["completedMax"] = completedMax
	return c
}

// CompletedMin sets the optional parameter "completedMin": Lower bound
// for a task's completion date (as a RFC 3339 timestamp) to filter by.
// The default is not to filter by completion date.
func (c *TasksListCall) CompletedMin(completedMin string) *TasksListCall {
	c.opt_["completedMin"] = completedMin
	return c
}

// DueMax sets the optional parameter "dueMax": Upper bound for a task's
// due date (as a RFC 3339 timestamp) to filter by.  The default is not
// to filter by due date.
func (c *TasksListCall) DueMax(dueMax string) *TasksListCall {
	c.opt_["dueMax"] = dueMax
	return c
}

// DueMin sets the optional parameter "dueMin": Lower bound for a task's
// due date (as a RFC 3339 timestamp) to filter by.  The default is not
// to filter by due date.
func (c *TasksListCall) DueMin(dueMin string) *TasksListCall {
	c.opt_["dueMin"] = dueMin
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of task lists returned on one page.  The default is 100.
func (c *TasksListCall) MaxResults(maxResults int64) *TasksListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Token specifying
// the result page to return.
func (c *TasksListCall) PageToken(pageToken string) *TasksListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// ShowCompleted sets the optional parameter "showCompleted": Flag
// indicating whether completed tasks are returned in the result.  The
// default is True.
func (c *TasksListCall) ShowCompleted(showCompleted bool) *TasksListCall {
	c.opt_["showCompleted"] = showCompleted
	return c
}

// ShowDeleted sets the optional parameter "showDeleted": Flag
// indicating whether deleted tasks are returned in the result.  The
// default is False.
func (c *TasksListCall) ShowDeleted(showDeleted bool) *TasksListCall {
	c.opt_["showDeleted"] = showDeleted
	return c
}

// ShowHidden sets the optional parameter "showHidden": Flag indicating
// whether hidden tasks are returned in the result.  The default is
// False.
func (c *TasksListCall) ShowHidden(showHidden bool) *TasksListCall {
	c.opt_["showHidden"] = showHidden
	return c
}

// UpdatedMin sets the optional parameter "updatedMin": Lower bound for
// a task's last modification time (as a RFC 3339 timestamp) to filter
// by.  The default is not to filter by last modification time.
func (c *TasksListCall) UpdatedMin(updatedMin string) *TasksListCall {
	c.opt_["updatedMin"] = updatedMin
	return c
}

func (c *TasksListCall) Do() (*Tasks, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["completedMax"]; ok {
		params.Set("completedMax", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["completedMin"]; ok {
		params.Set("completedMin", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["dueMax"]; ok {
		params.Set("dueMax", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["dueMin"]; ok {
		params.Set("dueMin", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["showCompleted"]; ok {
		params.Set("showCompleted", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["showDeleted"]; ok {
		params.Set("showDeleted", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["showHidden"]; ok {
		params.Set("showHidden", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["updatedMin"]; ok {
		params.Set("updatedMin", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
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
	//   "description": "Returns all tasks in the specified task list.",
	//   "httpMethod": "GET",
	//   "id": "tasks.tasks.list",
	//   "parameterOrder": [
	//     "tasklist"
	//   ],
	//   "parameters": {
	//     "completedMax": {
	//       "description": "Upper bound for a task's completion date (as a RFC 3339 timestamp) to filter by. Optional. The default is not to filter by completion date.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "completedMin": {
	//       "description": "Lower bound for a task's completion date (as a RFC 3339 timestamp) to filter by. Optional. The default is not to filter by completion date.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "dueMax": {
	//       "description": "Upper bound for a task's due date (as a RFC 3339 timestamp) to filter by. Optional. The default is not to filter by due date.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "dueMin": {
	//       "description": "Lower bound for a task's due date (as a RFC 3339 timestamp) to filter by. Optional. The default is not to filter by due date.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of task lists returned on one page. Optional. The default is 100.",
	//       "format": "int64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Token specifying the result page to return. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "showCompleted": {
	//       "description": "Flag indicating whether completed tasks are returned in the result. Optional. The default is True.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "showDeleted": {
	//       "description": "Flag indicating whether deleted tasks are returned in the result. Optional. The default is False.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "showHidden": {
	//       "description": "Flag indicating whether hidden tasks are returned in the result. Optional. The default is False.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updatedMin": {
	//       "description": "Lower bound for a task's last modification time (as a RFC 3339 timestamp) to filter by. Optional. The default is not to filter by last modification time.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "lists/{tasklist}/tasks",
	//   "response": {
	//     "$ref": "Tasks"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks",
	//     "https://www.googleapis.com/auth/tasks.readonly"
	//   ]
	// }

}

// method id "tasks.tasks.move":

type TasksMoveCall struct {
	s          *Service
	tasklistid string
	taskid     string
	opt_       map[string]interface{}
}

// Move: Moves the specified task to another position in the task list.
// This can include putting it as a child task under a new parent and/or
// move it to a different position among its sibling tasks.
func (r *TasksService) Move(tasklistid string, taskid string) *TasksMoveCall {
	c := &TasksMoveCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	c.taskid = taskid
	return c
}

// Parent sets the optional parameter "parent": New parent task
// identifier. If the task is moved to the top level, this parameter is
// omitted.
func (c *TasksMoveCall) Parent(parent string) *TasksMoveCall {
	c.opt_["parent"] = parent
	return c
}

// Previous sets the optional parameter "previous": New previous sibling
// task identifier. If the task is moved to the first position among its
// siblings, this parameter is omitted.
func (c *TasksMoveCall) Previous(previous string) *TasksMoveCall {
	c.opt_["previous"] = previous
	return c
}

func (c *TasksMoveCall) Do() (*Task, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["parent"]; ok {
		params.Set("parent", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["previous"]; ok {
		params.Set("previous", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks/{task}/move")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{task}", url.QueryEscape(c.taskid), 1)
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
	//   "description": "Moves the specified task to another position in the task list. This can include putting it as a child task under a new parent and/or move it to a different position among its sibling tasks.",
	//   "httpMethod": "POST",
	//   "id": "tasks.tasks.move",
	//   "parameterOrder": [
	//     "tasklist",
	//     "task"
	//   ],
	//   "parameters": {
	//     "parent": {
	//       "description": "New parent task identifier. If the task is moved to the top level, this parameter is omitted. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "previous": {
	//       "description": "New previous sibling task identifier. If the task is moved to the first position among its siblings, this parameter is omitted. Optional.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "task": {
	//       "description": "Task identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "lists/{tasklist}/tasks/{task}/move",
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks"
	//   ]
	// }

}

// method id "tasks.tasks.patch":

type TasksPatchCall struct {
	s          *Service
	tasklistid string
	taskid     string
	task       *Task
	opt_       map[string]interface{}
}

// Patch: Updates the specified task. This method supports patch
// semantics.
func (r *TasksService) Patch(tasklistid string, taskid string, task *Task) *TasksPatchCall {
	c := &TasksPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	c.taskid = taskid
	c.task = task
	return c
}

func (c *TasksPatchCall) Do() (*Task, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.task)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks/{task}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{task}", url.QueryEscape(c.taskid), 1)
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
	ret := new(Task)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the specified task. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "tasks.tasks.patch",
	//   "parameterOrder": [
	//     "tasklist",
	//     "task"
	//   ],
	//   "parameters": {
	//     "task": {
	//       "description": "Task identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "lists/{tasklist}/tasks/{task}",
	//   "request": {
	//     "$ref": "Task"
	//   },
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks"
	//   ]
	// }

}

// method id "tasks.tasks.update":

type TasksUpdateCall struct {
	s          *Service
	tasklistid string
	taskid     string
	task       *Task
	opt_       map[string]interface{}
}

// Update: Updates the specified task.
func (r *TasksService) Update(tasklistid string, taskid string, task *Task) *TasksUpdateCall {
	c := &TasksUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.tasklistid = tasklistid
	c.taskid = taskid
	c.task = task
	return c
}

func (c *TasksUpdateCall) Do() (*Task, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.task)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks/{task}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{tasklist}", url.QueryEscape(c.tasklistid), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{task}", url.QueryEscape(c.taskid), 1)
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
	ret := new(Task)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the specified task.",
	//   "httpMethod": "PUT",
	//   "id": "tasks.tasks.update",
	//   "parameterOrder": [
	//     "tasklist",
	//     "task"
	//   ],
	//   "parameters": {
	//     "task": {
	//       "description": "Task identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "tasklist": {
	//       "description": "Task list identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "lists/{tasklist}/tasks/{task}",
	//   "request": {
	//     "$ref": "Task"
	//   },
	//   "response": {
	//     "$ref": "Task"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/tasks"
	//   ]
	// }

}
