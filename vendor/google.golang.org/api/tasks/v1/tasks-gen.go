// Package tasks provides access to the Tasks API.
//
// See https://developers.google.com/google-apps/tasks/firstapp
//
// Usage example:
//
//   import "google.golang.org/api/tasks/v1"
//   ...
//   tasksService, err := tasks.New(oauthHttpClient)
package tasks // import "google.golang.org/api/tasks/v1"

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
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Tasklists *TasklistsService

	Tasks *TasksService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
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
	Completed *string `json:"completed,omitempty"`

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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Completed") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Completed") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Task) MarshalJSON() ([]byte, error) {
	type noMethod Task
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type TaskLinks struct {
	// Description: The description. In HTML speak: Everything between <a>
	// and </a>.
	Description string `json:"description,omitempty"`

	// Link: The URL.
	Link string `json:"link,omitempty"`

	// Type: Type of the link, e.g. "email".
	Type string `json:"type,omitempty"`

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

func (s *TaskLinks) MarshalJSON() ([]byte, error) {
	type noMethod TaskLinks
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *TaskList) MarshalJSON() ([]byte, error) {
	type noMethod TaskList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *TaskLists) MarshalJSON() ([]byte, error) {
	type noMethod TaskLists
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

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

func (s *Tasks) MarshalJSON() ([]byte, error) {
	type noMethod Tasks
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "tasks.tasklists.delete":

type TasklistsDeleteCall struct {
	s          *Service
	tasklistid string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes the authenticated user's specified task list.
func (r *TasklistsService) Delete(tasklistid string) *TasklistsDeleteCall {
	c := &TasklistsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasklistsDeleteCall) Fields(s ...googleapi.Field) *TasklistsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasklistsDeleteCall) Context(ctx context.Context) *TasklistsDeleteCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasklistsDeleteCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasklistsDeleteCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists/{tasklist}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasklists.delete" call.
func (c *TasklistsDeleteCall) Do(opts ...googleapi.CallOption) error {
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
	s            *Service
	tasklistid   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Returns the authenticated user's specified task list.
func (r *TasklistsService) Get(tasklistid string) *TasklistsGetCall {
	c := &TasklistsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasklistsGetCall) Fields(s ...googleapi.Field) *TasklistsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TasklistsGetCall) IfNoneMatch(entityTag string) *TasklistsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasklistsGetCall) Context(ctx context.Context) *TasklistsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasklistsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasklistsGetCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists/{tasklist}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasklists.get" call.
// Exactly one of *TaskList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *TaskList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TasklistsGetCall) Do(opts ...googleapi.CallOption) (*TaskList, error) {
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
	ret := &TaskList{
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
	s          *Service
	tasklist   *TaskList
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Creates a new task list and adds it to the authenticated
// user's task lists.
func (r *TasklistsService) Insert(tasklist *TaskList) *TasklistsInsertCall {
	c := &TasklistsInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklist = tasklist
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasklistsInsertCall) Fields(s ...googleapi.Field) *TasklistsInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasklistsInsertCall) Context(ctx context.Context) *TasklistsInsertCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasklistsInsertCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasklistsInsertCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tasklist)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasklists.insert" call.
// Exactly one of *TaskList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *TaskList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TasklistsInsertCall) Do(opts ...googleapi.CallOption) (*TaskList, error) {
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
	ret := &TaskList{
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
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Returns all the authenticated user's task lists.
func (r *TasklistsService) List() *TasklistsListCall {
	c := &TasklistsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of task lists returned on one page.  The default is 100.
func (c *TasklistsListCall) MaxResults(maxResults int64) *TasklistsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Token specifying
// the result page to return.
func (c *TasklistsListCall) PageToken(pageToken string) *TasklistsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasklistsListCall) Fields(s ...googleapi.Field) *TasklistsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TasklistsListCall) IfNoneMatch(entityTag string) *TasklistsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasklistsListCall) Context(ctx context.Context) *TasklistsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasklistsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasklistsListCall) doRequest(alt string) (*http.Response, error) {
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasklists.list" call.
// Exactly one of *TaskLists or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *TaskLists.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TasklistsListCall) Do(opts ...googleapi.CallOption) (*TaskLists, error) {
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
	ret := &TaskLists{
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

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *TasklistsListCall) Pages(ctx context.Context, f func(*TaskLists) error) error {
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

// method id "tasks.tasklists.patch":

type TasklistsPatchCall struct {
	s          *Service
	tasklistid string
	tasklist   *TaskList
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates the authenticated user's specified task list. This
// method supports patch semantics.
func (r *TasklistsService) Patch(tasklistid string, tasklist *TaskList) *TasklistsPatchCall {
	c := &TasklistsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	c.tasklist = tasklist
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasklistsPatchCall) Fields(s ...googleapi.Field) *TasklistsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasklistsPatchCall) Context(ctx context.Context) *TasklistsPatchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasklistsPatchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasklistsPatchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tasklist)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists/{tasklist}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasklists.patch" call.
// Exactly one of *TaskList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *TaskList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TasklistsPatchCall) Do(opts ...googleapi.CallOption) (*TaskList, error) {
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
	ret := &TaskList{
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
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates the authenticated user's specified task list.
func (r *TasklistsService) Update(tasklistid string, tasklist *TaskList) *TasklistsUpdateCall {
	c := &TasklistsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	c.tasklist = tasklist
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasklistsUpdateCall) Fields(s ...googleapi.Field) *TasklistsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasklistsUpdateCall) Context(ctx context.Context) *TasklistsUpdateCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasklistsUpdateCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasklistsUpdateCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tasklist)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/@me/lists/{tasklist}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasklists.update" call.
// Exactly one of *TaskList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *TaskList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *TasklistsUpdateCall) Do(opts ...googleapi.CallOption) (*TaskList, error) {
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
	ret := &TaskList{
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
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Clear: Clears all completed tasks from the specified task list. The
// affected tasks will be marked as 'hidden' and no longer be returned
// by default when retrieving all tasks for a task list.
func (r *TasksService) Clear(tasklistid string) *TasksClearCall {
	c := &TasksClearCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasksClearCall) Fields(s ...googleapi.Field) *TasksClearCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasksClearCall) Context(ctx context.Context) *TasksClearCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasksClearCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasksClearCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/clear")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasks.clear" call.
func (c *TasksClearCall) Do(opts ...googleapi.CallOption) error {
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
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Delete: Deletes the specified task from the task list.
func (r *TasksService) Delete(tasklistid string, taskid string) *TasksDeleteCall {
	c := &TasksDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	c.taskid = taskid
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks/{task}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
		"task":     c.taskid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasks.delete" call.
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
	s            *Service
	tasklistid   string
	taskid       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Returns the specified task.
func (r *TasksService) Get(tasklistid string, taskid string) *TasksGetCall {
	c := &TasksGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	c.taskid = taskid
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks/{task}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
		"task":     c.taskid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasks.get" call.
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
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Insert: Creates a new task on the specified task list.
func (r *TasksService) Insert(tasklistid string, task *Task) *TasksInsertCall {
	c := &TasksInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	c.task = task
	return c
}

// Parent sets the optional parameter "parent": Parent task identifier.
// If the task is created at the top level, this parameter is omitted.
func (c *TasksInsertCall) Parent(parent string) *TasksInsertCall {
	c.urlParams_.Set("parent", parent)
	return c
}

// Previous sets the optional parameter "previous": Previous sibling
// task identifier. If the task is created at the first position among
// its siblings, this parameter is omitted.
func (c *TasksInsertCall) Previous(previous string) *TasksInsertCall {
	c.urlParams_.Set("previous", previous)
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasks.insert" call.
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
	s            *Service
	tasklistid   string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Returns all tasks in the specified task list.
func (r *TasksService) List(tasklistid string) *TasksListCall {
	c := &TasksListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	return c
}

// CompletedMax sets the optional parameter "completedMax": Upper bound
// for a task's completion date (as a RFC 3339 timestamp) to filter by.
// The default is not to filter by completion date.
func (c *TasksListCall) CompletedMax(completedMax string) *TasksListCall {
	c.urlParams_.Set("completedMax", completedMax)
	return c
}

// CompletedMin sets the optional parameter "completedMin": Lower bound
// for a task's completion date (as a RFC 3339 timestamp) to filter by.
// The default is not to filter by completion date.
func (c *TasksListCall) CompletedMin(completedMin string) *TasksListCall {
	c.urlParams_.Set("completedMin", completedMin)
	return c
}

// DueMax sets the optional parameter "dueMax": Upper bound for a task's
// due date (as a RFC 3339 timestamp) to filter by.  The default is not
// to filter by due date.
func (c *TasksListCall) DueMax(dueMax string) *TasksListCall {
	c.urlParams_.Set("dueMax", dueMax)
	return c
}

// DueMin sets the optional parameter "dueMin": Lower bound for a task's
// due date (as a RFC 3339 timestamp) to filter by.  The default is not
// to filter by due date.
func (c *TasksListCall) DueMin(dueMin string) *TasksListCall {
	c.urlParams_.Set("dueMin", dueMin)
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of task lists returned on one page.  The default is 100.
func (c *TasksListCall) MaxResults(maxResults int64) *TasksListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Token specifying
// the result page to return.
func (c *TasksListCall) PageToken(pageToken string) *TasksListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// ShowCompleted sets the optional parameter "showCompleted": Flag
// indicating whether completed tasks are returned in the result.  The
// default is True.
func (c *TasksListCall) ShowCompleted(showCompleted bool) *TasksListCall {
	c.urlParams_.Set("showCompleted", fmt.Sprint(showCompleted))
	return c
}

// ShowDeleted sets the optional parameter "showDeleted": Flag
// indicating whether deleted tasks are returned in the result.  The
// default is False.
func (c *TasksListCall) ShowDeleted(showDeleted bool) *TasksListCall {
	c.urlParams_.Set("showDeleted", fmt.Sprint(showDeleted))
	return c
}

// ShowHidden sets the optional parameter "showHidden": Flag indicating
// whether hidden tasks are returned in the result.  The default is
// False.
func (c *TasksListCall) ShowHidden(showHidden bool) *TasksListCall {
	c.urlParams_.Set("showHidden", fmt.Sprint(showHidden))
	return c
}

// UpdatedMin sets the optional parameter "updatedMin": Lower bound for
// a task's last modification time (as a RFC 3339 timestamp) to filter
// by.  The default is not to filter by last modification time.
func (c *TasksListCall) UpdatedMin(updatedMin string) *TasksListCall {
	c.urlParams_.Set("updatedMin", updatedMin)
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasks.list" call.
// Exactly one of *Tasks or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Tasks.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TasksListCall) Do(opts ...googleapi.CallOption) (*Tasks, error) {
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

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *TasksListCall) Pages(ctx context.Context, f func(*Tasks) error) error {
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

// method id "tasks.tasks.move":

type TasksMoveCall struct {
	s          *Service
	tasklistid string
	taskid     string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Move: Moves the specified task to another position in the task list.
// This can include putting it as a child task under a new parent and/or
// move it to a different position among its sibling tasks.
func (r *TasksService) Move(tasklistid string, taskid string) *TasksMoveCall {
	c := &TasksMoveCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	c.taskid = taskid
	return c
}

// Parent sets the optional parameter "parent": New parent task
// identifier. If the task is moved to the top level, this parameter is
// omitted.
func (c *TasksMoveCall) Parent(parent string) *TasksMoveCall {
	c.urlParams_.Set("parent", parent)
	return c
}

// Previous sets the optional parameter "previous": New previous sibling
// task identifier. If the task is moved to the first position among its
// siblings, this parameter is omitted.
func (c *TasksMoveCall) Previous(previous string) *TasksMoveCall {
	c.urlParams_.Set("previous", previous)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TasksMoveCall) Fields(s ...googleapi.Field) *TasksMoveCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TasksMoveCall) Context(ctx context.Context) *TasksMoveCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TasksMoveCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TasksMoveCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks/{task}/move")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
		"task":     c.taskid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasks.move" call.
// Exactly one of *Task or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Task.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *TasksMoveCall) Do(opts ...googleapi.CallOption) (*Task, error) {
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
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Patch: Updates the specified task. This method supports patch
// semantics.
func (r *TasksService) Patch(tasklistid string, taskid string, task *Task) *TasksPatchCall {
	c := &TasksPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	c.taskid = taskid
	c.task = task
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
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.task)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks/{task}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
		"task":     c.taskid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasks.patch" call.
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
	urlParams_ gensupport.URLParams
	ctx_       context.Context
	header_    http.Header
}

// Update: Updates the specified task.
func (r *TasksService) Update(tasklistid string, taskid string, task *Task) *TasksUpdateCall {
	c := &TasksUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tasklistid = tasklistid
	c.taskid = taskid
	c.task = task
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
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.task)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "lists/{tasklist}/tasks/{task}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"tasklist": c.tasklistid,
		"task":     c.taskid,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "tasks.tasks.update" call.
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
