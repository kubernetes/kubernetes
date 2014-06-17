// Package appstate provides access to the Google App State API.
//
// See https://developers.google.com/games/services/web/api/states
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/appstate/v1"
//   ...
//   appstateService, err := appstate.New(oauthHttpClient)
package appstate

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

const apiId = "appstate:v1"
const apiName = "appstate"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/appstate/v1/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data for this application
	AppstateScope = "https://www.googleapis.com/auth/appstate"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.States = NewStatesService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	States *StatesService
}

func NewStatesService(s *Service) *StatesService {
	rs := &StatesService{s: s}
	return rs
}

type StatesService struct {
	s *Service
}

type GetResponse struct {
	// CurrentStateVersion: The current app state version.
	CurrentStateVersion string `json:"currentStateVersion,omitempty"`

	// Data: The requested data.
	Data string `json:"data,omitempty"`

	// Kind: Uniquely identifies the type of this resource. Value is always
	// the fixed string appstate#getResponse.
	Kind string `json:"kind,omitempty"`

	// StateKey: The key for the data.
	StateKey int64 `json:"stateKey,omitempty"`
}

type ListResponse struct {
	// Items: The app state data.
	Items []*GetResponse `json:"items,omitempty"`

	// Kind: Uniquely identifies the type of this resource. Value is always
	// the fixed string appstate#listResponse.
	Kind string `json:"kind,omitempty"`

	// MaximumKeyCount: The maximum number of keys allowed for this user.
	MaximumKeyCount int64 `json:"maximumKeyCount,omitempty"`
}

type UpdateRequest struct {
	// Data: The new app state data that your application is trying to
	// update with.
	Data string `json:"data,omitempty"`

	// Kind: Uniquely identifies the type of this resource. Value is always
	// the fixed string appstate#updateRequest.
	Kind string `json:"kind,omitempty"`
}

type WriteResult struct {
	// CurrentStateVersion: The version of the data for this key on the
	// server.
	CurrentStateVersion string `json:"currentStateVersion,omitempty"`

	// Kind: Uniquely identifies the type of this resource. Value is always
	// the fixed string appstate#writeResult.
	Kind string `json:"kind,omitempty"`

	// StateKey: The written key.
	StateKey int64 `json:"stateKey,omitempty"`
}

// method id "appstate.states.clear":

type StatesClearCall struct {
	s        *Service
	stateKey int64
	opt_     map[string]interface{}
}

// Clear: Clears (sets to empty) the data for the passed key if and only
// if the passed version matches the currently stored version. This
// method results in a conflict error on version mismatch.
func (r *StatesService) Clear(stateKey int64) *StatesClearCall {
	c := &StatesClearCall{s: r.s, opt_: make(map[string]interface{})}
	c.stateKey = stateKey
	return c
}

// CurrentDataVersion sets the optional parameter "currentDataVersion":
// The version of the data to be cleared. Version strings are returned
// by the server.
func (c *StatesClearCall) CurrentDataVersion(currentDataVersion string) *StatesClearCall {
	c.opt_["currentDataVersion"] = currentDataVersion
	return c
}

func (c *StatesClearCall) Do() (*WriteResult, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["currentDataVersion"]; ok {
		params.Set("currentDataVersion", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "states/{stateKey}/clear")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{stateKey}", strconv.FormatInt(c.stateKey, 10), 1)
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
	ret := new(WriteResult)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Clears (sets to empty) the data for the passed key if and only if the passed version matches the currently stored version. This method results in a conflict error on version mismatch.",
	//   "httpMethod": "POST",
	//   "id": "appstate.states.clear",
	//   "parameterOrder": [
	//     "stateKey"
	//   ],
	//   "parameters": {
	//     "currentDataVersion": {
	//       "description": "The version of the data to be cleared. Version strings are returned by the server.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "stateKey": {
	//       "description": "The key for the data to be retrieved.",
	//       "format": "int32",
	//       "location": "path",
	//       "maximum": "3",
	//       "minimum": "0",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "states/{stateKey}/clear",
	//   "response": {
	//     "$ref": "WriteResult"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/appstate"
	//   ]
	// }

}

// method id "appstate.states.delete":

type StatesDeleteCall struct {
	s        *Service
	stateKey int64
	opt_     map[string]interface{}
}

// Delete: Deletes a key and the data associated with it. The key is
// removed and no longer counts against the key quota. Note that since
// this method is not safe in the face of concurrent modifications, it
// should only be used for development and testing purposes. Invoking
// this method in shipping code can result in data loss and data
// corruption.
func (r *StatesService) Delete(stateKey int64) *StatesDeleteCall {
	c := &StatesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.stateKey = stateKey
	return c
}

func (c *StatesDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "states/{stateKey}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{stateKey}", strconv.FormatInt(c.stateKey, 10), 1)
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
	//   "description": "Deletes a key and the data associated with it. The key is removed and no longer counts against the key quota. Note that since this method is not safe in the face of concurrent modifications, it should only be used for development and testing purposes. Invoking this method in shipping code can result in data loss and data corruption.",
	//   "httpMethod": "DELETE",
	//   "id": "appstate.states.delete",
	//   "parameterOrder": [
	//     "stateKey"
	//   ],
	//   "parameters": {
	//     "stateKey": {
	//       "description": "The key for the data to be retrieved.",
	//       "format": "int32",
	//       "location": "path",
	//       "maximum": "3",
	//       "minimum": "0",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "states/{stateKey}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/appstate"
	//   ]
	// }

}

// method id "appstate.states.get":

type StatesGetCall struct {
	s        *Service
	stateKey int64
	opt_     map[string]interface{}
}

// Get: Retrieves the data corresponding to the passed key. If the key
// does not exist on the server, an HTTP 404 will be returned.
func (r *StatesService) Get(stateKey int64) *StatesGetCall {
	c := &StatesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.stateKey = stateKey
	return c
}

func (c *StatesGetCall) Do() (*GetResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "states/{stateKey}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{stateKey}", strconv.FormatInt(c.stateKey, 10), 1)
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
	ret := new(GetResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the data corresponding to the passed key. If the key does not exist on the server, an HTTP 404 will be returned.",
	//   "httpMethod": "GET",
	//   "id": "appstate.states.get",
	//   "parameterOrder": [
	//     "stateKey"
	//   ],
	//   "parameters": {
	//     "stateKey": {
	//       "description": "The key for the data to be retrieved.",
	//       "format": "int32",
	//       "location": "path",
	//       "maximum": "3",
	//       "minimum": "0",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "states/{stateKey}",
	//   "response": {
	//     "$ref": "GetResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/appstate"
	//   ]
	// }

}

// method id "appstate.states.list":

type StatesListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Lists all the states keys, and optionally the state data.
func (r *StatesService) List() *StatesListCall {
	c := &StatesListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// IncludeData sets the optional parameter "includeData": Whether to
// include the full data in addition to the version number
func (c *StatesListCall) IncludeData(includeData bool) *StatesListCall {
	c.opt_["includeData"] = includeData
	return c
}

func (c *StatesListCall) Do() (*ListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["includeData"]; ok {
		params.Set("includeData", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "states")
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
	ret := new(ListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all the states keys, and optionally the state data.",
	//   "httpMethod": "GET",
	//   "id": "appstate.states.list",
	//   "parameters": {
	//     "includeData": {
	//       "default": "false",
	//       "description": "Whether to include the full data in addition to the version number",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "states",
	//   "response": {
	//     "$ref": "ListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/appstate"
	//   ]
	// }

}

// method id "appstate.states.update":

type StatesUpdateCall struct {
	s             *Service
	stateKey      int64
	updaterequest *UpdateRequest
	opt_          map[string]interface{}
}

// Update: Update the data associated with the input key if and only if
// the passed version matches the currently stored version. This method
// is safe in the face of concurrent writes. Maximum per-key size is
// 128KB.
func (r *StatesService) Update(stateKey int64, updaterequest *UpdateRequest) *StatesUpdateCall {
	c := &StatesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.stateKey = stateKey
	c.updaterequest = updaterequest
	return c
}

// CurrentStateVersion sets the optional parameter
// "currentStateVersion": The version of the app state your application
// is attempting to update. If this does not match the current version,
// this method will return a conflict error. If there is no data stored
// on the server for this key, the update will succeed irrespective of
// the value of this parameter.
func (c *StatesUpdateCall) CurrentStateVersion(currentStateVersion string) *StatesUpdateCall {
	c.opt_["currentStateVersion"] = currentStateVersion
	return c
}

func (c *StatesUpdateCall) Do() (*WriteResult, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.updaterequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["currentStateVersion"]; ok {
		params.Set("currentStateVersion", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "states/{stateKey}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{stateKey}", strconv.FormatInt(c.stateKey, 10), 1)
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
	ret := new(WriteResult)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update the data associated with the input key if and only if the passed version matches the currently stored version. This method is safe in the face of concurrent writes. Maximum per-key size is 128KB.",
	//   "httpMethod": "PUT",
	//   "id": "appstate.states.update",
	//   "parameterOrder": [
	//     "stateKey"
	//   ],
	//   "parameters": {
	//     "currentStateVersion": {
	//       "description": "The version of the app state your application is attempting to update. If this does not match the current version, this method will return a conflict error. If there is no data stored on the server for this key, the update will succeed irrespective of the value of this parameter.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "stateKey": {
	//       "description": "The key for the data to be retrieved.",
	//       "format": "int32",
	//       "location": "path",
	//       "maximum": "3",
	//       "minimum": "0",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "states/{stateKey}",
	//   "request": {
	//     "$ref": "UpdateRequest"
	//   },
	//   "response": {
	//     "$ref": "WriteResult"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/appstate"
	//   ]
	// }

}
