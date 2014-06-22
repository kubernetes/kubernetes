/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cloudcfg

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"gopkg.in/v1/yaml"
)

func promptForString(field string) string {
	fmt.Printf("Please enter %s: ", field)
	var result string
	fmt.Scan(&result)
	return result
}

// Parse an AuthInfo object from a file path. Prompt user and create file if it doesn't exist.
func LoadAuthInfo(path string) (*client.AuthInfo, error) {
	var auth client.AuthInfo
	if _, err := os.Stat(path); os.IsNotExist(err) {
		auth.User = promptForString("Username")
		auth.Password = promptForString("Password")
		data, err := json.Marshal(auth)
		if err != nil {
			return &auth, err
		}
		err = ioutil.WriteFile(path, data, 0600)
		return &auth, err
	}
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(data, &auth)
	if err != nil {
		return nil, err
	}
	return &auth, err
}

// Perform a rolling update of a collection of pods.
// 'name' points to a replication controller.
// 'client' is used for updating pods.
// 'updatePeriod' is the time between pod updates.
func Update(name string, client client.ClientInterface, updatePeriod time.Duration) error {
	controller, err := client.GetReplicationController(name)
	if err != nil {
		return err
	}
	labels := controller.DesiredState.ReplicaSelector

	podList, err := client.ListPods(labels)
	if err != nil {
		return err
	}
	for _, pod := range podList.Items {
		_, err = client.UpdatePod(pod)
		if err != nil {
			return err
		}
		time.Sleep(updatePeriod)
	}
	return nil
}

// Server contains info locating a kubernetes api server.
// Example usage:
// auth, err := LoadAuth(filename)
// s := New(url, auth)
// resp, err := s.Verb("GET").
//	Path("api/v1beta1").
//	Path("pods").
//	Selector("area=staging").
//	Timeout(10*time.Second).
//	Do()
// list, ok := resp.(api.PodList)
type Server struct {
	auth   *client.AuthInfo
	rawUrl string
}

// Create a new server object.
func New(serverUrl string, auth *client.AuthInfo) *Server {
	return &Server{
		auth:   auth,
		rawUrl: serverUrl,
	}
}

// Begin a request with a verb (GET, POST, PUT, DELETE)
func (s *Server) Verb(verb string) *Request {
	return &Request{
		verb: verb,
		s:    s,
		path: "/",
	}
}

// Request allows for building up a request to a server in a chained fashion.
type Request struct {
	s        *Server
	err      error
	verb     string
	path     string
	body     interface{}
	selector labels.Selector
	timeout  time.Duration
}

// Append an item to the request path. You must call Path at least once.
func (r *Request) Path(item string) *Request {
	if r.err != nil {
		return r
	}
	r.path = path.Join(r.path, item)
	return r
}

// Use the given item as a resource label selector. Optional.
func (r *Request) Selector(item string) *Request {
	if r.err != nil {
		return r
	}
	r.selector, r.err = labels.ParseSelector(item)
	return r
}

// Use the given duration as a timeout. Optional.
func (r *Request) Timeout(d time.Duration) *Request {
	if r.err != nil {
		return r
	}
	r.timeout = d
	return r
}

// Use obj as the body of the request. Optional.
// If obj is a string, try to read a file of that name.
// If obj is a []byte, send it directly.
// Otherwise, assume obj is an api type and marshall it correctly.
func (r *Request) Body(obj interface{}) *Request {
	if r.err != nil {
		return r
	}
	r.body = obj
	return r
}

// Format and xecute the request. Returns the API object received, or an error.
func (r *Request) Do() (interface{}, error) {
	if r.err != nil {
		return nil, r.err
	}
	finalUrl := r.s.rawUrl + r.path
	query := url.Values{}
	if r.selector != nil {
		query.Add("labels", r.selector.String())
	}
	if r.timeout != 0 {
		query.Add("timeout", r.timeout.String())
	}
	finalUrl += "?" + query.Encode()
	var body io.Reader
	if r.body != nil {
		switch t := r.body.(type) {
		case string:
			data, err := ioutil.ReadFile(t)
			if err != nil {
				return nil, err
			}
			body = bytes.NewBuffer(data)
		case []byte:
			body = bytes.NewBuffer(t)
		default:
			data, err := api.Encode(r.body)
			if err != nil {
				return nil, err
			}
			body = bytes.NewBuffer(data)
		}
	}
	req, err := http.NewRequest(r.verb, finalUrl, body)
	if err != nil {
		return nil, err
	}
	str, err := DoRequest(req, r.s.auth)
	if err != nil {
		return nil, err
	}
	return api.Decode([]byte(str))
}

// RequestWithBody is a helper method that creates an HTTP request with the specified url, method
// and a body read from 'configFile'
// FIXME: need to be public API?
func RequestWithBody(configFile, url, method string) (*http.Request, error) {
	if len(configFile) == 0 {
		return nil, fmt.Errorf("empty config file.")
	}
	data, err := ioutil.ReadFile(configFile)
	if err != nil {
		return nil, err
	}
	return RequestWithBodyData(data, url, method)
}

// RequestWithBodyData is a helper method that creates an HTTP request with the specified url, method
// and body data
func RequestWithBodyData(data []byte, url, method string) (*http.Request, error) {
	request, err := http.NewRequest(method, url, bytes.NewBuffer(data))
	request.ContentLength = int64(len(data))
	return request, err
}

// Execute a request, adds authentication (if auth != nil), and HTTPS cert ignoring.
func DoRequest(request *http.Request, auth *client.AuthInfo) ([]byte, error) {
	if auth != nil {
		request.SetBasicAuth(auth.User, auth.Password)
	}
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	client := &http.Client{Transport: tr}
	response, err := client.Do(request)
	if err != nil {
		return []byte{}, err
	}
	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	return body, err
}

// StopController stops a controller named 'name' by setting replicas to zero
func StopController(name string, client client.ClientInterface) error {
	controller, err := client.GetReplicationController(name)
	if err != nil {
		return err
	}
	controller.DesiredState.Replicas = 0
	controllerOut, err := client.UpdateReplicationController(controller)
	if err != nil {
		return err
	}
	data, err := yaml.Marshal(controllerOut)
	if err != nil {
		return err
	}
	fmt.Print(string(data))
	return nil
}

// ResizeController resizes a controller named 'name' by setting replicas to 'replicas'
func ResizeController(name string, replicas int, client client.ClientInterface) error {
	controller, err := client.GetReplicationController(name)
	if err != nil {
		return err
	}
	controller.DesiredState.Replicas = replicas
	controllerOut, err := client.UpdateReplicationController(controller)
	if err != nil {
		return err
	}
	data, err := yaml.Marshal(controllerOut)
	if err != nil {
		return err
	}
	fmt.Print(string(data))
	return nil
}

func makePorts(spec string) []api.Port {
	parts := strings.Split(spec, ",")
	var result []api.Port
	for _, part := range parts {
		pieces := strings.Split(part, ":")
		if len(pieces) != 2 {
			log.Printf("Bad port spec: %s", part)
			continue
		}
		host, err := strconv.Atoi(pieces[0])
		if err != nil {
			log.Printf("Host part is not integer: %s %v", pieces[0], err)
			continue
		}
		container, err := strconv.Atoi(pieces[1])
		if err != nil {
			log.Printf("Container part is not integer: %s %v", pieces[1], err)
			continue
		}
		result = append(result, api.Port{ContainerPort: container, HostPort: host})
	}
	return result
}

// RunController creates a new replication controller named 'name' which creates 'replicas' pods running 'image'
func RunController(image, name string, replicas int, client client.ClientInterface, portSpec string, servicePort int) error {
	controller := api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: name,
		},
		DesiredState: api.ReplicationControllerState{
			Replicas: replicas,
			ReplicaSelector: map[string]string{
				"name": name,
			},
			PodTemplate: api.PodTemplate{
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Containers: []api.Container{
							{
								Image: image,
								Ports: makePorts(portSpec),
							},
						},
					},
				},
				Labels: map[string]string{
					"name": name,
				},
			},
		},
		Labels: map[string]string{
			"name": name,
		},
	}

	controllerOut, err := client.CreateReplicationController(controller)
	if err != nil {
		return err
	}
	data, err := yaml.Marshal(controllerOut)
	if err != nil {
		return err
	}
	fmt.Print(string(data))

	if servicePort > 0 {
		svc, err := createService(name, servicePort, client)
		if err != nil {
			return err
		}
		data, err = yaml.Marshal(svc)
		if err != nil {
			return err
		}
		fmt.Printf(string(data))
	}
	return nil
}

func createService(name string, port int, client client.ClientInterface) (api.Service, error) {
	svc := api.Service{
		JSONBase: api.JSONBase{ID: name},
		Port:     port,
		Labels: map[string]string{
			"name": name,
		},
		Selector: map[string]string{
			"name": name,
		},
	}
	svc, err := client.CreateService(svc)
	return svc, err
}

// DeleteController deletes a replication controller named 'name', requires that the controller
// already be stopped
func DeleteController(name string, client client.ClientInterface) error {
	controller, err := client.GetReplicationController(name)
	if err != nil {
		return err
	}
	if controller.DesiredState.Replicas != 0 {
		return fmt.Errorf("controller has non-zero replicas (%d), please stop it first", controller.DesiredState.Replicas)
	}
	return client.DeleteReplicationController(name)
}
