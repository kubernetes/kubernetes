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

package client

import (
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// ClientInterface holds the methods for clients of Kubenetes, an interface to allow mock testing
type ClientInterface interface {
	ListPods(selector labels.Selector) (api.PodList, error)
	GetPod(name string) (api.Pod, error)
	DeletePod(name string) error
	CreatePod(api.Pod) (api.Pod, error)
	UpdatePod(api.Pod) (api.Pod, error)

	GetReplicationController(name string) (api.ReplicationController, error)
	CreateReplicationController(api.ReplicationController) (api.ReplicationController, error)
	UpdateReplicationController(api.ReplicationController) (api.ReplicationController, error)
	DeleteReplicationController(string) error

	GetService(name string) (api.Service, error)
	CreateService(api.Service) (api.Service, error)
	UpdateService(api.Service) (api.Service, error)
	DeleteService(string) error
}

// StatusErr might get returned from an api call if your request is still being processed
// and hence the expected return data is not available yet.
type StatusErr struct {
	Status api.Status
}

func (s *StatusErr) Error() string {
	return fmt.Sprintf("Status: %v (%#v)", s.Status.Status, s)
}

// AuthInfo is used to store authorization information
type AuthInfo struct {
	User     string
	Password string
}

// Client is the actual implementation of a Kubernetes client.
// Host is the http://... base for the URL
type Client struct {
	host       string
	auth       *AuthInfo
	httpClient *http.Client
}

// Create a new client object.
func New(host string, auth *AuthInfo) *Client {
	return &Client{
		auth: auth,
		host: host,
		httpClient: &http.Client{
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{
					InsecureSkipVerify: true,
				},
			},
		},
	}
}

// Execute a request, adds authentication (if auth != nil), and HTTPS cert ignoring.
func (c *Client) doRequest(request *http.Request) ([]byte, error) {
	if c.auth != nil {
		request.SetBasicAuth(c.auth.User, c.auth.Password)
	}
	response, err := c.httpClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return body, err
	}
	if response.StatusCode < http.StatusOK || response.StatusCode > http.StatusPartialContent {
		return nil, fmt.Errorf("request [%#v] failed (%d) %s: %s", request, response.StatusCode, response.Status, string(body))
	}
	if response.StatusCode == http.StatusAccepted {
		var status api.Status
		if err := api.DecodeInto(body, &status); err == nil {
			if status.Status == api.StatusSuccess {
				return body, nil
			} else {
				return nil, &StatusErr{status}
			}
		}
		// Sometimes the server returns 202 even though it completely handled the request.
	}
	return body, err
}

// Underlying base implementation of performing a request.
// method is the HTTP method (e.g. "GET")
// path is the path on the host to hit
// requestBody is the body of the request. Can be nil.
// target the interface to marshal the JSON response into.  Can be nil.
func (c *Client) rawRequest(method, path string, requestBody io.Reader, target interface{}) ([]byte, error) {
	request, err := http.NewRequest(method, c.makeURL(path), requestBody)
	if err != nil {
		return nil, err
	}
	body, err := c.doRequest(request)
	if err != nil {
		return body, err
	}
	if target != nil {
		err = api.DecodeInto(body, target)
	}
	if err != nil {
		log.Printf("Failed to parse: %s\n", string(body))
		// FIXME: no need to return err here?
	}
	return body, err
}

func (client *Client) makeURL(path string) string {
	return client.host + "/api/v1beta1/" + path
}

// ListPods takes a selector, and returns the list of pods that match that selector
func (client *Client) ListPods(selector labels.Selector) (result api.PodList, err error) {
	err = client.Get().Path("pods").Selector(selector).Do().Into(&result)
	return
}

// GetPod takes the name of the pod, and returns the corresponding Pod object, and an error if it occurs
func (client *Client) GetPod(name string) (result api.Pod, err error) {
	err = client.Get().Path("pods").Path(name).Do().Into(&result)
	return
}

// DeletePod takes the name of the pod, and returns an error if one occurs
func (client *Client) DeletePod(name string) error {
	return client.Delete().Path("pods").Path(name).Do().Error()
}

// CreatePod takes the representation of a pod.  Returns the server's representation of the pod, and an error, if it occurs
func (client *Client) CreatePod(pod api.Pod) (result api.Pod, err error) {
	err = client.Post().Path("pods").Body(pod).Do().Into(&result)
	return
}

// UpdatePod takes the representation of a pod to update.  Returns the server's representation of the pod, and an error, if it occurs
func (client *Client) UpdatePod(pod api.Pod) (result api.Pod, err error) {
	err = client.Put().Path("pods").Path(pod.ID).Body(pod).Do().Into(&result)
	return
}

// GetReplicationController returns information about a particular replication controller
func (client *Client) GetReplicationController(name string) (result api.ReplicationController, err error) {
	err = client.Get().Path("replicationControllers").Path(name).Do().Into(&result)
	return
}

// CreateReplicationController creates a new replication controller
func (client *Client) CreateReplicationController(controller api.ReplicationController) (result api.ReplicationController, err error) {
	err = client.Post().Path("replicationControllers").Body(controller).Do().Into(&result)
	return
}

// UpdateReplicationController updates an existing replication controller
func (client *Client) UpdateReplicationController(controller api.ReplicationController) (result api.ReplicationController, err error) {
	err = client.Put().Path("replicationControllers").Path(controller.ID).Body(controller).Do().Into(&result)
	return
}

func (client *Client) DeleteReplicationController(name string) error {
	return client.Delete().Path("replicationControllers").Path(name).Do().Error()
}

// GetReplicationController returns information about a particular replication controller
func (client *Client) GetService(name string) (result api.Service, err error) {
	err = client.Get().Path("services").Path(name).Do().Into(&result)
	return
}

// CreateReplicationController creates a new replication controller
func (client *Client) CreateService(svc api.Service) (result api.Service, err error) {
	err = client.Post().Path("services").Body(svc).Do().Into(&result)
	return
}

// UpdateReplicationController updates an existing replication controller
func (client *Client) UpdateService(svc api.Service) (result api.Service, err error) {
	err = client.Put().Path("services").Path(svc.ID).Body(svc).Do().Into(&result)
	return
}

func (client *Client) DeleteService(name string) error {
	return client.Delete().Path("services").Path(name).Do().Error()
}
