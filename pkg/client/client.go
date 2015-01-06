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
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
)

// Interface holds the methods for clients of Kubernetes,
// an interface to allow mock testing.
type Interface interface {
	PodsNamespacer
	ReplicationControllersNamespacer
	ServicesNamespacer
	EndpointsNamespacer
	VersionInterface
	NodesInterface
	EventNamespacer
	BoundPodsNamespacer
}

func (c *Client) ReplicationControllers(namespace string) ReplicationControllerInterface {
	return newReplicationControllers(c, namespace)
}

func (c *Client) Nodes() NodeInterface {
	return newNodes(c, c.preV1Beta3)
}

func (c *Client) BoundPods() BoundPodsInterface {
	return newBoundPods(c)
}

func (c *Client) Events(namespace string) EventInterface {
	return newEvents(c, namespace)
}

func (c *Client) Endpoints(namespace string) EndpointsInterface {
	return newEndpoints(c, namespace)
}

func (c *Client) Pods(namespace string) PodInterface {
	return newPods(c, namespace)
}

func (c *Client) Services(namespace string) ServiceInterface {
	return newServices(c, namespace)
}

// VersionInterface has a method to retrieve the server version.
type VersionInterface interface {
	ServerVersion() (*version.Info, error)
	ServerAPIVersions() (*api.APIVersions, error)
}

// APIStatus is exposed by errors that can be converted to an api.Status object
// for finer grained details.
type APIStatus interface {
	Status() api.Status
}

// Client is the implementation of a Kubernetes client.
type Client struct {
	*RESTClient

	// preV1Beta3 is true for v1beta1 and v1beta2
	preV1Beta3 bool
}

// ServerVersion retrieves and parses the server's version.
func (c *Client) ServerVersion() (*version.Info, error) {
	body, err := c.Get().AbsPath("/version").Do().Raw()
	if err != nil {
		return nil, err
	}
	var info version.Info
	err = json.Unmarshal(body, &info)
	if err != nil {
		return nil, fmt.Errorf("got '%s': %v", string(body), err)
	}
	return &info, nil
}

// ServerAPIVersions retrieves and parses the list of API versions the server supports.
func (c *Client) ServerAPIVersions() (*api.APIVersions, error) {
	body, err := c.Get().AbsPath("/api").Do().Raw()
	if err != nil {
		return nil, err
	}
	var v api.APIVersions
	err = json.Unmarshal(body, &v)
	if err != nil {
		return nil, fmt.Errorf("got '%s': %v", string(body), err)
	}
	return &v, nil
}

// IsTimeout tests if this is a timeout error in the underlying transport.
// This is unbelievably ugly.
// See: http://stackoverflow.com/questions/23494950/specifically-check-for-timeout-error for details
func IsTimeout(err error) bool {
	if err == nil {
		return false
	}
	switch err := err.(type) {
	case *url.Error:
		if err, ok := err.Err.(net.Error); ok {
			return err.Timeout()
		}
	case net.Error:
		return err.Timeout()
	}

	if strings.Contains(err.Error(), "use of closed network connection") {
		return true
	}
	return false
}
