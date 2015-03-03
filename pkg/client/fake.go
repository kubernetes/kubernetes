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
	"net/http"
	"net/url"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type FakeAction struct {
	Action string
	Value  interface{}
}

// Fake implements Interface. Meant to be embedded into a struct to get a default
// implementation. This makes faking out just the method you want to test easier.
type Fake struct {
	Actions             []FakeAction
	PodsList            api.PodList
	CtrlList            api.ReplicationControllerList
	Ctrl                api.ReplicationController
	ServiceList         api.ServiceList
	EndpointsList       api.EndpointsList
	MinionsList         api.NodeList
	EventsList          api.EventList
	LimitRangesList     api.LimitRangeList
	ResourceQuotaStatus api.ResourceQuota
	ResourceQuotasList  api.ResourceQuotaList
	NamespacesList      api.NamespaceList
	SecretList          api.SecretList
	Secret              api.Secret
	Err                 error
	Watch               watch.Interface
	PersistentVolume          api.PersistentVolume
	PersistentVolumesList     api.PersistentVolumeList
	PersistentVolumeClaim     api.PersistentVolumeClaim
	PersistentVolumeClaimList api.PersistentVolumeClaimList	
}

func (c *Fake) LimitRanges(namespace string) LimitRangeInterface {
	return &FakeLimitRanges{Fake: c, Namespace: namespace}
}

func (c *Fake) ResourceQuotas(namespace string) ResourceQuotaInterface {
	return &FakeResourceQuotas{Fake: c, Namespace: namespace}
}

func (c *Fake) ReplicationControllers(namespace string) ReplicationControllerInterface {
	return &FakeReplicationControllers{Fake: c, Namespace: namespace}
}

func (c *Fake) Nodes() NodeInterface {
	return &FakeNodes{Fake: c}
}

func (c *Fake) Events(namespace string) EventInterface {
	return &FakeEvents{Fake: c}
}

func (c *Fake) Endpoints(namespace string) EndpointsInterface {
	return &FakeEndpoints{Fake: c, Namespace: namespace}
}

func (c *Fake) Pods(namespace string) PodInterface {
	return &FakePods{Fake: c, Namespace: namespace}
}

func (c *Fake) PersistentVolumes() PersistentVolumeInterface {
	return &FakePersistentVolumes{Fake: c}
}

func (c *Fake) PersistentVolumeClaims(namespace string) PersistentVolumeClaimInterface {
	return &FakePersistentVolumeClaims{Fake: c, Namespace: namespace}
}

func (c *Fake) Services(namespace string) ServiceInterface {
	return &FakeServices{Fake: c, Namespace: namespace}
}

func (c *Fake) Secrets(namespace string) SecretsInterface {
	return &FakeSecrets{Fake: c, Namespace: namespace}
}

func (c *Fake) Namespaces() NamespaceInterface {
	return &FakeNamespaces{Fake: c}
}

func (c *Fake) ServerVersion() (*version.Info, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "get-version", Value: nil})
	versionInfo := version.Get()
	return &versionInfo, nil
}

func (c *Fake) ServerAPIVersions() (*api.APIVersions, error) {
	c.Actions = append(c.Actions, FakeAction{Action: "get-apiversions", Value: nil})
	return &api.APIVersions{Versions: []string{"v1beta1", "v1beta2"}}, nil
}

type HTTPClientFunc func(*http.Request) (*http.Response, error)

func (f HTTPClientFunc) Do(req *http.Request) (*http.Response, error) {
	return f(req)
}

// FakeRESTClient provides a fake RESTClient interface.
type FakeRESTClient struct {
	Client HTTPClient
	Codec  runtime.Codec
	Legacy bool
	Req    *http.Request
	Resp   *http.Response
	Err    error
}

func (c *FakeRESTClient) Get() *Request {
	return NewRequest(c, "GET", &url.URL{Host: "localhost"}, testapi.Version(), c.Codec, c.Legacy, c.Legacy)
}

func (c *FakeRESTClient) Put() *Request {
	return NewRequest(c, "PUT", &url.URL{Host: "localhost"}, testapi.Version(), c.Codec, c.Legacy, c.Legacy)
}

func (c *FakeRESTClient) Post() *Request {
	return NewRequest(c, "POST", &url.URL{Host: "localhost"}, testapi.Version(), c.Codec, c.Legacy, c.Legacy)
}

func (c *FakeRESTClient) Delete() *Request {
	return NewRequest(c, "DELETE", &url.URL{Host: "localhost"}, testapi.Version(), c.Codec, c.Legacy, c.Legacy)
}

func (c *FakeRESTClient) Do(req *http.Request) (*http.Response, error) {
	c.Req = req
	if c.Client != HTTPClient(nil) {
		return c.Client.Do(req)
	}
	return c.Resp, c.Err
}
