/*
Copyright 2017 The Kubernetes Authors.

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

package proxy

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/pkg/api/v1"
	"net/http"
)

type serviceListerMock struct {
	services []*v1.Service
	err      error
}

func (s *serviceListerMock) List(selector labels.Selector) (ret []*v1.Service, err error) {
	return s.services, err
}

func (s *serviceListerMock) Services(namespace string) listersv1.ServiceNamespaceLister {
	return nil
}

func (s *serviceListerMock) GetPodServices(pod *v1.Pod) ([]*v1.Service, error) {
	return nil, nil
}

type endpointsListerMock struct {
	endpoints []*v1.Endpoints
	err       error
}

func (e *endpointsListerMock) List(selector labels.Selector) (ret []*v1.Endpoints, err error) {
	return e.endpoints, e.err
}

func (e *endpointsListerMock) Endpoints(namespace string) listersv1.EndpointsNamespaceLister {
	return endpointsNamespaceListMock{
		endpoints: e.endpoints,
		err:       e.err,
	}
}

type endpointsNamespaceListMock struct {
	endpoints []*v1.Endpoints
	err       error
}

func (e endpointsNamespaceListMock) List(selector labels.Selector) (ret []*v1.Endpoints, err error) {
	return e.endpoints, e.err
}

func (e endpointsNamespaceListMock) Get(name string) (*v1.Endpoints, error) {
	if len(e.endpoints) == 0 {
		return nil, e.err
	}
	return e.endpoints[0], e.err
}

func TestNoEndpointNoPort(t *testing.T) {
	services := &serviceListerMock{}
	endpoints := &endpointsListerMock{err: errors.NewNotFound(v1.Resource("endpoints"), "dummy-svc")}
	url, err := ResolveEndpoint(services, endpoints, "dummy-ns", "dummy-svc")
	if url != nil {
		t.Error("Should not have gotten back an URL")
	}
	if err == nil {
		t.Error("Should have gotten an error")
	}
	se, ok := err.(*errors.StatusError)
	if !ok {
		t.Error("Should have gotten a status error not %T", err)
	}
	if se.ErrStatus.Code != http.StatusNotFound {
		t.Error("Should have gotten a http 404 not %d", se.ErrStatus.Code)
	}
}

func TestOneEndpointNoPort(t *testing.T) {
	services := &serviceListerMock{}
	address := v1.EndpointAddress{Hostname: "dummy-host", IP: "127.0.0.1"}
	addresses := []v1.EndpointAddress{address}
	port := v1.EndpointPort{Port: 443}
	ports := []v1.EndpointPort{port}
	endpoint := v1.EndpointSubset{Addresses: addresses, Ports: ports}
	subsets := []v1.EndpointSubset{endpoint}
	one := &v1.Endpoints{Subsets: subsets}
	slice := []*v1.Endpoints{one}
	endpoints := &endpointsListerMock{endpoints: slice}
	url, err := ResolveEndpoint(services, endpoints, "dummy-ns", "dummy-svc")
	if err != nil {
		t.Errorf("Should not have gotten error %v", err)
	}
	if url == nil {
		t.Error("Should not have gotten back an URL")
	}
	if url.Host != "127.0.0.1:443" {
		t.Error("Should have gotten back a host of dummy-host not %s", url.Host)
	}

}
