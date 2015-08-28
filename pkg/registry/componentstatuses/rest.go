/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package componentstatuses

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/master/ports" //TODO(karlkfi): fix import cycle?
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/runtime"
	db_storage "k8s.io/kubernetes/pkg/storage"
	etcd_storage "k8s.io/kubernetes/pkg/storage/etcd"

	"github.com/golang/glog" //TODO: shouldn't be logging this deep
)

type namedServer struct {
	name   string
	server apiserver.Server
}

type REST struct {
	databaseStorage db_storage.Interface
	rt              http.RoundTripper
}

// NewStorage returns a new ComponentStatuses ReadableStorage.
func NewStorage(databaseStorage db_storage.Interface, rt http.RoundTripper) rest.ReadableStorage {
	return &REST{
		databaseStorage: databaseStorage,
		rt:              rt,
	}
}

func (rs *REST) New() runtime.Object {
	return &api.ComponentStatuses{}
}

func (rs *REST) NewList() runtime.Object {
	return &api.ComponentStatusesList{}
}

// Returns the list of component status. Note that the label and field are both ignored.
// Note that this call doesn't support labels or selectors.
func (rs *REST) List(ctx api.Context, label labels.Selector, field fields.Selector) (runtime.Object, error) {
	servers := rs.getServersToValidate()

	// TODO: This should be parallelized.
	reply := []api.ComponentStatuses{}
	for _, namedServer := range servers {
		status := rs.getComponentStatuses(namedServer)
		reply = append(reply, *status)
	}
	return &api.ComponentStatusesList{Items: reply}, nil
}

func (rs *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	servers := rs.getServersToValidate()

	for _, namedServer := range servers {
		if namedServer.name == name {
			return rs.getComponentStatuses(namedServer), nil
		}
	}

	return nil, fmt.Errorf("Component not found: %s", name)
}

func ToConditionStatus(s probe.Result) api.ConditionStatus {
	switch s {
	case probe.Success:
		return api.ConditionTrue
	case probe.Failure:
		return api.ConditionFalse
	default:
		return api.ConditionUnknown
	}
}

func (rs *REST) getComponentStatuses(ns namedServer) *api.ComponentStatuses {
	transport := rs.rt
	status, msg, err := ns.server.DoServerCheck(transport)
	var errorMsg string
	if err != nil {
		errorMsg = err.Error()
	} else {
		errorMsg = "nil"
	}

	c := &api.ComponentStatusesCondition{
		Type:    api.ComponentStatusesHealthy,
		Status:  ToConditionStatus(status),
		Message: msg,
		Error:   errorMsg,
	}

	retVal := &api.ComponentStatuses{
		Conditions: []api.ComponentStatusesCondition{*c},
	}
	retVal.Name = ns.name

	return retVal
}

func (rs *REST) getServersToValidate() []namedServer {
	serversToValidate := []namedServer{
		{
			name:   "controller-manager",
			server: apiserver.Server{Addr: "127.0.0.1", Port: ports.ControllerManagerPort, Path: "/healthz"},
		},
		{
			name:   "scheduler",
			server: apiserver.Server{Addr: "127.0.0.1", Port: ports.SchedulerPort, Path: "/healthz"},
		},
	}
	for ix, machine := range rs.databaseStorage.Backends() {
		etcdUrl, err := url.Parse(machine)
		if err != nil {
			glog.Errorf("Failed to parse etcd url for validation: %v", err)
			continue
		}
		var port int
		var addr string
		if strings.Contains(etcdUrl.Host, ":") {
			var portString string
			addr, portString, err = net.SplitHostPort(etcdUrl.Host)
			if err != nil {
				glog.Errorf("Failed to split host/port: %s (%v)", etcdUrl.Host, err)
				continue
			}
			port, _ = strconv.Atoi(portString)
		} else {
			addr = etcdUrl.Host
			port = 4001
		}
		serversToValidate = append(serversToValidate, namedServer{
			name:   fmt.Sprintf("etcd-%d", ix),
			server: apiserver.Server{Addr: addr, Port: port, Path: "/health", Validate: etcd_storage.EtcdHealthCheck},
		})
	}
	return serversToValidate
}
