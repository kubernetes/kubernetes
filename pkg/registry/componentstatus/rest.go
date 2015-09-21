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

package componentstatus

import (
	"fmt"
	"net/http"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apiserver"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/kubernetes/pkg/runtime"
)

type REST struct {
	GetServersToValidate func() map[string]apiserver.Server
	rt                   http.RoundTripper
}

// NewStorage returns a new REST.
func NewStorage(serverRetriever func() map[string]apiserver.Server) *REST {
	return &REST{
		GetServersToValidate: serverRetriever,
		rt:                   http.DefaultTransport,
	}
}

func (rs *REST) New() runtime.Object {
	return &api.ComponentStatus{}
}

func (rs *REST) NewList() runtime.Object {
	return &api.ComponentStatusList{}
}

// Returns the list of component status. Note that the label and field are both ignored.
// Note that this call doesn't support labels or selectors.
func (rs *REST) List(ctx api.Context, label labels.Selector, field fields.Selector) (runtime.Object, error) {
	servers := rs.GetServersToValidate()

	// TODO: This should be parallelized.
	reply := []api.ComponentStatus{}
	for name, server := range servers {
		status := rs.getComponentStatus(name, server)
		reply = append(reply, *status)
	}
	return &api.ComponentStatusList{Items: reply}, nil
}

func (rs *REST) Get(ctx api.Context, name string) (runtime.Object, error) {
	servers := rs.GetServersToValidate()

	if server, ok := servers[name]; !ok {
		return nil, fmt.Errorf("Component not found: %s", name)
	} else {
		return rs.getComponentStatus(name, server), nil
	}
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

func (rs *REST) getComponentStatus(name string, server apiserver.Server) *api.ComponentStatus {
	transport := rs.rt
	status, msg, err := server.DoServerCheck(transport)
	var errorMsg string
	if err != nil {
		errorMsg = err.Error()
	} else {
		errorMsg = "nil"
	}

	c := &api.ComponentCondition{
		Type:    api.ComponentHealthy,
		Status:  ToConditionStatus(status),
		Message: msg,
		Error:   errorMsg,
	}

	retVal := &api.ComponentStatus{
		Conditions: []api.ComponentCondition{*c},
	}
	retVal.Name = name

	return retVal
}
