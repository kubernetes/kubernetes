/*
Copyright 2015 The Kubernetes Authors.

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
	"context"
	"fmt"
	"sync"

	"k8s.io/apiserver/pkg/storage"

	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/rest"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/probe"
)

type REST struct {
	GetServersToValidate func() map[string]Server
	rest.TableConvertor
}

// NewStorage returns a new REST.
func NewStorage(serverRetriever func() map[string]Server) *REST {
	return &REST{
		GetServersToValidate: serverRetriever,
		TableConvertor:       printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
}

func (*REST) NamespaceScoped() bool {
	return false
}

func (rs *REST) New() runtime.Object {
	return &api.ComponentStatus{}
}

var _ rest.SingularNameProvider = &REST{}

func (rs *REST) GetSingularName() string {
	return "componentstatus"
}

// Destroy cleans up resources on shutdown.
func (r *REST) Destroy() {
	// Given no underlying store, we don't destroy anything
	// here explicitly.
}

func (rs *REST) NewList() runtime.Object {
	return &api.ComponentStatusList{}
}

// Returns the list of component status. Note that the label and field are both ignored.
// Note that this call doesn't support labels or selectors.
func (rs *REST) List(ctx context.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	servers := rs.GetServersToValidate()

	wait := sync.WaitGroup{}
	wait.Add(len(servers))
	statuses := make(chan api.ComponentStatus, len(servers))
	for k, v := range servers {
		go func(name string, server Server) {
			defer wait.Done()
			status := rs.getComponentStatus(name, server)
			statuses <- *status
		}(k, v)
	}
	wait.Wait()
	close(statuses)

	pred := componentStatusPredicate(options)

	reply := []api.ComponentStatus{}
	for status := range statuses {
		// ComponentStatus resources currently (v1.14) do not support labeling, however the filtering is executed
		// nonetheless in case the request contains Label or Field selectors (which will effectively filter out
		// all of the results and return an empty response).
		matches, err := pred.Matches(&status)
		if err != nil {
			return nil, err
		}
		if matches {
			reply = append(reply, status)
		}
	}
	return &api.ComponentStatusList{Items: reply}, nil
}

func componentStatusPredicate(options *metainternalversion.ListOptions) storage.SelectionPredicate {
	s := runtime.Selectors{}
	if options != nil {
		s.Labels = options.LabelSelector
		s.Fields = options.FieldSelector
	}
	return storage.DefaultPredicateFunc(s)
}

func (rs *REST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
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

func (rs *REST) getComponentStatus(name string, server Server) *api.ComponentStatus {
	status, msg, err := server.DoServerCheck()
	errorMsg := ""
	if err != nil {
		errorMsg = err.Error()
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

// Implement ShortNamesProvider
var _ rest.ShortNamesProvider = &REST{}

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"cs"}
}
