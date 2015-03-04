/*
Copyright 2015 Google Inc. All rights reserved.

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

package apiserver

import (
	"errors"
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// Interface for operations on services that are used by teardownHandler.
type serviceREST interface {
	RESTLister
	RESTDeleter
}

type teardownHandler struct {
	rest serviceREST
}

func NewTeardownHandler(rest serviceREST) (http.Handler, error) {
	return &teardownHandler{
		rest: rest,
	}, nil
}

func (t *teardownHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	errlist := t.teardown()
	if len(errlist) > 0 {
		w.WriteHeader(http.StatusInternalServerError)
		for _, e := range errlist {
			w.Write([]byte(e.Error()))
			w.Write([]byte(","))
		}
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ok"))
}

func (t *teardownHandler) teardown() []error {
	// Get a list of all services.
	errlist := []error{}
	ctx := api.WithNamespace(api.NewContext(), api.NamespaceAll)
	obj, err := t.rest.List(ctx, labels.Everything(), labels.Everything())
	if err != nil {
		errlist = append(errlist, err)
		return errlist
	}
	services, ok := obj.(*api.ServiceList)
	if !ok || services == nil {
		errlist = append(errlist, errors.New("cannot cast REST response to api.ServiceList in teardownHandler"))
		return errlist
	}
	// Delete services that use external load balancers.
	for _, s := range services.Items {
		if s.Spec.CreateExternalLoadBalancer {
			c := api.WithNamespace(api.NewContext(), s.Namespace)
			if _, e := t.rest.Delete(c, s.Name); e != nil {
				errlist = append(errlist, e)
			}
		}
	}
	return errlist
}
