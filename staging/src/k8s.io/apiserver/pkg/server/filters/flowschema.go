/*
Copyright 2016 The Kubernetes Authors.

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

package filters

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httputil"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v2"
	flowregistration "k8s.io/api/flowregistration/v1alpha1"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/klog"
	// "k8s.io/kubernetes/pkg/apis/flowregistration"
)

type FlowSchemaRunner struct {
	FlowSchema  flowregistration.FlowSchema
	FieldGetter flowregistration.FieldGetter
}

func (f *FlowSchemaRunner) ExecuteFullMatch(req *http.Request) (bool, error) {
	// iterate over matches
	matched := false
	for _, match := range f.FlowSchema.Spec.Match {
		fmatched, err := match.And.Execute(req, f.FieldGetter)
		if err != nil {
			return false, err
		}
		matched = matched || fmatched
	}
	return matched, nil
}

type TestFieldGetter struct {
	User      []string
	Groups    []string
	Namespace []string
	Resource  []string
}

// WithPanicRecovery wraps an http Handler to recover and log panics.
func WithFlowSchema(handler http.Handler) http.Handler {
	// init flowschemas
	// parse yaml string/objs
	fs := load()
	return withFlowSchema(handler, fs)
}

func loadfs(dir string) (flowregistration.FlowSchema, error) {
	data, err := ioutil.ReadFile(filepath.Join(dir, "flowschemas.yaml"))
	if err != nil {
		return flowregistration.FlowSchema{}, err
	}
	var fs flowregistration.FlowSchema
	if err := yaml.Unmarshal(data, &fs); err != nil {
		return flowregistration.FlowSchema{}, err
	}
	return fs, nil
}

type RequestFieldGetter struct {
	flowregistration.FieldGetter
}

func (fg *RequestFieldGetter) GetField(req *http.Request, field string) (bool, []string, error) {
	switch field {
	case "user":
		// get the request's users
		ctx := req.Context()
		requestor, exists := request.UserFrom(ctx)
		if !exists {
			// no user key was found in the ctx
			return false, []string{}, nil
		}
		klog.Errorf("aaron-prindle - user: %s", requestor.GetName())
		return true, []string{requestor.GetName()}, nil
	case "groups":
		// get the request's group
		ctx := req.Context()
		requestor, exists := request.UserFrom(ctx)
		if !exists {
			panic("no user found for request")
		}
		klog.Errorf("aaron-prindle - groups: %s", requestor.GetGroups())
		return true, requestor.GetGroups(), nil
	case "namespace":
		// get the request's namesapce
		urlarr := strings.Split(req.URL.String(), "/")
		for i, s := range urlarr {
			if s == "namespaces" && i+1 <= len(urlarr)-1 {
				return true, []string{urlarr[i+1]}, nil
			}
		}
		// TODO(aaron-prindle) - 'ok' is always false using this method in my tests
		// ctx := req.Context()
		// if namespace, ok := request.NamespaceFrom(ctx); ok {
		// 	return true, []string{namespace}
		// }
		return false, []string{}, nil
	case "resource":
		// get the request's resource
		// TODO(aaron-prindle) investigate other methods, how RBAC handles this?
		// get the current resource - from path for now
		// /apis/extensions/v1beta1/namespaces/default/deployments
		urlarr := strings.Split(req.URL.String(), "/")
		return false, []string{urlarr[len(urlarr)-1]}, nil
	default:
		// TODO(aaron-prindle) replace with better error
		return false, []string{}, fmt.Errorf("unknown field type")
	}
}

func load() flowregistration.FlowSchema {
	exampleDir := "/usr/local/google/home/aprindle/go/src/github.com/aaron-prindle/flowschema"
	klog.Errorf("aaron-prindle - Reading examples at %v\n", exampleDir)

	fs, _ := loadfs(exampleDir)
	fmt.Printf("fs: %v\n", fs)
	klog.Errorf("aaron-prindle - Reading examples at %v\n", exampleDir)
	return fs
}

func run(fs flowregistration.FlowSchema, fg flowregistration.FieldGetter, req *http.Request) {
	klog.Errorf("%v", fs.Spec.Match[0])
	fsr := FlowSchemaRunner{
		FlowSchema:  fs,
		FieldGetter: fg,
	}
	matched, err := fsr.ExecuteFullMatch(req)
	if err != nil {
		panic("Execute full match had an error")
	}
	klog.Errorf("aaron-prindle - matched: %t\n", matched)
}

func logreq(req *http.Request) {
	requestDump, err := httputil.DumpRequest(req, true)
	if err != nil {
		fmt.Println(err)
		klog.Errorf("aaron-prindle: error dumping request")
	}
	klog.Errorf("aaron-prindle - request-dump: %s", requestDump)
}

func withFlowSchema(handler http.Handler, fs flowregistration.FlowSchema) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		// Save a copy of this request for debugging.
		logreq(req)
		// take incoming request
		run(fs, &RequestFieldGetter{}, req)
		// Dispatch to the internal handler
		handler.ServeHTTP(w, req)
	})
}
