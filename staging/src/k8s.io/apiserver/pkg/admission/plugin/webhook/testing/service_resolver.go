/*
Copyright 2018 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"net/url"

	"k8s.io/apiserver/pkg/util/webhook"
)

type serviceResolver struct {
	base url.URL
}

// NewServiceResolver returns a static service resolve that return the given URL or
// an error for the failResolve namespace.
func NewServiceResolver(base url.URL) webhook.ServiceResolver {
	return &serviceResolver{base}
}

func (f serviceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	if namespace == "failResolve" {
		return nil, fmt.Errorf("couldn't resolve service location")
	}
	u := f.base
	return &u, nil
}
