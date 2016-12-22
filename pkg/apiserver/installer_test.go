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

package apiserver

import (
	"bytes"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"

	"github.com/emicklei/go-restful"
)

func TestScopeNamingGenerateLink(t *testing.T) {
	selfLinker := &setTestSelfLinker{
		t:           t,
		expectedSet: "/api/v1/namespaces/other/services/foo",
		name:        "foo",
		namespace:   "other",
	}
	s := scopeNaming{
		meta.RESTScopeNamespace,
		selfLinker,
		func(name, namespace string) bytes.Buffer {
			return *bytes.NewBufferString("/api/v1/namespaces/" + namespace + "/services/" + name)
		},
		true,
	}
	service := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: "other",
		},
		TypeMeta: metav1.TypeMeta{
			Kind: "Service",
		},
	}
	_, err := s.GenerateLink(&restful.Request{}, service)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
}
