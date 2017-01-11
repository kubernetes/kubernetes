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

package api

import (
	"bytes"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"

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

func TestIsVowel(t *testing.T) {
	tests := []struct {
		name string
		arg  rune
		want bool
	}{
		{
			name: "yes",
			arg:  'E',
			want: true,
		},
		{
			name: "no",
			arg:  'n',
			want: false,
		},
	}
	for _, tt := range tests {
		if got := isVowel(tt.arg); got != tt.want {
			t.Errorf("%q. IsVowel() = %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestGetArticleForNoun(t *testing.T) {
	type args struct {
	}
	tests := []struct {
		noun    string
		padding string
		want    string
	}{
		{
			noun:    "Frog",
			padding: " ",
			want:    " a ",
		},
		{
			noun:    "frogs",
			padding: " ",
			want:    " ",
		},
		{
			noun:    "apple",
			padding: "",
			want:    "an",
		},
		{
			noun:    "Apples",
			padding: " ",
			want:    " ",
		},
		{
			noun:    "Ingress",
			padding: " ",
			want:    " an ",
		},
		{
			noun:    "Class",
			padding: " ",
			want:    " a ",
		},
	}
	for _, tt := range tests {
		if got := getArticleForNoun(tt.noun, tt.padding); got != tt.want {
			t.Errorf("%q. GetArticleForNoun() = %v, want %v", tt.noun, got, tt.want)
		}
	}
}
