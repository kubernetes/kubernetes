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

package handlers

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// ScopeNamer handles accessing names from requests and objects
type ScopeNamer interface {
	// Namespace returns the appropriate namespace value from the request (may be empty) or an
	// error.
	Namespace(req *http.Request) (namespace string, err error)
	// Name returns the name from the request, and an optional namespace value if this is a namespace
	// scoped call. An error is returned if the name is not available.
	Name(req *http.Request) (namespace, name string, err error)
	// ObjectName returns the namespace and name from an object if they exist, or an error if the object
	// does not support names.
	ObjectName(obj runtime.Object) (namespace, name string, err error)
	// SetSelfLink sets the provided URL onto the object. The method should return nil if the object
	// does not support selfLinks.
	SetSelfLink(obj runtime.Object, url string) error
	// GenerateLink creates an encoded URI for a given runtime object that represents the canonical path
	// and query.
	GenerateLink(requestInfo *request.RequestInfo, obj runtime.Object) (uri string, err error)
	// GenerateListLink creates an encoded URI for a list that represents the canonical path and query.
	GenerateListLink(req *http.Request) (uri string, err error)
}

type ContextBasedNaming struct {
	SelfLinker    runtime.SelfLinker
	ClusterScoped bool

	SelfLinkPathPrefix string
	SelfLinkPathSuffix string
}

// ContextBasedNaming implements ScopeNamer
var _ ScopeNamer = ContextBasedNaming{}

func (n ContextBasedNaming) SetSelfLink(obj runtime.Object, url string) error {
	return n.SelfLinker.SetSelfLink(obj, url)
}

func (n ContextBasedNaming) Namespace(req *http.Request) (namespace string, err error) {
	requestInfo, ok := request.RequestInfoFrom(req.Context())
	if !ok {
		return "", fmt.Errorf("missing requestInfo")
	}
	return requestInfo.Namespace, nil
}

func (n ContextBasedNaming) Name(req *http.Request) (namespace, name string, err error) {
	requestInfo, ok := request.RequestInfoFrom(req.Context())
	if !ok {
		return "", "", fmt.Errorf("missing requestInfo")
	}

	if len(requestInfo.Name) == 0 {
		return "", "", errEmptyName
	}
	return requestInfo.Namespace, requestInfo.Name, nil
}

// fastURLPathEncode encodes the provided path as a URL path
func fastURLPathEncode(path string) string {
	for _, r := range []byte(path) {
		switch {
		case r >= '-' && r <= '9', r >= 'A' && r <= 'Z', r >= 'a' && r <= 'z':
			// characters within this range do not require escaping
		default:
			var u url.URL
			u.Path = path
			return u.EscapedPath()
		}
	}
	return path
}

func (n ContextBasedNaming) GenerateLink(requestInfo *request.RequestInfo, obj runtime.Object) (uri string, err error) {
	namespace, name, err := n.ObjectName(obj)
	if err == errEmptyName && len(requestInfo.Name) > 0 {
		name = requestInfo.Name
	} else if err != nil {
		return "", err
	}
	if len(namespace) == 0 && len(requestInfo.Namespace) > 0 {
		namespace = requestInfo.Namespace
	}

	if n.ClusterScoped {
		return n.SelfLinkPathPrefix + url.QueryEscape(name) + n.SelfLinkPathSuffix, nil
	}

	builder := strings.Builder{}
	builder.Grow(len(n.SelfLinkPathPrefix) + len(namespace) + len(requestInfo.Resource) + len(name) + len(n.SelfLinkPathSuffix) + 8)
	builder.WriteString(n.SelfLinkPathPrefix)
	builder.WriteString(namespace)
	builder.WriteByte('/')
	builder.WriteString(requestInfo.Resource)
	builder.WriteByte('/')
	builder.WriteString(name)
	builder.WriteString(n.SelfLinkPathSuffix)
	return fastURLPathEncode(builder.String()), nil
}

func (n ContextBasedNaming) GenerateListLink(req *http.Request) (uri string, err error) {
	if len(req.URL.RawPath) > 0 {
		return req.URL.RawPath, nil
	}
	return fastURLPathEncode(req.URL.Path), nil
}

func (n ContextBasedNaming) ObjectName(obj runtime.Object) (namespace, name string, err error) {
	name, err = n.SelfLinker.Name(obj)
	if err != nil {
		return "", "", err
	}
	if len(name) == 0 {
		return "", "", errEmptyName
	}
	namespace, err = n.SelfLinker.Namespace(obj)
	if err != nil {
		return "", "", err
	}
	return namespace, name, err
}

// errEmptyName is returned when API requests do not fill the name section of the path.
var errEmptyName = errors.NewBadRequest("name must be provided")
