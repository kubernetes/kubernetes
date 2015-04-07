/*
Copyright 2014 Google Inc. All rights reserved.

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

package authorizer

import (
	"net/http"
	"net/url"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
)

// APIAttributes is an interface used by an Authorizer to get information about a request
// that is used to make an authorization decision.
type APIAttributes interface {
	GetUserInfo() user.Info
	// The verb of the object, if a request is for a REST object.
	GetVerb() string
	// The namespace of the object, if a request is for a REST object.
	GetNamespace() string
	// The resource of object, if a request is for a REST object.
	GetResource() string
	// The subresource of object, if a request is for a REST object.
	GetSubresource() string
	// The kind of object, if a request is for a REST object.
	GetKind() string
	GetResourceName() string

	// GetExtendedInfo returns information that the attribute builder thought was useful.  It's type is dependent on the contract between the authorizer and the extension builder
	GetExtendedInfo() interface{}
}

type GenericAttributes interface {
	GetUserInfo() user.Info
	GetURL() url.URL
	GetMethod() string
}

// Authorizer makes an authorization decision based on information gained by making
// zero or more calls to methods of the Attributes interface.  It returns nil when an action is
// authorized, otherwise it returns an error.
type Authorizer interface {
	APIAuthorizer
	GenericAuthorizer
}
type APIAuthorizer interface {
	AuthorizeAPIRequest(a APIAttributes) (err error)
}
type GenericAuthorizer interface {
	AuthorizeGenericRequest(a GenericAttributes) (err error)
}

type APIAttributeExtensionBuilder interface {
	GetExtendedInfo(request *http.Request) (interface{}, error)
}

// APIAttributesRecord implements APIAttributes interface.
type APIAttributesRecord struct {
	UserInfo     user.Info
	Verb         string
	Namespace    string
	Resource     string
	Subresource  string
	Kind         string
	ResourceName string
	ExtendedInfo interface{}
}

func (a APIAttributesRecord) GetUserInfo() user.Info {
	return a.UserInfo
}

func (a APIAttributesRecord) GetVerb() string {
	return strings.ToLower(a.Verb)
}

func (a APIAttributesRecord) GetNamespace() string {
	return strings.ToLower(a.Namespace)
}

func (a APIAttributesRecord) GetResource() string {
	return strings.ToLower(a.Resource)
}

func (a APIAttributesRecord) GetSubresource() string {
	return strings.ToLower(a.Subresource)
}
func (a APIAttributesRecord) GetKind() string {
	return strings.ToLower(a.Kind)
}
func (a APIAttributesRecord) GetResourceName() string {
	return strings.ToLower(a.ResourceName)
}
func (a APIAttributesRecord) GetExtendedInfo() interface{} {
	return a.ExtendedInfo
}

// GenericAttributesRecord implements APIAttributes interface.
type GenericAttributesRecord struct {
	UserInfo user.Info
	URL      url.URL
	Method   string
}

func (a GenericAttributesRecord) GetUserInfo() user.Info {
	return a.UserInfo
}

func (a GenericAttributesRecord) GetURL() url.URL {
	return a.URL
}

func (a GenericAttributesRecord) GetMethod() string {
	return strings.ToLower(a.Method)
}
