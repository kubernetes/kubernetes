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

package event

import (
	"fmt"
	"net/url"

	"k8s.io/apiserver/pkg/apis/audit"
	authuser "k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

var _ authorizer.Attributes = &attributes{}

// attributes implements the authorizer attributes interface
// with event data. This is used for enforced audit backends
type attributes struct {
	event *audit.Event
	path  string
}

// NewAttributes returns a new attributes struct and parsed request uri
// if needed
func NewAttributes(event *audit.Event) (authorizer.Attributes, error) {
	a := attributes{
		event: event,
	}
	if event.ObjectRef == nil {
		u, err := url.ParseRequestURI(a.event.RequestURI)
		if err != nil {
			return nil, fmt.Errorf("could not parse url: %v", err)
		}
		a.path = u.Path
	}
	return &a, nil
}

// GetUser returns the user. This is only used for checking audit policy,
// and the audit policy user check is based off the original user,
// not the impersonated user.
func (a *attributes) GetUser() authuser.Info {
	return user(a.event.User)
}

// GetVerb returns the verb
func (a *attributes) GetVerb() string {
	return a.event.Verb
}

// IsReadOnly determines if the verb is a read only action
func (a *attributes) IsReadOnly() bool {
	return a.event.Verb == "get" || a.event.Verb == "list" || a.event.Verb == "watch"
}

// GetNamespace returns the object namespace if present
func (a *attributes) GetNamespace() string {
	if a.event.ObjectRef == nil {
		return ""
	}
	return a.event.ObjectRef.Namespace
}

// GetResource returns the object resource if present
func (a *attributes) GetResource() string {
	if a.event.ObjectRef == nil {
		return ""
	}
	return a.event.ObjectRef.Resource
}

// GetSubresource returns the object subresource if present
func (a *attributes) GetSubresource() string {
	if a.event.ObjectRef == nil {
		return ""
	}
	return a.event.ObjectRef.Subresource
}

// GetName returns the object name if present
func (a *attributes) GetName() string {
	if a.event.ObjectRef == nil {
		return ""
	}
	return a.event.ObjectRef.Name
}

// GetAPIGroup returns the object api group if present
func (a *attributes) GetAPIGroup() string {
	if a.event.ObjectRef == nil {
		return ""
	}
	return a.event.ObjectRef.APIGroup
}

// GetAPIVersion returns the object api version if present
func (a *attributes) GetAPIVersion() string {
	if a.event.ObjectRef == nil {
		return ""
	}
	return a.event.ObjectRef.APIVersion
}

// IsResourceRequest determines if the request was acted on a resource
func (a *attributes) IsResourceRequest() bool {
	return a.event.ObjectRef != nil
}

// GetPath returns the path uri accessed
func (a *attributes) GetPath() string {
	return a.path
}

// user represents the event user
type user audit.UserInfo

// GetName returns the user name
func (u user) GetName() string { return u.Username }

// GetUID returns the user uid
func (u user) GetUID() string { return u.UID }

// GetGroups returns the user groups
func (u user) GetGroups() []string { return u.Groups }

// GetExtra returns the user extra data
func (u user) GetExtra() map[string][]string {
	m := map[string][]string{}
	for k, v := range u.Extra {
		m[k] = []string(v)
	}
	return m
}
