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

package authorizerfactory

import (
	"errors"

	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// alwaysAllowAuthorizer is an implementation of authorizer.Attributes
// which always says yes to an authorization request.
// It is useful in tests and when using kubernetes in an open manner.
type alwaysAllowAuthorizer struct{}

func (alwaysAllowAuthorizer) Authorize(a authorizer.Attributes) (authorized bool, reason string, err error) {
	return true, "", nil
}

func NewAlwaysAllowAuthorizer() authorizer.Authorizer {
	return new(alwaysAllowAuthorizer)
}

// alwaysDenyAuthorizer is an implementation of authorizer.Attributes
// which always says no to an authorization request.
// It is useful in unit tests to force an operation to be forbidden.
type alwaysDenyAuthorizer struct{}

func (alwaysDenyAuthorizer) Authorize(a authorizer.Attributes) (authorized bool, reason string, err error) {
	return false, "Everything is forbidden.", nil
}

func NewAlwaysDenyAuthorizer() authorizer.Authorizer {
	return new(alwaysDenyAuthorizer)
}

// alwaysFailAuthorizer is an implementation of authorizer.Attributes
// which always says no to an authorization request.
// It is useful in unit tests to force an operation to fail with error.
type alwaysFailAuthorizer struct{}

func (alwaysFailAuthorizer) Authorize(a authorizer.Attributes) (authorized bool, reason string, err error) {
	return false, "", errors.New("Authorization failure.")
}

func NewAlwaysFailAuthorizer() authorizer.Authorizer {
	return new(alwaysFailAuthorizer)
}

type privilegedGroupAuthorizer struct {
	groups []string
}

func (r *privilegedGroupAuthorizer) Authorize(attr authorizer.Attributes) (bool, string, error) {
	if attr.GetUser() == nil {
		return false, "Error", errors.New("no user on request.")
	}
	for _, attr_group := range attr.GetUser().GetGroups() {
		for _, priv_group := range r.groups {
			if priv_group == attr_group {
				return true, "", nil
			}
		}
	}
	return false, "", nil
}

// NewPrivilegedGroups is for use in loopback scenarios
func NewPrivilegedGroups(groups ...string) *privilegedGroupAuthorizer {
	return &privilegedGroupAuthorizer{
		groups: groups,
	}
}
