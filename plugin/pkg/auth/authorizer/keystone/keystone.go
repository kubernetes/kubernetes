/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// Package keystone implements the authorizer.Authorizer interface allowing
// Keystone Authorization to be used.
package keystone

import (
	"errors"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/auth/authorizer"

	_ "k8s.io/kubernetes/pkg/apis/authorization/install"
)

// Ensure Keystone implements the authorizer.Authorizer interface.
var _ authorizer.Authorizer = (*KeystoneAuthorizer)(nil)

type KeystoneAuthorizer struct {
	ProjectId string
	Role      string
}

func New(projectId string, role string) (*KeystoneAuthorizer, error) {
	if projectId == "" {
		return nil, errors.New("Project ID needs to be set")
	} else if role == "" {
		return nil, errors.New("Role needs to be set")
	}
	return &KeystoneAuthorizer{projectId, role}, nil
}

func (ka *KeystoneAuthorizer) Authorize(attr authorizer.Attributes) (err error) {
	/* FIXME - kfox1111 If we want to restrict namespace to a project, we may need this:
	if attr.IsResourceRequest() {
		namespace := attr.GetNamespace()
	}
	*/
	glog.V(3).Infof("Checking Keystone AuthZ.")
	extra := attr.GetUserExtra()
	if v, ok := extra["alpha.kubernetes.io/keystone_project_id"]; ok {
		if ka.ProjectId != v[0] {
			glog.V(3).Infof("Project ID doesn't match. %s != %s.", ka.ProjectId, v[0])
			return errors.New("unauthorized")
		}
	} else {
		return errors.New("unauthorized")
	}
	if v, ok := extra["alpha.kubernetes.io/keystone_roles"]; ok {
		glog.V(3).Infof("Checking for Role %s in %v", ka.Role, v)
		not_found := true
		for i := 0; i < len(v); i++ {
			if v[i] == ka.Role {
				not_found = false
				glog.V(3).Infof("Found.")
				break
			}
		}
		if not_found {
			glog.V(3).Infof("Not Found.")
			return errors.New("unauthorized")
		}
		return nil
	}
	return errors.New("unauthorized")
}
