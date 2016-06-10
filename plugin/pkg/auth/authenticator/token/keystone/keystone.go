/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package keystone

import (
	"errors"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/auth/user"
)

// KeystoneAuthenticator contacts openstack keystone to validate user's credentials passed in the request.
// The keystone endpoint is passed during apiserver startup
type KeystoneAuthenticator struct {
	configPath       string
	apiCallValidator validator
}

func (keystoneAuthenticator *KeystoneAuthenticator) AuthenticateToken(token string) (user.Info, bool, error) {
	//FIXME kfox1111: To enhance performance, cache admin token/client.
	glog.V(3).Infof("Starting Keystone Token Auth.")
	userid, projectid, roles, valid, err := keystoneAuthenticator.apiCallValidator.validate(token)
	if err != nil {
		return nil, false, err
	}
	extra := make(map[string][]string, 3)
	extra["alpha.kubernetes.io/keystone_authn"] = []string{"true"}
	extra["alpha.kubernetes.io/keystone_project_id"] = []string{projectid}
	extra["alpha.kubernetes.io/keystone_roles"] = roles
	return &user.DefaultInfo{Name: userid, Extra: extra}, valid, nil
}

// NewKeystoneAuthenticator returns a token authenticator that validates credentials using OpenStack Keystone
func NewKeystoneAuthenticator(configPath string) (*KeystoneAuthenticator, error) {
	apiCallValidator, err := newAPICallValidator(configPath)
	if err != nil {
		glog.V(2).Infof("Initialization of Keystone api call validator failed: %s", err)
		return nil, err
	}
	if configPath != "" {
		return &KeystoneAuthenticator{configPath, apiCallValidator}, nil
	}
	return nil, errors.New("configPath is empty")
}
