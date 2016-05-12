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
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
)

type keystoneTokenAuthenticator struct {
	localValidator   validator
	apiCallValidator validator
}

// NewKeystoneTokenAuthenticator returns a token authenticator that validates token
// by local verification, if it does not support, then call openstack api
func NewKeystoneTokenAuthenticator(configFile string) (authenticator.Token, error) {
	glog.V(2).Infof("initialize keystone token authenticator")
	apiCallValidator, err := newAPICallValidator(configFile)
	// local token validator is allowed to be nil
	if err != nil {
		glog.V(2).Infof("initialize api call validator failed: %s", err)
	}
	localValidator, err := newLocalValidator(apiCallValidator.client.IdentityEndpoint)
	if err != nil {
		glog.Errorf("initialize local validator failed: %s", err)
		return nil, err
	}
	return &keystoneTokenAuthenticator{
		localValidator:   localValidator,
		apiCallValidator: apiCallValidator,
	}, nil
}

func (ka *keystoneTokenAuthenticator) AuthenticateToken(token string) (user.Info, bool, error) {
	var resp *response
	var err error
	// try local validator first
	if ka.localValidator != nil && ka.localValidator.support(token) {
		glog.V(4).Infof("authenticate via local token validation")
		resp, err = ka.localValidator.validate(token)
	} else {
		glog.V(4).Infof("authenticate via keystone api call validation")
		resp, err = ka.apiCallValidator.validate(token)
	}
	if err != nil {
		glog.Errorf("authentication failed: %s", err)
		return nil, false, err
	}
	// then token may be expired already
	expiredAt, err := time.Parse(time.RFC3339, resp.Access.Token.ExpiredAt)
	if err != nil {
		glog.Errorf("authentication failed: unable to parse expired time: %s", err)
		return nil, false, err
	}
	if expiredAt.Before(time.Now()) {
		glog.Errorf("authentication failed: token expired at %s", resp.Access.Token.ExpiredAt)
		return nil, false, fmt.Errorf("authentication failed: token expired at %s", resp.Access.Token.ExpiredAt)
	}
	return &user.DefaultInfo{
		Name:   resp.Access.User.Username,
		Groups: []string{resp.Access.Token.Tenant.Name},
	}, true, nil
}
