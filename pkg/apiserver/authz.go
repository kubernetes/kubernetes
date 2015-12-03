/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/authorizer/abac"
	"k8s.io/kubernetes/pkg/auth/authorizer/union"
)

// Attributes implements authorizer.Attributes interface.
type Attributes struct {
	// TODO: add fields and methods when authorizer.Attributes is completed.
}

// alwaysAllowAuthorizer is an implementation of authorizer.Attributes
// which always says yes to an authorization request.
// It is useful in tests and when using kubernetes in an open manner.
type alwaysAllowAuthorizer struct{}

func (alwaysAllowAuthorizer) Authorize(a authorizer.Attributes) (err error) {
	return nil
}

func NewAlwaysAllowAuthorizer() authorizer.Authorizer {
	return new(alwaysAllowAuthorizer)
}

// alwaysDenyAuthorizer is an implementation of authorizer.Attributes
// which always says no to an authorization request.
// It is useful in unit tests to force an operation to be forbidden.
type alwaysDenyAuthorizer struct{}

func (alwaysDenyAuthorizer) Authorize(a authorizer.Attributes) (err error) {
	return errors.New("Everything is forbidden.")
}

func NewAlwaysDenyAuthorizer() authorizer.Authorizer {
	return new(alwaysDenyAuthorizer)
}

const (
	ModeAlwaysAllow string = "AlwaysAllow"
	ModeAlwaysDeny  string = "AlwaysDeny"
	ModeABAC        string = "ABAC"
)

// Keep this list in sync with constant list above.
var AuthorizationModeChoices = []string{ModeAlwaysAllow, ModeAlwaysDeny, ModeABAC}

// NewAuthorizerFromAuthorizationConfig returns the right sort of union of multiple authorizer.Authorizer objects
// based on the authorizationMode or an error.  authorizationMode should be a comma separated values
// of AuthorizationModeChoices.
func NewAuthorizerFromAuthorizationConfig(authorizationModes []string, authorizationPolicyFile string) (authorizer.Authorizer, error) {

	if len(authorizationModes) == 0 {
		return nil, errors.New("Atleast one authorization mode should be passed")
	}

	var authorizers []authorizer.Authorizer
	authorizerMap := make(map[string]bool)

	for _, authorizationMode := range authorizationModes {
		if authorizerMap[authorizationMode] {
			return nil, fmt.Errorf("Authorization mode %s specified more than once", authorizationMode)
		}
		// Keep cases in sync with constant list above.
		switch authorizationMode {
		case ModeAlwaysAllow:
			authorizers = append(authorizers, NewAlwaysAllowAuthorizer())
		case ModeAlwaysDeny:
			authorizers = append(authorizers, NewAlwaysDenyAuthorizer())
		case ModeABAC:
			if authorizationPolicyFile == "" {
				return nil, errors.New("ABAC's authorization policy file not passed")
			}
			abacAuthorizer, err := abac.NewFromFile(authorizationPolicyFile)
			authorizerLock := &abac.PolicyAuthorizer{PL: &abacAuthorizer, Lock: &sync.RWMutex{}}
			go checkAndReloadPolicyFile(authorizerLock, authorizationPolicyFile)
			if err != nil {
				return nil, err
			}
			authorizers = append(authorizers, authorizerLock)
		default:
			return nil, fmt.Errorf("Unknown authorization mode %s specified", authorizationMode)
		}
		authorizerMap[authorizationMode] = true
	}

	if !authorizerMap[ModeABAC] && authorizationPolicyFile != "" {
		return nil, errors.New("Cannot specify --authorization-policy-file without mode ABAC")
	}

	return union.New(authorizers...), nil
}

func checkAndReloadPolicyFile(policyAuthz *abac.PolicyAuthorizer, file string) {
	lastModifyTime := time.Now()
	for {
		func() {
			defer func() {
				if x := recover(); x != nil {
					glog.Errorf("APIServer panic'd on checkAndReloadPolicyFile: %s, err: %v", file, x)
				}
			}()
			//var info os.FileInfo
			info, err := os.Stat(file)
			if err != nil {
				glog.Errorf("Stat authorizationPolicyFile: %s fail, err: %v, will retry to reload later", file, err)
				return
			}
			if info.ModTime().After(lastModifyTime) {
				lastModifyTime = info.ModTime()
				newPolicyList, err := abac.NewFromFile(file)
				if err != nil { // file format not correct
					glog.Errorf("Stat authorizationPolicyFile: %s fail, err: %v, will retry to reload later", file, err)
					return
				}

				glog.Infof("authorizationPolicyFile: %s is modified, reload finished, number of policy lines change from %d to %d", file, len(*policyAuthz.PL), len(newPolicyList))
				policyAuthz.Lock.Lock()
				*(policyAuthz.PL) = newPolicyList
				policyAuthz.Lock.Unlock()
			}
		}()
		time.Sleep(5 * time.Second)
	}
}
