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

	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/authorizer/abac"
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

// NewAuthorizerFromAuthorizationConfig returns the right sort of authorizer.Authorizer
// based on the authorizationMode xor an error.  authorizationMode should be one of AuthorizationModeChoices.
func NewAuthorizerFromAuthorizationConfig(authorizationMode string, authorizationPolicyFile string) (authorizer.Authorizer, error) {
	if authorizationPolicyFile != "" && authorizationMode != "ABAC" {
		return nil, errors.New("Cannot specify --authorization_policy_file without mode ABAC")
	}
	// Keep cases in sync with constant list above.
	switch authorizationMode {
	case ModeAlwaysAllow:
		return NewAlwaysAllowAuthorizer(), nil
	case ModeAlwaysDeny:
		return NewAlwaysDenyAuthorizer(), nil
	case ModeABAC:
		return abac.NewFromFile(authorizationPolicyFile)
	default:
		return nil, errors.New("Unknown authorization mode")
	}
}
