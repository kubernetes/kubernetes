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

package flags

import (
	"fmt"

	"github.com/spf13/pflag"
)

var modes = []string{
	"AlwaysAllow",
	"AlwaysDeny",
	"ABAC",
	"RBAC",
	"Webhook",
}

func NewAuthorizationModeFlag(mode *string) pflag.Value {
	return &authorizationModeValue{mode: mode}
}

type authorizationModeValue struct {
	mode *string
}

func (c *authorizationModeValue) String() string {
	return *c.mode
}

func (c *authorizationModeValue) Set(s string) error {
	if ValidateAuthorizationMode(s) {
		*c.mode = s
		return nil
	}

	return fmt.Errorf("authorization mode %q is not supported, you can use any of %v", s, modes)
}

func (c *authorizationModeValue) Type() string {
	return "mode"
}

func ValidateAuthorizationMode(mode string) bool {
	for _, supported := range modes {
		if mode == supported {
			return true
		}
	}
	return false
}
