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

package options

import (
	"k8s.io/kubernetes/pkg/apiserver/authenticator"
)

// AuthenticationRequestHeaderConfig returns an authenticator config object for these options
// if necessary.  nil otherwise.
func (s *ServerRunOptions) AuthenticationRequestHeaderConfig() *authenticator.RequestHeaderConfig {
	if len(s.RequestHeaderUsernameHeaders) == 0 {
		return nil
	}

	return &authenticator.RequestHeaderConfig{
		UsernameHeaders:    s.RequestHeaderUsernameHeaders,
		ClientCA:           s.RequestHeaderClientCAFile,
		AllowedClientNames: s.RequestHeaderAllowedNames,
	}
}
