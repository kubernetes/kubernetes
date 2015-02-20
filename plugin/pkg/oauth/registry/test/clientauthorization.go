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

package test

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
)

type ClientAuthorizationRegistry struct {
	registrytest.GenericRegistry
}

func NewClientAuthorizationRegistry() ClientAuthorizationRegistry {
	return ClientAuthorizationRegistry{
		GenericRegistry: *registrytest.NewGeneric(nil),
	}
}

func (*ClientAuthorizationRegistry) Name(userName, clientName string) string {
	return fmt.Sprintf("%s:%s", userName, clientName)
}
