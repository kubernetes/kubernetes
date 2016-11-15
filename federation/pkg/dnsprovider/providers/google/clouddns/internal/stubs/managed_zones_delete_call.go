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

package stubs

import (
	"fmt"

	"google.golang.org/api/googleapi"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
)

// Compile time check for interface adeherence
var _ interfaces.ManagedZonesDeleteCall = ManagedZonesDeleteCall{}

type ManagedZonesDeleteCall struct {
	Service  *ManagedZonesService
	Project  string
	ZoneName string
	Error    *error // Use this to override response for testing if required
}

func (call ManagedZonesDeleteCall) Do(opts ...googleapi.CallOption) error {
	if call.Error != nil { // Return the override value
		return *call.Error
	} else { // Just try to delete it from the in-memory array.
		project, ok := call.Service.Impl[call.Project]
		if ok {
			zone, ok := project[call.ZoneName]
			if ok {
				delete(project, zone.Name())
				return nil
			} else {
				return fmt.Errorf("Failed to find zone %s in project %s to delete it", call.ZoneName, call.Project)
			}
		} else {
			return fmt.Errorf("Failed to find project %s to delete zone %s from it", call.Project, call.ZoneName)
		}
	}
}
