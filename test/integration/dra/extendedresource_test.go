/*
Copyright The Kubernetes Authors.

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

package dra

import (
	"testing"

	"k8s.io/client-go/ktesting"
)

func TestExtendedResourcePodDeletedReschedule(t *testing.T) {
	testBindingRelease(ktesting.Init(t), "extended", false)
}

func TestBindingReleaseReschedule(t *testing.T) {
	for _, tc := range []struct {
		name, kind          string
		createAfterDeletion bool
	}{
		{name: "TemplateClaim", kind: "template"},
		{name: "StandaloneClaimDeleted", kind: "standalone-deleted"},
		{name: "StandaloneClaimTimeout", kind: "standalone-timeout"},
		{name: "StandaloneClaimDeallocated", kind: "standalone-deallocated"},
		{name: "SharedStandaloneClaim", kind: "shared", createAfterDeletion: true},
		{name: "ExtendedResourceMappingRenamed", kind: "extended-renamed"},
		{name: "ExtendedResourceMappingRemoved", kind: "extended-removed"},
	} {
		t.Run(tc.name, func(t *testing.T) { testBindingRelease(ktesting.Init(t), tc.kind, tc.createAfterDeletion) })
	}
}
