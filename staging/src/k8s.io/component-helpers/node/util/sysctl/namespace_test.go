/*
Copyright 2023 The Kubernetes Authors.

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

package sysctl

import (
	"testing"
)

func TestNamespacedOf(t *testing.T) {
	tests := map[string]Namespace{
		"kernel.shm_rmid_forced": IPCNamespace,
		"net.a.b.c":              NetNamespace,
		"fs.mqueue.a.b.c":        IPCNamespace,
		"foo":                    UnknownNamespace,
	}

	for sysctl, ns := range tests {
		if got := namespaceOf(sysctl); got != ns {
			t.Errorf("wrong namespace for %q: got=%s want=%s", sysctl, got, ns)
		}
	}
}
