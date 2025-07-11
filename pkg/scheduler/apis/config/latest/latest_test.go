/*
Copyright 2021 The Kubernetes Authors.

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

package latest

import (
	"testing"
)

func TestDefault(t *testing.T) {
	defaultIns, err := Default()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if defaultIns == nil {
		t.Fatal("expected non-nil default configuration")
	}
	if defaultIns.APIVersion != "kubescheduler.config.k8s.io/v1" {
		t.Errorf("expected default APIVersion 'kubescheduler.config.k8s.io/v1', got %s", defaultIns.APIVersion)
	}
}
