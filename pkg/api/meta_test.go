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

package api_test

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
)

// TestFillObjectMetaSystemFields validates that system populated fields are set on an object
func TestFillObjectMetaSystemFields(t *testing.T) {
	ctx := api.NewDefaultContext()
	resource := api.ObjectMeta{}
	api.FillObjectMetaSystemFields(ctx, &resource)
	if resource.CreationTimestamp.Time.IsZero() {
		t.Errorf("resource.CreationTimestamp is zero")
	} else if len(resource.UID) == 0 {
		t.Errorf("resource.UID missing")
	}
}

// TestHasObjectMetaSystemFieldValues validates that true is returned if and only if all fields are populated
func TestHasObjectMetaSystemFieldValues(t *testing.T) {
	ctx := api.NewDefaultContext()
	resource := api.ObjectMeta{}
	if api.HasObjectMetaSystemFieldValues(&resource) {
		t.Errorf("the resource does not have all fields yet populated, but incorrectly reports it does")
	}
	api.FillObjectMetaSystemFields(ctx, &resource)
	if !api.HasObjectMetaSystemFieldValues(&resource) {
		t.Errorf("the resource does have all fields populated, but incorrectly reports it does not")
	}
}
