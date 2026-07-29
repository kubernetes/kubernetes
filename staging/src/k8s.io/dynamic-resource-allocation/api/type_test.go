/*
Copyright 2025 The Kubernetes Authors.

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

package api

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

var slice = ResourceSlice{
	TypeMeta: metav1.TypeMeta{
		Kind: "ResourceSlice",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name: "slice",
	},
	Spec: ResourceSliceSpec{
		Driver: MakeUniqueString("driver-name"),
		Devices: []Device{{
			Name: MakeUniqueString("device-name"),
		}},
	},
}

func TestKlog(t *testing.T) {
	t.Logf("slice:\n%s", klog.Format(slice))
}
