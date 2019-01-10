/*
Copyright 2018 The Kubernetes Authors.

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

package v1alpha1

import (
	"reflect"
	"testing"

	componentconfigtesting "k8s.io/apimachinery/pkg/apis/config/testing"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestComponentConfigSetup(t *testing.T) {
	pkginfo := &componentconfigtesting.ComponentConfigPackage{
		ComponentName:      "kube-proxy",
		GroupName:          GroupName,
		SchemeGroupVersion: SchemeGroupVersion,
		AddToScheme:        AddToScheme,
		AllowedNonstandardJSONNames: map[reflect.Type]sets.String{
			reflect.TypeOf(KubeProxyConfiguration{}): sets.NewString(
				"iptables",
			),
		},
	}
	if err := componentconfigtesting.VerifyExternalTypePackage(pkginfo); err != nil {
		t.Errorf("failed TestComponentConfigSetup: %v", err)
	}
}
