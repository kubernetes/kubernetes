/*
Copyright 2019 The Kubernetes Authors.

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

package config

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	componentconfigtesting "k8s.io/component-base/config/testing"
	logsapi "k8s.io/component-base/logs/api/v1"
)

func TestComponentConfigSetup(t *testing.T) {
	pkginfo := &componentconfigtesting.ComponentConfigPackage{
		ComponentName:      "kube-proxy",
		GroupName:          GroupName,
		SchemeGroupVersion: SchemeGroupVersion,
		AddToScheme:        AddToScheme,
		AllowedTags: map[reflect.Type]bool{
			reflect.TypeOf(metav1.TypeMeta{}):              true,
			reflect.TypeOf(metav1.Duration{}):              true,
			reflect.TypeOf(logsapi.LoggingConfiguration{}): true,
		},
	}

	if err := componentconfigtesting.VerifyInternalTypePackage(pkginfo); err != nil {
		t.Errorf("failed TestComponentConfigSetup for kube-proxy: %v", err)
	}
}
