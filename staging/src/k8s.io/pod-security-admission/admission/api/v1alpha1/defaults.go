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

package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/pod-security-admission/api"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_PodSecurityDefaults(obj *PodSecurityDefaults) {
	if len(obj.Enforce) == 0 {
		obj.Enforce = string(api.LevelPrivileged)
	}
	if len(obj.Warn) == 0 {
		obj.Warn = string(api.LevelPrivileged)
	}
	if len(obj.Audit) == 0 {
		obj.Audit = string(api.LevelPrivileged)
	}

	if len(obj.EnforceVersion) == 0 {
		obj.EnforceVersion = string(api.VersionLatest)
	}
	if len(obj.WarnVersion) == 0 {
		obj.WarnVersion = string(api.VersionLatest)
	}
	if len(obj.AuditVersion) == 0 {
		obj.AuditVersion = string(api.VersionLatest)
	}
}
