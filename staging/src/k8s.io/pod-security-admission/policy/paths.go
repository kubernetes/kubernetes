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

package policy

import "k8s.io/apimachinery/pkg/util/validation/field"

var (
	annotationsPath            = field.NewPath("metadata", "annotations")
	specPath                   = field.NewPath("spec")
	initContainersFldPath      = specPath.Child("initContainers")
	containersFldPath          = specPath.Child("containers")
	ephemeralContainersFldPath = specPath.Child("ephemeralContainers")
	securityContextPath        = specPath.Child("securityContext")
	hostNetworkPath            = specPath.Child("hostNetwork")
	hostPIDPath                = specPath.Child("hostPID")
	hostIPCPath                = specPath.Child("hostIPC")
	volumesPath                = specPath.Child("volumes")
	runAsNonRootPath           = securityContextPath.Child("runAsNonRoot")
	runAsUserPath              = securityContextPath.Child("runAsUser")
	seccompProfileTypePath     = securityContextPath.Child("seccompProfile", "type")
	seLinuxOptionsTypePath     = securityContextPath.Child("seLinuxOptions", "type")
	seLinuxOptionsUserPath     = securityContextPath.Child("seLinuxOptions", "user")
	seLinuxOptionsRolePath     = securityContextPath.Child("seLinuxOptions", "role")
	sysctlsPath                = securityContextPath.Child("sysctls")
	hostProcessPath            = securityContextPath.Child("windowsOptions", "hostProcess")
	appArmorProfileTypePath    = securityContextPath.Child("appArmorProfile", "type")
)
