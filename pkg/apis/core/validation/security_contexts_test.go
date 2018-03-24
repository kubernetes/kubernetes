/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/capabilities"
)

func TestValidateSecurityContext(t *testing.T) {
	runAsUser := int64(1)
	fullValidSC := func() *core.SecurityContext {
		return &core.SecurityContext{
			Privileged: boolPtr(false),
			Capabilities: &core.Capabilities{
				Add:  []core.Capability{"foo"},
				Drop: []core.Capability{"bar"},
			},
			SELinuxOptions: &core.SELinuxOptions{
				User:  "user",
				Role:  "role",
				Type:  "type",
				Level: "level",
			},
			RunAsUser: &runAsUser,
		}
	}

	//setup data
	allSettings := fullValidSC()
	noCaps := fullValidSC()
	noCaps.Capabilities = nil

	noSELinux := fullValidSC()
	noSELinux.SELinuxOptions = nil

	noPrivRequest := fullValidSC()
	noPrivRequest.Privileged = nil

	noRunAsUser := fullValidSC()
	noRunAsUser.RunAsUser = nil

	successCases := map[string]struct {
		sc *core.SecurityContext
	}{
		"all settings":    {allSettings},
		"no capabilities": {noCaps},
		"no selinux":      {noSELinux},
		"no priv request": {noPrivRequest},
		"no run as user":  {noRunAsUser},
	}
	for k, v := range successCases {
		if errs := ValidateSecurityContext(v.sc, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("[%s] Expected success, got %v", k, errs)
		}
	}

	privRequestWithGlobalDeny := fullValidSC()
	privRequestWithGlobalDeny.Privileged = boolPtr(true)

	negativeRunAsUser := fullValidSC()
	negativeUser := int64(-1)
	negativeRunAsUser.RunAsUser = &negativeUser

	privWithoutEscalation := fullValidSC()
	privWithoutEscalation.Privileged = boolPtr(true)
	privWithoutEscalation.AllowPrivilegeEscalation = boolPtr(false)

	capSysAdminWithoutEscalation := fullValidSC()
	capSysAdminWithoutEscalation.Capabilities.Add = []core.Capability{"CAP_SYS_ADMIN"}
	capSysAdminWithoutEscalation.AllowPrivilegeEscalation = boolPtr(false)

	errorCases := map[string]struct {
		sc           *core.SecurityContext
		errorType    field.ErrorType
		errorDetail  string
		capAllowPriv bool
	}{
		"request privileged when capabilities forbids": {
			sc:          privRequestWithGlobalDeny,
			errorType:   "FieldValueForbidden",
			errorDetail: "disallowed by cluster policy",
		},
		"negative RunAsUser": {
			sc:          negativeRunAsUser,
			errorType:   "FieldValueInvalid",
			errorDetail: "must be between",
		},
		"with CAP_SYS_ADMIN and allowPrivilegeEscalation false": {
			sc:          capSysAdminWithoutEscalation,
			errorType:   "FieldValueInvalid",
			errorDetail: "cannot set `allowPrivilegeEscalation` to false and `capabilities.Add` CAP_SYS_ADMIN",
		},
		"with privileged and allowPrivilegeEscalation false": {
			sc:           privWithoutEscalation,
			errorType:    "FieldValueInvalid",
			errorDetail:  "cannot set `allowPrivilegeEscalation` to false and `privileged` to true",
			capAllowPriv: true,
		},
	}
	for k, v := range errorCases {
		capabilities.SetForTests(capabilities.Capabilities{
			AllowPrivileged: v.capAllowPriv,
		})
		if errs := ValidateSecurityContext(v.sc, field.NewPath("field")); len(errs) == 0 || errs[0].Type != v.errorType || !strings.Contains(errs[0].Detail, v.errorDetail) {
			t.Errorf("[%s] Expected error type %q with detail %q, got %v", k, v.errorType, v.errorDetail, errs)
		}
	}
}

func boolPtr(b bool) *bool {
	return &b
}
