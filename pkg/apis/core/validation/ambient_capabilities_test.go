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

package validation

import (
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

// Apart from the prohibited ambient grants, name validation follows Add.
// Unknown names are left to the runtime.
func TestValidateAmbientCapabilities(t *testing.T) {
	for _, tc := range []struct {
		name       string
		caps, drop []core.Capability
	}{
		{name: "individual", caps: []core.Capability{"NET_BIND_SERVICE"}},
		{name: "reset and add", caps: []core.Capability{"ALL", "NET_BIND_SERVICE"}, drop: []core.Capability{"ALL"}},
		{name: "empty name", caps: []core.Capability{""}},
		{name: "lowercase", caps: []core.Capability{"net_bind_service"}},
		{name: "unknown", caps: []core.Capability{"UNKNOWN"}},
		{name: "CAP prefix", caps: []core.Capability{"CAP_NET_BIND_SERVICE"}},
		{name: "duplicates", caps: []core.Capability{"CHOWN", "CHOWN"}},
		{name: "individual drop", caps: []core.Capability{"CHOWN"}, drop: []core.Capability{"CHOWN"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, allowEscalation := range []bool{true, false} {
				ordinary := &core.SecurityContext{Capabilities: &core.Capabilities{Add: tc.caps, Drop: tc.drop}, AllowPrivilegeEscalation: ptr.To(allowEscalation)}
				ambient := &core.SecurityContext{Capabilities: &core.Capabilities{Ambient: tc.caps, Drop: tc.drop}, AllowPrivilegeEscalation: ptr.To(allowEscalation)}
				ordinaryErrors := ValidateSecurityContext(ordinary, field.NewPath("securityContext"), true, false, false)
				ambientErrors := ValidateSecurityContext(ambient, field.NewPath("securityContext"), true, false, false)
				require.Len(t, ambientErrors, len(ordinaryErrors), "allowPrivilegeEscalation=%v: ordinary=%v ambient=%v", allowEscalation, ordinaryErrors, ambientErrors)
			}
		})
	}
}

func TestValidateAmbientCapabilitiesRestrictions(t *testing.T) {
	for _, tc := range []struct {
		name          string
		ambient, drop []core.Capability
		forbidden     []string
	}{
		{"sys admin", []core.Capability{"SYS_ADMIN"}, nil, []string{"SYS_ADMIN"}},
		{"dac override", []core.Capability{"DAC_OVERRIDE"}, nil, []string{"DAC_OVERRIDE"}},
		{"lowercase sys admin", []core.Capability{"sys_admin"}, nil, []string{"SYS_ADMIN"}},
		{"lowercase dac override", []core.Capability{"dac_override"}, nil, []string{"DAC_OVERRIDE"}},
		{"both", []core.Capability{"SYS_ADMIN", "DAC_OVERRIDE"}, nil, []string{"SYS_ADMIN", "DAC_OVERRIDE"}},
		{"all", []core.Capability{"ALL"}, nil, []string{"SYS_ADMIN", "DAC_OVERRIDE"}},
		{"lowercase all", []core.Capability{"all"}, nil, []string{"SYS_ADMIN", "DAC_OVERRIDE"}},
		{"drop sys admin", []core.Capability{"SYS_ADMIN"}, []core.Capability{"sys_admin"}, nil},
		{"drop dac override", []core.Capability{"DAC_OVERRIDE"}, []core.Capability{"dac_override"}, nil},
		{"all minus sys admin", []core.Capability{"ALL"}, []core.Capability{"SYS_ADMIN"}, []string{"DAC_OVERRIDE"}},
		{"all minus dac override", []core.Capability{"ALL"}, []core.Capability{"DAC_OVERRIDE"}, []string{"SYS_ADMIN"}},
		{"all minus both", []core.Capability{"ALL"}, []core.Capability{"sys_admin", "dac_override"}, nil},
		{"all reset", []core.Capability{"ALL"}, []core.Capability{"all"}, nil},
		{"explicit after reset", []core.Capability{"ALL", "SYS_ADMIN", "DAC_OVERRIDE"}, []core.Capability{"ALL"}, []string{"SYS_ADMIN", "DAC_OVERRIDE"}},
		{"explicit dropped after reset", []core.Capability{"ALL", "SYS_ADMIN", "DAC_OVERRIDE"}, []core.Capability{"ALL", "SYS_ADMIN", "DAC_OVERRIDE"}, nil},
		{"unrelated capability", []core.Capability{"NET_BIND_SERVICE"}, []core.Capability{"ALL"}, nil},
		{"empty", nil, nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, allowEscalation := range []*bool{nil, ptr.To(true), ptr.To(false)} {
				sc := &core.SecurityContext{Capabilities: &core.Capabilities{Ambient: tc.ambient, Drop: tc.drop}, AllowPrivilegeEscalation: allowEscalation}
				errs := ValidateSecurityContext(sc, field.NewPath("securityContext"), true, false, false)
				require.Len(t, errs, len(tc.forbidden))
				for i, capability := range tc.forbidden {
					require.Equal(t, "securityContext.capabilities.ambient", errs[i].Field)
					require.Contains(t, errs[i].Detail, capability)
				}
			}
		})
	}
}

func TestAmbientRestrictionsPreserveOrdinaryCapabilities(t *testing.T) {
	for _, allowEscalation := range []*bool{nil, ptr.To(true), ptr.To(false)} {
		sc := &core.SecurityContext{
			Capabilities:             &core.Capabilities{Add: []core.Capability{"SYS_ADMIN", "DAC_OVERRIDE"}},
			AllowPrivilegeEscalation: allowEscalation,
		}
		require.Empty(t, ValidateSecurityContext(sc, field.NewPath("securityContext"), true, false, false))
	}
}
