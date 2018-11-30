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

package policy

import (
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/api/auditregistration/v1alpha1"
	"k8s.io/apiserver/pkg/apis/audit"
)

func TestConvertDynamicPolicyToInternal(t *testing.T) {
	for _, test := range []struct {
		desc     string
		dynamic  *v1alpha1.Policy
		internal *audit.Policy
	}{
		{
			desc: "should convert full",
			dynamic: &v1alpha1.Policy{
				Level: v1alpha1.LevelMetadata,
				Stages: []v1alpha1.Stage{
					v1alpha1.StageResponseComplete,
				},
			},
			internal: &audit.Policy{
				Rules: []audit.PolicyRule{
					{
						Level: audit.LevelMetadata,
					},
				},
				OmitStages: []audit.Stage{
					audit.StageRequestReceived,
					audit.StageResponseStarted,
					audit.StagePanic,
				},
			},
		},
		{
			desc: "should convert missing stages",
			dynamic: &v1alpha1.Policy{
				Level: v1alpha1.LevelMetadata,
			},
			internal: &audit.Policy{
				Rules: []audit.PolicyRule{
					{
						Level: audit.LevelMetadata,
					},
				},
				OmitStages: []audit.Stage{
					audit.StageRequestReceived,
					audit.StageResponseStarted,
					audit.StageResponseComplete,
					audit.StagePanic,
				},
			},
		},
	} {
		t.Run(test.desc, func(t *testing.T) {
			d := ConvertDynamicPolicyToInternal(test.dynamic)
			require.ElementsMatch(t, test.internal.OmitStages, d.OmitStages)
			require.Equal(t, test.internal.Rules, d.Rules)
		})
	}
}
