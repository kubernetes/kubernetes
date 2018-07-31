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

package validation

import (
	"testing"

	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

func TestValidateKubeSchedulerConfiguration(t *testing.T) {
	type args struct {
		ksc *componentconfig.KubeSchedulerConfiguration
	}
	testCases := []struct {
		name   string
		args   args
		errLen int
	}{
		{
			name: "valid case",
			args: args{
				&componentconfig.KubeSchedulerConfiguration{
					HardPodAffinitySymmetricWeight: 40,
					HealthzBindAddress:             "127.0.0.1:10251",
					MetricsBindAddress:             "127.0.0.1:10252",
				},
			},
			errLen: 0,
		},
		{
			name: "invalid case with 3 error",
			args: args{
				&componentconfig.KubeSchedulerConfiguration{
					HardPodAffinitySymmetricWeight: -1,
					HealthzBindAddress:             "1.2.3.4.5:40",
					MetricsBindAddress:             "1.2.3.4:50-3",
				},
			},
			errLen: 3,
		},
	}
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateKubeSchedulerConfiguration(tt.args.ksc); len(err) != tt.errLen {
				t.Errorf("expected %d error, got %d error", tt.errLen, len(err))
			}
		})
	}
}
