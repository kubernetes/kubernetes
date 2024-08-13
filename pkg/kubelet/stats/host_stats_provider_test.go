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

package stats

import (
	"reflect"
	"testing"

	cadvisorapiv2 "github.com/google/cadvisor/info/v2"

	kubecontainertest "k8s.io/kubernetes/pkg/kubelet/container/testing"

	"k8s.io/apimachinery/pkg/types"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
)

func Test_hostStatsProvider_getPodEtcHostsStats(t *testing.T) {
	tests := []struct {
		name                string
		podEtcHostsPathFunc PodEtcHostsPathFunc
		podUID              types.UID
		rootFsInfo          *cadvisorapiv2.FsInfo
		want                *statsapi.FsStats
		wantErr             bool
	}{
		{
			name: "Should return nil for runtimes that do not support etc host file",
			podEtcHostsPathFunc: func(podUID types.UID) string {
				return ""
			},
			podUID:     "fake0001",
			rootFsInfo: nil,
			want:       nil,
			wantErr:    false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := hostStatsProvider{
				osInterface:         &kubecontainertest.FakeOS{},
				podEtcHostsPathFunc: tt.podEtcHostsPathFunc,
			}
			got, err := h.getPodEtcHostsStats(tt.podUID, tt.rootFsInfo)
			if (err != nil) != tt.wantErr {
				t.Errorf("getPodEtcHostsStats() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getPodEtcHostsStats() got = %v, want %v", got, tt.want)
			}
		})
	}
}
