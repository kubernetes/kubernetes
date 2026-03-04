/*
Copyright 2024 The Kubernetes Authors.

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

package kubelet

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	testingclock "k8s.io/utils/clock/testing"

	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/nodestatus"
	serverstats "k8s.io/kubernetes/pkg/kubelet/server/stats"
)

type fakeResourceAnalyzer struct {
	serverstats.ResourceAnalyzer
	summary *statsapi.Summary
	err     error
}

func (f *fakeResourceAnalyzer) Get(ctx context.Context, force bool) (*statsapi.Summary, error) {
	return f.summary, f.err
}

func TestGetSystemMemoryPSI(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PSINodeCondition, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletPSI, true)

	cases := []struct {
		desc      string
		threshold float64
		avg10     float64
		avg60     float64
		err       error
		expectErr bool
		expectNil bool
	}{
		{"below threshold", 0.05, 4.0, 4.0, nil, false, false},
		{"above threshold avg10", 0.05, 6.0, 4.0, nil, false, false},
		{"above threshold avg60", 0.05, 4.0, 6.0, nil, false, false},
		{"zero threshold", 0.0, 100.0, 100.0, nil, false, true},
		{"error from analyzer", 0.05, 0, 0, fmt.Errorf("analyzer error"), true, true},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			kubelet := &Kubelet{
				kubeletConfiguration: kubeletconfiginternal.KubeletConfiguration{
					SystemMemoryContentionThreshold: tc.threshold,
				},
				recorder: &record.FakeRecorder{},
				clock:    testingclock.NewFakeClock(time.Now()),
			}

			analyzer := &fakeResourceAnalyzer{
				err: tc.err,
			}
			if tc.err == nil && tc.threshold > 0.0 {
				analyzer.summary = &statsapi.Summary{
					Node: statsapi.NodeStats{
						Memory: &statsapi.MemoryStats{
							PSI: &statsapi.PSIStats{
								Full: statsapi.PSIData{Avg10: tc.avg10, Avg60: tc.avg60},
							},
						},
					},
				}
			}
			kubelet.resourceAnalyzer = analyzer

			ctx := context.Background()
			actual, err := kubelet.getSystemMemoryPSI(ctx)

			if tc.expectErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			if tc.expectNil {
				require.Nil(t, actual)
			} else {
				require.NotNil(t, actual)
				require.InEpsilon(t, tc.avg10, actual.Avg10, 0.0001)
				require.InEpsilon(t, tc.avg60, actual.Avg60, 0.0001)
			}
		})
	}
}

func TestGetSystemDiskPSI(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PSINodeCondition, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletPSI, true)

	cases := []struct {
		desc      string
		threshold float64
		avg10     float64
		avg60     float64
		err       error
		expectErr bool
		expectNil bool
	}{
		{"below threshold", 0.05, 4.0, 4.0, nil, false, false},
		{"above threshold avg10", 0.05, 6.0, 4.0, nil, false, false},
		{"above threshold avg60", 0.05, 4.0, 6.0, nil, false, false},
		{"zero threshold", 0.0, 100.0, 100.0, nil, false, true},
		{"error from analyzer", 0.05, 0, 0, fmt.Errorf("analyzer error"), true, true},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			kubelet := &Kubelet{
				kubeletConfiguration: kubeletconfiginternal.KubeletConfiguration{
					SystemDiskContentionThreshold: tc.threshold,
				},
				recorder: &record.FakeRecorder{},
				clock:    testingclock.NewFakeClock(time.Now()),
			}

			analyzer := &fakeResourceAnalyzer{
				err: tc.err,
			}
			if tc.err == nil && tc.threshold > 0.0 {
				analyzer.summary = &statsapi.Summary{
					Node: statsapi.NodeStats{
						IO: &statsapi.IOStats{
							PSI: &statsapi.PSIStats{
								Full: statsapi.PSIData{Avg10: tc.avg10, Avg60: tc.avg60},
							},
						},
					},
				}
			}
			kubelet.resourceAnalyzer = analyzer

			ctx := context.Background()
			actual, err := kubelet.getSystemDiskPSI(ctx)

			if tc.expectErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			if tc.expectNil {
				require.Nil(t, actual)
			} else {
				require.NotNil(t, actual)
				require.InEpsilon(t, tc.avg10, actual.Avg10, 0.0001)
				require.InEpsilon(t, tc.avg60, actual.Avg60, 0.0001)
			}
		})
	}
}

func TestGetKubepodsMemoryPSI(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PSINodeCondition, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletPSI, true)

	cases := []struct {
		desc      string
		threshold float64
		avg10     float64
		avg60     float64
		err       error
		expectErr bool
		expectNil bool
	}{
		{"below threshold", 0.05, 4.0, 4.0, nil, false, false},
		{"above threshold avg10", 0.05, 6.0, 4.0, nil, false, false},
		{"above threshold avg60", 0.05, 4.0, 6.0, nil, false, false},
		{"zero threshold", 0.0, 100.0, 100.0, nil, false, true},
		{"error from analyzer", 0.05, 0, 0, fmt.Errorf("analyzer error"), true, true},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			kubelet := &Kubelet{
				kubeletConfiguration: kubeletconfiginternal.KubeletConfiguration{
					KubepodsMemoryContentionThreshold: tc.threshold,
				},
				recorder: &record.FakeRecorder{},
				clock:    testingclock.NewFakeClock(time.Now()),
			}

			analyzer := &fakeResourceAnalyzer{
				err: tc.err,
			}
			if tc.err == nil && tc.threshold > 0.0 {
				analyzer.summary = &statsapi.Summary{
					Node: statsapi.NodeStats{
						SystemContainers: []statsapi.ContainerStats{
							{
								Name: statsapi.SystemContainerPods,
								Memory: &statsapi.MemoryStats{
									PSI: &statsapi.PSIStats{
										Full: statsapi.PSIData{Avg10: tc.avg10, Avg60: tc.avg60},
									},
								},
							},
						},
					},
				}
			}
			kubelet.resourceAnalyzer = analyzer

			ctx := context.Background()
			actual, err := kubelet.getKubepodsMemoryPSI(ctx)

			if tc.expectErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			if tc.expectNil {
				require.Nil(t, actual)
			} else {
				require.NotNil(t, actual)
				require.InEpsilon(t, tc.avg10, actual.Avg10, 0.0001)
				require.InEpsilon(t, tc.avg60, actual.Avg60, 0.0001)
			}
		})
	}
}

func TestGetKubepodsDiskPSI(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PSINodeCondition, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletPSI, true)

	cases := []struct {
		desc      string
		threshold float64
		avg10     float64
		avg60     float64
		err       error
		expectErr bool
		expectNil bool
	}{
		{"below threshold", 0.05, 4.0, 4.0, nil, false, false},
		{"above threshold avg10", 0.05, 6.0, 4.0, nil, false, false},
		{"above threshold avg60", 0.05, 4.0, 6.0, nil, false, false},
		{"zero threshold", 0.0, 100.0, 100.0, nil, false, true},
		{"error from analyzer", 0.05, 0, 0, fmt.Errorf("analyzer error"), true, true},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			kubelet := &Kubelet{
				kubeletConfiguration: kubeletconfiginternal.KubeletConfiguration{
					KubepodsDiskContentionThreshold: tc.threshold,
				},
				recorder: &record.FakeRecorder{},
				clock:    testingclock.NewFakeClock(time.Now()),
			}

			analyzer := &fakeResourceAnalyzer{
				err: tc.err,
			}
			if tc.err == nil && tc.threshold > 0.0 {
				analyzer.summary = &statsapi.Summary{
					Node: statsapi.NodeStats{
						SystemContainers: []statsapi.ContainerStats{
							{
								Name: statsapi.SystemContainerPods,
								IO: &statsapi.IOStats{
									PSI: &statsapi.PSIStats{
										Full: statsapi.PSIData{Avg10: tc.avg10, Avg60: tc.avg60},
									},
								},
							},
						},
					},
				}
			}
			kubelet.resourceAnalyzer = analyzer

			ctx := context.Background()
			actual, err := kubelet.getKubepodsDiskPSI(ctx)

			if tc.expectErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			if tc.expectNil {
				require.Nil(t, actual)
			} else {
				require.NotNil(t, actual)
				require.InEpsilon(t, tc.avg10, actual.Avg10, 0.0001)
				require.InEpsilon(t, tc.avg60, actual.Avg60, 0.0001)
			}
		})
	}
}

func TestNodeConditionFromSummaryAPI(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PSINodeCondition, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.KubeletPSI, true)

	kubelet := &Kubelet{
		kubeletConfiguration: kubeletconfiginternal.KubeletConfiguration{
			SystemMemoryContentionThreshold:   0.9,
			SystemDiskContentionThreshold:     0.9,
			KubepodsMemoryContentionThreshold: 0.9,
			KubepodsDiskContentionThreshold:   0.9,
		},
		recorder: &record.FakeRecorder{},
		clock:    testingclock.NewFakeClock(time.Now()),
	}

	analyzer := &fakeResourceAnalyzer{
		summary: &statsapi.Summary{
			Node: statsapi.NodeStats{
				Memory: &statsapi.MemoryStats{
					PSI: &statsapi.PSIStats{
						Full: statsapi.PSIData{Avg10: 100, Avg60: 100},
					},
				},
				IO: &statsapi.IOStats{
					PSI: &statsapi.PSIStats{
						Full: statsapi.PSIData{Avg10: 100, Avg60: 100},
					},
				},
				SystemContainers: []statsapi.ContainerStats{
					{
						Name: statsapi.SystemContainerPods,
						Memory: &statsapi.MemoryStats{
							PSI: &statsapi.PSIStats{
								Full: statsapi.PSIData{Avg10: 100, Avg60: 100},
							},
						},
						IO: &statsapi.IOStats{
							PSI: &statsapi.PSIStats{
								Full: statsapi.PSIData{Avg10: 100, Avg60: 100},
							},
						},
					},
				},
			},
		},
	}
	kubelet.resourceAnalyzer = analyzer

	// Manually register only the PSI condition setters.
	// We cannot use kubelet.defaultNodeStatusFuncs() because it registers numerous other
	// setters (like nodestatus.MachineInfo, nodestatus.VersionInfo, etc.) that depend on
	// kl.containerManager, kl.cadvisor, and kl.imageManager, which are intentionally left
	// as nil in this mock &Kubelet{} to isolate the testing of the PSI node condition logic.
	kubelet.setNodeStatusFuncs = []func(context.Context, *v1.Node) error{
		nodestatus.PSICondition(kubelet.clock.Now, v1.NodeSystemMemoryContentionPressure, kubelet.getSystemMemoryPSI, kubelet.kubeletConfiguration.SystemMemoryContentionThreshold, kubelet.recordNodeStatusEvent),
		nodestatus.PSICondition(kubelet.clock.Now, v1.NodeSystemDiskContentionPressure, kubelet.getSystemDiskPSI, kubelet.kubeletConfiguration.SystemDiskContentionThreshold, kubelet.recordNodeStatusEvent),
		nodestatus.PSICondition(kubelet.clock.Now, v1.NodeKubepodsMemoryContentionPressure, kubelet.getKubepodsMemoryPSI, kubelet.kubeletConfiguration.KubepodsMemoryContentionThreshold, kubelet.recordNodeStatusEvent),
		nodestatus.PSICondition(kubelet.clock.Now, v1.NodeKubepodsDiskContentionPressure, kubelet.getKubepodsDiskPSI, kubelet.kubeletConfiguration.KubepodsDiskContentionThreshold, kubelet.recordNodeStatusEvent),
	}

	ctx := context.Background()
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
	}

	kubelet.setNodeStatus(ctx, node)

	foundSystemMemory := false
	foundSystemDisk := false
	foundKubepodsMemory := false
	foundKubepodsDisk := false

	for _, condition := range node.Status.Conditions {
		if condition.Type == v1.NodeSystemMemoryContentionPressure && condition.Status == v1.ConditionTrue {
			foundSystemMemory = true
		}
		if condition.Type == v1.NodeSystemDiskContentionPressure && condition.Status == v1.ConditionTrue {
			foundSystemDisk = true
		}
		if condition.Type == v1.NodeKubepodsMemoryContentionPressure && condition.Status == v1.ConditionTrue {
			foundKubepodsMemory = true
		}
		if condition.Type == v1.NodeKubepodsDiskContentionPressure && condition.Status == v1.ConditionTrue {
			foundKubepodsDisk = true
		}
	}

	require.True(t, foundSystemMemory, "Expected NodeSystemMemoryContentionPressure to be True")
	require.True(t, foundSystemDisk, "Expected NodeSystemDiskContentionPressure to be True")
	require.True(t, foundKubepodsMemory, "Expected NodeKubepodsMemoryContentionPressure to be True")
	require.True(t, foundKubepodsDisk, "Expected NodeKubepodsDiskContentionPressure to be True")
}
