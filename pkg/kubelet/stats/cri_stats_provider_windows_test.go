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
	"time"

	"github.com/Microsoft/hcsshim"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	kubecontainertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/kuberuntime"
	"k8s.io/kubernetes/pkg/volume"
	testingclock "k8s.io/utils/clock/testing"
)

type fakeNetworkStatsProvider struct {
	containers []containerStats
}

type containerStats struct {
	container hcsshim.ContainerProperties
	hcsStats  []hcsshim.NetworkStats
}

func (s fakeNetworkStatsProvider) GetHNSEndpointStats(endpointName string) (*hcsshim.HNSEndpointStats, error) {
	eps := hcsshim.HNSEndpointStats{}
	for _, c := range s.containers {
		for _, stat := range c.hcsStats {
			if endpointName == stat.InstanceId {
				eps = hcsshim.HNSEndpointStats{
					EndpointID:      stat.EndpointId,
					BytesSent:       stat.BytesSent,
					BytesReceived:   stat.BytesReceived,
					PacketsReceived: stat.PacketsReceived,
					PacketsSent:     stat.PacketsSent,
				}
			}
		}
	}

	return &eps, nil
}

func (s fakeNetworkStatsProvider) HNSListEndpointRequest() ([]hcsshim.HNSEndpoint, error) {
	uniqueEndpoints := map[string]*hcsshim.HNSEndpoint{}

	for _, c := range s.containers {
		for _, stat := range c.hcsStats {
			e, found := uniqueEndpoints[stat.EndpointId]
			if found {
				// add the container
				e.SharedContainers = append(e.SharedContainers, c.container.ID)
				continue
			}

			uniqueEndpoints[stat.EndpointId] = &hcsshim.HNSEndpoint{
				Name:             stat.EndpointId,
				Id:               stat.EndpointId,
				SharedContainers: []string{c.container.ID},
			}
		}
	}

	eps := []hcsshim.HNSEndpoint{}
	for _, ep := range uniqueEndpoints {
		eps = append(eps, *ep)
	}

	return eps, nil
}

func Test_criStatsProvider_listContainerNetworkStats(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Time{})
	tests := []struct {
		name    string
		fields  fakeNetworkStatsProvider
		want    map[string]*statsapi.NetworkStats
		wantErr bool
		skipped bool
	}{
		{
			name: "basic example",
			fields: fakeNetworkStatsProvider{
				containers: []containerStats{
					{
						container: hcsshim.ContainerProperties{
							ID: "c1",
						}, hcsStats: []hcsshim.NetworkStats{
							{
								BytesReceived: 1,
								BytesSent:     10,
								EndpointId:    "test",
								InstanceId:    "test",
							},
						},
					},
					{
						container: hcsshim.ContainerProperties{
							ID: "c2",
						}, hcsStats: []hcsshim.NetworkStats{
							{
								BytesReceived: 2,
								BytesSent:     20,
								EndpointId:    "test2",
								InstanceId:    "test2",
							},
						},
					},
				},
			},
			want: map[string]*statsapi.NetworkStats{
				"c1": {
					Time: v1.NewTime(fakeClock.Now()),
					InterfaceStats: statsapi.InterfaceStats{
						Name:    "test",
						RxBytes: toP(1),
						TxBytes: toP(10),
					},
					Interfaces: []statsapi.InterfaceStats{
						{
							Name:    "test",
							RxBytes: toP(1),

							TxBytes: toP(10),
						},
					},
				},
				"c2": {
					Time: v1.Time{},
					InterfaceStats: statsapi.InterfaceStats{
						Name:    "test2",
						RxBytes: toP(2),
						TxBytes: toP(20),
					},
					Interfaces: []statsapi.InterfaceStats{
						{
							Name:    "test2",
							RxBytes: toP(2),
							TxBytes: toP(20),
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "multiple containers same endpoint",
			fields: fakeNetworkStatsProvider{
				containers: []containerStats{
					{
						container: hcsshim.ContainerProperties{
							ID: "c1",
						}, hcsStats: []hcsshim.NetworkStats{
							{
								BytesReceived: 1,
								BytesSent:     10,
								EndpointId:    "test",
								InstanceId:    "test",
							},
						},
					},
					{
						container: hcsshim.ContainerProperties{
							ID: "c2",
						}, hcsStats: []hcsshim.NetworkStats{
							{
								BytesReceived: 2,
								BytesSent:     20,
								EndpointId:    "test2",
								InstanceId:    "test2",
							},
						},
					},
					{
						container: hcsshim.ContainerProperties{
							ID: "c3",
						}, hcsStats: []hcsshim.NetworkStats{
							{
								BytesReceived: 3,
								BytesSent:     30,
								EndpointId:    "test2",
								InstanceId:    "test3",
							},
						},
					},
				},
			},
			want: map[string]*statsapi.NetworkStats{
				"c1": {
					Time: v1.NewTime(fakeClock.Now()),
					InterfaceStats: statsapi.InterfaceStats{
						Name:    "test",
						RxBytes: toP(1),
						TxBytes: toP(10),
					},
					Interfaces: []statsapi.InterfaceStats{
						{
							Name:    "test",
							RxBytes: toP(1),

							TxBytes: toP(10),
						},
					},
				},
				"c2": {
					Time: v1.Time{},
					InterfaceStats: statsapi.InterfaceStats{
						Name:    "test2",
						RxBytes: toP(2),
						TxBytes: toP(20),
					},
					Interfaces: []statsapi.InterfaceStats{
						{
							Name:    "test2",
							RxBytes: toP(2),
							TxBytes: toP(20),
						},
					},
				},
				"c3": {
					Time: v1.Time{},
					InterfaceStats: statsapi.InterfaceStats{
						Name:    "test2",
						RxBytes: toP(2),
						TxBytes: toP(20),
					},
					Interfaces: []statsapi.InterfaceStats{
						{
							Name:    "test2",
							RxBytes: toP(2),
							TxBytes: toP(20),
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "multiple stats instances of same interface only picks up first",
			fields: fakeNetworkStatsProvider{
				containers: []containerStats{
					{
						container: hcsshim.ContainerProperties{
							ID: "c1",
						}, hcsStats: []hcsshim.NetworkStats{
							{
								BytesReceived: 1,
								BytesSent:     10,
								EndpointId:    "test",
								InstanceId:    "test",
							},
							{
								BytesReceived: 3,
								BytesSent:     30,
								EndpointId:    "test",
								InstanceId:    "test3",
							},
						},
					},
					{
						container: hcsshim.ContainerProperties{
							ID: "c2",
						}, hcsStats: []hcsshim.NetworkStats{
							{
								BytesReceived: 2,
								BytesSent:     20,
								EndpointId:    "test2",
								InstanceId:    "test2",
							},
						},
					},
				},
			},
			want: map[string]*statsapi.NetworkStats{
				"c1": {
					Time: v1.NewTime(fakeClock.Now()),
					InterfaceStats: statsapi.InterfaceStats{
						Name:    "test",
						RxBytes: toP(1),
						TxBytes: toP(10),
					},
					Interfaces: []statsapi.InterfaceStats{
						{
							Name:    "test",
							RxBytes: toP(1),

							TxBytes: toP(10),
						},
					},
				},
				"c2": {
					Time: v1.Time{},
					InterfaceStats: statsapi.InterfaceStats{
						Name:    "test2",
						RxBytes: toP(2),
						TxBytes: toP(20),
					},
					Interfaces: []statsapi.InterfaceStats{
						{
							Name:    "test2",
							RxBytes: toP(2),
							TxBytes: toP(20),
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "multiple endpoints per container",
			fields: fakeNetworkStatsProvider{
				containers: []containerStats{
					{
						container: hcsshim.ContainerProperties{
							ID: "c1",
						}, hcsStats: []hcsshim.NetworkStats{
							{
								BytesReceived: 1,
								BytesSent:     10,
								EndpointId:    "test",
								InstanceId:    "test",
							},
							{
								BytesReceived: 3,
								BytesSent:     30,
								EndpointId:    "test3",
								InstanceId:    "test3",
							},
						},
					},
					{
						container: hcsshim.ContainerProperties{
							ID: "c2",
						}, hcsStats: []hcsshim.NetworkStats{
							{
								BytesReceived: 2,
								BytesSent:     20,
								EndpointId:    "test2",
								InstanceId:    "test2",
							},
						},
					},
				},
			},
			want: map[string]*statsapi.NetworkStats{
				"c1": {
					Time: v1.NewTime(fakeClock.Now()),
					InterfaceStats: statsapi.InterfaceStats{
						Name:    "test",
						RxBytes: toP(1),
						TxBytes: toP(10),
					},
					Interfaces: []statsapi.InterfaceStats{
						{
							Name:    "test",
							RxBytes: toP(1),

							TxBytes: toP(10),
						},
						{
							Name:    "test3",
							RxBytes: toP(3),

							TxBytes: toP(30),
						},
					},
				},
				"c2": {
					Time: v1.Time{},
					InterfaceStats: statsapi.InterfaceStats{
						Name:    "test2",
						RxBytes: toP(2),
						TxBytes: toP(20),
					},
					Interfaces: []statsapi.InterfaceStats{
						{
							Name:    "test2",
							RxBytes: toP(2),
							TxBytes: toP(20),
						},
					},
				},
			},
			wantErr: false,
			skipped: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// TODO: Remove skip once https://github.com/kubernetes/kubernetes/issues/116692 is fixed.
			if tt.skipped {
				t.Skip("Test temporarily skipped.")
			}
			p := &criStatsProvider{
				windowsNetworkStatsProvider: fakeNetworkStatsProvider{
					containers: tt.fields.containers,
				},
				clock: fakeClock,
			}
			got, err := p.listContainerNetworkStats()
			if (err != nil) != tt.wantErr {
				t.Errorf("listContainerNetworkStats() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("listContainerNetworkStats() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func toP(i uint64) *uint64 {
	return &i
}

func Test_criStatsProvider_makeWinContainerStats(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Time{})
	containerStartTime := fakeClock.Now()

	cpuUsageTimestamp := int64(555555)
	cpuUsageNanoSeconds := uint64(0x123456)
	cpuUsageNanoCores := uint64(0x4000)
	memoryUsageTimestamp := int64(666666)
	memoryUsageWorkingSetBytes := uint64(0x11223344)
	memoryUsageAvailableBytes := uint64(0x55667788)
	memoryUsagePageFaults := uint64(200)
	logStatsUsed := uint64(5000)
	logStatsInodesUsed := uint64(5050)

	// getPodContainerLogStats is called during makeWindowsContainerStats to populate ContainerStats.Logs
	c0LogStats := &volume.Metrics{
		Used:       resource.NewQuantity(int64(logStatsUsed), resource.BinarySI),
		InodesUsed: resource.NewQuantity(int64(logStatsInodesUsed), resource.BinarySI),
	}
	fakeStats := map[string]*volume.Metrics{
		kuberuntime.BuildContainerLogsDirectory(testPodLogDirectory, "sb0-ns", "sb0-name", types.UID("sb0-uid"), "c0"): c0LogStats,
	}
	fakeOS := &kubecontainertest.FakeOS{}
	fakeHostStatsProvider := NewFakeHostStatsProviderWithData(fakeStats, fakeOS)

	p := &criStatsProvider{
		clock:             fakeClock,
		hostStatsProvider: fakeHostStatsProvider,
	}

	inputStats := &runtimeapi.WindowsContainerStats{
		Attributes: &runtimeapi.ContainerAttributes{
			Metadata: &runtimeapi.ContainerMetadata{
				Name: "c0",
			},
		},
		Cpu: &runtimeapi.WindowsCpuUsage{
			Timestamp: cpuUsageTimestamp,
			UsageCoreNanoSeconds: &runtimeapi.UInt64Value{
				Value: cpuUsageNanoSeconds,
			},
			UsageNanoCores: &runtimeapi.UInt64Value{
				Value: cpuUsageNanoCores,
			},
		},
		Memory: &runtimeapi.WindowsMemoryUsage{
			Timestamp: memoryUsageTimestamp,
			AvailableBytes: &runtimeapi.UInt64Value{
				Value: memoryUsageAvailableBytes,
			},
			WorkingSetBytes: &runtimeapi.UInt64Value{
				Value: memoryUsageWorkingSetBytes,
			},
			PageFaults: &runtimeapi.UInt64Value{
				Value: memoryUsagePageFaults,
			},
		},
	}

	inputContainer := &runtimeapi.Container{
		CreatedAt: containerStartTime.Unix(),
		Metadata: &runtimeapi.ContainerMetadata{
			Name: "c0",
		},
	}

	inputRootFsInfo := &cadvisorapiv2.FsInfo{}

	// Used by the getPodContainerLogStats() call in makeWinContainerStats()
	inputPodSandboxMetadata := &runtimeapi.PodSandboxMetadata{
		Namespace: "sb0-ns",
		Name:      "sb0-name",
		Uid:       "sb0-uid",
	}

	got, err := p.makeWinContainerStats(inputStats, inputContainer, inputRootFsInfo, make(map[runtimeapi.FilesystemIdentifier]*cadvisorapiv2.FsInfo), inputPodSandboxMetadata)

	expected := &statsapi.ContainerStats{
		Name:      "c0",
		StartTime: v1.NewTime(time.Unix(0, containerStartTime.Unix())),
		CPU: &statsapi.CPUStats{
			Time:                 v1.NewTime(time.Unix(0, cpuUsageTimestamp)),
			UsageNanoCores:       toP(cpuUsageNanoCores),
			UsageCoreNanoSeconds: toP(cpuUsageNanoSeconds),
		},
		Memory: &statsapi.MemoryStats{
			Time:            v1.NewTime(time.Unix(0, memoryUsageTimestamp)),
			AvailableBytes:  toP(memoryUsageAvailableBytes),
			WorkingSetBytes: toP(memoryUsageWorkingSetBytes),
			PageFaults:      toP(memoryUsagePageFaults),
		},
		Rootfs: &statsapi.FsStats{},
		Logs: &statsapi.FsStats{
			Time:       c0LogStats.Time,
			UsedBytes:  toP(logStatsUsed),
			InodesUsed: toP(logStatsInodesUsed),
		},
	}

	if err != nil {
		t.Fatalf("makeWinContainerStats() error = %v, expected no error", err)
	}

	if !reflect.DeepEqual(got.CPU, expected.CPU) {
		t.Errorf("makeWinContainerStats() CPU = %v, expected %v", got.CPU, expected.CPU)
	}

	if !reflect.DeepEqual(got.Memory, expected.Memory) {
		t.Errorf("makeWinContainerStats() Memory = %v, want %v", got.Memory, expected.Memory)
	}

	if !reflect.DeepEqual(got.Rootfs, expected.Rootfs) {
		t.Errorf("makeWinContainerStats() Rootfs = %v, want %v", got.Rootfs, expected.Rootfs)
	}

	// Log stats contain pointers to calculated resource values so we cannot use DeepEqual here
	assert.Equal(t, *got.Logs.UsedBytes, logStatsUsed, "Logs.UsedBytes does not match expected value")
	assert.Equal(t, *got.Logs.InodesUsed, logStatsInodesUsed, "Logs.InodesUsed does not match expected value")
}
