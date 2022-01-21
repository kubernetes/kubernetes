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
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
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
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
