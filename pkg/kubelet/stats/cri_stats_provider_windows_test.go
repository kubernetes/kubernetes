package stats

import (
	"reflect"
	"testing"
	"time"

	"github.com/Microsoft/hcsshim"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
)



type fakeConatiner struct {
	stat hcsshim.Statistics
}

func (f fakeConatiner) Start() error {
	return nil
}

func (f fakeConatiner) Shutdown() error {
	return nil
}

func (f fakeConatiner) Terminate() error {
	return nil
}

func (f fakeConatiner) Wait() error {
	return nil
}

func (f fakeConatiner) WaitTimeout(duration time.Duration) error {
	return nil
}

func (f fakeConatiner) Pause() error {
	return nil
}

func (f fakeConatiner) Resume() error {
	return nil
}

func (f fakeConatiner) HasPendingUpdates() (bool, error) {
	return false, nil
}

func (f fakeConatiner) Statistics() (hcsshim.Statistics, error) {
	return f.stat, nil
}

func (f fakeConatiner) ProcessList() ([]hcsshim.ProcessListItem, error) {
	return []hcsshim.ProcessListItem{}, nil
}

func (f fakeConatiner) MappedVirtualDisks() (map[int]hcsshim.MappedVirtualDiskController, error) {
	return map[int]hcsshim.MappedVirtualDiskController{}, nil
}

func (f fakeConatiner) CreateProcess(c *hcsshim.ProcessConfig) (hcsshim.Process, error) {
	return nil, nil
}

func (f fakeConatiner) OpenProcess(pid int) (hcsshim.Process, error) {
	return nil, nil
}

func (f fakeConatiner) Close() error {
	return nil
}

func (f fakeConatiner) Modify(config *hcsshim.ResourceModificationRequestResponse) error {
	return nil
}

func (s fakehcsshim) GetContainers(q hcsshim.ComputeSystemQuery) ([]hcsshim.ContainerProperties, error) {
	cp := []hcsshim.ContainerProperties{}
	for _, c := range s.containers  {
		cp = append(cp, c.container)
	}

	return cp, nil
}

func (s fakehcsshim) GetHNSEndpointByID(endpointID string) (*hcsshim.HNSEndpoint, error) {
	e := hcsshim.HNSEndpoint{
		Name: endpointID,
	}
	return &e, nil
}

func (s fakehcsshim) OpenContainer(id string) (hcsshim.Container, error) {
	fc := fakeConatiner{}
	for _, c := range s.containers {
		if c.container.ID == id {
			for _, s := range c.hcsStats{
				fc.stat.Network = append(fc.stat.Network, s)
			}
		}
	}

	return fc, nil
}

type fakehcsshim struct {
	containers       []containerStats
}

type containerStats struct {
	container       hcsshim.ContainerProperties
	hcsStats         []hcsshim.NetworkStats
}


func Test_criStatsProvider_listContainerNetworkStats(t *testing.T) {
	tests := []struct {
		name    string
		fields  fakehcsshim
		want    map[string]*statsapi.NetworkStats
		wantErr bool
	}{
		{
			name: "basic example",
			fields: fakehcsshim{
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
				"c1": &statsapi.NetworkStats{
					Time: v1.Time{},
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
				"c2": &statsapi.NetworkStats{
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
			fields: fakehcsshim{
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
				"c1": &statsapi.NetworkStats{
					Time: v1.Time{},
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
				"c2": &statsapi.NetworkStats{
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
				"c3": &statsapi.NetworkStats{
					Time: v1.Time{},
					InterfaceStats: statsapi.InterfaceStats{
						Name:    "test2",
						RxBytes: toP(3),
						TxBytes: toP(30),
					},
					Interfaces: []statsapi.InterfaceStats{
						{
							Name:    "test2",
							RxBytes: toP(3),
							TxBytes: toP(30),
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "multiple stats instances of same interface only picks up first",
			fields: fakehcsshim{
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
				"c1": &statsapi.NetworkStats{
					Time: v1.Time{},
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
				"c2": &statsapi.NetworkStats{
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
			fields: fakehcsshim{
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
				"c1": &statsapi.NetworkStats{
					Time: v1.Time{},
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
				"c2": &statsapi.NetworkStats{
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
				hcsshimInterface: fakehcsshim{
					containers:       tt.fields.containers,
				},
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
