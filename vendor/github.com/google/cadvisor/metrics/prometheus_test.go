// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metrics

import (
	"errors"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	info "github.com/google/cadvisor/info/v1"

	"github.com/prometheus/client_golang/prometheus"
)

type testSubcontainersInfoProvider struct{}

func (p testSubcontainersInfoProvider) GetVersionInfo() (*info.VersionInfo, error) {
	return &info.VersionInfo{
		KernelVersion:      "4.1.6-200.fc22.x86_64",
		ContainerOsVersion: "Fedora 22 (Twenty Two)",
		DockerVersion:      "1.8.1",
		CadvisorVersion:    "0.16.0",
		CadvisorRevision:   "abcdef",
	}, nil
}

func (p testSubcontainersInfoProvider) GetMachineInfo() (*info.MachineInfo, error) {
	return &info.MachineInfo{
		NumCores:       4,
		MemoryCapacity: 1024,
	}, nil
}

func (p testSubcontainersInfoProvider) SubcontainersInfo(string, *info.ContainerInfoRequest) ([]*info.ContainerInfo, error) {
	return []*info.ContainerInfo{
		{
			ContainerReference: info.ContainerReference{
				Name:    "testcontainer",
				Aliases: []string{"testcontaineralias"},
			},
			Spec: info.ContainerSpec{
				Image:  "test",
				HasCpu: true,
				Cpu: info.CpuSpec{
					Limit:  1000,
					Period: 100000,
					Quota:  10000,
				},
				Memory: info.MemorySpec{
					Limit:       2048,
					Reservation: 1024,
					SwapLimit:   4096,
				},
				CreationTime: time.Unix(1257894000, 0),
				Labels: map[string]string{
					"foo.label": "bar",
				},
				Envs: map[string]string{
					"foo+env": "prod",
				},
			},
			Stats: []*info.ContainerStats{
				{
					Cpu: info.CpuStats{
						Usage: info.CpuUsage{
							Total:  1,
							PerCpu: []uint64{2, 3, 4, 5},
							User:   6,
							System: 7,
						},
						CFS: info.CpuCFS{
							Periods:          723,
							ThrottledPeriods: 18,
							ThrottledTime:    1724314000,
						},
						LoadAverage: 2,
					},
					Memory: info.MemoryStats{
						Usage:      8,
						MaxUsage:   8,
						WorkingSet: 9,
						ContainerData: info.MemoryStatsMemoryData{
							Pgfault:    10,
							Pgmajfault: 11,
						},
						HierarchicalData: info.MemoryStatsMemoryData{
							Pgfault:    12,
							Pgmajfault: 13,
						},
						Cache: 14,
						RSS:   15,
						Swap:  8192,
					},
					Network: info.NetworkStats{
						InterfaceStats: info.InterfaceStats{
							Name:      "eth0",
							RxBytes:   14,
							RxPackets: 15,
							RxErrors:  16,
							RxDropped: 17,
							TxBytes:   18,
							TxPackets: 19,
							TxErrors:  20,
							TxDropped: 21,
						},
						Interfaces: []info.InterfaceStats{
							{
								Name:      "eth0",
								RxBytes:   14,
								RxPackets: 15,
								RxErrors:  16,
								RxDropped: 17,
								TxBytes:   18,
								TxPackets: 19,
								TxErrors:  20,
								TxDropped: 21,
							},
						},
						Tcp: info.TcpStat{
							Established: 13,
							SynSent:     0,
							SynRecv:     0,
							FinWait1:    0,
							FinWait2:    0,
							TimeWait:    0,
							Close:       0,
							CloseWait:   0,
							LastAck:     0,
							Listen:      3,
							Closing:     0,
						},
						Udp: info.UdpStat{
							Listen:   0,
							Dropped:  0,
							RxQueued: 0,
							TxQueued: 0,
						},
					},
					Filesystem: []info.FsStats{
						{
							Device:          "sda1",
							InodesFree:      524288,
							Inodes:          2097152,
							Limit:           22,
							Usage:           23,
							ReadsCompleted:  24,
							ReadsMerged:     25,
							SectorsRead:     26,
							ReadTime:        27,
							WritesCompleted: 28,
							WritesMerged:    39,
							SectorsWritten:  40,
							WriteTime:       41,
							IoInProgress:    42,
							IoTime:          43,
							WeightedIoTime:  44,
						},
						{
							Device:          "sda2",
							InodesFree:      262144,
							Inodes:          2097152,
							Limit:           37,
							Usage:           38,
							ReadsCompleted:  39,
							ReadsMerged:     40,
							SectorsRead:     41,
							ReadTime:        42,
							WritesCompleted: 43,
							WritesMerged:    44,
							SectorsWritten:  45,
							WriteTime:       46,
							IoInProgress:    47,
							IoTime:          48,
							WeightedIoTime:  49,
						},
					},
					Accelerators: []info.AcceleratorStats{
						{
							Make:        "nvidia",
							Model:       "tesla-p100",
							ID:          "GPU-deadbeef-1234-5678-90ab-feedfacecafe",
							MemoryTotal: 20304050607,
							MemoryUsed:  2030405060,
							DutyCycle:   12,
						},
						{
							Make:        "nvidia",
							Model:       "tesla-k80",
							ID:          "GPU-deadbeef-0123-4567-89ab-feedfacecafe",
							MemoryTotal: 10203040506,
							MemoryUsed:  1020304050,
							DutyCycle:   6,
						},
					},
					TaskStats: info.LoadStats{
						NrSleeping:        50,
						NrRunning:         51,
						NrStopped:         52,
						NrUninterruptible: 53,
						NrIoWait:          54,
					},
				},
			},
		},
	}, nil
}

var (
	includeRe = regexp.MustCompile(`^(?:(?:# HELP |# TYPE )?container_|cadvisor_version_info\{)`)
	ignoreRe  = regexp.MustCompile(`^container_last_seen\{`)
)

func TestPrometheusCollector(t *testing.T) {
	c := NewPrometheusCollector(testSubcontainersInfoProvider{}, func(container *info.ContainerInfo) map[string]string {
		s := DefaultContainerLabels(container)
		s["zone.name"] = "hello"
		return s
	})
	prometheus.MustRegister(c)
	defer prometheus.Unregister(c)

	testPrometheusCollector(t, c, "testdata/prometheus_metrics")
}

func testPrometheusCollector(t *testing.T, c *PrometheusCollector, metricsFile string) {
	rw := httptest.NewRecorder()
	prometheus.Handler().ServeHTTP(rw, &http.Request{})

	wantMetrics, err := ioutil.ReadFile(metricsFile)
	if err != nil {
		t.Fatalf("unable to read input test file %s", metricsFile)
	}

	wantLines := strings.Split(string(wantMetrics), "\n")
	gotLines := strings.Split(string(rw.Body.String()), "\n")

	// Until the Prometheus Go client library offers better testability
	// (https://github.com/prometheus/client_golang/issues/58), we simply compare
	// verbatim text-format metrics outputs, but ignore certain metric lines
	// whose value depends on the current time or local circumstances.
	for i, want := range wantLines {
		if !includeRe.MatchString(want) || ignoreRe.MatchString(want) {
			continue
		}
		if want != gotLines[i] {
			t.Fatalf("unexpected metric line\nwant: %s\nhave: %s", want, gotLines[i])
		}
	}
}

type erroringSubcontainersInfoProvider struct {
	successfulProvider testSubcontainersInfoProvider
	shouldFail         bool
}

func (p *erroringSubcontainersInfoProvider) GetVersionInfo() (*info.VersionInfo, error) {
	if p.shouldFail {
		return nil, errors.New("Oops 1")
	}
	return p.successfulProvider.GetVersionInfo()
}

func (p *erroringSubcontainersInfoProvider) GetMachineInfo() (*info.MachineInfo, error) {
	if p.shouldFail {
		return nil, errors.New("Oops 2")
	}
	return p.successfulProvider.GetMachineInfo()
}

func (p *erroringSubcontainersInfoProvider) SubcontainersInfo(
	a string, r *info.ContainerInfoRequest) ([]*info.ContainerInfo, error) {
	if p.shouldFail {
		return []*info.ContainerInfo{}, errors.New("Oops 3")
	}
	return p.successfulProvider.SubcontainersInfo(a, r)
}

func TestPrometheusCollector_scrapeFailure(t *testing.T) {
	provider := &erroringSubcontainersInfoProvider{
		successfulProvider: testSubcontainersInfoProvider{},
		shouldFail:         true,
	}

	c := NewPrometheusCollector(provider, func(container *info.ContainerInfo) map[string]string {
		s := DefaultContainerLabels(container)
		s["zone.name"] = "hello"
		return s
	})
	prometheus.MustRegister(c)
	defer prometheus.Unregister(c)

	testPrometheusCollector(t, c, "testdata/prometheus_metrics_failure")

	provider.shouldFail = false

	testPrometheusCollector(t, c, "testdata/prometheus_metrics")
}
