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
				Image:        "test",
				CreationTime: time.Unix(1257894000, 0),
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
					},
					Memory: info.MemoryStats{
						Usage:      8,
						WorkingSet: 9,
						ContainerData: info.MemoryStatsMemoryData{
							Pgfault:    10,
							Pgmajfault: 11,
						},
						HierarchicalData: info.MemoryStatsMemoryData{
							Pgfault:    12,
							Pgmajfault: 13,
						},
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
					},
					Filesystem: []info.FsStats{
						{
							Device:          "sda1",
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

func TestPrometheusCollector(t *testing.T) {
	prometheus.MustRegister(NewPrometheusCollector(testSubcontainersInfoProvider{}))

	rw := httptest.NewRecorder()
	prometheus.Handler().ServeHTTP(rw, &http.Request{})

	metricsFile := "testdata/prometheus_metrics"
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
	includeRe := regexp.MustCompile("^(# HELP |# TYPE |)container_")
	ignoreRe := regexp.MustCompile("^container_last_seen{")
	for i, want := range wantLines {
		if !includeRe.MatchString(want) || ignoreRe.MatchString(want) {
			continue
		}
		if want != gotLines[i] {
			t.Fatalf("want %s, got %s", want, gotLines[i])
		}
	}
}
