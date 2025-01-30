// Copyright 2020 Google Inc. All Rights Reserved.
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
	"time"

	info "github.com/google/cadvisor/info/v1"
	v2 "github.com/google/cadvisor/info/v2"
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
		Timestamp:        time.Unix(1395066363, 0),
		NumCores:         4,
		NumPhysicalCores: 1,
		NumSockets:       1,
		MemoryCapacity:   1024,
		MemoryByType: map[string]*info.MemoryInfo{
			"Non-volatile-RAM": {Capacity: 2168421613568, DimmCount: 8},
			"Unbuffered-DDR4":  {Capacity: 412316860416, DimmCount: 12},
		},
		NVMInfo: info.NVMInfo{
			MemoryModeCapacity:    429496729600,
			AppDirectModeCapacity: 1735166787584,
		},
		MachineID:  "machine-id-test",
		SystemUUID: "system-uuid-test",
		BootID:     "boot-id-test",
		Topology: []info.Node{
			{
				Id:     0,
				Memory: 33604804608,
				HugePages: []info.HugePagesInfo{
					{
						PageSize: uint64(1048576),
						NumPages: uint64(0),
					},
					{
						PageSize: uint64(2048),
						NumPages: uint64(0),
					},
				},
				Cores: []info.Core{
					{
						Id:      0,
						Threads: []int{0, 1},
						Caches: []info.Cache{
							{
								Size:  32768,
								Type:  "Data",
								Level: 1,
							},
							{
								Size:  32768,
								Type:  "Instruction",
								Level: 1,
							},
							{
								Size:  262144,
								Type:  "Unified",
								Level: 2,
							},
						},
					},
					{
						Id:      1,
						Threads: []int{2, 3},
						Caches: []info.Cache{
							{
								Size:  32764,
								Type:  "Data",
								Level: 1,
							},
							{
								Size:  32764,
								Type:  "Instruction",
								Level: 1,
							},
							{
								Size:  262148,
								Type:  "Unified",
								Level: 2,
							},
						},
					},

					{
						Id:      2,
						Threads: []int{4, 5},
						Caches: []info.Cache{
							{
								Size:  32768,
								Type:  "Data",
								Level: 1,
							},
							{
								Size:  32768,
								Type:  "Instruction",
								Level: 1,
							},
							{
								Size:  262144,
								Type:  "Unified",
								Level: 2,
							},
						},
					},
					{
						Id:      3,
						Threads: []int{6, 7},
						Caches: []info.Cache{
							{
								Size:  32764,
								Type:  "Data",
								Level: 1,
							},
							{
								Size:  32764,
								Type:  "Instruction",
								Level: 1,
							},
							{
								Size:  262148,
								Type:  "Unified",
								Level: 2,
							},
						},
					},
				},
				Distances: []uint64{
					10,
					12,
				},
			},
			{
				Id:     1,
				Memory: 33604804606,
				HugePages: []info.HugePagesInfo{
					{
						PageSize: uint64(1048576),
						NumPages: uint64(2),
					},
					{
						PageSize: uint64(2048),
						NumPages: uint64(4),
					},
				},
				Cores: []info.Core{
					{
						Id:      4,
						Threads: []int{8, 9},
						Caches: []info.Cache{
							{
								Size:  32768,
								Type:  "Data",
								Level: 1,
							},
							{
								Size:  32768,
								Type:  "Instruction",
								Level: 1,
							},
							{
								Size:  262144,
								Type:  "Unified",
								Level: 2,
							},
						},
					},
					{
						Id:      5,
						Threads: []int{10, 11},
						Caches: []info.Cache{
							{
								Size:  32764,
								Type:  "Data",
								Level: 1,
							},
							{
								Size:  32764,
								Type:  "Instruction",
								Level: 1,
							},
							{
								Size:  262148,
								Type:  "Unified",
								Level: 2,
							},
						},
					},
					{
						Id:      6,
						Threads: []int{12, 13},
						Caches: []info.Cache{
							{
								Size:  32768,
								Type:  "Data",
								Level: 1,
							},
							{
								Size:  32768,
								Type:  "Instruction",
								Level: 1,
							},
							{
								Size:  262144,
								Type:  "Unified",
								Level: 2,
							},
						},
					},
					{
						Id:      7,
						Threads: []int{14, 15},
						Caches: []info.Cache{
							{
								Size:  32764,
								Type:  "Data",
								Level: 1,
							},
							{
								Size:  32764,
								Type:  "Instruction",
								Level: 1,
							},
							{
								Size:  262148,
								Type:  "Unified",
								Level: 2,
							},
						},
					},
				},
				Caches: []info.Cache{
					{
						Size:  8388608,
						Type:  "Unified",
						Level: 3,
					},
				},
				Distances: []uint64{
					12,
					10,
				},
			},
		},
	}, nil
}

func (p testSubcontainersInfoProvider) GetRequestedContainersInfo(string, v2.RequestOptions) (map[string]*info.ContainerInfo, error) {
	return map[string]*info.ContainerInfo{
		"testcontainer": {
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
				HasHugetlb:   true,
				HasProcesses: true,
				Processes: info.ProcessSpec{
					Limit: 100,
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
					Timestamp: time.Unix(1395066363, 0),
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
						Schedstat: info.CpuSchedstat{
							RunTime:      53643567,
							RunqueueTime: 479424566378,
							RunPeriods:   984285,
						},
						LoadAverage:  2,
						LoadDAverage: 2,
					},
					Memory: info.MemoryStats{
						Usage:             8,
						MaxUsage:          8,
						WorkingSet:        9,
						TotalActiveFile:   7,
						TotalInactiveFile: 6,
						ContainerData: info.MemoryStatsMemoryData{
							Pgfault:    10,
							Pgmajfault: 11,
							NumaStats: info.MemoryNumaStats{
								File:        map[uint8]uint64{0: 16649, 1: 10000},
								Anon:        map[uint8]uint64{0: 10000, 1: 7109},
								Unevictable: map[uint8]uint64{0: 8900, 1: 10000},
							},
						},
						HierarchicalData: info.MemoryStatsMemoryData{
							Pgfault:    12,
							Pgmajfault: 13,
							NumaStats: info.MemoryNumaStats{
								File:        map[uint8]uint64{0: 36649, 1: 10000},
								Anon:        map[uint8]uint64{0: 20000, 1: 7109},
								Unevictable: map[uint8]uint64{0: 8900, 1: 20000},
							},
						},
						Cache:       14,
						RSS:         15,
						MappedFile:  16,
						KernelUsage: 17,
						Swap:        8192,
					},
					Hugetlb: map[string]info.HugetlbStats{
						"2Mi": {
							Usage:    4,
							MaxUsage: 10,
							Failcnt:  1,
						},
						"1Gi": {
							Usage:    0,
							MaxUsage: 0,
							Failcnt:  0,
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
						Tcp6: info.TcpStat{
							Established: 11,
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
						TcpAdvanced: info.TcpAdvancedStat{
							TCPFullUndo:               2361,
							TCPMD5NotFound:            0,
							TCPDSACKRecv:              83680,
							TCPSackShifted:            2,
							TCPSackShiftFallback:      298,
							PFMemallocDrop:            0,
							EstabResets:               37,
							InSegs:                    140370590,
							TCPPureAcks:               24251339,
							TCPDSACKOldSent:           15633,
							IPReversePathFilter:       0,
							TCPFastOpenPassiveFail:    0,
							InCsumErrors:              0,
							TCPRenoFailures:           43414,
							TCPMemoryPressuresChrono:  0,
							TCPDeferAcceptDrop:        0,
							TW:                        10436427,
							TCPSpuriousRTOs:           0,
							TCPDSACKIgnoredNoUndo:     71885,
							RtoMax:                    120000,
							ActiveOpens:               11038621,
							EmbryonicRsts:             0,
							RcvPruned:                 0,
							TCPLossProbeRecovery:      401,
							TCPHPHits:                 56096478,
							TCPPartialUndo:            3,
							TCPAbortOnMemory:          0,
							AttemptFails:              48997,
							RetransSegs:               462961,
							SyncookiesFailed:          0,
							OfoPruned:                 0,
							TCPAbortOnLinger:          0,
							TCPAbortFailed:            0,
							TCPRenoReorder:            839,
							TCPRcvCollapsed:           0,
							TCPDSACKIgnoredOld:        0,
							TCPReqQFullDrop:           0,
							OutOfWindowIcmps:          0,
							TWKilled:                  0,
							TCPLossProbes:             88648,
							TCPRenoRecoveryFail:       394,
							TCPFastOpenCookieReqd:     0,
							TCPHPAcks:                 21490641,
							TCPSACKReneging:           0,
							TCPTSReorder:              3,
							TCPSlowStartRetrans:       290832,
							MaxConn:                   -1,
							SyncookiesRecv:            0,
							TCPSackFailures:           60,
							DelayedACKLocked:          90,
							TCPDSACKOfoSent:           1,
							TCPSynRetrans:             988,
							TCPDSACKOfoRecv:           10,
							TCPSACKDiscard:            0,
							TCPMD5Unexpected:          0,
							TCPSackMerged:             6,
							RtoMin:                    200,
							CurrEstab:                 22,
							TCPTimeWaitOverflow:       0,
							ListenOverflows:           0,
							DelayedACKs:               503975,
							TCPLossUndo:               61374,
							TCPOrigDataSent:           130698387,
							TCPBacklogDrop:            0,
							TCPReqQFullDoCookies:      0,
							TCPFastOpenPassive:        0,
							PAWSActive:                0,
							OutRsts:                   91699,
							TCPSackRecoveryFail:       2,
							DelayedACKLost:            18843,
							TCPAbortOnData:            8,
							TCPMinTTLDrop:             0,
							PruneCalled:               0,
							TWRecycled:                0,
							ListenDrops:               0,
							TCPAbortOnTimeout:         0,
							SyncookiesSent:            0,
							TCPSACKReorder:            11,
							TCPDSACKUndo:              33,
							TCPMD5Failure:             0,
							TCPLostRetransmit:         0,
							TCPAbortOnClose:           7,
							TCPFastOpenListenOverflow: 0,
							OutSegs:                   211580512,
							InErrs:                    31,
							TCPTimeouts:               27422,
							TCPLossFailures:           729,
							TCPSackRecovery:           159,
							RtoAlgorithm:              1,
							PassiveOpens:              59,
							LockDroppedIcmps:          0,
							TCPRenoRecovery:           3519,
							TCPFACKReorder:            0,
							TCPFastRetrans:            11794,
							TCPRetransFail:            0,
							TCPMemoryPressures:        0,
							TCPFastOpenActive:         0,
							TCPFastOpenActiveFail:     0,
							PAWSEstab:                 0,
						},
						Udp: info.UdpStat{
							Listen:   0,
							Dropped:  0,
							RxQueued: 0,
							TxQueued: 0,
						},
						Udp6: info.UdpStat{
							Listen:   0,
							Dropped:  0,
							RxQueued: 0,
							TxQueued: 0,
						},
					},
					DiskIo: info.DiskIoStats{
						IoServiceBytes: []info.PerDiskStats{{
							Device: "/dev/sdb",
							Major:  8,
							Minor:  0,
							Stats: map[string]uint64{
								"Async":   1,
								"Discard": 2,
								"Read":    3,
								"Sync":    4,
								"Total":   5,
								"Write":   6,
							},
						}},
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
					Processes: info.ProcessStats{
						ProcessCount:   1,
						FdCount:        5,
						SocketCount:    3,
						ThreadsCurrent: 5,
						ThreadsMax:     100,
						Ulimits: []info.UlimitSpec{
							{
								Name:      "max_open_files",
								SoftLimit: 16384,
								HardLimit: 16384,
							},
						},
					},
					TaskStats: info.LoadStats{
						NrSleeping:        50,
						NrRunning:         51,
						NrStopped:         52,
						NrUninterruptible: 53,
						NrIoWait:          54,
					},
					CustomMetrics: map[string][]info.MetricVal{
						"container_custom_app_metric_1": {
							{
								FloatValue: float64(1.1),
								Timestamp:  time.Now(),
								Label:      "testlabel_1_1_1",
								Labels:     map[string]string{"test_label": "1_1", "test_label_2": "2_1"},
							},
							{
								FloatValue: float64(1.2),
								Timestamp:  time.Now(),
								Label:      "testlabel_1_1_2",
								Labels:     map[string]string{"test_label": "1_2", "test_label_2": "2_2"},
							},
						},
						"container_custom_app_metric_2": {
							{
								FloatValue: float64(2),
								Timestamp:  time.Now(),
								Label:      "testlabel2",
								Labels:     map[string]string{"test_label": "test_value"},
							},
						},
						"container_custom_app_metric_3": {
							{
								FloatValue: float64(3),
								Timestamp:  time.Now(),
								Label:      "testlabel3",
								Labels:     map[string]string{"test_label": "test_value"},
							},
						},
					},
					PerfStats: []info.PerfStat{
						{
							PerfValue: info.PerfValue{
								ScalingRatio: 1.0,
								Value:        123,
								Name:         "instructions",
							},
							Cpu: 0,
						},
						{
							PerfValue: info.PerfValue{
								ScalingRatio: 0.5,
								Value:        456,
								Name:         "instructions",
							},
							Cpu: 1,
						},
						{
							PerfValue: info.PerfValue{
								ScalingRatio: 0.66666666666,
								Value:        321,
								Name:         "instructions_retired",
							},
							Cpu: 0,
						},
						{
							PerfValue: info.PerfValue{
								ScalingRatio: 0.33333333333,
								Value:        789,
								Name:         "instructions_retired",
							},
							Cpu: 1,
						},
					},
					PerfUncoreStats: []info.PerfUncoreStat{
						{
							PerfValue: info.PerfValue{
								ScalingRatio: 1.0,
								Value:        1231231512.0,
								Name:         "cas_count_read",
							},
							Socket: 0,
							PMU:    "uncore_imc_0",
						},
						{
							PerfValue: info.PerfValue{
								ScalingRatio: 1.0,
								Value:        1111231331.0,
								Name:         "cas_count_read",
							},
							Socket: 1,
							PMU:    "uncore_imc_0",
						},
					},
					ReferencedMemory: 1234,
					Resctrl: info.ResctrlStats{
						MemoryBandwidth: []info.MemoryBandwidthStats{
							{
								TotalBytes: 4512312,
								LocalBytes: 2390393,
							},
							{
								TotalBytes: 2173713,
								LocalBytes: 1231233,
							},
						},
						Cache: []info.CacheStats{
							{
								LLCOccupancy: 162626,
							},
							{
								LLCOccupancy: 213777,
							},
						},
					},
					CpuSet: info.CPUSetStats{MemoryMigrate: 1},
				},
			},
		},
	}, nil
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

func (p *erroringSubcontainersInfoProvider) GetRequestedContainersInfo(
	a string, opt v2.RequestOptions) (map[string]*info.ContainerInfo, error) {
	if p.shouldFail {
		return map[string]*info.ContainerInfo{}, errors.New("Oops 3")
	}
	return p.successfulProvider.GetRequestedContainersInfo(a, opt)
}
