// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build !windows

package sysfs

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestFibreChannelClass(t *testing.T) {
	fs, err := NewFS(sysTestFixtures)
	if err != nil {
		t.Fatal(err)
	}

	got, err := fs.FibreChannelClass()
	if err != nil {
		t.Fatal(err)
	}

	want := FibreChannelClass{
		"host0": FibreChannelHost{
			Name:             "host0",
			Speed:            "16 Gbit",
			PortState:        "Online",
			PortType:         "Point-To-Point (direct nport connection)",
			PortName:         "1000e0071bce95f2",
			SymbolicName:     "Emulex SN1100E2P FV12.4.270.3 DV12.4.0.0. HN:gotest. OS:Linux",
			NodeName:         "2000e0071bce95f2",
			PortID:           "000002",
			FabricName:       "0",
			DevLossTMO:       "30",
			SupportedClasses: "Class 3",
			SupportedSpeeds:  "4 Gbit, 8 Gbit, 16 Gbit",
			Counters: FibreChannelCounters{
				DumpedFrames:          ^uint64(0),
				ErrorFrames:           0,
				InvalidCRCCount:       0x2,
				RXFrames:              0x3,
				RXWords:               0x4,
				TXFrames:              0x5,
				TXWords:               0x6,
				SecondsSinceLastReset: 0x7,
				InvalidTXWordCount:    0x8,
				LinkFailureCount:      0x9,
				LossOfSyncCount:       0x10,
				LossOfSignalCount:     0x11,
				NosCount:              0x12,
				FCPPacketAborts:       0x13,
			},
		},
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("unexpected FibreChannel class (-want +got):\n%s", diff)
	}
}
