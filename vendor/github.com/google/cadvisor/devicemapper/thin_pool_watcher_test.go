// Copyright 2016 Google Inc. All Rights Reserved.
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
package devicemapper

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/google/cadvisor/devicemapper/fake"
)

func TestRefresh(t *testing.T) {
	usage := map[string]uint64{
		"1": 12345,
		"2": 23456,
		"3": 34567,
	}

	cases := []struct {
		name            string
		dmsetupCommands []fake.DmsetupCommand
		thinLsOutput    map[string]uint64
		thinLsErr       error
		expectedError   bool
		deviceId        string
		expectedUsage   uint64
	}{
		{
			name: "check reservation fails",
			dmsetupCommands: []fake.DmsetupCommand{
				{Name: "status", Result: "", Err: fmt.Errorf("not gonna work")},
			},
			expectedError: true,
		},
		{
			name: "no existing reservation - ok with minimum # of fields",
			dmsetupCommands: []fake.DmsetupCommand{
				{Name: "status", Result: "0 75497472 thin-pool 65 327/524288 14092/589824 -", Err: nil}, // status check
				{Name: "message", Result: "", Err: nil},                                                 // make reservation
				{Name: "message", Result: "", Err: nil},                                                 // release reservation
			},
			thinLsOutput:  usage,
			expectedError: false,
			deviceId:      "2",
			expectedUsage: 23456,
		},
		{
			name: "no existing reservation - ok",
			dmsetupCommands: []fake.DmsetupCommand{
				{Name: "status", Result: "0 75497472 thin-pool 65 327/524288 14092/589824 - rw no_discard_passdown error_if_no_space - ", Err: nil}, // status check
				{Name: "message", Result: "", Err: nil},                                                                                             // make reservation
				{Name: "message", Result: "", Err: nil},                                                                                             // release reservation
			},
			thinLsOutput:  usage,
			expectedError: false,
			deviceId:      "2",
			expectedUsage: 23456,
		},
		{
			name: "existing reservation - ok",
			dmsetupCommands: []fake.DmsetupCommand{
				// status check
				{Name: "status", Result: "0 75497472 thin-pool 65 327/524288 14092/589824 39 rw no_discard_passdown error_if_no_space - ", Err: nil},
				// release reservation
				{Name: "message", Result: "", Err: nil},
				// make reservation
				{Name: "message", Result: "", Err: nil},
				// release reservation
				{Name: "message", Result: "", Err: nil},
			},
			thinLsOutput:  usage,
			expectedError: false,
			deviceId:      "3",
			expectedUsage: 34567,
		},
		{
			name: "failure releasing existing reservation",
			dmsetupCommands: []fake.DmsetupCommand{
				// status check
				{Name: "status", Result: "0 75497472 thin-pool 65 327/524288 14092/589824 39 rw no_discard_passdown error_if_no_space - ", Err: nil},
				// release reservation
				{Name: "message", Result: "", Err: fmt.Errorf("not gonna work")},
			},
			expectedError: true,
		},
		{
			name: "failure making reservation",
			dmsetupCommands: []fake.DmsetupCommand{
				// status check
				{Name: "status", Result: "0 75497472 thin-pool 65 327/524288 14092/589824 39 rw no_discard_passdown error_if_no_space - ", Err: nil},
				// release reservation
				{Name: "message", Result: "", Err: nil},
				// make reservation
				{Name: "message", Result: "", Err: fmt.Errorf("not gonna work")},
			},
			expectedError: true,
		},
		{
			name: "failure running thin_ls",
			dmsetupCommands: []fake.DmsetupCommand{
				// status check
				{Name: "status", Result: "0 75497472 thin-pool 65 327/524288 14092/589824 39 rw no_discard_passdown error_if_no_space - ", Err: nil},
				// release reservation
				{Name: "message", Result: "", Err: nil},
				// make reservation
				{Name: "message", Result: "", Err: nil},
				// release reservation
				{Name: "message", Result: "", Err: nil},
			},
			thinLsErr:     fmt.Errorf("not gonna work"),
			expectedError: true,
		},
	}

	for _, tc := range cases {
		dmsetup := fake.NewFakeDmsetupClient(t, tc.dmsetupCommands...)
		thinLsClient := fake.NewFakeThinLsClient(tc.thinLsOutput, tc.thinLsErr)
		watcher := &ThinPoolWatcher{
			poolName:       "test pool name",
			metadataDevice: "/dev/mapper/metadata-device",
			lock:           &sync.RWMutex{},
			period:         15 * time.Second,
			stopChan:       make(chan struct{}),
			dmsetup:        dmsetup,
			thinLsClient:   thinLsClient,
		}

		err := watcher.Refresh()
		if err != nil {
			if !tc.expectedError {
				t.Errorf("%v: unexpected error: %v", tc.name, err)
			}
			continue
		} else if tc.expectedError {
			t.Errorf("%v: unexpected success", tc.name)
			continue
		}

		actualUsage, err := watcher.GetUsage(tc.deviceId)
		if err != nil {
			t.Errorf("%v: device ID not found: %v", tc.deviceId, err)
			continue
		}

		if e, a := tc.expectedUsage, actualUsage; e != a {
			t.Errorf("%v: actual usage did not match expected usage: expected: %v got: %v", tc.name, e, a)
		}
	}
}

func TestCheckReservation(t *testing.T) {
	cases := []struct {
		name           string
		statusResult   string
		statusErr      error
		expectedResult bool
		expectedErr    error
	}{
		{
			name:           "existing reservation 1",
			statusResult:   "0 75497472 thin-pool 65 327/524288 14092/589824 36 rw no_discard_passdown queue_if_no_space - ",
			expectedResult: true,
		},
		{
			name:           "existing reservation 2",
			statusResult:   "0 12345 thin-pool 65 327/45678 14092/45678 36 rw discard_passdown error_if_no_space needs_check ",
			expectedResult: true,
		},
		{
			name:           "no reservation 1",
			statusResult:   "0 75497472 thin-pool 65 327/524288 14092/589824 - rw no_discard_passdown error_if_no_space - ",
			expectedResult: false,
		},
		{
			name:           "no reservation 2",
			statusResult:   "0 75 thin-pool 65 327/12345 14092/589824 - rw no_discard_passdown queue_if_no_space - ",
			expectedResult: false,
		},
		{
			name:           "no reservation 2",
			statusResult:   "0 75 thin-pool 65 327/12345 14092/589824 - rw no_discard_passdown queue_if_no_space - ",
			expectedResult: false,
		},
	}

	for _, tc := range cases {
		fakeDmsetupClient := fake.NewFakeDmsetupClient(t)
		fakeDmsetupClient.AddCommand("status", tc.statusResult, tc.statusErr)
		watcher := &ThinPoolWatcher{dmsetup: fakeDmsetupClient}
		actualResult, err := watcher.checkReservation("test pool")
		if err != nil {
			if tc.expectedErr == nil {
				t.Errorf("%v: unexpected error running checkReservation: %v", tc.name, err)
			}
		} else if tc.expectedErr != nil {
			t.Errorf("%v: unexpected success running checkReservation", tc.name)
		}

		if e, a := tc.expectedResult, actualResult; e != a {
			t.Errorf("%v: unexpected result from checkReservation: expected: %v got: %v", tc.name, e, a)
		}
	}
}
