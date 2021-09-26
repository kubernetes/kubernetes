// Copyright 2020 The Prometheus Authors
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

package procfs

import (
	"net"
	"reflect"
	"testing"
)

func Test_parseNetUDPLine(t *testing.T) {
	tests := []struct {
		fields  []string
		name    string
		want    *netUDPLine
		wantErr bool
	}{
		{
			name:   "reading valid lines, no issue should happened",
			fields: []string{"11:", "00000000:0000", "00000000:0000", "0A", "00000017:0000002A", "0:0", "0", "1000"},
			want: &netUDPLine{
				Sl:        11,
				LocalAddr: net.IP{0, 0, 0, 0},
				LocalPort: 0,
				RemAddr:   net.IP{0, 0, 0, 0},
				RemPort:   0,
				St:        10,
				TxQueue:   23,
				RxQueue:   42,
				UID:       1000,
			},
		},
		{
			name:    "error case - invalid line - number of fields/columns < 8",
			fields:  []string{"1:", "00000000:0000", "00000000:0000", "07", "0:0", "0"},
			want:    nil,
			wantErr: true,
		},
		{
			name:    "error case - parse sl - not a valid uint",
			fields:  []string{"a:", "00000000:0000", "00000000:0000", "07", "00000000:00000001", "0:0", "0", "0"},
			want:    nil,
			wantErr: true,
		},
		{
			name:    "error case - parse local_address - not a valid hex",
			fields:  []string{"1:", "0000000O:0000", "00000000:0000", "07", "00000000:00000001", "0:0", "0", "0"},
			want:    nil,
			wantErr: true,
		},
		{
			name:    "error case - parse rem_address - not a valid hex",
			fields:  []string{"1:", "00000000:0000", "0000000O:0000", "07", "00000000:00000001", "0:0", "0", "0"},
			want:    nil,
			wantErr: true,
		},
		{
			name:    "error case - cannot parse line - missing colon",
			fields:  []string{"1:", "00000000:0000", "00000000:0000", "07", "0000000000000001", "0:0", "0", "0"},
			want:    nil,
			wantErr: true,
		},
		{
			name:    "error case - parse tx_queue - not a valid hex",
			fields:  []string{"1:", "00000000:0000", "00000000:0000", "07", "DEADCODE:00000001", "0:0", "0", "0"},
			want:    nil,
			wantErr: true,
		},
		{
			name:    "error case - parse rx_queue - not a valid hex",
			fields:  []string{"1:", "00000000:0000", "00000000:0000", "07", "00000000:FEEDCODE", "0:0", "0", "0"},
			want:    nil,
			wantErr: true,
		},
		{
			name:    "error case - parse UID - not a valid uint",
			fields:  []string{"1:", "00000000:0000", "00000000:0000", "07", "00000000:00000001", "0:0", "0", "-10"},
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseNetUDPLine(tt.fields)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseNetUDPLine() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.want == nil && got != nil {
				t.Errorf("parseNetUDPLine() = %v, want %v", got, tt.want)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseNetUDPLine() = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func Test_newNetUDP(t *testing.T) {
	tests := []struct {
		name    string
		file    string
		want    NetUDP
		wantErr bool
	}{
		{
			name: "udp file found, no error should come up",
			file: "fixtures/proc/net/udp",
			want: []*netUDPLine{
				&netUDPLine{
					Sl:        0,
					LocalAddr: net.IP{10, 0, 0, 5},
					LocalPort: 22,
					RemAddr:   net.IP{0, 0, 0, 0},
					RemPort:   0,
					St:        10,
					TxQueue:   0,
					RxQueue:   1,
					UID:       0,
				},
				&netUDPLine{
					Sl:        1,
					LocalAddr: net.IP{0, 0, 0, 0},
					LocalPort: 22,
					RemAddr:   net.IP{0, 0, 0, 0},
					RemPort:   0,
					St:        10,
					TxQueue:   1,
					RxQueue:   0,
					UID:       0,
				},
				&netUDPLine{
					Sl:        2,
					LocalAddr: net.IP{0, 0, 0, 0},
					LocalPort: 22,
					RemAddr:   net.IP{0, 0, 0, 0},
					RemPort:   0,
					St:        10,
					TxQueue:   1,
					RxQueue:   1,
					UID:       0,
				},
			},
			wantErr: false,
		},
		{
			name: "udp6 file found, no error should come up",
			file: "fixtures/proc/net/udp6",
			want: []*netUDPLine{
				&netUDPLine{
					Sl:        1315,
					LocalAddr: net.IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					LocalPort: 5355,
					RemAddr:   net.IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					RemPort:   0,
					St:        7,
					TxQueue:   0,
					RxQueue:   0,
					UID:       981,
				},
				&netUDPLine{
					Sl:        6073,
					LocalAddr: net.IP{0, 0, 128, 254, 0, 0, 0, 0, 255, 173, 225, 86, 9, 102, 124, 254},
					LocalPort: 51073,
					RemAddr:   net.IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					RemPort:   0,
					St:        7,
					TxQueue:   0,
					RxQueue:   0,
					UID:       1000,
				},
			},
			wantErr: false,
		},
		{
			name:    "error case - file not found",
			file:    "somewhere over the rainbow",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "error case - parse error",
			file:    "fixtures/proc/net/udp_broken",
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := newNetUDP(tt.file)
			if (err != nil) != tt.wantErr {
				t.Errorf("newNetUDP() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("newNetUDP() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_newNetUDPSummary(t *testing.T) {
	tests := []struct {
		name    string
		file    string
		want    *NetUDPSummary
		wantErr bool
	}{
		{
			name:    "udp file found, no error should come up",
			file:    "fixtures/proc/net/udp",
			want:    &NetUDPSummary{TxQueueLength: 2, RxQueueLength: 2, UsedSockets: 3},
			wantErr: false,
		},
		{
			name:    "udp6 file found, no error should come up",
			file:    "fixtures/proc/net/udp6",
			want:    &NetUDPSummary{TxQueueLength: 0, RxQueueLength: 0, UsedSockets: 2},
			wantErr: false,
		},
		{
			name:    "error case - file not found",
			file:    "somewhere over the rainbow",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "error case - parse error",
			file:    "fixtures/proc/net/udp_broken",
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := newNetUDPSummary(tt.file)
			if (err != nil) != tt.wantErr {
				t.Errorf("newNetUDPSummary() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("newNetUDPSummary() = %v, want %v", got, tt.want)
			}
		})
	}
}
