/*
Copyright 2019 The Kubernetes Authors.

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

// Partially copied from https://github.com/bowei/lighthouse/blob/master/pkg/probe/tcp_test.go

package main

import (
	"testing"
)

func TestTCPChecksummer(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		desc string
		data [][]byte
		want uint16
	}{
		{
			desc: "empty",
			data: [][]byte{},
			want: 0xffff,
		},
		{
			desc: "1 byte",
			data: [][]byte{{0x55}},
			want: 0xffaa,
		},
		{
			desc: "2 bytes",
			data: [][]byte{{0x55, 0x88}},
			want: 0x77aa,
		},
		{
			desc: "3 bytes",
			data: [][]byte{{0x55, 0x88, 0x99}},
			want: 0x7711,
		},
		{
			desc: "3 bytes / 1 at a time",
			data: [][]byte{{0x55}, {0x88}, {0x99}},
			want: 0x7711,
		},
		{
			desc: "3 bytes / 2 1",
			data: [][]byte{{0x55, 0x88}, {0x99}},
			want: 0x7711,
		},
		{
			desc: "simple packet",
			data: [][]byte{
				{
					0x7f, 0x00, 0x00, 0x01, // 127.0.0.1
					0x7f, 0x00, 0x00, 0x01, // 127.0.0.1
					0x00, 0x06, // TCP proto 6
					0x00, 0x14, // Size = 20 bytes
					0x00, 0x50, 0x1f, 0x90, 0x00,
					0x00, 0x00, 0x01, 0x00, 0x00,
					0x00, 0x00, 0x50, 0x02, 0x00,
					0x00, 0x00, 0x00, 0x00, 0x00,
				},
			},
			want: 0xff91,
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			c := &tcpChecksummer{}
			for _, b := range tc.data {
				c.add(b)
			}
			got := c.finalize()
			if got != tc.want {
				t.Errorf("c.finalize() = %x, want %x; bytes: %v", got, tc.want, tc.data)
			}
		})
	}
}
