/*
Copyright The Kubernetes Authors.

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

package service

import (
	"flag"
	"io"
	"strings"
	"testing"

	utilnet "k8s.io/apimachinery/pkg/util/net"
)

func TestServiceNodePortRangeFlag(t *testing.T) {
	originalRange := NodePortRange
	t.Cleanup(func() {
		NodePortRange = originalRange
		initializeStaticPortAllocator()
	})

	tests := []struct {
		name                string
		value               string
		want                utilnet.PortRange
		wantAllocatorLength int32
	}{
		{
			name:                "hyphen notation",
			value:               "20000-22767",
			want:                utilnet.PortRange{Base: 20000, Size: 2768},
			wantAllocatorLength: 86,
		},
		{
			name:                "offset notation",
			value:               "20000+2767",
			want:                utilnet.PortRange{Base: 20000, Size: 2768},
			wantAllocatorLength: 86,
		},
		{
			name:                "small range",
			value:               "30000-30001",
			want:                utilnet.PortRange{Base: 30000, Size: 2},
			wantAllocatorLength: 2,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			flags := flag.NewFlagSet("test", flag.ContinueOnError)
			flags.SetOutput(io.Discard)
			RegisterFlags(flags)
			if err := flags.Parse([]string{"--service-node-port-range=" + test.value}); err != nil {
				t.Fatalf("parse flag: %v", err)
			}

			if NodePortRange != test.want {
				t.Fatalf("NodePortRange = %#v, want %#v", NodePortRange, test.want)
			}
			if staticPortAllocator.baseport != int32(test.want.Base) {
				t.Errorf("static allocator base = %d, want %d", staticPortAllocator.baseport, test.want.Base)
			}
			if staticPortAllocator.length != test.wantAllocatorLength {
				t.Errorf("static allocator length = %d, want %d", staticPortAllocator.length, test.wantAllocatorLength)
			}

			port, err := staticPortAllocator.getUnusedPort()
			if err != nil {
				t.Fatalf("get unused port: %v", err)
			}
			if !test.want.Contains(int(port)) {
				t.Errorf("unused port %d is outside configured range %s", port, test.want.String())
			}

			lastPort := int32(test.want.Base) + test.wantAllocatorLength - 1
			if !staticPortAllocator.reservePort(lastPort) {
				t.Errorf("expected static allocator to reserve upper-bound port %d", lastPort)
			}
			if firstOutOfRangePort := lastPort + 1; staticPortAllocator.reservePort(firstOutOfRangePort) {
				t.Errorf("static allocator reserved out-of-range port %d", firstOutOfRangePort)
			}
		})
	}
}

func TestServiceNodePortRangeFlagHelpIncludesDefault(t *testing.T) {
	flags := flag.NewFlagSet("test", flag.ContinueOnError)
	RegisterFlags(flags)

	var output strings.Builder
	flags.SetOutput(&output)
	flags.PrintDefaults()
	want := "(default " + NodePortRange.String() + ")"
	if !strings.Contains(output.String(), want) {
		t.Fatalf("flag help does not contain %q:\n%s", want, output.String())
	}
}

func TestServiceNodePortRangeFlagRejectsInvalidRange(t *testing.T) {
	originalRange := NodePortRange
	t.Cleanup(func() {
		NodePortRange = originalRange
		initializeStaticPortAllocator()
	})

	tests := []struct {
		name  string
		value string
	}{
		{name: "malformed", value: "invalid"},
		{name: "empty", value: ""},
		{name: "includes port zero", value: "0-1"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			flags := flag.NewFlagSet("test", flag.ContinueOnError)
			flags.SetOutput(io.Discard)
			RegisterFlags(flags)
			if err := flags.Parse([]string{"--service-node-port-range=" + test.value}); err == nil {
				t.Fatalf("expected range %q to be rejected", test.value)
			}
			if NodePortRange != originalRange {
				t.Fatalf("NodePortRange changed after invalid input: got %#v, want %#v", NodePortRange, originalRange)
			}
		})
	}
}
