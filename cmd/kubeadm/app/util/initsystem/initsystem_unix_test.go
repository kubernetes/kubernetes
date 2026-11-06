//go:build !windows

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

package initsystem

import (
	"fmt"
	"strings"
	"testing"
)

// row renders one line the way rc-update does: the service in a padded column, then
// " %s" per runlevel, blanked when it is not in one.
func row(service string, runlevels ...string) string {
	out := fmt.Sprintf(" %20s |", service)
	for _, runlevel := range runlevels {
		out += " " + runlevel
	}
	return out + "\n"
}

func TestOpenrcServiceIsEnabled(t *testing.T) {
	// What verbose output prints for a service that is in no runlevel.
	blank := strings.Repeat(" ", len("default"))

	tests := []struct {
		name    string
		service string // what to ask about, "kubelet" when empty
		listing string // what "rc-update show default" printed
		want    bool
	}{
		{
			name:    "listed in the default runlevel",
			listing: row("kubelet", "default"),
			want:    true,
		},
		{
			name:    "another service whose name starts with the same text",
			listing: row("kubelet-debug", "default"),
		},
		{
			name:    "another service whose name ends with the same text",
			listing: row("my-kubelet", "default"),
		},
		{
			name:    "listed among other services",
			listing: row("kubelet-debug", "default") + row("kubelet", "default"),
			want:    true,
		},
		{
			name:    "a service other than kubelet",
			service: "containerd",
			listing: row("containerd", "default"),
			want:    true,
		},
		{
			name:    "listed with the runlevel column blanked",
			listing: row("kubelet", blank),
		},
		{
			name:    "listed with a different runlevel",
			listing: row("kubelet", "boot"),
		},
		{
			name:    "a name containing the column separator",
			listing: row("kubelet|debug", "default"),
		},
		{
			// The reverse of the case above: kubelet|debug asking about itself.
			name:    "the name asked about contains the separator",
			service: "kubelet|debug",
			listing: row("kubelet|debug", "default"),
			want:    true,
		},
		{
			name:    "test with a wider column",
			listing: fmt.Sprintf(" %26s | default\n", "kubelet"),
			want:    true,
		},
		{
			name:    "a line without a separator",
			listing: "kubelet\n",
		},
		{
			name:    "nothing in the default runlevel",
			listing: "",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			service := tc.service
			if service == "" {
				service = "kubelet"
			}
			if got := openrcServiceIsEnabled(tc.listing, service); got != tc.want {
				t.Errorf("openrcServiceIsEnabled(%q) = %v, want %v", service, got, tc.want)
			}
		})
	}
}
