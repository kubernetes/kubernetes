//go:build linux

/*
Copyright 2024 The Kubernetes Authors.

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

package fc

import (
	"testing"

	"k8s.io/kubernetes/pkg/volume/util"
)

func TestSearchDiskMultipathDevice(t *testing.T) {
	tests := []struct {
		name        string
		wwns        []string
		lun         string
		expectError bool
	}{
		{
			name: "Non PCI disk 0",
			wwns: []string{"500507681021a537"},
			lun:  "0",
		},
		{
			name: "Non PCI disk 1",
			wwns: []string{"500507681022a554"},
			lun:  "2",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fakeMounter := fcDiskMounter{
				fcDisk: &fcDisk{
					wwns: test.wwns,
					lun:  test.lun,
					io:   &fakeIOHandler{},
				},
				deviceUtil: util.NewDeviceHandler(&fakeIOHandler{}),
			}
			devicePath, err := searchDisk(fakeMounter)
			if test.expectError && err == nil {
				t.Errorf("expected error but got none")
			}
			if !test.expectError && err != nil {
				t.Errorf("got unexpected error: %s", err)
			}
			// if no disk matches input wwn and lun, exit
			if devicePath == "" && !test.expectError {
				t.Errorf("no fc disk found")
			}
			if devicePath != "/dev/dm-1" {
				t.Errorf("multipath device not found dm-1 expected got [%s]", devicePath)
			}
		})
	}
}
