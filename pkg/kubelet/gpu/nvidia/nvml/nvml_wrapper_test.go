/*
Copyright 2017 The Kubernetes Authors.

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

package nvml

import (
	"testing"
)

func TestNvmlWrapper(t *testing.T) {
	nvmlWrapper := NewNvmlWrapper()
	var err error
	// test NvmlInit(), this must be the first testing.
	// TODO: Return directly if nvml lib not exists. Will add conditional testing logic for nvml then.
	if err = nvmlWrapper.NvmlInit(); err != nil {
		t.Logf("NvmlInit() returns error: %v", err)
		return
	}

	// test NvmlGetDriverVersion()
	var version string
	if version, err = nvmlWrapper.NvmlGetDriverVersion(); err != nil {
		t.Errorf("NvmlGetDriverVersion() returns error: %v", err)
	}
	t.Logf("GPU driver version: %v", version)

	// test NvmlGetDeviceCount()
	var num uint
	if num, err = nvmlWrapper.NvmlGetDeviceCount(); err != nil {
		t.Errorf("NvmlGetDeviceCount() returns error: %v", err)
	}
	t.Logf("GPU number: %v", num)

	var idx uint
	for idx = 0; idx < num; idx++ {
		// test GetDeviceNameByIdx()
		var name string
		if name, err = nvmlWrapper.NvmlGetDeviceNameByIdx(idx); err != nil {
			t.Errorf("GetDeviceNameByIdx() returns error: %v", err)
		}
		t.Logf("GPU index %v, name: %v", idx, name)
		// test GetDeviceMinorByIdx()
		var minor int
		if minor, err = nvmlWrapper.NvmlGetDeviceMinorByIdx(idx); err != nil {
			t.Errorf("GetDeviceMinorByIdx() returns error: %v", err)
		}
		t.Logf("GPU index %v, minor: %v", idx, minor)
	}

	// test NvmlShutdown(), this must be the last testing.
	if err = nvmlWrapper.NvmlShutdown(); err != nil {
		t.Errorf("NvmlShutdown() returns error: %v", err)
	}
}
