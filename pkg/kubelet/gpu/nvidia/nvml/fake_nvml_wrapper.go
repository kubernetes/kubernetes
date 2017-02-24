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

type FakeNvmlWrapper struct {
	fakeNumber  uint
	fakeName    string
	fakeVersion string
}

/* Helper functions */
// NvmlInit() initializes nvml lib.
func (nvml *FakeNvmlWrapper) NvmlInit() error {
	return nil
}

// NvmlShutdown() shutdowns nvml lib.
func (nvml *FakeNvmlWrapper) NvmlShutdown() error {
	return nil
}

// NvmlGetDriverVersion() returns GPU driver version.
func (nvml *FakeNvmlWrapper) NvmlGetDriverVersion() (string, error) {
	return nvml.fakeVersion, nil
}

// NvmlGetDeviceCount() returns GPU numbers detected by nvml
func (nvml *FakeNvmlWrapper) NvmlGetDeviceCount() (uint, error) {
	return nvml.fakeNumber, nil
}

// NvmlGetDeviceNameByIdx(idx uint) returns GPU name (such as "TeslaK80") by its index.
func (nvml *FakeNvmlWrapper) NvmlGetDeviceNameByIdx(idx uint) (string, error) {
	return nvml.fakeName, nil
}

// NvmlGetDeviceMinorByIdx(idx uint) returns GPU minor number X used in "/dev/nvidiaX" by its index. -1 for error.
func (nvml *FakeNvmlWrapper) NvmlGetDeviceMinorByIdx(idx uint) (int, error) {
	return int(idx), nil
}

func NewFakeNvmlWrapper(fakeNumber uint, fakeName string, fakeVersion string) Nvml {
	return &FakeNvmlWrapper{
		fakeNumber:  fakeNumber,
		fakeName:    fakeName,
		fakeVersion: fakeVersion,
	}
}
