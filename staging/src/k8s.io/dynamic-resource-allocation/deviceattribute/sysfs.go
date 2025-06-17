/*
Copyright 2025 The Kubernetes Authors.

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

package deviceattribute

import "path/filepath"

const (
	defaultSysfsRoot = "/sys"
)

// sysfs provides methods to construct sysfs paths for various subsystems.
// It is used to abstract the sysfs path construction
// and can be replaced with a mock in tests.
type sysfs string

func (s sysfs) Devices(path string) string {
	return filepath.Join(string(s), "devices", path)
}

func (s sysfs) Bus(path string) string {
	return filepath.Join(string(s), "bus", path)
}

func (s sysfs) Block(path string) string {
	return filepath.Join(string(s), "block", path)
}

func (s sysfs) Class(path string) string {
	return filepath.Join(string(s), "class", path)
}

func (s sysfs) Dev(path string) string {
	return filepath.Join(string(s), "dev", path)
}

func (s sysfs) Firmware(path string) string {
	return filepath.Join(string(s), "firmware", path)
}
func (s sysfs) Kernel(path string) string {
	return filepath.Join(string(s), "kernel", path)
}

func (s sysfs) Module(path string) string {
	return filepath.Join(string(s), "module", path)
}
