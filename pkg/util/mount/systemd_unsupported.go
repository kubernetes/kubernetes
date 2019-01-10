// +build !linux

/*
Copyright 2018 The Kubernetes Authors.

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

package mount

// DetectSystemd returns true if OS runs with systemd as init.
func DetectSystemd(exec Exec, systemdRunPath string) bool {
	return false
}

// AddSystemdScope adds "system-run --scope" to given command line.
func AddSystemdScope(systemdRunPath string, description, command string, args []string) (string, []string) {
	return command, args
}
