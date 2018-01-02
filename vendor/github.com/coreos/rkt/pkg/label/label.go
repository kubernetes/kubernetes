// Copyright 2014,2015 Red Hat, Inc
// Copyright 2014,2015 Docker, Inc
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

// +build !selinux !linux

package label

// InitLabels returns the process label and file labels to be used within
// the container.  A list of options can be passed into this function to alter
// the labels.
func InitLabels(mcsdir string, options []string) (string, string, error) {
	return "", "", nil
}

func FormatMountLabel(src string, mountLabel string) string {
	return src
}

func SetProcessLabel(processLabel string) error {
	return nil
}

func SetFileLabel(path string, fileLabel string) error {
	return nil
}

func SetFileCreateLabel(fileLabel string) error {
	return nil
}

func Relabel(path string, fileLabel string, relabel string) error {
	return nil
}

func GetPidLabel(pid int) (string, error) {
	return "", nil
}

func Init() {
}

func ReserveLabel(label string) error {
	return nil
}

func UnreserveLabel(label string) error {
	return nil
}

// DupSecOpt takes an process label and returns security options that
// can be used to set duplicate labels on future container processes
func DupSecOpt(src string) []string {
	return nil
}

// DisableSecOpt returns a security opt that can disable labeling
// support for future container processes
func DisableSecOpt() []string {
	return nil
}
