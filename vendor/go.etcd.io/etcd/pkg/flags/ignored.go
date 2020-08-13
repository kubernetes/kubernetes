// Copyright 2018 The etcd Authors
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

package flags

// IgnoredFlag encapsulates a flag that may have been previously valid but is
// now ignored. If an IgnoredFlag is set, a warning is printed and
// operation continues.
type IgnoredFlag struct {
	Name string
}

// IsBoolFlag is defined to allow the flag to be defined without an argument
func (f *IgnoredFlag) IsBoolFlag() bool {
	return true
}

func (f *IgnoredFlag) Set(s string) error {
	plog.Warningf(`flag "-%s" is no longer supported - ignoring.`, f.Name)
	return nil
}

func (f *IgnoredFlag) String() string {
	return ""
}
