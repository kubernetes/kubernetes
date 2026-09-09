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

package flag

// OmitEmpty is an interface for flags to report whether their underlying value
// is "empty." If a flag implements OmitEmpty and returns true for a call to Empty(),
// it is assumed that flag may be omitted from the command line.
type OmitEmpty interface {
	Empty() bool
}
