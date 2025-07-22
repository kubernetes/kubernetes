//go:build windows
// +build windows

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

package eviction

// DefaultEvictionHard includes default options for hard eviction for Windows.
// Note: On Linux, the default eviction hard threshold is 100Mi for memory.available
// but Windows generally requires more memory reserved for system processes so we'll
// set the default threshold to 500Mi.
var DefaultEvictionHard = map[string]string{
	"memory.available":  "500Mi",
	"nodefs.available":  "10%",
	"imagefs.available": "15%",
}
