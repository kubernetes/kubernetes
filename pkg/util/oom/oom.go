/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package oom

// This is a struct instead of an interface to allow injection of process ID listers and
// applying OOM score in tests.
// TODO: make this an interface, and inject a mock ioutil struct for testing.
type OomAdjuster struct {
	pidLister                 func(cgroupName string) ([]int, error)
	ApplyOomScoreAdj          func(pid int, oomScoreAdj int) error
	ApplyOomScoreAdjContainer func(cgroupName string, oomScoreAdj, maxTries int) error
}
