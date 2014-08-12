// Copyright 2014 Google Inc. All Rights Reserved.
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

package info

// This struct describes one type of relationship between containers: One
// container, antagonist, interferes the performance of other
// containers, victims.
type Interference struct {
	// Absolute name of the antagonist container name. This field
	// should not be empty.
	Antagonist string `json:"antagonist"`

	// The absolute path of the victims. This field should not be empty.
	Victims []string `json:"victims"`

	// The name of the detector used to detect this antagonism. This field
	// should not be empty
	Detector string `json:"detector"`

	// Human readable description of this interference
	Description string `json:"description,omitempty"`
}
