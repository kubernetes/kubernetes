// Copyright (c) 2020-2022 Denis Tingaikin
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package goheader

import "fmt"

type Location struct {
	Line     int
	Position int
}

func (l Location) String() string {
	return fmt.Sprintf("%v:%v", l.Line+1, l.Position)
}

func (l Location) Add(other Location) Location {
	return Location{
		Line:     l.Line + other.Line,
		Position: l.Position + other.Position,
	}
}
