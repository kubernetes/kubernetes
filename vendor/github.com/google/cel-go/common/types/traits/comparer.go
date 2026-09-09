// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package traits

import (
	"github.com/google/cel-go/common/types/ref"
)

// Comparer interface for ordering comparisons between values in order to
// support '<', '<=', '>=', '>' overloads.
type Comparer interface {
	// Compare this value to the input other value, returning an Int:
	//
	//    this < other  -> Int(-1)
	//    this == other ->  Int(0)
	//    this > other  ->  Int(1)
	//
	// If the comparison cannot be made or is not supported, an error should
	// be returned.
	Compare(other ref.Val) ref.Val
}
