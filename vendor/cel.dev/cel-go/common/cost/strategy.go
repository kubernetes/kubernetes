// Copyright 2026 Google LLC
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

package cost

import (
	"cel.dev/cel-go/common/types/ref"
)

// SizingStrategy defines how sizes and container element sizes are estimated and tracked.
type SizingStrategy interface {
	// EstimateSize computes the size estimate for an AST node during cost estimation.
	// For lists and maps, the returned SizeEstimate can recursively include Elem and Key sizes.
	EstimateSize(ctx EstimateContext, node AstNode) (SizeEstimate, bool)

	// TrackSize computes the actual runtime size of a value during cost tracking.
	TrackSize(ctx TrackContext, value ref.Val) (uint64, bool)
}
