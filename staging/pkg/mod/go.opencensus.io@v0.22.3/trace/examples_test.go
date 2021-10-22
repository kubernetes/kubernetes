// Copyright 2017, OpenCensus Authors
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

package trace_test

import (
	"fmt"

	"context"
	"go.opencensus.io/trace"
)

// This example shows how to use StartSpan and (*Span).End to capture
// a function execution in a Span. It assumes that the function
// has a context.Context argument.
func ExampleStartSpan() {
	printEvens := func(ctx context.Context) {
		ctx, span := trace.StartSpan(ctx, "my/package.Function")
		defer span.End()

		for i := 0; i < 10; i++ {
			if i%2 == 0 {
				fmt.Printf("Even!\n")
			}
		}
	}

	ctx := context.Background()
	printEvens(ctx)
}
