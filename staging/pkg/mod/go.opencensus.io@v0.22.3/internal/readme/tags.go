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

package readme

import (
	"context"
	"log"

	"go.opencensus.io/tag"
)

func tagsExamples() {
	ctx := context.Background()

	osKey := tag.MustNewKey("example.com/keys/user-os")
	userIDKey := tag.MustNewKey("example.com/keys/user-id")

	// START new
	ctx, err := tag.New(ctx,
		tag.Insert(osKey, "macOS-10.12.5"),
		tag.Upsert(userIDKey, "cde36753ed"),
	)
	if err != nil {
		log.Fatal(err)
	}
	// END new

	// START profiler
	ctx, err = tag.New(ctx,
		tag.Insert(osKey, "macOS-10.12.5"),
		tag.Insert(userIDKey, "fff0989878"),
	)
	if err != nil {
		log.Fatal(err)
	}
	tag.Do(ctx, func(ctx context.Context) {
		// Do work.
		// When profiling is on, samples will be
		// recorded with the key/values from the tag map.
	})
	// END profiler
}
