// Copyright 2018, OpenCensus Authors
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

// +build go1.9

package tag

import (
	"context"
	"runtime/pprof"
)

func do(ctx context.Context, f func(ctx context.Context)) {
	m := FromContext(ctx)
	keyvals := make([]string, 0, 2*len(m.m))
	for k, v := range m.m {
		keyvals = append(keyvals, k.Name(), v)
	}
	pprof.Do(ctx, pprof.Labels(keyvals...), f)
}
