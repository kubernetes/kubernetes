// Copyright 2019, OpenCensus Authors
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
//

package tag

const (
	// valueTTLNoPropagation prevents tag from propagating.
	valueTTLNoPropagation = 0

	// valueTTLUnlimitedPropagation allows tag to propagate without any limits on number of hops.
	valueTTLUnlimitedPropagation = -1
)

// TTL is metadata that specifies number of hops a tag can propagate.
// Details about TTL metadata is specified at https://github.com/census-instrumentation/opencensus-specs/blob/master/tags/TagMap.md#tagmetadata
type TTL struct {
	ttl int
}

var (
	// TTLUnlimitedPropagation is TTL metadata that allows tag to propagate without any limits on number of hops.
	TTLUnlimitedPropagation = TTL{ttl: valueTTLUnlimitedPropagation}

	// TTLNoPropagation is TTL metadata that prevents tag from propagating.
	TTLNoPropagation = TTL{ttl: valueTTLNoPropagation}
)

type metadatas struct {
	ttl TTL
}

// Metadata applies metadatas specified by the function.
type Metadata func(*metadatas)

// WithTTL applies metadata with provided ttl.
func WithTTL(ttl TTL) Metadata {
	return func(m *metadatas) {
		m.ttl = ttl
	}
}
