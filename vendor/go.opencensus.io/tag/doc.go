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
//

/*
Package tag contains OpenCensus tags.

Tags are key-value pairs. Tags provide additional cardinality to
the OpenCensus instrumentation data.

Tags can be propagated on the wire and in the same
process via context.Context. Encode and Decode should be
used to represent tags into their binary propagation form.
*/
package tag // import "go.opencensus.io/tag"
