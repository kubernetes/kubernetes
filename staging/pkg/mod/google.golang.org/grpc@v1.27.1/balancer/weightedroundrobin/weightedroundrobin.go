/*
 *
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package weightedroundrobin defines a weighted roundrobin balancer.
package weightedroundrobin

// Name is the name of weighted_round_robin balancer.
const Name = "weighted_round_robin"

// AddrInfo will be stored inside Address metadata in order to use weighted roundrobin
// balancer.
type AddrInfo struct {
	Weight uint32
}
