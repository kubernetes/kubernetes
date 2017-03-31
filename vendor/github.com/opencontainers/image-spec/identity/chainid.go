// Copyright 2016 The Linux Foundation
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

// Package identity provides implementations of subtle calculations pertaining
// to image and layer identity.  The primary item present here is the ChainID
// calculation used in identifying the result of subsequent layer applications.
//
// Helpers are also provided here to ease transition to the
// github.com/opencontainers/go-digest package, but that package may be used
// directly.
package identity

import "github.com/opencontainers/go-digest"

// ChainID takes a slice of digests and returns the ChainID corresponding to
// the last entry. Typically, these are a list of layer DiffIDs, with the
// result providing the ChainID identifying the result of sequential
// application of the preceding layers.
func ChainID(dgsts []digest.Digest) digest.Digest {
	chainIDs := make([]digest.Digest, len(dgsts))
	copy(chainIDs, dgsts)
	ChainIDs(chainIDs)

	if len(chainIDs) == 0 {
		return ""
	}
	return chainIDs[len(chainIDs)-1]
}

// ChainIDs calculates the recursively applied chain id for each identifier in
// the slice. The result is written direcly back into the slice such that the
// ChainID for each item will be in the respective position.
//
// By definition of ChainID, the zeroth element will always be the same before
// and after the call.
//
// As an example, given the chain of ids `[A, B, C]`, the result `[A,
// ChainID(A|B), ChainID(A|B|C)]` will be written back to the slice.
//
// The input is provided as a return value for convenience.
//
// Typically, these are a list of layer DiffIDs, with the
// result providing the ChainID for each the result of each layer application
// sequentially.
func ChainIDs(dgsts []digest.Digest) []digest.Digest {
	if len(dgsts) < 2 {
		return dgsts
	}

	parent := digest.FromBytes([]byte(dgsts[0] + " " + dgsts[1]))
	next := dgsts[1:]
	next[0] = parent
	ChainIDs(next)

	return dgsts
}
