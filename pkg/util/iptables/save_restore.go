/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package iptables

import (
	"bytes"
	"fmt"
)

// MakeChainLine return an iptables-save/restore formatted chain line given a Chain
func MakeChainLine(chain Chain) string {
	return fmt.Sprintf(":%s - [0:0]", chain)
}

// GetChainsFromTable parses iptables-save data to find the chains that are defined. It
// assumes that save contains a single table's data, and returns a map with keys for every
// chain defined in that table.
func GetChainsFromTable(save []byte) map[Chain]struct{} {
	chainsMap := make(map[Chain]struct{})

	for {
		i := bytes.Index(save, []byte("\n:"))
		if i == -1 {
			break
		}
		start := i + 2
		save = save[start:]
		end := bytes.Index(save, []byte(" "))
		if i == -1 {
			// shouldn't happen, but...
			break
		}
		chain := Chain(save[:end])
		chainsMap[chain] = struct{}{}
		save = save[end:]
	}
	return chainsMap
}
