// Copyright 2017 The etcd Authors
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

package namespace

func prefixInterval(pfx string, key, end []byte) (pfxKey []byte, pfxEnd []byte) {
	pfxKey = make([]byte, len(pfx)+len(key))
	copy(pfxKey[copy(pfxKey, pfx):], key)

	if len(end) == 1 && end[0] == 0 {
		// the edge of the keyspace
		pfxEnd = make([]byte, len(pfx))
		copy(pfxEnd, pfx)
		ok := false
		for i := len(pfxEnd) - 1; i >= 0; i-- {
			if pfxEnd[i]++; pfxEnd[i] != 0 {
				ok = true
				break
			}
		}
		if !ok {
			// 0xff..ff => 0x00
			pfxEnd = []byte{0}
		}
	} else if len(end) >= 1 {
		pfxEnd = make([]byte, len(pfx)+len(end))
		copy(pfxEnd[copy(pfxEnd, pfx):], end)
	}

	return pfxKey, pfxEnd
}
