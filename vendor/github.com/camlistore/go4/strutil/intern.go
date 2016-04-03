/*
Copyright 2013 The Camlistore Authors

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

package strutil

var internStr = map[string]string{}

// RegisterCommonString adds common strings to the interned string
// table.  This should be called during init from the main
// goroutine, not later at runtime.
func RegisterCommonString(s ...string) {
	for _, v := range s {
		internStr[v] = v
	}
}

// StringFromBytes returns string(v), minimizing copies for common values of v
// as previously registered with RegisterCommonString.
func StringFromBytes(v []byte) string {
	// In Go 1.3, this string conversion in the map lookup does not allocate
	// to make a new string. We depend on Go 1.3, so this is always free:
	if s, ok := internStr[string(v)]; ok {
		return s
	}
	return string(v)
}
