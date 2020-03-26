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

package tag

// Key represents a tag key.
type Key struct {
	name string
}

// NewKey creates or retrieves a string key identified by name.
// Calling NewKey consequently with the same name returns the same key.
func NewKey(name string) (Key, error) {
	if !checkKeyName(name) {
		return Key{}, errInvalidKeyName
	}
	return Key{name: name}, nil
}

// MustNewKey creates or retrieves a string key identified by name.
// An invalid key name raises a panic.
func MustNewKey(name string) Key {
	k, err := NewKey(name)
	if err != nil {
		panic(err)
	}
	return k
}

// Name returns the name of the key.
func (k Key) Name() string {
	return k.name
}
