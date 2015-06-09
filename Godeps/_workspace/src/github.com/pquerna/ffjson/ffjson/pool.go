package ffjson

/**
 *  Copyright 2015 Paul Querna, Klaus Post
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

import (
	fflib "github.com/pquerna/ffjson/fflib/v1"
)

// Send a buffer to the Pool to reuse for other instances.
//
// On servers where you have a lot of concurrent encoding going on,
// you can hand back the byte buffer you get marshalling once you are done using it.
//
// You may no longer utilize the content of the buffer, since it may be used
// by other goroutines.
func Pool(b []byte) {
	fflib.Pool(b)
}
