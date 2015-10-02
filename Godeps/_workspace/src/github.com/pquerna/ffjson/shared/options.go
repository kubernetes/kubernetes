/**
 *  Copyright 2014 Paul Querna, Klaus Post
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

package shared

type StructOptions struct {
	SkipDecoder bool
	SkipEncoder bool
}

type InceptionType struct {
	Obj     interface{}
	Options StructOptions
}
type Feature int

const (
	Nothing     Feature = 0
	MustDecoder         = 1 << 1
	MustEncoder         = 1 << 2
	MustEncDec          = MustDecoder | MustEncoder
)

func (i InceptionType) HasFeature(f Feature) bool {
	return i.HasFeature(f)
}

func (s StructOptions) HasFeature(f Feature) bool {
	hasNeeded := true
	if f&MustDecoder != 0 && s.SkipDecoder {
		hasNeeded = false
	}
	if f&MustEncoder != 0 && s.SkipEncoder {
		hasNeeded = false
	}
	return hasNeeded
}
