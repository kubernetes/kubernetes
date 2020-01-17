/*
Package to provides helpers to ease working with pointer values of marshalled structures.
*/
package to

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// String returns a string value for the passed string pointer. It returns the empty string if the
// pointer is nil.
func String(s *string) string {
	if s != nil {
		return *s
	}
	return ""
}

// StringPtr returns a pointer to the passed string.
func StringPtr(s string) *string {
	return &s
}

// StringSlice returns a string slice value for the passed string slice pointer. It returns a nil
// slice if the pointer is nil.
func StringSlice(s *[]string) []string {
	if s != nil {
		return *s
	}
	return nil
}

// StringSlicePtr returns a pointer to the passed string slice.
func StringSlicePtr(s []string) *[]string {
	return &s
}

// StringMap returns a map of strings built from the map of string pointers. The empty string is
// used for nil pointers.
func StringMap(msp map[string]*string) map[string]string {
	ms := make(map[string]string, len(msp))
	for k, sp := range msp {
		if sp != nil {
			ms[k] = *sp
		} else {
			ms[k] = ""
		}
	}
	return ms
}

// StringMapPtr returns a pointer to a map of string pointers built from the passed map of strings.
func StringMapPtr(ms map[string]string) *map[string]*string {
	msp := make(map[string]*string, len(ms))
	for k, s := range ms {
		msp[k] = StringPtr(s)
	}
	return &msp
}

// Bool returns a bool value for the passed bool pointer. It returns false if the pointer is nil.
func Bool(b *bool) bool {
	if b != nil {
		return *b
	}
	return false
}

// BoolPtr returns a pointer to the passed bool.
func BoolPtr(b bool) *bool {
	return &b
}

// Int returns an int value for the passed int pointer. It returns 0 if the pointer is nil.
func Int(i *int) int {
	if i != nil {
		return *i
	}
	return 0
}

// IntPtr returns a pointer to the passed int.
func IntPtr(i int) *int {
	return &i
}

// Int32 returns an int value for the passed int pointer. It returns 0 if the pointer is nil.
func Int32(i *int32) int32 {
	if i != nil {
		return *i
	}
	return 0
}

// Int32Ptr returns a pointer to the passed int32.
func Int32Ptr(i int32) *int32 {
	return &i
}

// Int64 returns an int value for the passed int pointer. It returns 0 if the pointer is nil.
func Int64(i *int64) int64 {
	if i != nil {
		return *i
	}
	return 0
}

// Int64Ptr returns a pointer to the passed int64.
func Int64Ptr(i int64) *int64 {
	return &i
}

// Float32 returns an int value for the passed int pointer. It returns 0.0 if the pointer is nil.
func Float32(i *float32) float32 {
	if i != nil {
		return *i
	}
	return 0.0
}

// Float32Ptr returns a pointer to the passed float32.
func Float32Ptr(i float32) *float32 {
	return &i
}

// Float64 returns an int value for the passed int pointer. It returns 0.0 if the pointer is nil.
func Float64(i *float64) float64 {
	if i != nil {
		return *i
	}
	return 0.0
}

// Float64Ptr returns a pointer to the passed float64.
func Float64Ptr(i float64) *float64 {
	return &i
}

// ByteSlicePtr returns a pointer to the passed byte slice.
func ByteSlicePtr(b []byte) *[]byte {
	return &b
}
