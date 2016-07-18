// Copyright 2015 flannel authors
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

package ip

// Taken from a patch by David Anderson who submitted it
// but got rejected by the golang team

import (
	"encoding/binary"
	"unsafe"
)

// NativeEndian is the ByteOrder of the current system.
var NativeEndian binary.ByteOrder

func init() {
	// Examine the memory layout of an int16 to determine system
	// endianness.
	var one int16 = 1
	b := (*byte)(unsafe.Pointer(&one))
	if *b == 0 {
		NativeEndian = binary.BigEndian
	} else {
		NativeEndian = binary.LittleEndian
	}
}

func NativelyLittle() bool {
	return NativeEndian == binary.LittleEndian
}
