// Copyright 2016 The appc Authors
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

// +build linux

package device

// with glibc/sysdeps/unix/sysv/linux/sys/sysmacros.h as reference

func Major(rdev uint64) uint {
	return uint((rdev>>8)&0xfff) | (uint(rdev>>32) & ^uint(0xfff))
}

func Minor(rdev uint64) uint {
	return uint(rdev&0xff) | uint(uint32(rdev>>12) & ^uint32(0xff))
}

func Makedev(maj uint, min uint) uint64 {
	return uint64(min&0xff) | (uint64(maj&0xfff) << 8) |
		((uint64(min) & ^uint64(0xff)) << 12) |
		((uint64(maj) & ^uint64(0xfff)) << 32)
}
