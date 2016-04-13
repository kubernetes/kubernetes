// Copyright 2015 The appc Authors
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

// +build linux freebsd netbsd openbsd darwin

package device

/*
#define _BSD_SOURCE
#define _DEFAULT_SOURCE
#include <sys/types.h>

unsigned int
my_major(dev_t dev)
{
  return major(dev);
}

unsigned int
my_minor(dev_t dev)
{
  return minor(dev);
}

dev_t
my_makedev(unsigned int maj, unsigned int min)
{
       return makedev(maj, min);
}
*/
import "C"

func Major(rdev uint64) uint {
	major := C.my_major(C.dev_t(rdev))
	return uint(major)
}

func Minor(rdev uint64) uint {
	minor := C.my_minor(C.dev_t(rdev))
	return uint(minor)
}

func Makedev(maj uint, min uint) uint64 {
	dev := C.my_makedev(C.uint(maj), C.uint(min))
	return uint64(dev)
}
