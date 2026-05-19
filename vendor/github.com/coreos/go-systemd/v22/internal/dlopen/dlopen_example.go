// Copyright 2015 CoreOS, Inc.
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
//go:build linux
// +build linux

package dlopen

// #include <string.h>
// #include <stdlib.h>
//
// int
// my_strlen(void *f, const char *s)
// {
//   size_t (*strlen)(const char *);
//
//   strlen = (size_t (*)(const char *))f;
//   return strlen(s);
// }
import "C"

import (
	"fmt"
	"unsafe"
)

func strlen(libs []string, s string) (int, error) {
	h, err := GetHandle(libs)
	if err != nil {
		return -1, fmt.Errorf(`couldn't get a handle to the library: %v`, err)
	}
	defer h.Close()

	f := "strlen"
	cs := C.CString(s)
	defer C.free(unsafe.Pointer(cs))

	strlen, err := h.GetSymbolPointer(f)
	if err != nil {
		return -1, fmt.Errorf(`couldn't get symbol %q: %v`, f, err)
	}

	len := C.my_strlen(strlen, cs)

	return int(len), nil
}
