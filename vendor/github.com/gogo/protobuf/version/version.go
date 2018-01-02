// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2013, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package version

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

func Get() string {
	versionBytes, _ := exec.Command("protoc", "--version").CombinedOutput()
	version := strings.TrimSpace(string(versionBytes))
	versions := strings.Split(version, " ")
	if len(versions) != 2 {
		panic("version string returned from protoc is seperated with a space: " + version)
	}
	return versions[1]
}

func parseVersion(version string) (int, error) {
	versions := strings.Split(version, ".")
	if len(versions) != 3 {
		return 0, fmt.Errorf("version does not have 3 numbers seperated by dots: %s", version)
	}
	n := 0
	for _, v := range versions {
		i, err := strconv.Atoi(v)
		if err != nil {
			return 0, err
		}
		n = n*10 + i
	}
	return n, nil
}

func less(this, that string) bool {
	thisNum, err := parseVersion(this)
	if err != nil {
		panic(err)
	}
	thatNum, err := parseVersion(that)
	if err != nil {
		panic(err)
	}
	return thisNum <= thatNum
}

func AtLeast(v string) bool {
	return less(v, Get())
}
