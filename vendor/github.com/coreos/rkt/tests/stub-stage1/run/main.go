// Copyright 2016 The rkt Authors
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

package main

import (
	"fmt"
	"os"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/sys"
	stage1common "github.com/coreos/rkt/stage1/common"
)

func main() {
	os.Exit(run())
}

func run() int {
	lfd, err := common.GetRktLockFD()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to get rkt lock fd: %v\n", err)
		return 254
	}

	if err := sys.CloseOnExec(lfd, true); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to set FD_CLOEXEC on rkt lock: %v\n", err)
		return 254
	}

	if err := stage1common.WritePid(os.Getpid(), "ppid"); err != nil {
		fmt.Fprintf(os.Stderr, "write ppid: %v", err)
		return 254
	}
	fmt.Println("success, stub stage1 would at this point switch to stage2")
	return 0
}
