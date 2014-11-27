/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// flags2yaml is a tool to generate flat YAML from command-line flags
//
// $ flags2yaml name=foo image=busybox | simplegen - | kubectl apply -

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/golang/glog"
)

const usage = "usage: flags2yaml key1=value1 [key2=value2 ...]"

func main() {
	for i := 1; i < len(os.Args); i++ {
		pieces := strings.Split(os.Args[i], "=")
		if len(pieces) != 2 {
			glog.Fatalf("Bad arg: %s", os.Args[i])
		}
		fmt.Printf("%s: %s\n", pieces[0], pieces[1])
	}
}
