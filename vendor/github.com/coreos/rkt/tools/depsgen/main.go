// Copyright 2015 The rkt Authors
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
	"sort"
	"strings"

	"github.com/coreos/rkt/tools/common"
)

var cmds = make(map[string]func([]string) string)

func main() {
	depTypes := getAllDepTypes()
	depTypesString := fmt.Sprintf("'%s'", strings.Join(depTypes, "', '"))
	if len(os.Args) < 2 {
		common.Die("Expected a deps type (possible values: %s)", depTypesString)
	}
	depType := os.Args[1]
	cmdArgs := os.Args[2:]
	if f, ok := cmds[depType]; ok {
		fmt.Print(f(cmdArgs))
	} else if depType == "--help" || depType == "-help" {
		common.Warn("Run %s with one of the following commands: %s\nE.g. %s %s --help", appName(), depTypesString, os.Args[0], depTypes[0])
	} else {
		common.Die("Unknown deps type: %q, expected one of %s", depType, depTypesString)
	}
}

// getAllDepTypes returns a sorted list of names of all dep type
// commands.
func getAllDepTypes() []string {
	depTypes := make([]string, 0, len(cmds))
	for depType := range cmds {
		depTypes = append(depTypes, depType)
	}
	sort.Strings(depTypes)
	return depTypes
}
