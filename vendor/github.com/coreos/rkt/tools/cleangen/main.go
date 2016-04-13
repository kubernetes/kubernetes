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
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/coreos/rkt/tools/common"
	"github.com/coreos/rkt/tools/common/filelist"
)

type mkPair struct {
	name string
	data *[]string
}

func main() {
	filename, mapTo := getValidArgs()
	list := getListing(filename)
	pairs := getPairs(list)
	for _, pair := range pairs {
		toClean := common.MapFilesToDirectories(*pair.data, mapTo)
		fmt.Printf("%s += %s\n", pair.name, strings.Join(toClean, " "))
	}
}

func getValidArgs() (string, []string) {
	filename := ""
	var mapTo []string
	mapToWrapper := common.StringSliceWrapper{
		Slice: &mapTo,
	}

	flag.StringVar(&filename, "filelist", "", "Name of the source (path to the directory or to the filelist")
	flag.Var(&mapToWrapper, "map-to", "Map contents of traversed source to this directory, can be used multiple times")

	flag.Parse()
	if filename == "" {
		common.Die("No --filelist parameter passed")
	}
	if len(mapTo) < 1 {
		common.Die("No --map-to parameter passed, at least one is required")
		os.Exit(1)
	}

	filename = common.MustAbs(filename)
	for i := 0; i < len(mapTo); i++ {
		mapTo[i] = common.MustAbs(mapTo[i])
	}
	return filename, mapTo
}

func getListing(filename string) *filelist.Lists {
	fl, err := os.Open(filename)
	if err != nil {
		common.Die("Failed to open filelist %q: %v", filename, err)
	}
	defer fl.Close()

	list := &filelist.Lists{}
	if err := list.ParseFilelist(fl); err != nil {
		common.Die("Error during getting listing from filelist %q: %v\n", filename, err)
	}
	return list
}

func getPairs(list *filelist.Lists) []mkPair {
	return []mkPair{
		{name: "CLEAN_FILES", data: &list.Files},
		{name: "CLEAN_SYMLINKS", data: &list.Symlinks},
		{name: "CLEAN_DIRS", data: &list.Dirs},
	}
}
