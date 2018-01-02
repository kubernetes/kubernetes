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
	"path/filepath"
	"strings"

	"github.com/coreos/rkt/tools/common"
	"github.com/coreos/rkt/tools/common/filelist"
	"github.com/hashicorp/errwrap"
)

const (
	globCmd = "glob"
	// globMakeFunction is a template for generating all files for
	// given set of wildcards. See globMakeWildcard.
	globMakeFunction = `$(strip \
        $(eval _DEPS_GEN_FG_ := $(strip !!!WILDCARDS!!!)) \
        $(if $(_DEPS_GEN_FG_),$(shell LC_ALL=C stat --format "%n: %F" $(_DEPS_GEN_FG_) | grep -e 'regular file$$' | cut -f1 -d:)))
`
	// globMakeWildcard is a template for call wildcard function
	// for in a given directory with a given suffix. This wildcard
	// is for normal files.
	globMakeWildcard = "$(wildcard !!!DIR!!!/*!!!SUFFIX!!!)"
	// globMakeHiddenWildcard is a template for call wildcard
	// function for in a given directory with a given suffix. This
	// wildcard is for files beginning with a dot, which are
	// normally not taken into account by wildcard.
	globMakeHiddenWildcard = "$(wildcard !!!DIR!!!/.*!!!SUFFIX!!!)"
)

type globMode int

const (
	globNormal globMode = iota
	globDotFiles
	globAll
)

type globArgs struct {
	target   string
	suffix   string
	mode     globMode
	filelist string
	mapTo    []string
}

func init() {
	cmds[globCmd] = globDeps
}

func globDeps(args []string) string {
	parsedArgs := globGetArgs(args)
	files := globGetFiles(parsedArgs)
	makeFunction := globGetMakeFunction(files, parsedArgs.suffix, parsedArgs.mode)
	return GenerateFileDeps(parsedArgs.target, makeFunction, files)
}

// globGetArgs parses given parameters and returns a target, a suffix
// and a list of files.
func globGetArgs(args []string) globArgs {
	f, target := standardFlags(globCmd)
	suffix := f.String("suffix", "", "File suffix (example: .go)")
	globbingMode := f.String("glob-mode", "all", "Which files to glob (normal, dot-files, all [default])")
	filelist := f.String("filelist", "", "Read all the files from this file")
	var mapTo []string
	mapToWrapper := common.StringSliceWrapper{Slice: &mapTo}
	f.Var(&mapToWrapper, "map-to", "Map contents of filelist to this directory, can be used multiple times")

	f.Parse(args)
	if *target == "" {
		common.Die("--target parameter must be specified and cannot be empty")
	}
	mode := globModeFromString(*globbingMode)
	if *filelist == "" {
		common.Die("--filelist parameter must be specified and cannot be empty")
	}
	if len(mapTo) < 1 {
		common.Die("--map-to parameter must be specified at least once")
	}
	return globArgs{
		target:   *target,
		suffix:   *suffix,
		mode:     mode,
		filelist: *filelist,
		mapTo:    mapTo,
	}
}

func globModeFromString(mode string) globMode {
	switch mode {
	case "normal":
		return globNormal
	case "dot-files":
		return globDotFiles
	case "all":
		return globAll
	default:
		common.Die("Unknown glob mode %q", mode)
	}
	panic("Should not happen")
}

func globGetFiles(args globArgs) []string {
	f, err := globGetFilesFromFilelist(args.filelist)
	if err != nil {
		common.Die("Failed to get files from filelist %q: %v", args.filelist, err)
	}
	return common.MapFilesToDirectories(f, args.mapTo)
}

func globGetFilesFromFilelist(filename string) ([]string, error) {
	fl, err := os.Open(filename)
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("failed to open filelist %q", filename), err)
	}
	defer fl.Close()
	lists := filelist.Lists{}
	if err := lists.ParseFilelist(fl); err != nil {
		return nil, err
	}
	return lists.Files, nil
}

// globGetMakeFunction returns a make snippet which calls wildcard
// function in all directories where given files are and with a given
// suffix.
func globGetMakeFunction(files []string, suffix string, mode globMode) string {
	dirs := map[string]struct{}{}
	for _, file := range files {
		dirs[filepath.Dir(file)] = struct{}{}
	}
	makeWildcards := make([]string, 0, len(dirs))
	wildcard := globGetMakeSnippet(mode)
	for dir := range dirs {
		str := replacePlaceholders(wildcard, "SUFFIX", suffix, "DIR", dir)
		makeWildcards = append(makeWildcards, str)
	}
	return replacePlaceholders(globMakeFunction, "WILDCARDS", strings.Join(makeWildcards, " "))
}

func globGetMakeSnippet(mode globMode) string {
	switch mode {
	case globNormal:
		return globMakeWildcard
	case globDotFiles:
		return globMakeHiddenWildcard
	case globAll:
		return fmt.Sprintf("%s %s", globMakeWildcard, globMakeHiddenWildcard)
	}
	panic("Should not happen")
}
