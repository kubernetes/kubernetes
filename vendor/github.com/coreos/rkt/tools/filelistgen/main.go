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
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/coreos/rkt/tools/common"
	"github.com/coreos/rkt/tools/common/filelist"
)

type fsWalker struct {
	dir  string
	list *filelist.Lists
}

func main() {
	dir := ""
	suffix := ""
	empty := false
	flag.StringVar(&dir, "directory", "", "Directory path")
	flag.StringVar(&suffix, "suffix", "", "Suffix for files in directory, when passed it basically does ${--dir}/*${suffix} (dotfiles are ignored)")
	flag.BoolVar(&empty, "empty", false, "Generate empty filelist")

	flag.Parse()
	if !empty && dir == "" {
		common.Die("No --directory parameter passed")
	}

	list := getListing(empty, dir, suffix)
	if err := list.GenerateFilelist(os.Stdout); err != nil {
		common.Die("Failed to generate a filelist: %v", err)
	}
}

func getListing(empty bool, dir, suffix string) *filelist.Lists {
	if empty {
		return &filelist.Lists{}
	}
	if suffix == "" {
		return getDeepListing(dir)
	}
	return getShallowListing(dir, suffix)
}

func getDeepListing(dir string) *filelist.Lists {
	walker := newFsWalker(common.MustAbs(dir))
	list, err := walker.getListing()
	if err != nil {
		common.Die("Error during getting listing from directory %q: %v", dir, err)
	}
	return list
}

func getShallowListing(dir, suffix string) *filelist.Lists {
	list := &filelist.Lists{}
	fiList, err := ioutil.ReadDir(dir)
	if err != nil {
		common.Die("Failed to read directory %q: %v", dir, err)
	}
	for _, fi := range fiList {
		name := fi.Name()
		if !strings.HasSuffix(name, suffix) {
			continue
		}
		if strings.HasPrefix(name, ".") {
			continue
		}
		if err := categorizeEntry(name, fi.Mode(), list); err != nil {
			common.Die("%v", err)
		}
	}
	return list
}

func newFsWalker(dir string) *fsWalker {
	return &fsWalker{
		dir:  dir,
		list: &filelist.Lists{},
	}
}

func (w *fsWalker) getListing() (*filelist.Lists, error) {
	src := w.dir
	if err := filepath.Walk(src, w.getWalkFunc(src)); err != nil {
		return nil, err
	}
	return w.list, nil
}

func (w *fsWalker) getWalkFunc(src string) filepath.WalkFunc {
	return func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rootLess := filepath.Clean(path[len(src):])
		// filepath.Clean guarantees that rootLess is never empty.
		if rootLess[0] == filepath.Separator {
			rootLess = rootLess[1:]
		}
		if rootLess == "." {
			// skip the toplevel directory
			return nil
		}
		if err := categorizeEntry(rootLess, info.Mode(), w.list); err != nil {
			return err
		}
		return nil
	}
}

func categorizeEntry(path string, mode os.FileMode, list *filelist.Lists) error {
	switch {
	case mode.IsDir():
		list.Dirs = append(list.Dirs, path)
	case mode.IsRegular():
		list.Files = append(list.Files, path)
	case isSymlink(mode):
		list.Symlinks = append(list.Symlinks, path)
	default:
		return fmt.Errorf("unsupported file mode: %d (not a file, directory or symlink)", mode&os.ModeType)
	}
	return nil
}

func isSymlink(mode os.FileMode) bool {
	return mode&os.ModeSymlink == os.ModeSymlink
}
