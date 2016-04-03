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

package main

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/appc/spec/aci"
	"github.com/appc/spec/schema"
)

var (
	buildNocompress bool
	buildOverwrite  bool
	buildOwnerRoot  bool
	cmdBuild        = &Command{
		Name: "build",
		Description: `Build an ACI from a given directory. The directory should
contain an Image Layout. The Image Layout will be validated
before the ACI is created. The produced ACI will be
gzip-compressed by default.`,
		Summary: "Build an ACI from an Image Layout (experimental)",
		Usage:   `[--overwrite] [--no-compression] [--owner-root] DIRECTORY OUTPUT_FILE`,
		Run:     runBuild,
	}
)

func init() {
	cmdBuild.Flags.BoolVar(&buildOverwrite, "overwrite", false, "Overwrite target file if it already exists")
	cmdBuild.Flags.BoolVar(&buildOwnerRoot, "owner-root", false, "Force ownership to root:root on all files")
	cmdBuild.Flags.BoolVar(&buildNocompress, "no-compression", false, "Do not gzip-compress the produced ACI")
}

func runBuild(args []string) (exit int) {
	if len(args) != 2 {
		stderr("build: Must provide directory and output file")
		return 1
	}

	root := args[0]
	tgt := args[1]
	ext := filepath.Ext(tgt)
	if ext != schema.ACIExtension {
		stderr("build: Extension must be %s (given %s)", schema.ACIExtension, ext)
		return 1
	}

	// TODO(jonboulle): stream the validation so we don't have to walk the rootfs twice
	if err := aci.ValidateLayout(root); err != nil {
		if e, ok := err.(aci.ErrOldVersion); ok {
			stderr("build: Warning: %v. Please update your manifest.", e)
		} else {
			stderr("build: Layout failed validation: %v", err)
			return 1
		}
	}

	mode := os.O_CREATE | os.O_WRONLY
	if buildOverwrite {
		mode |= os.O_TRUNC
	} else {
		mode |= os.O_EXCL
	}
	fh, err := os.OpenFile(tgt, mode, 0644)
	if err != nil {
		if os.IsExist(err) {
			stderr("build: Target file exists (try --overwrite)")
		} else {
			stderr("build: Unable to open target %s: %v", tgt, err)
		}
		return 1
	}

	var gw *gzip.Writer
	var r io.WriteCloser = fh
	if !buildNocompress {
		gw = gzip.NewWriter(fh)
		r = gw
	}
	tr := tar.NewWriter(r)

	defer func() {
		tr.Close()
		if !buildNocompress {
			gw.Close()
		}
		fh.Close()
		if exit != 0 && !buildOverwrite {
			os.Remove(tgt)
		}
	}()

	mpath := filepath.Join(root, aci.ManifestFile)
	b, err := ioutil.ReadFile(mpath)
	if err != nil {
		stderr("build: Unable to read Image Manifest: %v", err)
		return 1
	}
	var im schema.ImageManifest
	if err := im.UnmarshalJSON(b); err != nil {
		stderr("build: Unable to load Image Manifest: %v", err)
		return 1
	}
	iw := aci.NewImageWriter(im, tr)

	var walkerCb aci.TarHeaderWalkFunc
	if buildOwnerRoot {
		walkerCb = func(hdr *tar.Header) bool {
			hdr.Uid, hdr.Gid = 0, 0
			hdr.Uname, hdr.Gname = "root", "root"
			return true
		}
	}

	err = filepath.Walk(root, aci.BuildWalker(root, iw, walkerCb))
	if err != nil {
		stderr("build: Error walking rootfs: %v", err)
		return 1
	}

	err = iw.Close()
	if err != nil {
		stderr("build: Unable to close image %s: %v", tgt, err)
		return 1
	}

	return
}
