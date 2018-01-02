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
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"

	"github.com/coreos/rkt/pkg/fileutil"
	"github.com/coreos/rkt/pkg/tar"
	"github.com/coreos/rkt/pkg/user"
	"github.com/coreos/rkt/store/imagestore"

	"github.com/spf13/cobra"
)

var (
	cmdImageExtract = &cobra.Command{
		Use:   "extract IMAGE OUTPUT_DIR",
		Short: "Extract a stored image to a directory",
		Long: `IMAGE should be a string referencing an image: either a ID or an image name.

Note that in order to make cleaning up easy (just rm -rf), extract does not use
overlayfs or any other mechanism.`,
		Run: runWrapper(runImageExtract),
	}
	flagExtractRootfsOnly bool
	flagExtractOverwrite  bool
)

func init() {
	cmdImage.AddCommand(cmdImageExtract)
	cmdImageExtract.Flags().BoolVar(&flagExtractRootfsOnly, "rootfs-only", false, "extract rootfs only")
	cmdImageExtract.Flags().BoolVar(&flagExtractOverwrite, "overwrite", false, "overwrite output directory")
}

func runImageExtract(cmd *cobra.Command, args []string) (exit int) {
	if len(args) != 2 {
		cmd.Usage()
		return 254
	}
	outputDir := args[1]

	s, err := imagestore.NewStore(storeDir())
	if err != nil {
		stderr.PrintE("cannot open store", err)
		return 254
	}

	key, err := getStoreKeyFromAppOrHash(s, args[0])
	if err != nil {
		stderr.Error(err)
		return 254
	}

	aci, err := s.ReadStream(key)
	if err != nil {
		stderr.PrintE("error reading ACI from the store", err)
		return 254
	}

	// ExtractTar needs an absolute path
	absOutputDir, err := filepath.Abs(outputDir)
	if err != nil {
		stderr.PrintE("error converting output to an absolute path", err)
		return 254
	}

	if _, err := os.Stat(absOutputDir); err == nil {
		if !flagExtractOverwrite {
			stderr.Print("output directory exists (try --overwrite)")
			return 254
		}

		// don't allow the user to delete the root filesystem by mistake
		if absOutputDir == "/" {
			stderr.Print("this would delete your root filesystem. Refusing.")
			return 254
		}

		if err := os.RemoveAll(absOutputDir); err != nil {
			stderr.PrintE("error removing existing output dir", err)
			return 254
		}
	}

	// if the user only asks for the rootfs we extract the image to a temporary
	// directory and then move/copy the rootfs to the output directory, if not
	// we just extract the image to the output directory
	extractDir := absOutputDir
	if flagExtractRootfsOnly {
		rktTmpDir, err := s.TmpDir()
		if err != nil {
			stderr.PrintE("error creating rkt temporary directory", err)
			return 254
		}

		tmpDir, err := ioutil.TempDir(rktTmpDir, "rkt-image-extract-")
		if err != nil {
			stderr.PrintE("error creating temporary directory", err)
			return 254
		}
		defer os.RemoveAll(tmpDir)

		extractDir = tmpDir
	} else {
		if err := os.MkdirAll(absOutputDir, 0755); err != nil {
			stderr.PrintE("error creating output directory", err)
			return 254
		}
	}

	if err := tar.ExtractTar(aci, extractDir, false, user.NewBlankUidRange(), nil); err != nil {
		stderr.PrintE("error extracting ACI", err)
		return 254
	}

	if flagExtractRootfsOnly {
		rootfsDir := filepath.Join(extractDir, "rootfs")
		if err := os.Rename(rootfsDir, absOutputDir); err != nil {
			if e, ok := err.(*os.LinkError); ok && e.Err == syscall.EXDEV {
				// it's on a different device, fall back to copying
				if err := fileutil.CopyTree(rootfsDir, absOutputDir, user.NewBlankUidRange()); err != nil {
					stderr.PrintE("error copying ACI rootfs", err)
					return 254
				}
			} else {
				stderr.PrintE("error moving ACI rootfs", err)
				return 254
			}
		}
	}

	return 0
}
