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
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/coreos/rkt/pkg/fileutil"
	"github.com/coreos/rkt/pkg/user"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/coreos/rkt/store/treestore"

	"github.com/spf13/cobra"
)

var (
	cmdImageRender = &cobra.Command{
		Use:   "render IMAGE OUTPUT_DIR",
		Short: "Render a stored image to a directory with all its dependencies",
		Long: `IMAGE should be a string referencing an image: either a ID or an image name.

This differs from extract in that the rendered image is in the state the app
would see when running in rkt, dependencies and all.

Note that in order to make cleaning up easy (just rm -rf), this does not use
overlayfs or any other mechanism.`,
		Run: runWrapper(runImageRender),
	}
	flagRenderRootfsOnly bool
	flagRenderOverwrite  bool
)

func init() {
	cmdImage.AddCommand(cmdImageRender)
	cmdImageRender.Flags().BoolVar(&flagRenderRootfsOnly, "rootfs-only", false, "render rootfs only")
	cmdImageRender.Flags().BoolVar(&flagRenderOverwrite, "overwrite", false, "overwrite output directory")
}

func runImageRender(cmd *cobra.Command, args []string) (exit int) {
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

	ts, err := treestore.NewStore(treeStoreDir(), s)
	if err != nil {
		stderr.PrintE("cannot open store", err)
		return
	}

	key, err := getStoreKeyFromAppOrHash(s, args[0])
	if err != nil {
		stderr.Error(err)
		return 254
	}

	id, _, err := ts.Render(key, false)
	if err != nil {
		stderr.PrintE("error rendering ACI", err)
		return 254
	}
	if _, err := ts.Check(id); err != nil {
		stderr.Print("warning: tree cache is in a bad state. Rebuilding...")
		var err error
		if id, _, err = ts.Render(key, true); err != nil {
			stderr.PrintE("error rendering ACI", err)
			return 254
		}
	}

	if _, err := os.Stat(outputDir); err == nil {
		if !flagRenderOverwrite {
			stderr.Print("output directory exists (try --overwrite)")
			return 254
		}

		// don't allow the user to delete the root filesystem by mistake
		if outputDir == "/" {
			stderr.Print("this would delete your root filesystem. Refusing.")
			return 254
		}

		if err := os.RemoveAll(outputDir); err != nil {
			stderr.PrintE("error removing existing output dir", err)
			return 254
		}
	}
	rootfsOutDir := outputDir
	if !flagRenderRootfsOnly {
		if err := os.MkdirAll(outputDir, 0755); err != nil {
			stderr.PrintE("error creating output directory", err)
			return 254
		}
		rootfsOutDir = filepath.Join(rootfsOutDir, "rootfs")

		manifest, err := s.GetImageManifest(key)
		if err != nil {
			stderr.PrintE("error getting manifest", err)
			return 254
		}

		mb, err := json.Marshal(manifest)
		if err != nil {
			stderr.PrintE("error marshalling image manifest", err)
			return 254
		}

		if err := ioutil.WriteFile(filepath.Join(outputDir, "manifest"), mb, 0700); err != nil {
			stderr.PrintE("error writing image manifest", err)
			return 254
		}
	}

	cachedTreePath := ts.GetRootFS(id)
	if err := fileutil.CopyTree(cachedTreePath, rootfsOutDir, user.NewBlankUidRange()); err != nil {
		stderr.PrintE("error copying ACI rootfs", err)
		return 254
	}

	return 0
}
