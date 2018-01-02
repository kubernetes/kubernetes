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

	"github.com/coreos/rkt/store/imagestore"
	"github.com/spf13/cobra"
)

var (
	cmdImageRm = &cobra.Command{
		Use:   "rm IMAGE...",
		Short: "Remove one or more images with the given IDs or image names from the local store",
		Long:  `Unlike image gc, image rm allows users to remove specific images.`,
		Run:   runWrapper(runRmImage),
	}
)

func init() {
	cmdImage.AddCommand(cmdImageRm)
}

func rmImages(s *imagestore.Store, images []string) error {
	imageMap := make(map[string]string)

	for _, img := range images {
		key, err := getStoreKeyFromAppOrHash(s, img)
		if err != nil {
			stderr.Error(err)
			continue
		}

		aciinfo, err := s.GetACIInfoWithBlobKey(key)
		if err != nil {
			stderr.PrintE(fmt.Sprintf("error retrieving aci infos for image %q", key), err)
			continue
		}

		imageMap[key] = aciinfo.Name
	}

	done := 0
	errors := 0
	staleErrors := 0

	for key, name := range imageMap {
		if err := s.RemoveACI(key); err != nil {
			if serr, ok := err.(*imagestore.StoreRemovalError); ok {
				staleErrors++
				stderr.PrintE(fmt.Sprintf("some files cannot be removed for image %q (%q)", key, name), serr)
			} else {
				errors++
				stderr.PrintE(fmt.Sprintf("error removing aci for image %q (%q)", key, name), err)
				continue
			}
		}
		stdout.Printf("successfully removed aci for image: %q", key)
		done++
	}

	if done > 0 {
		stderr.Printf("%d image(s) successfully removed", done)
	}

	// If anything didn't complete, return exit status of 1
	if (errors + staleErrors) > 0 {
		if staleErrors > 0 {
			stderr.Printf("%d image(s) removed but left some stale files", staleErrors)
		}
		if errors > 0 {
			stderr.Printf("%d image(s) cannot be removed", errors)
		}
		return fmt.Errorf("error(s) found while removing images")
	}

	return nil
}

func runRmImage(cmd *cobra.Command, args []string) (exit int) {
	if len(args) < 1 {
		stderr.Print("must provide at least one image ID")
		return 254
	}

	s, err := imagestore.NewStore(storeDir())
	if err != nil {
		stderr.PrintE("cannot open store", err)
		return 254
	}

	if err := rmImages(s, args); err != nil {
		stderr.Error(err)
		return 254
	}

	return 0
}
