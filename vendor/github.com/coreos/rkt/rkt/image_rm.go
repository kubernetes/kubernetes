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

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/store"
	"github.com/spf13/cobra"
)

var (
	cmdImageRm = &cobra.Command{
		Use:   "rm IMAGE...",
		Short: "Remove image(s) with the given ID(s) or name(s) from the local store",
		Long:  `Unlike image gc, image rm allows users to remove specific images.`,
		Run:   runWrapper(runRmImage),
	}
)

func init() {
	cmdImage.AddCommand(cmdImageRm)
}

func rmImages(s *store.Store, images []string) error {
	done := 0
	errors := 0
	staleErrors := 0
	imageMap := make(map[string]string)
	imageCounter := make(map[string]int)

	for _, pkey := range images {
		errors++
		h, err := types.NewHash(pkey)
		if err != nil {
			var found bool
			keys, found, err := s.ResolveName(pkey)
			if len(keys) > 0 {
				errors += len(keys) - 1
			}
			if err != nil {
				stderr.Error(err)
				continue
			}
			if !found {
				stderr.Printf("image name %q not found", pkey)
				continue
			}
			for _, key := range keys {
				imageMap[key] = pkey
				imageCounter[key]++
			}
		} else {
			key, err := s.ResolveKey(h.String())
			if err != nil {
				stderr.PrintE(fmt.Sprintf("image ID %q not valid", pkey), err)
				continue
			}
			if key == "" {
				stderr.Printf("image ID %q doesn't exist", pkey)
				continue
			}

			aciinfo, err := s.GetACIInfoWithBlobKey(key)
			if err != nil {
				stderr.PrintE(fmt.Sprintf("error retrieving aci infos for image %q", key), err)
				continue
			}
			imageMap[key] = aciinfo.Name
			imageCounter[key]++
		}
	}

	// Adjust the error count by subtracting duplicate IDs from it,
	// therefore allowing only one error per ID.
	for _, c := range imageCounter {
		if c > 1 {
			errors -= c - 1
		}
	}

	for key, name := range imageMap {
		if err := s.RemoveACI(key); err != nil {
			if serr, ok := err.(*store.StoreRemovalError); ok {
				staleErrors++
				stderr.PrintE(fmt.Sprintf("some files cannot be removed for image %q (%q)", key, name), serr)
			} else {
				stderr.PrintE(fmt.Sprintf("error removing aci for image %q (%q)", key, name), err)
				continue
			}
		}
		stdout.Printf("successfully removed aci for image: %q", key)
		errors--
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
		return 1
	}

	s, err := store.NewStore(getDataDir())
	if err != nil {
		stderr.PrintE("cannot open store", err)
		return 1
	}

	if err := rmImages(s, args); err != nil {
		stderr.Error(err)
		return 1
	}

	return 0
}
