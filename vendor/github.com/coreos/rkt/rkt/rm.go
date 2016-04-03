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

//+build linux

package main

import (
	"os"

	"github.com/appc/spec/schema/types"
	"github.com/spf13/cobra"
)

var (
	cmdRm = &cobra.Command{
		Use:   "rm [--uuid-file=FILE] UUID ...",
		Short: "Remove all files and resources associated with an exited pod",
		Long:  `Unlike gc, rm allows users to remove specific pods.`,
		Run:   ensureSuperuser(runWrapper(runRm)),
	}
	flagUUIDFile string
)

func init() {
	cmdRkt.AddCommand(cmdRm)
	cmdRm.Flags().StringVar(&flagUUIDFile, "uuid-file", "", "read pod UUID from file instead of argument")
}

func runRm(cmd *cobra.Command, args []string) (exit int) {
	var podUUID *types.UUID
	var podUUIDs []*types.UUID
	var err error

	switch {
	case len(args) == 0 && flagUUIDFile != "":
		podUUID, err = readUUIDFromFile(flagUUIDFile)
		if err != nil {
			stderr.PrintE("unable to read UUID from file", err)
			return 1
		}
		podUUIDs = append(podUUIDs, podUUID)

	case len(args) > 0 && flagUUIDFile == "":
		for _, uuid := range args {
			podUUID, err := resolveUUID(uuid)
			if err != nil {
				stderr.PrintE("unable to resolve UUID", err)
			} else {
				podUUIDs = append(podUUIDs, podUUID)
			}
		}

	default:
		cmd.Usage()
		return 1
	}

	ret := 0
	for _, podUUID = range podUUIDs {
		p, err := getPod(podUUID)
		if err != nil {
			ret = 1
			stderr.PrintE("cannot get pod", err)
		}

		if removePod(p) {
			stdout.Printf("%q", p.uuid)
		} else {
			ret = 1
		}
	}

	if ret == 1 {
		stderr.Print("failed to remove one or more pods")
	}

	return ret
}

func removePod(p *pod) bool {
	switch {
	case p.isRunning():
		stderr.Printf("pod %q is currently running", p.uuid)
		return false

	case p.isEmbryo, p.isPreparing:
		stderr.Printf("pod %q is currently being prepared", p.uuid)
		return false

	case p.isExitedDeleting, p.isDeleting:
		stderr.Printf("pod %q is currently being deleted", p.uuid)
		return false

	case p.isAbortedPrepare:
		stderr.Printf("moving failed prepare %q to garbage", p.uuid)
		if err := p.xToGarbage(); err != nil && err != os.ErrNotExist {
			stderr.PrintE("rename error", err)
			return false
		}

	case p.isPrepared:
		stderr.Printf("moving expired prepared pod %q to garbage", p.uuid)
		if err := p.xToGarbage(); err != nil && err != os.ErrNotExist {
			stderr.PrintE("rename error", err)
			return false
		}

	// p.isExitedGarbage and p.isExited can be true at the same time. Test
	// the most specific case first.
	case p.isExitedGarbage, p.isGarbage:

	case p.isExited:
		if err := p.xToExitedGarbage(); err != nil && err != os.ErrNotExist {
			stderr.PrintE("rename error", err)
			return false
		}
	}

	if err := p.ExclusiveLock(); err != nil {
		stderr.PrintE("unable to acquire exclusive lock", err)
		return false
	}

	deletePod(p)

	return true
}
