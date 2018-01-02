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

	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/spf13/cobra"
)

var (
	cmdRm = &cobra.Command{
		Use:   "rm --uuid-file=FILE | UUID ...",
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
	var podUUIDs []string
	var ret int

	switch {
	case len(args) == 0 && flagUUIDFile != "":
		podUUID, err := pkgPod.ReadUUIDFromFile(flagUUIDFile)
		if err != nil {
			stderr.PrintE("unable to resolve UUID from file", err)
			ret = 254
		} else {
			podUUIDs = append(podUUIDs, podUUID)
		}

	case len(args) > 0 && flagUUIDFile == "":
		podUUIDs = args

	default:
		cmd.Usage()
		return 254
	}

	for _, podUUID := range podUUIDs {
		p, err := pkgPod.PodFromUUIDString(getDataDir(), podUUID)
		if err != nil {
			ret = 254
			stderr.PrintE("cannot get pod", err)
			continue
		}
		defer p.Close()

		if removePod(p) {
			stdout.Printf("%q", p.UUID)
		} else {
			ret = 254
		}
	}

	if ret == 254 {
		stderr.Print("failed to remove one or more pods")
	}

	return ret
}

func removePod(p *pkgPod.Pod) bool {
	switch p.State() {
	case pkgPod.Running:
		stderr.Printf("pod %q is currently running", p.UUID)
		return false

	case pkgPod.Embryo, pkgPod.Preparing:
		stderr.Printf("pod %q is currently being prepared", p.UUID)
		return false

	case pkgPod.Deleting:
		stderr.Printf("pod %q is currently being deleted", p.UUID)
		return false

	case pkgPod.AbortedPrepare:
		stderr.Printf("moving failed prepare %q to garbage", p.UUID)
		if err := p.ToGarbage(); err != nil && err != os.ErrNotExist {
			stderr.PrintE("rename error", err)
			return false
		}

	case pkgPod.Prepared:
		stderr.Printf("moving expired prepared pod %q to garbage", p.UUID)
		if err := p.ToGarbage(); err != nil && err != os.ErrNotExist {
			stderr.PrintE("rename error", err)
			return false
		}

	// p.isExitedGarbage and p.isExited can be true at the same time. Test
	// the most specific case first.
	case pkgPod.ExitedGarbage, pkgPod.Garbage:

	case pkgPod.Exited:
		if err := p.ToExitedGarbage(); err != nil && err != os.ErrNotExist {
			stderr.PrintE("rename error", err)
			return false
		}
	}

	if err := p.ExclusiveLock(); err != nil {
		stderr.PrintE("unable to acquire exclusive lock", err)
		return false
	}

	return deletePod(p)
}
