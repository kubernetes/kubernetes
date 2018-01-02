// Copyright 2014 The rkt Authors
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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"time"

	"github.com/coreos/rkt/common"
	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/coreos/rkt/stage0"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/coreos/rkt/store/treestore"
	"github.com/hashicorp/errwrap"
	"github.com/spf13/cobra"
)

const (
	defaultGracePeriod        = 30 * time.Minute
	defaultPreparedExpiration = 24 * time.Hour
)

var (
	cmdGC = &cobra.Command{
		Use:   "gc [--grace-period=duration] [--expire-prepared=duration]",
		Short: "Garbage collect rkt pods no longer in use",
		Long: `This is intended to be run periodically from a timer or cron job.

Garbage collection is a 2-step process. First, stopped pods are moved to the
garbage by one invocation of the gc command. A subsequent invocation will clean
up the pod, assuming the pod has been in the garbage for more time than the
specified grace period.

Use --grace-period=0s to effectively disable the grace-period.`,
		Run: ensureSuperuser(runWrapper(runGC)),
	}
	flagGracePeriod        time.Duration
	flagPreparedExpiration time.Duration
	flagMarkOnly           bool
)

func init() {
	cmdRkt.AddCommand(cmdGC)
	cmdGC.Flags().DurationVar(&flagGracePeriod, "grace-period", defaultGracePeriod, "duration to wait before discarding inactive pods from garbage")
	cmdGC.Flags().DurationVar(&flagPreparedExpiration, "expire-prepared", defaultPreparedExpiration, "duration to wait before expiring prepared pods")
	cmdGC.Flags().BoolVar(&flagMarkOnly, "mark-only", false, "if set to true, then the exited/aborted pods will be moved to the garbage directories without actually deleting them, this is useful for marking the exit time of a pod")
}

func runGC(cmd *cobra.Command, args []string) (exit int) {
	if err := renameExited(); err != nil {
		stderr.PrintE("failed to rename exited pods", err)
		return 254
	}

	if err := renameAborted(); err != nil {
		stderr.PrintE("failed to rename aborted pods", err)
		return 254
	}

	if flagMarkOnly {
		return
	}

	if err := renameExpired(flagPreparedExpiration); err != nil {
		stderr.PrintE("failed to rename expired prepared pods", err)
		return 254
	}

	if err := emptyExitedGarbage(flagGracePeriod); err != nil {
		stderr.PrintE("failed to empty exitedGarbage", err)
		return 254
	}

	if err := emptyGarbage(); err != nil {
		stderr.PrintE("failed to empty garbage", err)
		return 254
	}

	return
}

// renameExited renames exited pods to the exitedGarbage directory
func renameExited() error {
	if err := pkgPod.WalkPods(getDataDir(), pkgPod.IncludeRunDir, func(p *pkgPod.Pod) {
		if p.State() == pkgPod.Exited {
			stderr.Printf("moving pod %q to garbage", p.UUID)
			if err := p.ToExitedGarbage(); err != nil && err != os.ErrNotExist {
				stderr.PrintE("rename error", err)
			}
		}
	}); err != nil {
		return err
	}

	return nil
}

// emptyExitedGarbage discards sufficiently aged pods from exitedGarbageDir()
func emptyExitedGarbage(gracePeriod time.Duration) error {
	if err := pkgPod.WalkPods(getDataDir(), pkgPod.IncludeExitedGarbageDir, func(p *pkgPod.Pod) {
		gp := p.Path()
		st := &syscall.Stat_t{}
		if err := syscall.Lstat(gp, st); err != nil {
			if err != syscall.ENOENT {
				stderr.PrintE(fmt.Sprintf("unable to stat %q, ignoring", gp), err)
			}
			return
		}

		if expiration := time.Unix(st.Ctim.Unix()).Add(gracePeriod); time.Now().After(expiration) {
			if err := p.ExclusiveLock(); err != nil {
				return
			}
			stdout.Printf("Garbage collecting pod %q", p.UUID)

			deletePod(p)
		} else {
			stderr.Printf("pod %q not removed: still within grace period (%s)", p.UUID, gracePeriod)
		}
	}); err != nil {
		return err
	}

	return nil
}

// renameAborted renames failed prepares to the garbage directory
func renameAborted() error {
	if err := pkgPod.WalkPods(getDataDir(), pkgPod.IncludePrepareDir, func(p *pkgPod.Pod) {
		if p.State() == pkgPod.AbortedPrepare {
			stderr.Printf("moving failed prepare %q to garbage", p.UUID)
			if err := p.ToGarbage(); err != nil && err != os.ErrNotExist {
				stderr.PrintE("rename error", err)
			}
		}
	}); err != nil {
		return err
	}
	return nil
}

// renameExpired renames expired prepared pods to the garbage directory
func renameExpired(preparedExpiration time.Duration) error {
	if err := pkgPod.WalkPods(getDataDir(), pkgPod.IncludePreparedDir, func(p *pkgPod.Pod) {
		st := &syscall.Stat_t{}
		pp := p.Path()
		if err := syscall.Lstat(pp, st); err != nil {
			if err != syscall.ENOENT {
				stderr.PrintE(fmt.Sprintf("unable to stat %q, ignoring", pp), err)
			}
			return
		}

		if expiration := time.Unix(st.Ctim.Unix()).Add(preparedExpiration); time.Now().After(expiration) {
			stderr.Printf("moving expired prepared pod %q to garbage", p.UUID)
			if err := p.ToGarbage(); err != nil && err != os.ErrNotExist {
				stderr.PrintE("rename error", err)
			}
		}
	}); err != nil {
		return err
	}
	return nil
}

// emptyGarbage discards everything from garbageDir()
func emptyGarbage() error {
	if err := pkgPod.WalkPods(getDataDir(), pkgPod.IncludeGarbageDir, func(p *pkgPod.Pod) {
		if err := p.ExclusiveLock(); err != nil {
			return
		}
		stdout.Printf("Garbage collecting pod %q", p.UUID)

		deletePod(p)
	}); err != nil {
		return err
	}

	return nil
}

func mountPodStage1(ts *treestore.Store, p *pkgPod.Pod) error {
	if !p.UsesOverlay() {
		return nil
	}

	s1Id, err := p.GetStage1TreeStoreID()
	if err != nil {
		return errwrap.Wrap(errors.New("error getting stage1 treeStoreID"), err)
	}
	s1rootfs := ts.GetRootFS(s1Id)

	stage1Dir := common.Stage1RootfsPath(p.Path())
	overlayDir := filepath.Join(p.Path(), "overlay")
	imgDir := filepath.Join(overlayDir, s1Id)
	upperDir := filepath.Join(imgDir, "upper")
	workDir := filepath.Join(imgDir, "work")

	opts := fmt.Sprintf("lowerdir=%s,upperdir=%s,workdir=%s", s1rootfs, upperDir, workDir)
	if err := syscall.Mount("overlay", stage1Dir, "overlay", 0, opts); err != nil {
		return errwrap.Wrap(errors.New("error mounting stage1"), err)
	}

	return nil
}

// deletePod cleans up files and resource associated with the pod
// pod must be under exclusive lock and be in either ExitedGarbage
// or Garbage state
func deletePod(p *pkgPod.Pod) bool {
	podState := p.State()
	if podState != pkgPod.ExitedGarbage && podState != pkgPod.Garbage && podState != pkgPod.ExitedDeleting {
		stderr.Errorf("non-garbage pod %q (status %q), skipped", p.UUID, p.State())
		return false
	}

	if podState == pkgPod.ExitedGarbage {
		s, err := imagestore.NewStore(storeDir())
		if err != nil {
			stderr.PrintE("cannot open store", err)
			return false
		}
		defer s.Close()

		ts, err := treestore.NewStore(treeStoreDir(), s)
		if err != nil {
			stderr.PrintE("cannot open store", err)
			return false
		}

		if globalFlags.Debug {
			stage0.InitDebug()
		}

		if err := mountPodStage1(ts, p); err == nil {
			if err = stage0.GC(p.Path(), p.UUID, globalFlags.LocalConfigDir); err != nil {
				stderr.PrintE(fmt.Sprintf("problem performing stage1 GC on %q", p.UUID), err)
			}
			// an overlay fs can be mounted over itself, let's unmount it here
			// if we mounted it before to avoid problems when running
			// stage0.MountGC
			if p.UsesOverlay() {
				stage1Mnt := common.Stage1RootfsPath(p.Path())
				if err := syscall.Unmount(stage1Mnt, 0); err != nil {
					stderr.PrintE("error unmounting stage1", err)
				}
			}
		} else {
			stderr.PrintE("skipping stage1 GC", err)
		}

		// unmount all leftover mounts
		if err := stage0.MountGC(p.Path(), p.UUID.String()); err != nil {
			stderr.PrintE(fmt.Sprintf("GC of leftover mounts for pod %q failed", p.UUID), err)
			return false
		}
	}

	// remove the rootfs first; if this fails (eg. due to busy mountpoints), pod manifest
	// is left in place and clean-up can be re-tried later.
	rootfsPath, err := p.Stage1RootfsPath()
	if err == nil {
		if e := os.RemoveAll(rootfsPath); e != nil {
			stderr.PrintE(fmt.Sprintf("unable to remove pod rootfs %q", p.UUID), e)
			return false
		}
	}

	// finally remove all remaining pieces
	if err := os.RemoveAll(p.Path()); err != nil {
		stderr.PrintE(fmt.Sprintf("unable to remove pod %q", p.UUID), err)
		return false
	}

	return true
}
