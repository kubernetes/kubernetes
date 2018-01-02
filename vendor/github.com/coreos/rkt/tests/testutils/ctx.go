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

package testutils

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/tests/testutils/logger"
)

// dirDesc structure manages one directory and provides an option for
// rkt invocations
type dirDesc struct {
	dir    string // directory path
	desc   string // directory description, mostly for failure cases
	prefix string // temporary directory prefix
	option string // rkt option for given directory
}

// newDirDesc creates dirDesc instance managing a temporary directory.
func newDirDesc(prefix, desc, option string) *dirDesc {
	dir := &dirDesc{
		dir:    "",
		desc:   desc,
		prefix: prefix,
		option: option,
	}
	dir.reset()
	return dir
}

// reset removes the managed directory and recreates it
func (d *dirDesc) reset() {
	d.cleanup()
	dir, err := ioutil.TempDir("", d.prefix)
	if err != nil {
		panic(fmt.Sprintf("Failed to create temporary %s directory: %v", d.desc, err))
	}
	d.dir = dir
}

// cleanup removes the managed directory. After cleanup this instance
// cannot be used for anything, until it is reset.
func (d *dirDesc) cleanup() {
	if d.dir == "" {
		return
	}
	if err := os.RemoveAll(d.dir); err != nil && !os.IsNotExist(err) {
		panic(fmt.Sprintf("Failed to remove temporary %s directory %q: %s", d.desc, d.dir, err))
	}
	d.dir = ""
}

// rktOption returns option for rkt invocation
func (d *dirDesc) rktOption() string {
	d.ensureValid()
	return fmt.Sprintf("--%s=%s", d.option, d.dir)
}

func (d *dirDesc) ensureValid() {
	if d.dir == "" {
		panic(fmt.Sprintf("A temporary %s directory is not set up", d.desc))
	}
}

// TODO(emil): Document exported.

type RktRunCtx struct {
	directories []*dirDesc
	useDefaults bool
	mds         *exec.Cmd
	children    []*gexpect.ExpectSubprocess
}

func NewRktRunCtx() *RktRunCtx {
	return &RktRunCtx{
		directories: []*dirDesc{
			newDirDesc("datadir-", "data", "dir"),
			newDirDesc("localdir-", "local configuration", "local-config"),
			newDirDesc("systemdir-", "system configuration", "system-config"),
			newDirDesc("userdir-", "user configuration", "user-config"),
		},
	}
}

func (ctx *RktRunCtx) SetupDataDir() error {
	return setupDataDir(ctx.DataDir())
}

func (ctx *RktRunCtx) LaunchMDS() error {
	ctx.mds = exec.Command(ctx.rktBin(), "metadata-service")
	return ctx.mds.Start()
}

func (ctx *RktRunCtx) DataDir() string {
	return ctx.dir(0)
}

func (ctx *RktRunCtx) LocalDir() string {
	return ctx.dir(1)
}

func (ctx *RktRunCtx) SystemDir() string {
	return ctx.dir(2)
}

func (ctx *RktRunCtx) UserDir() string {
	return ctx.dir(3)
}

func (ctx *RktRunCtx) dir(idx int) string {
	ctx.ensureValid()
	if idx < len(ctx.directories) {
		return ctx.directories[idx].dir
	}
	panic("Directory index out of bounds")
}

func (ctx *RktRunCtx) Reset() {
	ctx.cleanupChildren()
	ctx.RunGC()
	for _, d := range ctx.directories {
		d.reset()
	}
}

func (ctx *RktRunCtx) cleanupChildren() error {
	for _, child := range ctx.children {
		if child.Cmd == nil ||
			child.Cmd.Process == nil ||
			child.Cmd.ProcessState == nil {
			continue
		}

		if child.Cmd.ProcessState.Exited() {
			logger.Logf("Child %q already exited", child.Cmd.Path)
			continue
		}
		logger.Logf("Shutting down child %q", child.Cmd.Path)
		if err := child.Cmd.Process.Kill(); err != nil {
			return err
		}
		if _, err := child.Cmd.Process.Wait(); err != nil {
			return err
		}
	}
	return nil
}

func (ctx *RktRunCtx) Cleanup() {
	if ctx.mds != nil {
		ctx.mds.Process.Kill()
		ctx.mds.Wait()
		os.Remove("/run/rkt/metadata-svc.sock")
	}
	if err := ctx.cleanupChildren(); err != nil {
		logger.Logf("Error during child cleanup: %v", err)
	}
	ctx.RunGC()
	for _, d := range ctx.directories {
		d.cleanup()
	}
}

func (ctx *RktRunCtx) RunGC() {
	rktArgs := append(ctx.rktOptions(),
		"gc",
		"--grace-period=0s",
	)
	if err := exec.Command(ctx.rktBin(), rktArgs...).Run(); err != nil {
		panic(fmt.Sprintf("Failed to run gc: %v", err))
	}
}

func (ctx *RktRunCtx) Cmd() string {
	return fmt.Sprintf("%s %s",
		ctx.rktBin(),
		strings.Join(ctx.rktOptions(), " "),
	)
}

func (ctx *RktRunCtx) ExecCmd(arg ...string) *exec.Cmd {
	args := ctx.rktOptions()
	args = append(args, arg...)
	return exec.Command(ctx.rktBin(), args...)
}

// TODO(jonboulle): clean this up
func (ctx *RktRunCtx) CmdNoConfig() string {
	return fmt.Sprintf("%s %s",
		ctx.rktBin(),
		ctx.directories[0].rktOption(),
	)
}

func (ctx *RktRunCtx) rktBin() string {
	rkt := GetValueFromEnvOrPanic("RKT")
	abs, err := filepath.Abs(rkt)
	if err != nil {
		abs = rkt
	}
	return abs
}

func (ctx *RktRunCtx) GetUidGidRktBinOwnerNotRoot() (int, int) {
	s, err := os.Stat(ctx.rktBin())
	if err != nil {
		return GetUnprivilegedUidGid()
	}

	uid := int(s.Sys().(*syscall.Stat_t).Uid)
	gid := int(s.Sys().(*syscall.Stat_t).Gid)

	// If owner is root, fallback to user "nobody"
	if uid == 0 {
		return GetUnprivilegedUidGid()
	}

	return uid, gid
}

func (ctx *RktRunCtx) rktOptions() []string {
	ctx.ensureValid()
	opts := make([]string, 0, len(ctx.directories))
	for _, d := range ctx.directories {
		opts = append(opts, d.rktOption())
	}
	return opts
}

func (ctx *RktRunCtx) ensureValid() {
	for _, d := range ctx.directories {
		d.ensureValid()
	}
}

func (ctx *RktRunCtx) RegisterChild(child *gexpect.ExpectSubprocess) {
	ctx.children = append(ctx.children, child)
}
