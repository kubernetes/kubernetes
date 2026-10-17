/*
Copyright 2023 The Kubernetes Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package xtables

import (
	"context"
	"path/filepath"
	"strings"

	utilexec "k8s.io/utils/exec"
)

// Mode represents the two different modes iptables can be configured in: nft or legacy.
// The string value is identical to the form used in iptables command names (e.g.
// "iptables-legacy", "ip6tables-nft-save", "xtables-nft-multi")
type Mode string

const (
	LegacyMode Mode = "legacy"
	NFTMode    Mode = "nft"
)

// IPTablesBinaries is the list of aliases for the xtables-*-multi binaries
var IPTablesBinaries = []string{
	"iptables",
	"iptables-save",
	"iptables-restore",
	"ip6tables",
	"ip6tables-save",
	"ip6tables-restore",
}

// MultiBinaryPath returns the path to the `xtables-<mode>-multi` binary
func MultiBinaryPath(sbinPath string, mode Mode) string {
	return filepath.Join(sbinPath, "xtables-"+string(mode)+"-multi")
}

// ExecWrapper wraps a utilexec.Interface and reimplements it with support for rewriting
// iptables commands to always use a particular mode.
type ExecWrapper struct {
	execer   utilexec.Interface
	mappings map[string]string
}

var _ utilexec.Interface = &ExecWrapper{}

// NewExecWrapper returns an ExecWrapper that uses the given iptables mode
func NewExecWrapper(execer utilexec.Interface, mode Mode) *ExecWrapper {
	// If DetectBinaryDir fails and returns "", filepath.Join() will just ignore it.
	sbinDir, _ := DetectBinaryDir(execer)

	mappings := map[string]string{}
	for _, binary := range IPTablesBinaries {
		mapped := filepath.Join(sbinDir, strings.Replace(binary, "tables", "tables-"+string(mode), 1))
		mappings[binary] = mapped
		mappings[filepath.Join("/sbin", binary)] = mapped
		mappings[filepath.Join("/usr/sbin", binary)] = mapped
	}

	return &ExecWrapper{
		execer:   execer,
		mappings: mappings,
	}
}

func (mew *ExecWrapper) mapCommand(cmd string) string {
	if mapped := mew.mappings[cmd]; mapped != "" {
		return mapped
	}
	return cmd
}

// Command is part of utilexec.Interface
func (mew *ExecWrapper) Command(cmd string, args ...string) utilexec.Cmd {
	return mew.execer.Command(mew.mapCommand(cmd), args...)
}

// CommandContext is part of utilexec.Interface
func (mew *ExecWrapper) CommandContext(ctx context.Context, cmd string, args ...string) utilexec.Cmd {
	return mew.execer.CommandContext(ctx, mew.mapCommand(cmd), args...)
}

// LookPath is part of utilexec.Interface
func (mew *ExecWrapper) LookPath(file string) (string, error) {
	return mew.execer.LookPath(file)
}
