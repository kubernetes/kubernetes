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
	"bufio"
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	utilexec "k8s.io/utils/exec"
)

// DetectBinaryDir tries to detect the `iptables` location in
// either /usr/sbin or /sbin. If it's not there, it returns an error.
func DetectBinaryDir(execer utilexec.Interface) (string, error) {
	if path, _ := execer.LookPath("/usr/sbin/iptables"); path != "" {
		return "/usr/sbin", nil
	} else if path, _ := execer.LookPath("/sbin/iptables"); path != "" {
		return "/sbin", nil
	} else {
		return "", errors.New("iptables is not present in either /usr/sbin or /sbin")
	}
}

var detectMutex sync.Mutex
var detectedMode Mode

// DetectMode returns the best guess about the current iptables mode.
//
// The first time it is called, it uses execer to run iptables-legacy and iptables-nft. If
// it is unable to make a guess, it will return an error (though it will also return a
// default value for the mode in that case, so you can just ignore the error).
//
// After successfully determining the mode once, if it is called again after that, it just
// returns the cached value from the previous check.
func DetectMode(ctx context.Context, execer utilexec.Interface) (Mode, error) {
	detectMutex.Lock()
	defer detectMutex.Unlock()

	var err error
	mode := detectedMode
	if mode == "" {
		mode, err = realDetectMode(ctx, execer)
		if ctx.Err() != nil {
			// realDetectMode will have ignored the errors and returned a guess
			// anyway, but in this case, the context error is more relevant.
			err = fmt.Errorf("unable to detect iptables mode: %w", ctx.Err())
		}
		if err == nil {
			detectedMode = mode
		}
	}
	return mode, err
}

func realDetectMode(ctx context.Context, execer utilexec.Interface) (Mode, error) {
	// If DetectBinaryDir() fails then sbinPath will be "". Since filepath.Join()
	// ignores empty strings, this means we will end up just using the plain binary
	// names and falling back to searching $PATH.
	sbinPath, _ := DetectBinaryDir(execer)

	iptablesLegacySave, _ := execer.LookPath(filepath.Join(sbinPath, "iptables-legacy-save"))
	ip6tablesLegacySave, _ := execer.LookPath(filepath.Join(sbinPath, "ip6tables-legacy-save"))
	haveIPTablesLegacy := iptablesLegacySave != "" || ip6tablesLegacySave != ""

	iptablesNFTSave, _ := execer.LookPath(filepath.Join(sbinPath, "iptables-nft-save"))
	ip6tablesNFTSave, _ := execer.LookPath(filepath.Join(sbinPath, "ip6tables-nft-save"))
	haveIPTablesNFT := iptablesNFTSave != "" || ip6tablesNFTSave != ""

	if haveIPTablesNFT && !haveIPTablesLegacy {
		return NFTMode, nil
	} else if haveIPTablesLegacy && !haveIPTablesNFT {
		return LegacyMode, nil
	} else if !haveIPTablesLegacy && !haveIPTablesNFT {
		return NFTMode, fmt.Errorf("could not find any iptables binaries")
	}

	// In kubernetes 1.17 and later, kubelet will have created at least
	// one chain in the "mangle" table (either "KUBE-IPTABLES-HINT" or
	// "KUBE-KUBELET-CANARY"), so check that first, against
	// iptables-nft, because we can check that more efficiently and
	// it's more common these days.
	if iptablesNFTSave != "" && execAndScanForKubeletChains(ctx, execer, iptablesNFTSave, "-t", "mangle") {
		return NFTMode, nil
	}
	if ip6tablesNFTSave != "" && execAndScanForKubeletChains(ctx, execer, ip6tablesNFTSave, "-t", "mangle") {
		return NFTMode, nil
	}

	// Check for kubernetes 1.17-or-later with iptables-legacy. We
	// can't pass "-t mangle" to iptables-legacy-save because it would
	// cause the kernel to create that table if it didn't already
	// exist, which we don't want. So we have to grab all the rules.
	if iptablesLegacySave != "" && execAndScanForKubeletChains(ctx, execer, iptablesLegacySave) {
		return LegacyMode, nil
	}
	if ip6tablesLegacySave != "" && execAndScanForKubeletChains(ctx, execer, ip6tablesLegacySave) {
		return LegacyMode, nil
	}

	// If we don't recognize any rules, default to nft.
	return NFTMode, fmt.Errorf("unable to determine system iptables mode; could not find any kubelet iptables chains")
}

// for unit tests
func unDetectMode() {
	detectMutex.Lock()
	defer detectMutex.Unlock()
	detectedMode = ""
}

// Especially on a restart, there may already be lots of iptables rules, so rather than
// using CombinedOutput() or something simple like that that might end up using a lot of
// memory, we stream the command output, only keeping a line at a time, and stop as soon
// as we find a match.
func execAndScanForKubeletChains(ctx context.Context, execer utilexec.Interface, command string, args ...string) bool {
	cmd := execer.CommandContext(ctx, command, args...)
	stdout, _ := cmd.StdoutPipe()
	if err := cmd.Start(); err != nil {
		return false
	}
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, ":KUBE-IPTABLES-HINT ") ||
			strings.HasPrefix(line, ":KUBE-KUBELET-CANARY ") {
			// SIGPIPE the iptables process
			_ = stdout.Close()
			_ = cmd.Wait()
			return true
		}
	}
	_ = cmd.Wait()
	return false
}
