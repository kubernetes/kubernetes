/*
Copyright 2019 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"io"
	"os/exec"
	"strings"
)

// getCmd uses the given environment to form the ginkgo command to run tests. It will
// set the stdout/stderr to the given writer.
func getCmd(env Getenver, w io.Writer) *exec.Cmd {
	ginkgoArgs := []string{}

	// The logic of the parallel env var impacting the skip value necessitates it
	// being placed before the rest of the flag resolution.
	skip := env.Getenv(skipEnvKey)
	switch env.Getenv(parallelEnvKey) {
	case "y", "Y", "true":
		ginkgoArgs = append(ginkgoArgs, "--p")
		if len(skip) == 0 {
			skip = serialTestsRegexp
		}
	}

	ginkgoArgs = append(ginkgoArgs, []string{
		"--focus=" + env.Getenv(focusEnvKey),
		"--skip=" + skip,
		"--noColor=true",
	}...)

	extraArgs := []string{
		"--disable-log-dump",
		"--repo-root=/kubernetes",
		"--provider=" + env.Getenv(providerEnvKey),
		"--report-dir=" + env.Getenv(resultsDirEnvKey),
		"--kubeconfig=" + env.Getenv(kubeconfigEnvKey),
	}

	// Extra args handling
	sep := " "
	if len(env.Getenv(extraArgsSeparaterEnvKey)) > 0 {
		sep = env.Getenv(extraArgsSeparaterEnvKey)
	}

	if len(env.Getenv(extraGinkgoArgsEnvKey)) > 0 {
		ginkgoArgs = append(ginkgoArgs, strings.Split(env.Getenv(extraGinkgoArgsEnvKey), sep)...)
	}

	if len(env.Getenv(extraArgsEnvKey)) > 0 {
		fmt.Printf("sep is %q args are %q", sep, env.Getenv(extraArgsEnvKey))
		fmt.Println("split", strings.Split(env.Getenv(extraArgsEnvKey), sep))
		extraArgs = append(extraArgs, strings.Split(env.Getenv(extraArgsEnvKey), sep)...)
	}

	if len(env.Getenv(dryRunEnvKey)) > 0 {
		ginkgoArgs = append(ginkgoArgs, "--dryRun=true")
	}
	// NOTE: Ginkgo's default timeout has been reduced from 24h to 1h in V2, set it as "24h" for backward compatibility
	// if this is not set by env of extraGinkgoArgsEnvKey.
	exists := func(args []string) bool {
		for _, arg := range args {
			if strings.Contains(arg, "--timeout") {
				return true
			}
		}
		return false
	}(ginkgoArgs)

	if !exists {
		ginkgoArgs = append(ginkgoArgs, "--timeout=24h")
	}

	args := []string{}
	args = append(args, ginkgoArgs...)
	args = append(args, env.Getenv(testBinEnvKey))
	args = append(args, "--")
	args = append(args, extraArgs...)

	cmd := exec.Command(env.Getenv(ginkgoEnvKey), args...)
	cmd.Stdout = w
	cmd.Stderr = w
	return cmd
}

// cmdInfo generates a useful look at what the command is for printing/debug.
func cmdInfo(cmd *exec.Cmd) string {
	return fmt.Sprintf(
		`Command env: %v
Run from directory: %v
Executable path: %v
Args (comma-delimited): %v`, cmd.Env, cmd.Dir, cmd.Path, strings.Join(cmd.Args, ","),
	)
}
