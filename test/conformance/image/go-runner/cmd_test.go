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
	"os"
	"reflect"
	"testing"
)

func TestGetCmd(t *testing.T) {
	testCases := []struct {
		desc       string
		env        Getenver
		expectArgs []string
	}{
		{
			desc: "Default",
			env: &explicitEnv{
				vals: map[string]string{
					ginkgoEnvKey:  "ginkgobin",
					testBinEnvKey: "testbin",
				},
			},
			expectArgs: []string{
				"ginkgobin",
				"--focus=", "--skip=",
				"--noColor=true", "--timeout=24h", "testbin", "--",
				"--disable-log-dump", "--repo-root=/kubernetes",
				"--provider=", "--report-dir=", "--kubeconfig=",
			},
		}, {
			desc: "Filling in defaults",
			env: &explicitEnv{
				vals: map[string]string{
					ginkgoEnvKey:     "ginkgobin",
					testBinEnvKey:    "testbin",
					focusEnvKey:      "focus",
					skipEnvKey:       "skip",
					providerEnvKey:   "provider",
					resultsDirEnvKey: "results",
					kubeconfigEnvKey: "kubeconfig",
				},
			},
			expectArgs: []string{
				"ginkgobin",
				"--focus=focus", "--skip=skip",
				"--noColor=true", "--timeout=24h", "testbin", "--",
				"--disable-log-dump", "--repo-root=/kubernetes",
				"--provider=provider", "--report-dir=results", "--kubeconfig=kubeconfig",
			},
		}, {
			desc: "Parallel gets set and skips serial",
			env: &explicitEnv{
				vals: map[string]string{
					ginkgoEnvKey:   "ginkgobin",
					testBinEnvKey:  "testbin",
					parallelEnvKey: "true",
				},
			},
			expectArgs: []string{
				"ginkgobin", "--p",
				"--focus=", "--skip=\\[Serial\\]",
				"--noColor=true", "--timeout=24h", "testbin", "--",
				"--disable-log-dump", "--repo-root=/kubernetes",
				"--provider=", "--report-dir=", "--kubeconfig=",
			},
		}, {
			desc: "Arbitrary options before and after double dash split by space",
			env: &explicitEnv{
				vals: map[string]string{
					ginkgoEnvKey:          "ginkgobin",
					testBinEnvKey:         "testbin",
					extraArgsEnvKey:       "--extra=1 --extra=2",
					extraGinkgoArgsEnvKey: "--ginkgo1 --ginkgo2",
				},
			},
			expectArgs: []string{
				"ginkgobin", "--focus=", "--skip=",
				"--noColor=true", "--ginkgo1", "--ginkgo2", "--timeout=24h",
				"testbin", "--",
				"--disable-log-dump", "--repo-root=/kubernetes",
				"--provider=", "--report-dir=", "--kubeconfig=",
				"--extra=1", "--extra=2",
			},
		}, {
			desc: "Arbitrary options can be split by other tokens",
			env: &explicitEnv{
				vals: map[string]string{
					ginkgoEnvKey:             "ginkgobin",
					testBinEnvKey:            "testbin",
					extraArgsEnvKey:          "--extra=value with spaces:--extra=value with % anything!$$",
					extraGinkgoArgsEnvKey:    `--ginkgo='with "quotes" and ':--ginkgo2=true$(foo)`,
					extraArgsSeparaterEnvKey: ":",
				},
			},
			expectArgs: []string{
				"ginkgobin", "--focus=", "--skip=",
				"--noColor=true", `--ginkgo='with "quotes" and '`, "--ginkgo2=true$(foo)", "--timeout=24h",
				"testbin", "--",
				"--disable-log-dump", "--repo-root=/kubernetes",
				"--provider=", "--report-dir=", "--kubeconfig=",
				"--extra=value with spaces", "--extra=value with % anything!$$",
			},
		}, {
			desc: "Set Ginkgo timeout in env",
			env: &explicitEnv{
				vals: map[string]string{
					ginkgoEnvKey:          "ginkgobin",
					testBinEnvKey:         "testbin",
					extraGinkgoArgsEnvKey: "--timeout=10h",
				},
			},
			expectArgs: []string{
				"ginkgobin", "--focus=", "--skip=",
				"--noColor=true", "--timeout=10h", "testbin", "--",
				"--disable-log-dump", "--repo-root=/kubernetes",
				"--provider=", "--report-dir=", "--kubeconfig=",
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			c := getCmd(tc.env, os.Stdout)
			if !reflect.DeepEqual(c.Args, tc.expectArgs) {
				t.Errorf("Expected args %q but got %q", tc.expectArgs, c.Args)
			}
		})
	}
}
