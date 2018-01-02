// Copyright 2016 CNI authors
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

// Package legacy_examples contains sample code from prior versions of
// the CNI library, for use in verifying backwards compatibility.
package legacy_examples

import (
	"io/ioutil"
	"net"
	"path/filepath"
	"sync"

	"github.com/containernetworking/cni/pkg/types"
	"github.com/containernetworking/cni/pkg/types/020"
	"github.com/containernetworking/cni/pkg/version/testhelpers"
)

// An Example is a Git reference to the CNI repo and a Golang CNI plugin that
// builds against that version of the repo.
//
// By convention, every Example plugin returns an ADD result that is
// semantically equivalent to the ExpectedResult.
type Example struct {
	Name          string
	CNIRepoGitRef string
	PluginSource  string
}

var buildDir = ""
var buildDirLock sync.Mutex

func ensureBuildDirExists() error {
	buildDirLock.Lock()
	defer buildDirLock.Unlock()

	if buildDir != "" {
		return nil
	}

	var err error
	buildDir, err = ioutil.TempDir("", "cni-example-plugins")
	return err
}

// Build builds the example, returning the path to the binary
func (e Example) Build() (string, error) {
	if err := ensureBuildDirExists(); err != nil {
		return "", err
	}

	outBinPath := filepath.Join(buildDir, e.Name)

	if err := testhelpers.BuildAt([]byte(e.PluginSource), e.CNIRepoGitRef, outBinPath); err != nil {
		return "", err
	}
	return outBinPath, nil
}

// V010 acts like a CNI plugin from the v0.1.0 era
var V010 = Example{
	Name:          "example_v010",
	CNIRepoGitRef: "2c482f4",
	PluginSource: `package main

import (
	"net"

	"github.com/containernetworking/cni/pkg/skel"
	"github.com/containernetworking/cni/pkg/types"
)

var result = types.Result{
	IP4: &types.IPConfig{
		IP: net.IPNet{
			IP:   net.ParseIP("10.1.2.3"),
			Mask: net.CIDRMask(24, 32),
		},
		Gateway: net.ParseIP("10.1.2.1"),
		Routes: []types.Route{
			types.Route{
				Dst: net.IPNet{
					IP:   net.ParseIP("0.0.0.0"),
					Mask: net.CIDRMask(0, 32),
				},
				GW: net.ParseIP("10.1.0.1"),
			},
		},
	},
	DNS: types.DNS{
		Nameservers: []string{"8.8.8.8"},
		Domain:      "example.com",
	},
}

func c(_ *skel.CmdArgs) error { result.Print(); return nil }

func main() { skel.PluginMain(c, c) }
`,
}

// ExpectedResult is the current representation of the plugin result
// that is expected from each of the examples.
//
// As we change the CNI spec, the Result type and this value may change.
// The text of the example plugins should not.
var ExpectedResult = &types020.Result{
	IP4: &types020.IPConfig{
		IP: net.IPNet{
			IP:   net.ParseIP("10.1.2.3"),
			Mask: net.CIDRMask(24, 32),
		},
		Gateway: net.ParseIP("10.1.2.1"),
		Routes: []types.Route{
			types.Route{
				Dst: net.IPNet{
					IP:   net.ParseIP("0.0.0.0"),
					Mask: net.CIDRMask(0, 32),
				},
				GW: net.ParseIP("10.1.0.1"),
			},
		},
	},
	DNS: types.DNS{
		Nameservers: []string{"8.8.8.8"},
		Domain:      "example.com",
	},
}
