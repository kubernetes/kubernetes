/*
Copyright 2016 The Kubernetes Authors.

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

package common

import (
	"fmt"
	"io"
	"net/http"
	"strings"

	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/client-go/discovery"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econfig "k8s.io/kubernetes/test/e2e/framework/config"
	"k8s.io/kubernetes/test/e2e/upgrades"
)

var (
	upgradeTarget = e2econfig.Flags.String("upgrade-target", "ci/latest", "Version to upgrade to (e.g. 'release/stable', 'release/latest', 'ci/latest', '0.19.1', '0.19.1-669-gabac8c8') if doing an upgrade test.")
	upgradeImage  = e2econfig.Flags.String("upgrade-image", "", "Image to upgrade to (e.g. 'container_vm' or 'gci') if doing an upgrade test.")
)

// GetUpgradeContext return UpgradeContext for GCP provider.
func GetUpgradeContext(c discovery.DiscoveryInterface) (*upgrades.UpgradeContext, error) {
	current, err := c.ServerVersion()
	if err != nil {
		return nil, err
	}

	curVer, err := utilversion.ParseSemantic(current.String())
	if err != nil {
		return nil, err
	}

	upgCtx := &upgrades.UpgradeContext{
		Versions: []upgrades.VersionContext{
			{
				Version:   *curVer,
				NodeImage: framework.TestContext.NodeOSDistro,
			},
		},
	}

	if len(*upgradeTarget) == 0 {
		return upgCtx, nil
	}

	next, err := realVersion(*upgradeTarget)
	if err != nil {
		return nil, err
	}

	nextVer, err := utilversion.ParseSemantic(next)
	if err != nil {
		return nil, err
	}

	upgCtx.Versions = append(upgCtx.Versions, upgrades.VersionContext{
		Version:   *nextVer,
		NodeImage: *upgradeImage,
	})

	return upgCtx, nil
}

// realVersion turns a version constants into a version string deployable on
// GKE. This corresponds to "hack/get-build.sh -v", which is implemented by:
// https://github.com/kubernetes/kubernetes/blob/e61430919e9655aa6f798609458a0d9f0850c005/cluster/common.sh#L278-L296
func realVersion(s string) (string, error) {
	framework.Logf("Getting real version for %q", s)
	v := s
	if strings.Contains(s, "/") {
		url := fmt.Sprintf("https://dl.k8s.io/%s.txt", s)
		resp, err := http.Get(url)
		if err != nil {
			return "", fmt.Errorf("GET %s: %v", url, err)
		}
		if resp.Body == nil {
			return "", fmt.Errorf("%s: no response", url)
		}
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", fmt.Errorf("%s: read response: %v", url, err)
		}
		s = string(data)
	}

	v = strings.TrimPrefix(strings.TrimSpace(v), "v")
	framework.Logf("Version for %q is %q", s, v)
	return v, nil
}
