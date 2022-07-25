// Copyright 2020 The etcd Authors
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

package membership

import (
	"github.com/coreos/go-semver/semver"
	"go.etcd.io/etcd/api/v3/version"
	"go.uber.org/zap"
)

type DowngradeInfo struct {
	// TargetVersion is the target downgrade version, if the cluster is not under downgrading,
	// the targetVersion will be an empty string
	TargetVersion string `json:"target-version"`
	// Enabled indicates whether the cluster is enabled to downgrade
	Enabled bool `json:"enabled"`
}

func (d *DowngradeInfo) GetTargetVersion() *semver.Version {
	return semver.Must(semver.NewVersion(d.TargetVersion))
}

// isValidDowngrade verifies whether the cluster can be downgraded from verFrom to verTo
func isValidDowngrade(verFrom *semver.Version, verTo *semver.Version) bool {
	return verTo.Equal(*AllowedDowngradeVersion(verFrom))
}

// mustDetectDowngrade will detect unexpected downgrade when the local server is recovered.
func mustDetectDowngrade(lg *zap.Logger, cv *semver.Version, d *DowngradeInfo) {
	lv := semver.Must(semver.NewVersion(version.Version))
	// only keep major.minor version for comparison against cluster version
	lv = &semver.Version{Major: lv.Major, Minor: lv.Minor}

	// if the cluster enables downgrade, check local version against downgrade target version.
	if d != nil && d.Enabled && d.TargetVersion != "" {
		if lv.Equal(*d.GetTargetVersion()) {
			if cv != nil {
				lg.Info(
					"cluster is downgrading to target version",
					zap.String("target-cluster-version", d.TargetVersion),
					zap.String("determined-cluster-version", version.Cluster(cv.String())),
					zap.String("current-server-version", version.Version),
				)
			}
			return
		}
		lg.Fatal(
			"invalid downgrade; server version is not allowed to join when downgrade is enabled",
			zap.String("current-server-version", version.Version),
			zap.String("target-cluster-version", d.TargetVersion),
		)
	}

	// if the cluster disables downgrade, check local version against determined cluster version.
	// the validation passes when local version is not less than cluster version
	if cv != nil && lv.LessThan(*cv) {
		lg.Fatal(
			"invalid downgrade; server version is lower than determined cluster version",
			zap.String("current-server-version", version.Version),
			zap.String("determined-cluster-version", version.Cluster(cv.String())),
		)
	}
}

func AllowedDowngradeVersion(ver *semver.Version) *semver.Version {
	// Todo: handle the case that downgrading from higher major version(e.g. downgrade from v4.0 to v3.x)
	return &semver.Version{Major: ver.Major, Minor: ver.Minor - 1}
}
