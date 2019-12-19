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

package metrics

import (
	"fmt"
	"regexp"

	"github.com/blang/semver"

	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

const (
	versionRegexpString = `^v(\d+\.\d+\.\d+)`
)

var (
	versionRe = regexp.MustCompile(versionRegexpString)
)

func parseSemver(s string) *semver.Version {
	if s != "" {
		sv := semver.MustParse(s)
		return &sv
	}
	return nil
}
func parseVersion(ver apimachineryversion.Info) semver.Version {
	matches := versionRe.FindAllStringSubmatch(ver.String(), -1)

	if len(matches) != 1 {
		panic(fmt.Sprintf("version string \"%v\" doesn't match expected regular expression: \"%v\"", ver.String(), versionRe.String()))
	}
	return semver.MustParse(matches[0][1])
}
