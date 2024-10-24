/*
Copyright 2024 The Kubernetes Authors.

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

package statusz

import (
	"time"

	"github.com/blang/semver/v4"
	"k8s.io/component-base/version"
	"k8s.io/klog/v2"

	utilversion "k8s.io/apiserver/pkg/util/version"
	compbasemetrics "k8s.io/component-base/metrics"
)

type statuszRegistry interface {
	processStartTime() time.Time
	goVersion() string
	binaryVersion() semver.Version
	emulationVersion() semver.Version
	minCompatibilityVersion() semver.Version
	usefulLinks() map[string]string
}

type registry struct{}

func (registry) processStartTime() time.Time {
	start, err := compbasemetrics.GetProcessStart()
	if err != nil {
		klog.Errorf("Could not get process start time, %v", err)
	}

	return time.Unix(int64(start), 0)
}

func (registry) goVersion() string {
	return version.Get().GoVersion
}

func (registry) binaryVersion() semver.Version {
	var binaryVersion semver.Version
	binaryVersion, err := semver.ParseTolerant(utilversion.DefaultComponentGlobalsRegistry.EffectiveVersionFor(utilversion.DefaultKubeComponent).BinaryVersion().String())
	if err != nil {
		klog.Errorf("err parsing binary version: %v", err)
	}

	return binaryVersion
}

func (registry) emulationVersion() semver.Version {
	var emulationVersion semver.Version
	emulationVersion, err := semver.ParseTolerant(utilversion.DefaultComponentGlobalsRegistry.EffectiveVersionFor(utilversion.DefaultKubeComponent).EmulationVersion().String())
	if err != nil {
		klog.Errorf("err parsing emulationVersion version: %v", err)
	}

	return emulationVersion
}

func (registry) minCompatibilityVersion() semver.Version {
	var minCompatVersion semver.Version
	minCompatVersion, err := semver.ParseTolerant(utilversion.DefaultComponentGlobalsRegistry.EffectiveVersionFor(utilversion.DefaultKubeComponent).MinCompatibilityVersion().String())
	if err != nil {
		klog.Errorf("err parsing min compatibility version: %v", err)
	}

	return minCompatVersion
}

func (registry) usefulLinks() map[string]string {
	return map[string]string{
		"healthz":     "/healthz",
		"livez":       "/livez",
		"readyz":      "/readyz",
		"metrics":     "/metrics",
		"sli metrics": "/metrics/slis",
	}
}
