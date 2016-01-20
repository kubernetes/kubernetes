/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package rkt

import (
	"fmt"

	"github.com/coreos/go-semver/semver"
	rktapi "github.com/coreos/rkt/api/v1alpha"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

// rktVersion implementes kubecontainer.Version interface by implementing
// Compare() and String() (which is implemented by the underlying semver.Version)
type rktVersion struct {
	*semver.Version
}

func newRktVersion(version string) (rktVersion, error) {
	sem, err := semver.NewVersion(version)
	if err != nil {
		return rktVersion{}, err
	}
	return rktVersion{sem}, nil
}

func (r rktVersion) Compare(other string) (int, error) {
	v, err := semver.NewVersion(other)
	if err != nil {
		return -1, err
	}

	if r.LessThan(*v) {
		return -1, nil
	}
	if v.LessThan(*r.Version) {
		return 1, nil
	}
	return 0, nil
}

// checkVersion tests whether the rkt/systemd/rkt-api-service that meet the version requirement.
// If all version requirements are met, it returns nil.
func (r *Runtime) checkVersion(minimumRktBinVersion, recommendedRktBinVersion, minimumAppcVersion, minimumRktApiVersion, minimumSystemdVersion string) error {
	// Check systemd version.
	var err error
	r.systemdVersion, err = r.systemd.Version()
	if err != nil {
		return err
	}
	result, err := r.systemdVersion.Compare(minimumSystemdVersion)
	if err != nil {
		return err
	}
	if result < 0 {
		return fmt.Errorf("rkt: systemd version(%v) is too old, requires at least %v", r.systemdVersion, minimumSystemdVersion)
	}

	// Example for the version strings returned by GetInfo():
	// RktVersion:"0.10.0+gitb7349b1" AppcVersion:"0.7.1" ApiVersion:"1.0.0-alpha"
	resp, err := r.apisvc.GetInfo(context.Background(), &rktapi.GetInfoRequest{})
	if err != nil {
		return err
	}

	// Check rkt binary version.
	r.binVersion, err = newRktVersion(resp.Info.RktVersion)
	if err != nil {
		return err
	}
	result, err = r.binVersion.Compare(minimumRktBinVersion)
	if err != nil {
		return err
	}
	if result < 0 {
		return fmt.Errorf("rkt: binary version is too old(%v), requires at least %v", resp.Info.RktVersion, minimumRktBinVersion)
	}
	result, err = r.binVersion.Compare(recommendedRktBinVersion)
	if err != nil {
		return err
	}
	if result != 0 {
		// TODO(yifan): Record an event to expose the information.
		glog.Warningf("rkt: current binary version %q is not recommended (recommended version %q)", resp.Info.RktVersion, recommendedRktBinVersion)
	}

	// Check Appc version.
	r.appcVersion, err = newRktVersion(resp.Info.AppcVersion)
	if err != nil {
		return err
	}
	result, err = r.appcVersion.Compare(minimumAppcVersion)
	if err != nil {
		return err
	}
	if result < 0 {
		return fmt.Errorf("rkt: appc version is too old(%v), requires at least %v", resp.Info.AppcVersion, minimumAppcVersion)
	}

	// Check rkt API version.
	r.apiVersion, err = newRktVersion(resp.Info.ApiVersion)
	if err != nil {
		return err
	}
	result, err = r.apiVersion.Compare(minimumRktApiVersion)
	if err != nil {
		return err
	}
	if result < 0 {
		return fmt.Errorf("rkt: API version is too old(%v), requires at least %v", resp.Info.ApiVersion, minimumRktApiVersion)
	}
	return nil
}
