/*
Copyright 2015 The Kubernetes Authors.

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
	"sync"

	rktapi "github.com/coreos/rkt/api/v1alpha"
	"golang.org/x/net/context"

	utilversion "k8s.io/kubernetes/pkg/util/version"
)

type versions struct {
	sync.RWMutex
	binVersion     *utilversion.Version
	apiVersion     *utilversion.Version
	systemdVersion systemdVersion
}

func newRktVersion(version string) (*utilversion.Version, error) {
	return utilversion.ParseSemantic(version)
}

func (r *Runtime) getVersions() error {
	r.versions.Lock()
	defer r.versions.Unlock()

	// Get systemd version.
	var err error
	r.versions.systemdVersion, err = r.systemd.Version()
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), r.requestTimeout)
	defer cancel()
	// Example for the version strings returned by GetInfo():
	// RktVersion:"0.10.0+gitb7349b1" AppcVersion:"0.7.1" ApiVersion:"1.0.0-alpha"
	resp, err := r.apisvc.GetInfo(ctx, &rktapi.GetInfoRequest{})
	if err != nil {
		return err
	}

	// Get rkt binary version.
	r.versions.binVersion, err = newRktVersion(resp.Info.RktVersion)
	if err != nil {
		return err
	}

	// Get rkt API version.
	r.versions.apiVersion, err = newRktVersion(resp.Info.ApiVersion)
	if err != nil {
		return err
	}
	return nil
}

// checkVersion tests whether the rkt/systemd/rkt-api-service that meet the version requirement.
// If all version requirements are met, it returns nil.
func (r *Runtime) checkVersion(minimumRktBinVersion, minimumRktApiVersion, minimumSystemdVersion string) error {
	if err := r.getVersions(); err != nil {
		return err
	}

	r.versions.RLock()
	defer r.versions.RUnlock()

	// Check systemd version.
	result, err := r.versions.systemdVersion.Compare(minimumSystemdVersion)
	if err != nil {
		return err
	}
	if result < 0 {
		return fmt.Errorf("rkt: systemd version(%v) is too old, requires at least %v", r.versions.systemdVersion, minimumSystemdVersion)
	}

	// Check rkt binary version.
	result, err = r.versions.binVersion.Compare(minimumRktBinVersion)
	if err != nil {
		return err
	}
	if result < 0 {
		return fmt.Errorf("rkt: binary version is too old(%v), requires at least %v", r.versions.binVersion, minimumRktBinVersion)
	}

	// Check rkt API version.
	result, err = r.versions.apiVersion.Compare(minimumRktApiVersion)
	if err != nil {
		return err
	}
	if result < 0 {
		return fmt.Errorf("rkt: API version is too old(%v), requires at least %v", r.versions.apiVersion, minimumRktApiVersion)
	}
	return nil
}
