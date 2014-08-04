/*
Copyright 2014 Google Inc. All rights reserved.

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

package build

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/build/buildapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
)

// TODO: Need to add a reconciler loop that makes sure that things in pods are reflected into
//       kubelet (and vice versa)

// EtcdRegistry implements BuildRegistry backed by etcd.
type EtcdRegistry struct {
	etcdClient tools.EtcdClient
}

// MakeEtcdRegistry creates an etcd registry.
// 'client' is the connection to etcd
func MakeEtcdRegistry(client tools.EtcdClient) *EtcdRegistry {
	registry := &EtcdRegistry{
		etcdClient: client,
	}
	return registry
}

func (registry *EtcdRegistry) helper() *tools.EtcdHelper {
	return &tools.EtcdHelper{registry.etcdClient}
}

func makeBuildKey(id string) string {
	return "/builds/" + id
}

// ListBuilds obtains a list of Builds.
func (registry *EtcdRegistry) ListBuilds() (buildapi.BuildList, error) {
	var list buildapi.BuildList
	err := registry.helper().ExtractList("/builds", &list.Items)
	return list, err
}

// GetBuild gets a specific Build specified by its ID.
func (registry *EtcdRegistry) GetBuild(buildID string) (*buildapi.Build, error) {
	var build *buildapi.Build
	err := registry.helper().ExtractObj(makeBuildKey(buildID), &build, false)
	if tools.IsEtcdNotFound(err) {
		return nil, apiserver.NewNotFoundErr("build", buildID)
	}
	if err != nil {
		return nil, err
	}
	return build, nil
}

// CreateBuild creates a new Build.
func (registry *EtcdRegistry) CreateBuild(build buildapi.Build) error {
	return registry.UpdateBuild(build)
}

// UpdateBuild replaces an existing Build.
func (registry *EtcdRegistry) UpdateBuild(build buildapi.Build) error {
	return registry.helper().SetObj(makeBuildKey(build.ID), build)
}

// DeleteBuild deletes a Build specified by its ID.
func (registry *EtcdRegistry) DeleteBuild(buildID string) error {
	key := makeBuildKey(buildID)
	_, err := registry.etcdClient.Delete(key, true)
	if tools.IsEtcdNotFound(err) {
		return apiserver.NewNotFoundErr("build", buildID)
	}
	return err
}
