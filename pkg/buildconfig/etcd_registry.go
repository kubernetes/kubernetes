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

package buildconfig

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/buildconfig/buildconfigapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
)

// TODO: Need to add a reconciler loop that makes sure that things in pods are reflected into
//       kubelet (and vice versa)

// EtcdRegistry implements BuildConfigRegistry backed by etcd.
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

func makeBuildConfigKey(id string) string {
	return "/build-configs/" + id
}

// ListBuildConfigs obtains a list of BuildConfigs.
func (registry *EtcdRegistry) ListBuildConfigs() (buildconfigapi.BuildConfigList, error) {
	var list buildconfigapi.BuildConfigList
	err := registry.helper().ExtractList("/build-configs", &list.Items)
	return list, err
}

// GetBuildConfig gets a specific BuildConfig specified by its ID.
func (registry *EtcdRegistry) GetBuildConfig(buildConfigID string) (*buildconfigapi.BuildConfig, error) {
	var buildConfig buildconfigapi.BuildConfig
	err := registry.helper().ExtractObj(makeBuildConfigKey(buildConfigID), &buildConfig, false)
	if tools.IsEtcdNotFound(err) {
		return nil, apiserver.NewNotFoundErr("buildconfig", buildConfigID)
	}
	if err != nil {
		return nil, err
	}
	return &buildConfig, nil
}

// CreateBuildConfig creates a new BuildConfig.
func (registry *EtcdRegistry) CreateBuildConfig(buildConfig buildconfigapi.BuildConfig) error {
	return registry.UpdateBuildConfig(buildConfig)
}

// UpdateBuildConfig replaces an existing BuildConfig.
func (registry *EtcdRegistry) UpdateBuildConfig(buildConfig buildconfigapi.BuildConfig) error {
	return registry.helper().SetObj(makeBuildConfigKey(buildConfig.ID), buildConfig)
}

// DeleteBuildConfig deletes a BuildConfig specified by its ID.
func (registry *EtcdRegistry) DeleteBuildConfig(buildConfigID string) error {
	key := makeBuildConfigKey(buildConfigID)
	_, err := registry.etcdClient.Delete(key, true)
	if tools.IsEtcdNotFound(err) {
		return apiserver.NewNotFoundErr("buildconfig", buildConfigID)
	}
	return err
}
