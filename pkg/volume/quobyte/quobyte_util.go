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

package quobyte

import (
	"net"
	"path"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/golang/glog"
	quobyte_api "github.com/quobyte/api"
)

type quobyteVolumeManager struct {
	config *quobyteAPIConfig
}

func (manager *quobyteVolumeManager) createVolume(provisioner *quobyteVolumeProvisioner) (quobyte *api.QuobyteVolumeSource, size int, err error) {
	capacity := provisioner.options.PVC.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)]
	volumeSize := int(volume.RoundUpSize(capacity.Value(), 1024*1024*1024))
	// Quobyte has the concept of Volumes which doen't have a specific size (they can grow unlimited)
	// to simulate a size constraint we could set here a Quota
	volumeRequest := &quobyte_api.CreateVolumeRequest{
		Name:              provisioner.volume,
		RootUserID:        provisioner.user,
		RootGroupID:       provisioner.group,
		TenantID:          provisioner.tenant,
		ConfigurationName: provisioner.config,
	}

	if _, err := manager.createQuobyteClient().CreateVolume(volumeRequest); err != nil {
		return &api.QuobyteVolumeSource{}, volumeSize, err
	}

	glog.V(4).Infof("Created Quobyte volume %s", provisioner.volume)
	return &api.QuobyteVolumeSource{
		Registry: provisioner.registry,
		Volume:   provisioner.volume,
		User:     provisioner.user,
		Group:    provisioner.group,
	}, volumeSize, nil
}

func (manager *quobyteVolumeManager) deleteVolume(deleter *quobyteVolumeDeleter) error {
	return manager.createQuobyteClient().DeleteVolumeByName(deleter.volume, deleter.tenant)
}

func (manager *quobyteVolumeManager) createQuobyteClient() *quobyte_api.QuobyteClient {
	return quobyte_api.NewQuobyteClient(
		manager.config.quobyteAPIServer,
		manager.config.quobyteUser,
		manager.config.quobytePassword,
	)
}

func (mounter *quobyteMounter) pluginDirIsMounted(pluginDir string) (bool, error) {
	mounts, err := mounter.mounter.List()
	if err != nil {
		return false, err
	}

	for _, mountPoint := range mounts {
		if strings.HasPrefix(mountPoint.Type, "quobyte") {
			continue
		}

		if mountPoint.Path == pluginDir {
			glog.V(4).Infof("quobyte: found mountpoint %s in /proc/mounts", mountPoint.Path)
			return true, nil
		}
	}

	return false, nil
}

func (mounter *quobyteMounter) correctTraillingSlash(regStr string) string {
	return path.Clean(regStr) + "/"
}

func validateRegistry(registry string) bool {
	if len(registry) == 0 {
		return false
	}

	for _, hostPortPair := range strings.Split(registry, ",") {
		if _, _, err := net.SplitHostPort(hostPortPair); err != nil {
			return false
		}
	}

	return true
}
