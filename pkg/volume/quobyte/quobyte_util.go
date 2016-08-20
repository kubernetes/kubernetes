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
	"fmt"
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
	volumeSize := int(volume.RoundUpSize(provisioner.options.Capacity.Value(), 1024*1024*1024))
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
		manager.config.quobyteApiServer,
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

// ADD Annotations to a PersistentVolume all information needed to delete the Volume
func addVolumeAnnotations(apiServer, adminSecret, adminSecretNamespace string, pv *api.PersistentVolume) {
	if pv.Annotations == nil {
		pv.Annotations = map[string]string{}
	}
	pv.Annotations[annotationQuobyteAPIServer] = apiServer
	pv.Annotations[annotationQuobyteAPISecret] = adminSecret
	pv.Annotations[annotationQuobyteAPISecretNamespace] = adminSecretNamespace
}

func parseVolumeAnnotations(pv *api.PersistentVolume) (apiServer, adminSecret, adminSecretNamespace string, err error) {
	var ok bool

	if pv.Annotations == nil {
		err = fmt.Errorf("cannot parse volume annotations: no annotations found")
		return
	}

	if apiServer, ok = pv.Annotations[annotationQuobyteAPIServer]; !ok {
		err = fmt.Errorf("cannot parse volume annotations: annotation %q not found", annotationQuobyteAPIServer)
		return
	}

	if adminSecret, ok = pv.Annotations[annotationQuobyteAPISecret]; !ok {
		err = fmt.Errorf("cannot parse volume annotations: annotation %q not found", annotationQuobyteAPISecret)
		return
	}

	if adminSecretNamespace, ok = pv.Annotations[annotationQuobyteAPISecretNamespace]; !ok {
		err = fmt.Errorf("cannot parse volume annotations: annotation %q not found", annotationQuobyteAPISecretNamespace)
		return
	}

	return
}

func (plugin *quobytePlugin) getUserAndPasswordFromSecret(namespace, secretName string) (string, string, error) {
	var user, password string
	kubeClient := plugin.host.GetKubeClient()
	if kubeClient == nil {
		return user, password, fmt.Errorf("Cannot get kube client")
	}

	secrets, err := kubeClient.Core().Secrets(namespace).Get(secretName)
	if err != nil {
		return user, password, err
	}

	for name, data := range secrets.Data {
		if string(name) == "user" {
			user = string(data)
		}
		if string(name) == "password" {
			password = string(data)
		}

	}

	//sanity check
	if user == "" {
		return user, password, fmt.Errorf("Missing \"user\" in secret")
	}

	if password == "" {
		return user, password, fmt.Errorf("Missing \"password\" in secret")
	}

	glog.V(4).Infof("quobyte secret [%q/%q] created User: %s %s", namespace, secretName, user, password)

	return user, password, nil
}
