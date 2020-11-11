// +build !dockerless

/*
Copyright 2020 The Kubernetes Authors.

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

package kubelet

import (
	"k8s.io/klog/v2"

	kubeletconfiginternal "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/dockershim"
	dockerremote "k8s.io/kubernetes/pkg/kubelet/dockershim/remote"
)

func runDockershim(kubeCfg *kubeletconfiginternal.KubeletConfiguration,
	kubeDeps *Dependencies,
	crOptions *config.ContainerRuntimeOptions,
	runtimeCgroups string,
	remoteRuntimeEndpoint string,
	remoteImageEndpoint string,
	nonMasqueradeCIDR string) error {
	pluginSettings := dockershim.NetworkPluginSettings{
		HairpinMode:        kubeletconfiginternal.HairpinMode(kubeCfg.HairpinMode),
		NonMasqueradeCIDR:  nonMasqueradeCIDR,
		PluginName:         crOptions.NetworkPluginName,
		PluginConfDir:      crOptions.CNIConfDir,
		PluginBinDirString: crOptions.CNIBinDir,
		PluginCacheDir:     crOptions.CNICacheDir,
		MTU:                int(crOptions.NetworkPluginMTU),
	}

	// Create and start the CRI shim running as a grpc server.
	streamingConfig := getStreamingConfig(kubeCfg, kubeDeps, crOptions)
	dockerClientConfig := &dockershim.ClientConfig{
		DockerEndpoint:            kubeDeps.DockerOptions.DockerEndpoint,
		RuntimeRequestTimeout:     kubeDeps.DockerOptions.RuntimeRequestTimeout,
		ImagePullProgressDeadline: kubeDeps.DockerOptions.ImagePullProgressDeadline,
	}
	ds, err := dockershim.NewDockerService(dockerClientConfig, crOptions.PodSandboxImage, streamingConfig,
		&pluginSettings, runtimeCgroups, kubeCfg.CgroupDriver, crOptions.DockershimRootDirectory)
	if err != nil {
		return err
	}

	// The unix socket for kubelet <-> dockershim communication, dockershim start before runtime service init.
	klog.V(5).Infof("RemoteRuntimeEndpoint: %q, RemoteImageEndpoint: %q",
		remoteRuntimeEndpoint,
		remoteImageEndpoint)
	klog.V(2).Infof("Starting the GRPC server for the docker CRI shim.")
	dockerServer := dockerremote.NewDockerServer(remoteRuntimeEndpoint, ds)
	if err := dockerServer.Start(); err != nil {
		return err
	}

	// Create dockerLegacyService when the logging driver is not supported.
	supported, err := ds.IsCRISupportedLogDriver()
	if err != nil {
		return err
	}
	if !supported {
		kubeDeps.dockerLegacyService = ds
	}

	return nil
}
