/*
Copyright 2018 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"path/filepath"

	"k8s.io/apimachinery/pkg/runtime/serializer"
	apiserver "k8s.io/apiserver/pkg/server"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
	kubectrlmgrscheme "k8s.io/kubernetes/pkg/controller/apis/config/scheme"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

// Config is the main context object for the controller manager.
type Config struct {
	ComponentConfig kubectrlmgrconfig.KubeControllerManagerConfiguration

	SecureServing *apiserver.SecureServingInfo
	// LoopbackClientConfig is a config for a privileged loopback connection
	LoopbackClientConfig *restclient.Config

	// TODO: remove deprecated insecure serving
	InsecureServing *apiserver.DeprecatedInsecureServingInfo
	Authentication  apiserver.AuthenticationInfo
	Authorization   apiserver.AuthorizationInfo

	// the general kube client
	Client *clientset.Clientset

	// the client only used for leader election
	LeaderElectionClient *clientset.Clientset

	// the rest config for the master
	Kubeconfig *restclient.Config

	// the event sink
	EventRecorder record.EventRecorder
}

type completedConfig struct {
	*Config
}

// CompletedConfig same as Config, just to swap private object.
type CompletedConfig struct {
	// Embed a private pointer that cannot be instantiated outside of this package.
	*completedConfig
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (c *Config) Complete() *CompletedConfig {
	cc := completedConfig{c}

	apiserver.AuthorizeClientBearerToken(c.LoopbackClientConfig, &c.Authentication, &c.Authorization)

	return &CompletedConfig{&cc}
}

func loadConfigFile(name string) (*kubectrlmgrconfig.KubeControllerManagerConfiguration, error) {
	const errFmt = "failed to load controller manager config file %q: %v"
	f, err := filepath.Abs(name)
	if err != nil {
		return nil, fmt.Errorf(errFmt, name, err)
	}
	loader, err := NewFsLoader(utilfs.DefaultFs{}, f)
	if err != nil {
		return nil, fmt.Errorf(errFmt, name, err)
	}
	cmc, err := loader.Load()
	if err != nil {
		return nil, fmt.Errorf(errFmt, name, err)
	}

	return cmc, nil
}

// Loader loads configuration from a storage layer
type Loader interface {
	// Load loads and returns the KubeletConfiguration from the storage layer,
	// or an error if a configuration could not be loaded
	Load() (*kubectrlmgrconfig.KubeControllerManagerConfiguration, error)
}

// fsLoader loads configuration from `configDir`
type fsLoader struct {
	// fs is the filesystem where the config files exist; can be mocked for
	// testing
	fs utilfs.Filesystem
	// controllerManagerCodecs is the scheme used to decode config files
	controllerManagerCodecs *serializer.CodecFactory
	// controllerManagerFile is an absolute path to the file containing a
	// serialized controller manager configuration
	controllerManagerFile string
}

func (loader *fsLoader) Load() (*kubectrlmgrconfig.KubeControllerManagerConfiguration, error) {
	data, err := loader.fs.ReadFile(loader.controllerManagerFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read controller manager config file %q: %v", loader.controllerManagerFile, err)
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("controller manager config file %q was empty", loader.controllerManagerFile)
	}

	// TODO: handle strict decoding error
	obj, gvk, err := loader.controllerManagerCodecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decode controller manager config: %v", err)
	}
	internalCMC, ok := obj.(*kubectrlmgrconfig.KubeControllerManagerConfiguration)
	if !ok {
		return nil, fmt.Errorf("failed to cast object to KubeControllerManagerConfiguration, unexpected type: %v", gvk)
	}
	return internalCMC, nil
}

// NewFsLoader returns a Loader that loads a KubeletConfiguration from the `kubeletFile`
func NewFsLoader(fs utilfs.Filesystem, kubeletFile string) (Loader, error) {
	// TODO: make strict
	_, controllerManagerCodes, err := kubectrlmgrscheme.NewSchemeAndCodecs()
	if err != nil {
		return nil, err
	}

	return &fsLoader{
		fs:                      fs,
		controllerManagerCodecs: controllerManagerCodes,
		controllerManagerFile:   kubeletFile,
	}, nil
}
