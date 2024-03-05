/*
Copyright 2017 The Kubernetes Authors.

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

// KubeletConfigurationPathRefs returns pointers to all of the KubeletConfiguration fields that contain filepaths.
// You might use this, for example, to resolve all relative paths against some common root before
// passing the configuration to the application. This method must be kept up to date as new fields are added.
func KubeletConfigurationPathRefs(kc *KubeletConfiguration) []*string {
	paths := []*string{}
	paths = append(paths, &kc.StaticPodPath)
	paths = append(paths, &kc.Authentication.X509.ClientCAFile)
	paths = append(paths, &kc.TLSCertFile)
	paths = append(paths, &kc.TLSPrivateKeyFile)
	paths = append(paths, &kc.ResolverConfig)
	paths = append(paths, &kc.VolumePluginDir)
	paths = append(paths, &kc.PodLogsDir)
	return paths
}
