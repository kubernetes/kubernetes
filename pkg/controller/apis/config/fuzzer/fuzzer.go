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

package fuzzer

import (
	"fmt"

	"sigs.k8s.io/randfill"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	kubectrlmgrconfig "k8s.io/kubernetes/pkg/controller/apis/config"
)

// Funcs returns the fuzzer functions for the kube-controller manager apis.
func Funcs(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *kubectrlmgrconfig.KubeControllerManagerConfiguration, c randfill.Continue) {
			c.FillNoCustom(obj)
			obj.Generic.Address = fmt.Sprintf("%d.%d.%d.%d", c.Intn(256), c.Intn(256), c.Intn(256), c.Intn(256))
			obj.Generic.ClientConnection.ContentType = fmt.Sprintf("%s/%s.%s.%s", c.String(0), c.String(0), c.String(0), c.String(0))
			if obj.Generic.LeaderElection.ResourceLock == "" {
				obj.Generic.LeaderElection.ResourceLock = "leases"
			}
			obj.Generic.Controllers = []string{c.String(0)}
			if obj.KubeCloudShared.ClusterName == "" {
				obj.KubeCloudShared.ClusterName = "kubernetes"
			}
			obj.CSRSigningController.ClusterSigningCertFile = fmt.Sprintf("/%s", c.String(0))
			obj.CSRSigningController.ClusterSigningKeyFile = fmt.Sprintf("/%s", c.String(0))
			obj.PersistentVolumeBinderController.VolumeConfiguration.FlexVolumePluginDir = fmt.Sprintf("/%s", c.String(0))
			obj.TTLAfterFinishedController.ConcurrentTTLSyncs = c.Int31()
		},
	}
}
