/*
Copyright 2021 The Kubernetes Authors.

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

package componentconfigs

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	kubeletconfig "k8s.io/kubelet/config/v1beta1"
	"k8s.io/utils/ptr"
)

// Mutate modifies absolute path fields in the KubeletConfiguration to be Windows compatible absolute paths.
func (kc *kubeletConfig) Mutate() error {
	// When "kubeadm join" downloads the KubeletConfiguration from the cluster on Windows
	// nodes, it would contain absolute paths that may lack drive letters, since the config
	// could have been generated on a Linux control-plane node. On Windows the
	// Golang filepath.IsAbs() function returns false unless the path contains a drive letter.
	// This trips client-go and the kubelet, creating problems on Windows nodes.
	// Fixing it in client-go or the kubelet is a breaking change to existing Windows
	// users that rely on relative paths:
	//   https://github.com/kubernetes/kubernetes/pull/77710#issuecomment-491989621
	//
	// Thus, a workaround here is to adapt the KubeletConfiguration paths for Windows.
	// Note this is currently bound to KubeletConfiguration v1beta1.
	klog.V(2).Infoln("[componentconfig] Adapting the paths in the KubeletConfiguration for Windows...")

	// Get the drive from where the kubeadm binary was called.
	exe, err := os.Executable()
	if err != nil {
		return errors.Wrap(err, "could not obtain information about the kubeadm executable")
	}
	drive := filepath.VolumeName(filepath.Dir(exe))
	klog.V(2).Infof("[componentconfig] Assuming Windows drive %q", drive)

	// Mutate the paths in the config.
	mutatePaths(&kc.config, drive)
	return nil
}

func mutatePaths(cfg *kubeletconfig.KubeletConfiguration, drive string) {
	mutateStringField := func(name string, field *string) {
		// filepath.IsAbs() is not reliable here in the Windows runtime, so check if the
		// path starts with "/" instead. This means the path originated from a Unix node and
		// is an absolute path.
		if !strings.HasPrefix(*field, "/") {
			return
		}
		// Prepend the drive letter to the path and update the field.
		*field = filepath.Join(drive, *field)
		klog.V(2).Infof("[componentconfig] kubelet/Windows: adapted path for field %q to %q", name, *field)
	}

	// Mutate the fields we care about.
	klog.V(2).Infof("[componentconfig] kubelet/Windows: changing field \"resolverConfig\" to empty")
	cfg.ResolverConfig = ptr.To("")
	mutateStringField("staticPodPath", &cfg.StaticPodPath)
	mutateStringField("authentication.x509.clientCAFile", &cfg.Authentication.X509.ClientCAFile)
}
