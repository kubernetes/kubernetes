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

package images

import (
	"fmt"

	kubeadmapiv1alpha1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	utilsexec "k8s.io/utils/exec"
)

// Puller is an interface for pulling images
type Puller interface {
	Pull(string) error
}

// ImagePuller is a struct that can pull images and hides the implementation (crictl vs docker)
type ImagePuller struct {
	criSocket  string
	exec       utilsexec.Interface
	crictlPath string
}

// NewImagePuller returns a ready to go ImagePuller
func NewImagePuller(execer utilsexec.Interface, criSocket string) (*ImagePuller, error) {
	crictlPath, err := execer.LookPath("crictl")
	if err != nil && criSocket != kubeadmapiv1alpha1.DefaultCRISocket {
		return nil, fmt.Errorf("crictl is required for non docker container runtimes: %v", err)
	}
	return &ImagePuller{
		exec:       execer,
		criSocket:  criSocket,
		crictlPath: crictlPath,
	}, nil
}

// Pull pulls the actual image using either crictl or docker
func (ip *ImagePuller) Pull(image string) error {
	if ip.criSocket != kubeadmapiv1alpha1.DefaultCRISocket {
		return ip.exec.Command(ip.crictlPath, "-r", ip.criSocket, "pull", image).Run()
	}
	return ip.exec.Command("sh", "-c", fmt.Sprintf("docker pull %v", image)).Run()
}
