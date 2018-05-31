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

	kubeadmapiv1alpha2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2"
	utilsexec "k8s.io/utils/exec"
)

// Puller is an interface for pulling images
type Puller interface {
	Pull(string) error
}

// Existence is an interface to determine if an image exists on the system
// A nil error means the image was found
type Existence interface {
	Exists(string) error
}

// Images defines the set of behaviors needed for images relating to the CRI
type Images interface {
	Puller
	Existence
}

// CRInterfacer is a struct that interfaces with the container runtime
type CRInterfacer struct {
	criSocket  string
	exec       utilsexec.Interface
	crictlPath string
	dockerPath string
}

// NewCRInterfacer sets up and returns a CRInterfacer
func NewCRInterfacer(execer utilsexec.Interface, criSocket string) (*CRInterfacer, error) {
	var crictlPath, dockerPath string
	var err error
	if criSocket != kubeadmapiv1alpha2.DefaultCRISocket {
		if crictlPath, err = execer.LookPath("crictl"); err != nil {
			return nil, fmt.Errorf("crictl is required for non docker container runtimes: %v", err)
		}
	} else {
		// use the dockershim
		if dockerPath, err = execer.LookPath("docker"); err != nil {
			return nil, fmt.Errorf("`docker` is required when docker is the container runtime and the kubelet is not running: %v", err)
		}
	}

	return &CRInterfacer{
		exec:       execer,
		criSocket:  criSocket,
		crictlPath: crictlPath,
		dockerPath: dockerPath,
	}, nil
}

// Pull pulls the actual image using either crictl or docker
func (cri *CRInterfacer) Pull(image string) error {
	if cri.criSocket != kubeadmapiv1alpha2.DefaultCRISocket {
		return cri.exec.Command(cri.crictlPath, "-r", cri.criSocket, "pull", image).Run()
	}
	return cri.exec.Command(cri.dockerPath, "pull", image).Run()
}

// Exists checks to see if the image exists on the system already
// Returns an error if the image is not found.
func (cri *CRInterfacer) Exists(image string) error {
	if cri.criSocket != kubeadmapiv1alpha2.DefaultCRISocket {
		return cri.exec.Command(cri.crictlPath, "-r", cri.criSocket, "inspecti", image).Run()
	}
	return cri.exec.Command(cri.dockerPath, "inspect", image).Run()
}
