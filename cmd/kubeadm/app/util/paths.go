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

package util

import (
	"fmt"
	"io/ioutil"
	"os"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// Paths stores all the directory paths
type Paths struct {
	certificatesDir string
	kubernetesDir   string
	manifestDir     string
	kubeletDir      string
}

// InitPaths initializes all the directory paths
func InitPaths(dryRun bool, certsDir string) (*Paths, error) {
	if dryRun {
		dryRunDir, err := ioutil.TempDir("", "kubeadm-init-dryrun")
		if err != nil {
			return nil, fmt.Errorf("couldn't create a temporary directory: %v", err)
		}
		// Use the same temp dir for all
		return &Paths{
			dryRunDir, dryRunDir, dryRunDir, dryRunDir,
		}, nil
	}

	if _, err := os.Stat(certsDir); os.IsNotExist(err) {
		return nil, err
	}

	return &Paths{
		certificatesDir: certsDir,
		kubernetesDir:   kubeadmconstants.KubernetesDir,
		manifestDir:     kubeadmconstants.GetStaticPodDirectory(),
		kubeletDir:      kubeadmconstants.KubeletRunDirectory,
	}, nil
}

func (p *Paths) CertificateDir() string {
	return p.certificatesDir
}

func (p *Paths) ManifestDir() string {
	return p.manifestDir
}

func (p *Paths) KubeletDir() string {
	return p.kubeletDir
}

func (p *Paths) KubernetesDir() string {
	return p.kubernetesDir
}
