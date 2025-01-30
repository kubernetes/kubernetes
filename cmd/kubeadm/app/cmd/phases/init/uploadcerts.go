/*
Copyright 2019 The Kubernetes Authors.

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

package phases

import (
	"fmt"

	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/copycerts"
)

// NewUploadCertsPhase returns the uploadCerts phase
func NewUploadCertsPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "upload-certs",
		Short: fmt.Sprintf("Upload certificates to %s", kubeadmconstants.KubeadmCertsSecret),
		Long:  fmt.Sprintf("Upload control plane certificates to the %s Secret", kubeadmconstants.KubeadmCertsSecret),
		Run:   runUploadCerts,
		InheritFlags: []string{
			options.CfgPath,
			options.KubeconfigPath,
			options.UploadCerts,
			options.CertificateKey,
			options.SkipCertificateKeyPrint,
			options.DryRun,
		},
	}
}

func runUploadCerts(c workflow.RunData) error {
	data, ok := c.(InitData)
	if !ok {
		return errors.New("upload-certs phase invoked with an invalid data struct")
	}

	if !data.UploadCerts() {
		fmt.Printf("[upload-certs] Skipping phase. Please see --%s\n", options.UploadCerts)
		return nil
	}
	client, err := data.Client()
	if err != nil {
		return err
	}

	if len(data.CertificateKey()) == 0 {
		certificateKey, err := copycerts.CreateCertificateKey()
		if err != nil {
			return err
		}
		data.SetCertificateKey(certificateKey)
	}

	if err := copycerts.UploadCerts(client, data.Cfg(), data.CertificateKey()); err != nil {
		return errors.Wrap(err, "error uploading certs")
	}
	if !data.SkipCertificateKeyPrint() {
		fmt.Printf("[upload-certs] Using certificate key:\n%s\n", data.CertificateKey())
	}
	return nil
}
