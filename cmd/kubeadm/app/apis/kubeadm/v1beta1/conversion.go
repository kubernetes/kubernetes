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

package v1beta1

import (
	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func Convert_kubeadm_InitConfiguration_To_v1beta1_InitConfiguration(in *kubeadm.InitConfiguration, out *InitConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_InitConfiguration_To_v1beta1_InitConfiguration(in, out, s); err != nil {
		return err
	}

	if in.CertificateKey != "" {
		return errors.New("certificateKey field is not supported by v1beta1 config format")
	}

	return nil
}

func Convert_kubeadm_JoinControlPlane_To_v1beta1_JoinControlPlane(in *kubeadm.JoinControlPlane, out *JoinControlPlane, s conversion.Scope) error {
	if err := autoConvert_kubeadm_JoinControlPlane_To_v1beta1_JoinControlPlane(in, out, s); err != nil {
		return err
	}

	if in.CertificateKey != "" {
		return errors.New("certificateKey field is not supported by v1beta1 config format")
	}

	return nil
}

func Convert_kubeadm_NodeRegistrationOptions_To_v1beta1_NodeRegistrationOptions(in *kubeadm.NodeRegistrationOptions, out *NodeRegistrationOptions, s conversion.Scope) error {
	if err := autoConvert_kubeadm_NodeRegistrationOptions_To_v1beta1_NodeRegistrationOptions(in, out, s); err != nil {
		return err
	}

	if len(in.IgnorePreflightErrors) > 0 {
		return errors.New("ignorePreflightErrors field is not supported by v1beta1 config format")
	}

	return nil
}
