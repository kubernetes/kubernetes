/*
Copyright 2016 The Kubernetes Authors.

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

package kubemaster

import (
	"bytes"
	"fmt"
	"os"
	"path"

	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	kubeadmutil "k8s.io/kubernetes/pkg/kubeadm/util"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/uuid"
)

func generateTokenIfNeeded(params *kubeadmapi.BootstrapParams) error {
	ok, err := kubeadmutil.UseGivenTokenIfValid(params)
	if !ok {
		if err != nil {
			return err
		}
		err = kubeadmutil.GenerateToken(params)
		if err != nil {
			return err
		}
		fmt.Printf("<master/tokens> generated token: %q\n", params.Discovery.GivenToken)
	} else {
		fmt.Println("<master/tokens> accepted provided token")
	}

	return nil
}

func CreateTokenAuthFile(params *kubeadmapi.BootstrapParams) error {
	tokenAuthFilePath := path.Join(params.EnvParams["host_pki_path"], "tokens.csv")
	if err := generateTokenIfNeeded(params); err != nil {
		return fmt.Errorf("<master/tokens> failed to generate token(s) [%s]", err)
	}
	if err := os.MkdirAll(params.EnvParams["host_pki_path"], 0700); err != nil {
		return fmt.Errorf("<master/tokens> failed to create directory %q [%s]", params.EnvParams["host_pki_path"], err)
	}
	serialized := []byte(fmt.Sprintf("%s,kubeadm-node-csr,%s,system:kubelet-bootstrap\n", params.Discovery.BearerToken, uuid.NewUUID()))
	if err := cmdutil.DumpReaderToFile(bytes.NewReader(serialized), tokenAuthFilePath); err != nil {
		return fmt.Errorf("<master/tokens> failed to save token auth file (%q) [%s]", tokenAuthFilePath, err)
	}
	return nil
}
