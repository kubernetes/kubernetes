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

package master

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

func generateTokenIfNeeded(s *kubeadmapi.KubeadmConfig) error {
	ok, err := kubeadmutil.UseGivenTokenIfValid(s)
	if !ok {
		if err != nil {
			return err
		}
		err = kubeadmutil.GenerateToken(s)
		if err != nil {
			return err
		}
		fmt.Printf("<master/tokens> generated token: %q\n", s.Secrets.GivenToken)
	} else {
		fmt.Println("<master/tokens> accepted provided token")
	}

	return nil
}

func CreateTokenAuthFile(s *kubeadmapi.KubeadmConfig) error {
	tokenAuthFilePath := path.Join(s.EnvParams["host_pki_path"], "tokens.csv")
	if err := generateTokenIfNeeded(s); err != nil {
		return fmt.Errorf("<master/tokens> failed to generate token(s) [%s]", err)
	}
	if err := os.MkdirAll(s.EnvParams["host_pki_path"], 0700); err != nil {
		return fmt.Errorf("<master/tokens> failed to create directory %q [%s]", s.EnvParams["host_pki_path"], err)
	}
	serialized := []byte(fmt.Sprintf("%s,kubeadm-node-csr,%s,system:kubelet-bootstrap\n", s.Secrets.BearerToken, uuid.NewUUID()))
	if err := cmdutil.DumpReaderToFile(bytes.NewReader(serialized), tokenAuthFilePath); err != nil {
		return fmt.Errorf("<master/tokens> failed to save token auth file (%q) [%s]", tokenAuthFilePath, err)
	}
	return nil
}
