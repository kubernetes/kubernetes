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

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/uuid"
)

func generateTokenIfNeeded(s *kubeadmapi.Secrets) error {
	ok, err := kubeadmutil.UseGivenTokenIfValid(s)
	// TODO(phase1+) @krousey: I know it won't happen with the way it is currently implemented, but this doesn't handle case where ok is true and err is non-nil.
	if !ok {
		if err != nil {
			return err
		}
		err = kubeadmutil.GenerateToken(s)
		if err != nil {
			return err
		}
		fmt.Printf("<master/tokens> generated token: %q\n", s.GivenToken)
	} else {
		fmt.Println("<master/tokens> accepted provided token")
	}

	return nil
}

func CreateTokenAuthFile(s *kubeadmapi.Secrets) error {
	tokenAuthFilePath := path.Join(kubeadmapi.GlobalEnvParams.HostPKIPath, "tokens.csv")
	if err := generateTokenIfNeeded(s); err != nil {
		return fmt.Errorf("<master/tokens> failed to generate token(s) [%v]", err)
	}
	if err := os.MkdirAll(kubeadmapi.GlobalEnvParams.HostPKIPath, 0700); err != nil {
		return fmt.Errorf("<master/tokens> failed to create directory %q [%v]", kubeadmapi.GlobalEnvParams.HostPKIPath, err)
	}
	serialized := []byte(fmt.Sprintf("%s,kubeadm-node-csr,%s,system:kubelet-bootstrap\n", s.BearerToken, uuid.NewUUID()))
	// DumpReaderToFile create a file with mode 0600
	if err := cmdutil.DumpReaderToFile(bytes.NewReader(serialized), tokenAuthFilePath); err != nil {
		return fmt.Errorf("<master/tokens> failed to save token auth file (%q) [%v]", tokenAuthFilePath, err)
	}
	return nil
}
