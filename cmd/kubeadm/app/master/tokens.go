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

	"k8s.io/apimachinery/pkg/util/uuid"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

const (
	// TODO: prefix with kubeadm prefix
	KubeletBootstrapUser = "kubeadm-node-csr"

	KubeletBootstrapGroup = "kubeadm:kubelet-bootstrap"
)

func CreateTokenAuthFile(bt string) error {
	tokenAuthFilePath := path.Join(kubeadmapi.GlobalEnvParams.HostPKIPath, "tokens.csv")
	if err := os.MkdirAll(kubeadmapi.GlobalEnvParams.HostPKIPath, 0700); err != nil {
		return fmt.Errorf("failed to create directory %q [%v]", kubeadmapi.GlobalEnvParams.HostPKIPath, err)
	}
	serialized := []byte(fmt.Sprintf("%s,%s,%s,%s\n", bt, KubeletBootstrapUser, uuid.NewUUID(), KubeletBootstrapGroup))
	// DumpReaderToFile create a file with mode 0600
	if err := cmdutil.DumpReaderToFile(bytes.NewReader(serialized), tokenAuthFilePath); err != nil {
		return fmt.Errorf("failed to save token auth file (%q) [%v]", tokenAuthFilePath, err)
	}
	return nil
}
