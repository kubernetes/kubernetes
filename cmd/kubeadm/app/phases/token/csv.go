/*
Copyright 2017 The Kubernetes Authors.

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

package token

import (
	"bytes"
	"fmt"
	"os"
	"path"

	"k8s.io/apimachinery/pkg/util/uuid"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

// CreateTokenAuthFile creates the CSV file that can be used for allowing users with tokens access to the API Server
func CreateTokenAuthFile(bt string) error {
	tokenAuthFilePath := path.Join(kubeadmapi.GlobalEnvParams.HostPKIPath, kubeadmconstants.CSVTokenFileName)
	if err := os.MkdirAll(kubeadmapi.GlobalEnvParams.HostPKIPath, 0700); err != nil {
		return fmt.Errorf("failed to create directory %q [%v]", kubeadmapi.GlobalEnvParams.HostPKIPath, err)
	}
	serialized := []byte(fmt.Sprintf("%s,%s,%s,%s\n", bt, kubeadmconstants.CSVTokenBootstrapUser, uuid.NewUUID(), kubeadmconstants.CSVTokenBootstrapGroup))
	// DumpReaderToFile create a file with mode 0600
	if err := cmdutil.DumpReaderToFile(bytes.NewReader(serialized), tokenAuthFilePath); err != nil {
		return fmt.Errorf("failed to save token auth file (%q) [%v]", tokenAuthFilePath, err)
	}
	return nil
}
