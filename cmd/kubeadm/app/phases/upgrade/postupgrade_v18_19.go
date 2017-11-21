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

package upgrade

import (
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	"k8s.io/apimachinery/pkg/util/errors"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/pkg/util/version"
)

var v190 = version.MustParseSemantic("v1.9.0")

// backupAPIServerCertAndKey backups the old cert and key of kube-apiserver to a specified directory.
func backupAPIServerCertAndKey() error {
	subDir := "expired"
	filesToMove := map[string]string{
		filepath.Join(kubeadmapiext.DefaultCertificatesDir, constants.APIServerCertName): filepath.Join(kubeadmapiext.DefaultCertificatesDir, subDir, constants.APIServerCertName),
		filepath.Join(kubeadmapiext.DefaultCertificatesDir, constants.APIServerKeyName):  filepath.Join(kubeadmapiext.DefaultCertificatesDir, subDir, constants.APIServerKeyName),
	}

	return moveFiles(filesToMove)
}

// moveFiles moves files from one directory to another.
func moveFiles(files map[string]string) error {
	filesToRecover := map[string]string{}
	for from, to := range files {
		if err := os.Rename(from, to); err != nil {
			return rollbackFiles(filesToRecover, err)
		}
		filesToRecover[to] = from
	}
	return nil
}

// rollbackFiles moves the files back to the original directory.
func rollbackFiles(files map[string]string, originalErr error) error {
	errs := []error{originalErr}
	for from, to := range files {
		if err := os.Rename(from, to); err != nil {
			errs = append(errs, err)
		}
	}
	return fmt.Errorf("couldn't move these files: %v. Got errors: %v", files, errors.NewAggregate(errs))
}

// shouldBackupAPIServerCertAndKey check if the new k8s version is at least 1.9.0
// and kube-apiserver will be expired in 60 days.
func shouldBackupAPIServerCertAndKey(newK8sVer *version.Version) (bool, error) {
	if !newK8sVer.LessThan(v190) {
		return false, nil
	}

	apiServerCert := filepath.Join(kubeadmapiext.DefaultCertificatesDir, constants.APIServerCertName)
	data, err := ioutil.ReadFile(apiServerCert)
	if err != nil {
		return false, fmt.Errorf("failed to read kube-apiserver certificate from disk: %v", err)
	}

	block, _ := pem.Decode(data)
	if block == nil {
		return false, fmt.Errorf("expected the kube-apiserver certificate to be PEM encoded")
	}

	certs, err := x509.ParseCertificates(block.Bytes)
	if err != nil {
		return false, fmt.Errorf("unable to parse certificate data: %v", err)
	}
	if len(certs) == 0 {
		return false, fmt.Errorf("no certificate data found")
	}

	if certs[0].NotAfter.Sub(time.Now()) < 60*24*time.Hour {
		return true, nil
	}

	return false, nil
}
