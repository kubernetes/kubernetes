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

package factory

import (
	"fmt"
	"io/ioutil"
	"path/filepath"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

// MasterConfigurationFactory provides a factory with the responsibility to create/and store the kubeadm masterConfiguration used during kubeadm init.
// Additionally it exposes convenience methods for accessing paths derived from such masterConfiguration/used during kubeadm init.
type MasterConfigurationFactory struct {
	masterConfigurationInstance *kubeadmapi.MasterConfiguration

	certsDir            string
	certsDirToWriteTo   string
	kubeConfigDir       string
	manifestDir         string
	adminKubeConfigPath string
}

// InitMasterConfiguration initializes the masterConfiguration starting from the v1alpha1 configuration passed to the command;
// it includes also a validation of all the configuration entry, defaulting for missing values and computation of paths used during kubeadm init.
func (f *MasterConfigurationFactory) InitMasterConfiguration(v1alpha1Cfg *kubeadmapiext.MasterConfiguration, cfgPath string, dryRun bool) error {

	cfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(cfgPath, v1alpha1Cfg)
	if err != nil {
		return err
	}

	certsDirToWriteTo, kubeConfigDir, manifestDir, err := getDirectoriesToUse(dryRun, cfg.CertificatesDir)
	if err != nil {
		return err
	}

	f.masterConfigurationInstance = cfg
	f.certsDir = cfg.CertificatesDir
	f.certsDirToWriteTo = certsDirToWriteTo
	f.kubeConfigDir = kubeConfigDir
	f.manifestDir = manifestDir
	f.adminKubeConfigPath = filepath.Join(kubeConfigDir, constants.AdminKubeConfigFileName)

	return nil
}

// getDirectoriesToUse returns the (in order) certificates, kubeconfig and Static Pod manifest directories, followed by a possible error
// This behaves differently when dry-running vs the normal flow
func getDirectoriesToUse(dryRun bool, defaultPkiDir string) (string, string, string, error) {
	if dryRun {
		dryRunDir, err := ioutil.TempDir("", "kubeadm-init-dryrun")
		if err != nil {
			return "", "", "", fmt.Errorf("couldn't create a temporary directory: %v", err)
		}
		// Use the same temp dir for all
		return dryRunDir, dryRunDir, dryRunDir, nil
	}

	return defaultPkiDir, constants.KubernetesDir, constants.GetStaticPodDirectory(), nil
}

// MasterConfiguration returns the MasterConfiguration instance.
func (f *MasterConfigurationFactory) MasterConfiguration() *kubeadmapi.MasterConfiguration {
	if f.masterConfigurationInstance == nil {
		panic("Invalid operation. InitMasterConfiguration must be executed before GetMasterConfiguration")
	}
	return f.masterConfigurationInstance
}

// CertsDir returns the path of directory where certificate are stored.
func (f *MasterConfigurationFactory) CertsDir() string {
	if f.masterConfigurationInstance == nil {
		panic("Invalid operation. InitMasterDirectory must be executed before GetCertsDir")
	}
	return f.certsDir
}

// CertsDirToWriteTo returns path of directory where certificate should be written.
// NB. CertsDir is different from CertsDirToWriteTo only in case of dry running
func (f *MasterConfigurationFactory) CertsDirToWriteTo() string {
	if f.masterConfigurationInstance == nil {
		panic("Invalid operation. InitMasterDirectory must be executed before GetCertsDirToWriteTo")
	}
	return f.certsDirToWriteTo
}

// KubeConfigDir returns path of directory for kubeconfig files.
func (f *MasterConfigurationFactory) KubeConfigDir() string {
	if f.masterConfigurationInstance == nil {
		panic("Invalid operation. InitMasterDirectory must be executed before GetKubeConfigDir")
	}
	return f.kubeConfigDir
}

// ManifestDir returns path of directory for manifest files.
func (f *MasterConfigurationFactory) ManifestDir() string {
	if f.masterConfigurationInstance == nil {
		panic("Invalid operation. InitMasterDirectory must be executed before GetManifestDir")
	}
	return f.manifestDir
}

// AdminKubeConfigPath returns name of the admin kubeconfig file.
func (f *MasterConfigurationFactory) AdminKubeConfigPath() string {
	if f.masterConfigurationInstance == nil {
		panic("Invalid operation. adminKubeConfigPath must be executed before GetAdminKubeConfigPath")
	}
	return f.adminKubeConfigPath
}
