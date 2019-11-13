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

package installer

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"

	addoninstall "sigs.k8s.io/addon-operators/installer/install"
)

// ApplyAddonConfiguration invokes the sigs.k8s.io/addon-operators/installer library to apply addons to the cluster.
// It fetches the AddonInstallerConfiguration component config from the InitConfiguration.
// `realKubeConfigPath` should be the path of a potential kubeconfig for an APIServer.
// If `realKubeConfigPath` does not stat, the installer's runtime.ServerDryRun will not be enabled.
func ApplyAddonConfiguration(cfg *kubeadmapi.InitConfiguration, client clientset.Interface, realReadOnlyClient clientset.Interface, dryRun bool, realKubeConfigPath string) (err error) {
	installCfg := cfg.ClusterConfiguration.ComponentConfigs.AddonInstaller
	if installCfg == nil {
		return errors.New("addoninstaller phase invoked with nil AddonInstaller ComponentConfig")
	}

	// Override the AddonInstallerConfiguration dryRun field with the kubeadm dryRun runtime value.
	// Note that this makes the dryRun field of any user-specified ComponentConfig file meaningless when used /w kubeadm.
	// It's possible to make kubeadm dry-run different from the addon-installer dry-run with legacyFlag style logic,
	// but that requires more work and my not make sense.
	if installCfg.DryRun != dryRun {
		fmt.Fprintf(os.Stderr, "[addon installer] WARNING: overriding AddonInstallerConfiguration with dryRun: %t\n", dryRun)
		installCfg.DryRun = dryRun
	}

	realKubeConfigPathExists := false
	if _, err := os.Stat(realKubeConfigPath); err == nil {
		realKubeConfigPathExists = true
	}

	if !installCfg.DryRun || (realKubeConfigPathExists && realReadOnlyClient != nil) {
		k8sVersion := version.MustParseGeneric(cfg.ClusterConfiguration.KubernetesVersion)
		// We can use realReadOnlyClient to download the configmap from a real apiserver if it exists, even in dryRun mode
		prevInstallCfg, err := DownloadConfiguration(realReadOnlyClient, k8sVersion)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[addon installer] ERROR: downloading previous config: %v\n", err)
		}
		if prevInstallCfg != nil {
			// TODO: diff addons and prune (delete) ones that are no longer declared
		}
	}

	r := addoninstall.Runtime{
		Config: installCfg,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
	if realKubeConfigPathExists {
		r.KubeConfigPath = realKubeConfigPath
		r.ServerDryRun = true
	}

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		errs := r.HandleSignal(<-sigs)
		for _, err := range errs {
			fmt.Fprintf(os.Stderr, "[addon installer] ERROR: %v\n", err)
		}
	}()

	err = r.CheckDeps()
	if err != nil {
		return err
	}
	err = r.CheckConfig()
	if err != nil {
		return err
	}
	err = r.InstallAddons()
	if err != nil {
		return err
	}

	uploadConfiguration(cfg.ComponentConfigs.AddonInstaller, client)

	return nil
}
