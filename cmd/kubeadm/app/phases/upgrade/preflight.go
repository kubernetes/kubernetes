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

package upgrade

import (
	"fmt"
	"os"
	"strings"

	"github.com/coredns/corefile-migration/migration"

	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

// CoreDNSCheck validates installed kubelet version
type CoreDNSCheck struct {
	name   string
	client clientset.Interface
	f      func(clientset.Interface) error
}

// Name is part of the preflight.Checker interface
func (c CoreDNSCheck) Name() string {
	return c.name
}

// Check is part of the preflight.Checker interface
func (c CoreDNSCheck) Check() (warnings, errors []error) {
	if err := c.f(c.client); err != nil {
		return nil, []error{err}
	}
	return nil, nil
}

// RunCoreDNSMigrationCheck initializes checks related to CoreDNS migration.
func RunCoreDNSMigrationCheck(client clientset.Interface, ignorePreflightErrors sets.Set[string]) error {
	migrationChecks := []preflight.Checker{
		&CoreDNSCheck{
			name:   "CoreDNSUnsupportedPlugins",
			client: client,
			f:      checkUnsupportedPlugins,
		},
		&CoreDNSCheck{
			name:   "CoreDNSMigration",
			client: client,
			f:      checkMigration,
		},
	}

	return preflight.RunChecks(migrationChecks, os.Stderr, ignorePreflightErrors)
}

// checkUnsupportedPlugins checks if there are any plugins included in the current configuration
// that are unsupported for migration.
func checkUnsupportedPlugins(client clientset.Interface) error {
	klog.V(1).Infoln("validating if there are any unsupported CoreDNS plugins in the Corefile")
	_, corefile, currentInstalledCoreDNSversion, err := dns.GetCoreDNSInfo(client)
	if err != nil {
		return err
	}

	currentInstalledCoreDNSversion = strings.TrimLeft(currentInstalledCoreDNSversion, "v")
	targetCoreDNSVersion := strings.TrimLeft(kubeadmconstants.CoreDNSVersion, "v")
	if currentInstalledCoreDNSversion == targetCoreDNSVersion {
		return nil
	}
	unsupportedCoreDNS, err := migration.Unsupported(currentInstalledCoreDNSversion, targetCoreDNSVersion, corefile)
	if err != nil {
		return err
	}
	if len(unsupportedCoreDNS) != 0 {
		var UnsupportedPlugins []string
		for _, unsup := range unsupportedCoreDNS {
			UnsupportedPlugins = append(UnsupportedPlugins, unsup.ToString())
		}
		fmt.Println("[preflight] The corefile contains plugins that kubeadm/CoreDNS does not know how to migrate. " +
			"Each plugin listed should be manually verified for compatibility with the newer version of CoreDNS. " +
			"Once ready, the upgrade can be initiated by skipping the preflight check. During the upgrade, " +
			"kubeadm will migrate the configuration while leaving the listed plugin configs untouched, " +
			"but cannot guarantee that they will work with the newer version of CoreDNS.")
		return errors.Errorf("CoreDNS cannot migrate the following plugins:\n%s", UnsupportedPlugins)
	}
	return nil
}

// checkMigration validates if migration of the current CoreDNS ConfigMap is possible.
func checkMigration(client clientset.Interface) error {
	klog.V(1).Infoln("validating if migration can be done for the current CoreDNS release.")
	_, corefile, currentInstalledCoreDNSversion, err := dns.GetCoreDNSInfo(client)
	if err != nil {
		return err
	}

	currentInstalledCoreDNSversion = strings.TrimLeft(currentInstalledCoreDNSversion, "v")
	_, err = migration.Migrate(currentInstalledCoreDNSversion, strings.TrimLeft(kubeadmconstants.CoreDNSVersion, "v"), corefile, false)
	return errors.Wrap(err, "CoreDNS will not be upgraded")
}
