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
	"os"

	"github.com/coredns/corefile-migration/migration"
	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
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
func RunCoreDNSMigrationCheck(client clientset.Interface, ignorePreflightErrors sets.String, dnsType kubeadmapi.DNSAddOnType) error {
	if dnsType != kubeadmapi.CoreDNS {
		return nil
	}
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
	unsupportedCoreDNS, err := migration.Unsupported(currentInstalledCoreDNSversion, kubeadmconstants.CoreDNSVersion, corefile)
	if err != nil {
		return err
	}
	if unsupportedCoreDNS != nil {
		var UnsupportedPlugins, UnsupportedVersion string
		for _, unsup := range unsupportedCoreDNS {
			UnsupportedPlugins = unsup.Plugin
			UnsupportedVersion = unsup.Version

		}
		if UnsupportedPlugins != "" || UnsupportedVersion != "" {
			return errors.New("there are unsupported plugins in the CoreDNS Corefile")
		}
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

	_, err = migration.Migrate(currentInstalledCoreDNSversion, kubeadmconstants.CoreDNSVersion, corefile, false)
	if err != nil {
		return errors.Wrap(err, "CoreDNS will not be upgraded")
	}
	return nil
}
