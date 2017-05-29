/*
Copyright 2014 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"os"
	"path"
	"regexp"
	"strings"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	validationutil "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
)

const FederatedDefaultTestTimeout = 5 * time.Minute

// Detects whether the federation namespace exists in the underlying cluster
func SkipUnlessFederated(c clientset.Interface) {
	federationNS := framework.FederationSystemNamespace()

	_, err := c.Core().Namespaces().Get(federationNS, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			framework.Skipf("Could not find federation namespace %s: skipping federated test", federationNS)
		} else {
			framework.Failf("Unexpected error getting namespace: %v", err)
		}
	}
}

// WaitForFederationApiserverReady waits for the federation apiserver to be ready.
// It tests the readiness by sending a GET request and expecting a non error response.
func WaitForFederationApiserverReady(c *federation_clientset.Clientset) error {
	return wait.PollImmediate(time.Second, 1*time.Minute, func() (bool, error) {
		_, err := c.Federation().Clusters().List(metav1.ListOptions{})
		if err != nil {
			return false, nil
		}
		return true, nil
	})
}

func LoadFederationClientset() (*federation_clientset.Clientset, error) {
	config, err := LoadFederatedConfig(&clientcmd.ConfigOverrides{})
	if err != nil {
		return nil, err
	}

	c, err := federation_clientset.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("error creating federation clientset: %v", err.Error())
	}
	return c, nil
}

func LoadFederatedConfig(overrides *clientcmd.ConfigOverrides) (*restclient.Config, error) {
	c, err := framework.RestclientConfig(framework.TestContext.FederatedKubeContext)
	if err != nil {
		return nil, fmt.Errorf("error creating federation client config: %v", err.Error())
	}
	cfg, err := clientcmd.NewDefaultClientConfig(*c, overrides).ClientConfig()
	if cfg != nil {
		//TODO(colhom): this is only here because https://github.com/kubernetes/kubernetes/issues/25422
		cfg.NegotiatedSerializer = api.Codecs
	}
	if err != nil {
		return cfg, fmt.Errorf("error creating federation default client config: %v", err.Error())
	}
	return cfg, nil
}

// GetValidDNSSubdomainName massages the given name to be a valid dns subdomain name.
// Most resources (such as secrets, clusters) require the names to be valid dns subdomain.
// This is a generic function (not specific to federation). Should be moved to a more generic location if others want to use it.
func GetValidDNSSubdomainName(name string) (string, error) {
	// "_" are not allowed. Replace them by "-".
	name = regexp.MustCompile("_").ReplaceAllLiteralString(name, "-")
	maxLength := validationutil.DNS1123SubdomainMaxLength
	if len(name) > maxLength {
		name = name[0 : maxLength-1]
	}
	// Verify that name now passes the validation.
	if errors := validation.NameIsDNSSubdomain(name, false); len(errors) != 0 {
		return "", fmt.Errorf("errors in converting name to a valid DNS subdomain %s", errors)
	}
	return name, nil
}

func FederationControlPlaneUpgrade(version string) error {
	version = "v" + version
	_, _, err := framework.RunCmd(path.Join(framework.TestContext.RepoRoot, "federation/cluster/upgrade.sh"), version)
	return err
}

func CheckFederationVersion(c federation_clientset.Interface, want string) error {
	framework.Logf("Checking federation version")
	v, err := c.Discovery().ServerVersion()
	if err != nil {
		return fmt.Errorf("CheckFederationVersion() couldn't get the master version: %v", err)
	}
	// We do prefix trimming and then matching because:
	// want looks like:  0.19.3-815-g50e67d4
	// got  looks like: v0.19.3-815-g50e67d4034e858-dirty
	got := strings.TrimPrefix(v.GitVersion, "v")
	if !strings.HasPrefix(got, want) {
		return fmt.Errorf("federation had apiserver version %s which does not start with %s",
			got, want)
	}
	framework.Logf("Federation is at version %s", want)
	return nil
}

func MasterUpgrade(context, version string) error {
	switch framework.TestContext.Provider {
	case "gce":
		return masterUpgradeGCE(context, version)
	default:
		return fmt.Errorf("MasterUpgrade() is not implemented for provider %s", framework.TestContext.Provider)
	}
}

func masterUpgradeGCE(context, rawVersion string) error {
	version := "v" + rawVersion
	// TODO: this breaks if we want to upgrade 2 clusters in same zone. use alternate methods in future to get zone of a cluster
	zone := strings.TrimPrefix(context, "federation-e2e-"+framework.TestContext.Provider+"-")

	env := append(os.Environ(), "KUBE_CONTEXT="+context, "ZONE="+zone)
	_, _, err := framework.RunCmdEnv(env, path.Join(framework.TestContext.RepoRoot, "cluster/gce/upgrade.sh"), "-M", version)
	return err
}
