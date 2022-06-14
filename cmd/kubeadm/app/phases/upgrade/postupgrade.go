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
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptoken "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
)

// PerformPostUpgradeTasks runs nearly the same functions as 'kubeadm init' would do
// Note that the mark-control-plane phase is left out, not needed, and no token is created as that doesn't belong to the upgrade
func PerformPostUpgradeTasks(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, patchesDir string, dryRun bool, out io.Writer) error {
	errs := []error{}

	// Upload currently used configuration to the cluster
	// Note: This is done right in the beginning of cluster initialization; as we might want to make other phases
	// depend on centralized information from this source in the future
	if err := uploadconfig.UploadConfiguration(cfg, client); err != nil {
		errs = append(errs, err)
	}

	// Create the new, version-branched kubelet ComponentConfig ConfigMap
	if err := kubeletphase.CreateConfigMap(&cfg.ClusterConfiguration, client); err != nil {
		errs = append(errs, errors.Wrap(err, "error creating kubelet configuration ConfigMap"))
	}

	// Write the new kubelet config down to disk and the env file if needed
	if err := writeKubeletConfigFiles(client, cfg, patchesDir, dryRun, out); err != nil {
		errs = append(errs, err)
	}

	// Annotate the node with the crisocket information, sourced either from the InitConfiguration struct or
	// --cri-socket.
	// TODO: In the future we want to use something more official like NodeStatus or similar for detecting this properly
	if err := patchnodephase.AnnotateCRISocket(client, cfg.NodeRegistration.Name, cfg.NodeRegistration.CRISocket); err != nil {
		errs = append(errs, errors.Wrap(err, "error uploading crisocket"))
	}

	// Create RBAC rules that makes the bootstrap tokens able to get nodes
	if err := nodebootstraptoken.AllowBoostrapTokensToGetNodes(client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the bootstrap tokens able to post CSRs
	if err := nodebootstraptoken.AllowBootstrapTokensToPostCSRs(client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the bootstrap tokens able to get their CSRs approved automatically
	if err := nodebootstraptoken.AutoApproveNodeBootstrapTokens(client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the nodes to rotate certificates and get their CSRs approved automatically
	if err := nodebootstraptoken.AutoApproveNodeCertificateRotation(client); err != nil {
		errs = append(errs, err)
	}

	// TODO: Is this needed to do here? I think that updating cluster info should probably be separate from a normal upgrade
	// Create the cluster-info ConfigMap with the associated RBAC rules
	// if err := clusterinfo.CreateBootstrapConfigMapIfNotExists(client, kubeadmconstants.GetAdminKubeConfigPath()); err != nil {
	// 	return err
	//}
	// Create/update RBAC rules that makes the cluster-info ConfigMap reachable
	if err := clusterinfo.CreateClusterInfoRBACRules(client); err != nil {
		errs = append(errs, err)
	}

	// If the coredns ConfigMap is missing, show a warning and assume that the
	// DNS addon was skipped during "kubeadm init", and that its redeployment on upgrade is not desired.
	//
	// TODO: remove this once "kubeadm upgrade apply" phases are supported:
	//   https://github.com/kubernetes/kubeadm/issues/1318
	var missingCoreDNSConfigMap bool
	if _, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(
		context.TODO(),
		kubeadmconstants.CoreDNSConfigMap,
		metav1.GetOptions{},
	); err != nil && apierrors.IsNotFound(err) {
		missingCoreDNSConfigMap = true
	}
	if missingCoreDNSConfigMap {
		klog.Warningf("the ConfigMaps %q in the namespace %q were not found. "+
			"Assuming that a DNS server was not deployed for this cluster. "+
			"Note that once 'kubeadm upgrade apply' supports phases you "+
			"will have to skip the DNS upgrade manually",
			kubeadmconstants.CoreDNSConfigMap,
			metav1.NamespaceSystem)
	} else {
		// Upgrade CoreDNS
		if err := dns.EnsureDNSAddon(&cfg.ClusterConfiguration, client, out, false); err != nil {
			errs = append(errs, err)
		}
	}

	// If the kube-proxy ConfigMap is missing, show a warning and assume that kube-proxy
	// was skipped during "kubeadm init", and that its redeployment on upgrade is not desired.
	//
	// TODO: remove this once "kubeadm upgrade apply" phases are supported:
	//   https://github.com/kubernetes/kubeadm/issues/1318
	if _, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(
		context.TODO(),
		kubeadmconstants.KubeProxyConfigMap,
		metav1.GetOptions{},
	); err != nil && apierrors.IsNotFound(err) {
		klog.Warningf("the ConfigMap %q in the namespace %q was not found. "+
			"Assuming that kube-proxy was not deployed for this cluster. "+
			"Note that once 'kubeadm upgrade apply' supports phases you "+
			"will have to skip the kube-proxy upgrade manually",
			kubeadmconstants.KubeProxyConfigMap,
			metav1.NamespaceSystem)
	} else {
		// Upgrade kube-proxy
		if err := proxy.EnsureProxyAddon(&cfg.ClusterConfiguration, &cfg.LocalAPIEndpoint, client, out, false); err != nil {
			errs = append(errs, err)
		}
	}

	return errorsutil.NewAggregate(errs)
}

func writeKubeletConfigFiles(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, patchesDir string, dryRun bool, out io.Writer) error {
	kubeletDir, err := GetKubeletDir(dryRun)
	if err != nil {
		// The error here should never occur in reality, would only be thrown if /tmp doesn't exist on the machine.
		return err
	}
	errs := []error{}
	// Write the configuration for the kubelet down to disk so the upgraded kubelet can start with fresh config
	if err := kubeletphase.WriteConfigToDisk(&cfg.ClusterConfiguration, kubeletDir, patchesDir, out); err != nil {
		errs = append(errs, errors.Wrap(err, "error writing kubelet configuration to file"))
	}

	if dryRun { // Print what contents would be written
		dryrunutil.PrintDryRunFile(kubeadmconstants.KubeletConfigurationFileName, kubeletDir, kubeadmconstants.KubeletRunDirectory, os.Stdout)
	}
	return errorsutil.NewAggregate(errs)
}

// GetKubeletDir gets the kubelet directory based on whether the user is dry-running this command or not.
func GetKubeletDir(dryRun bool) (string, error) {
	if dryRun {
		return kubeadmconstants.CreateTempDirForKubeadm("", "kubeadm-upgrade-dryrun")
	}
	return kubeadmconstants.KubeletRunDirectory, nil
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
	return errors.Errorf("couldn't move these files: %v. Got errors: %v", files, errorsutil.NewAggregate(errs))
}

// RemoveOldControlPlaneTaint finds all nodes with the new "control-plane" node-role label
// and removes the old "control-plane" taint to them.
// TODO: https://github.com/kubernetes/kubeadm/issues/2200
func RemoveOldControlPlaneTaint(client clientset.Interface) error {
	selectorControlPlane := labels.SelectorFromSet(labels.Set(map[string]string{
		kubeadmconstants.LabelNodeRoleControlPlane: "",
	}))
	nodes, err := client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{
		LabelSelector: selectorControlPlane.String(),
	})
	if err != nil {
		return errors.Wrapf(err, "could not list nodes labeled with %q", kubeadmconstants.LabelNodeRoleControlPlane)
	}

	for _, n := range nodes.Items {
		// Check if the node has the old taint
		hasOldTaint := false
		taints := []v1.Taint{}
		for _, t := range n.Spec.Taints {
			if t.String() == kubeadmconstants.OldControlPlaneTaint.String() {
				hasOldTaint = true
				continue
			}
			// Collect all other taints
			taints = append(taints, t)
		}
		// If the old taint is present remove it
		if hasOldTaint {
			err = apiclient.PatchNode(client, n.Name, func(n *v1.Node) {
				n.Spec.Taints = taints
			})
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func updateKubeletDynamicEnvFileWithURLScheme(str string) string {
	const (
		flag   = "container-runtime-endpoint"
		scheme = kubeadmapiv1.DefaultContainerRuntimeURLScheme + "://"
	)
	// Trim the prefix
	str = strings.TrimLeft(str, fmt.Sprintf("%s=\"", kubeadmconstants.KubeletEnvFileVariableName))

	// Flags are managed by kubeadm as pairs of key=value separated by space.
	// Split them, find the one containing the flag of interest and update
	// its value to have the scheme prefix.
	split := strings.Split(str, " ")
	for i, s := range split {
		if !strings.Contains(s, flag) {
			continue
		}
		keyValue := strings.Split(s, "=")
		if len(keyValue) < 2 {
			// Post init/join, the user may have edited the file and has flags that are not
			// followed by "=". If that is the case the next argument must be the value
			// of the endpoint flag and if its not a flag itself. Update that argument with
			// the scheme instead.
			if i+1 < len(split) {
				nextArg := split[i+1]
				if !strings.HasPrefix(nextArg, "-") && !strings.HasPrefix(nextArg, scheme) {
					split[i+1] = scheme + nextArg
				}
			}
			continue
		}
		if len(keyValue[1]) == 0 || strings.HasPrefix(keyValue[1], scheme) {
			continue // The flag value already has the URL scheme prefix or is empty
		}
		// Missing prefix. Add it and update the key=value pair
		keyValue[1] = scheme + keyValue[1]
		split[i] = strings.Join(keyValue, "=")
	}
	str = strings.Join(split, " ")
	return fmt.Sprintf("%s=\"%s", kubeadmconstants.KubeletEnvFileVariableName, str)
}
