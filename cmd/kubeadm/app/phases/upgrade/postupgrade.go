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
	"path/filepath"

	"github.com/pkg/errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptoken "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	kubeletphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubelet"
	patchnodephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/patchnode"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
)

// PerformPostUpgradeTasks runs nearly the same functions as 'kubeadm init' would do
// Note that the mark-control-plane phase is left out, not needed, and no token is created as that doesn't belong to the upgrade
func PerformPostUpgradeTasks(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, patchesDir string, dryRun bool, out io.Writer) error {
	var errs []error

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
	if err := WriteKubeletConfigFiles(cfg, patchesDir, dryRun, out); err != nil {
		errs = append(errs, err)
	}

	// Annotate the node with the crisocket information, sourced either from the InitConfiguration struct or
	// --cri-socket.
	// TODO: In the future we want to use something more official like NodeStatus or similar for detecting this properly
	if err := patchnodephase.AnnotateCRISocket(client, cfg.NodeRegistration.Name, cfg.NodeRegistration.CRISocket); err != nil {
		errs = append(errs, errors.Wrap(err, "error uploading crisocket"))
	}

	// Create RBAC rules that makes the bootstrap tokens able to get nodes
	if err := nodebootstraptoken.AllowBootstrapTokensToGetNodes(client); err != nil {
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

	if err := PerformAddonsUpgrade(client, cfg, patchesDir, out); err != nil {
		errs = append(errs, err)
	}

	if features.Enabled(cfg.FeatureGates, features.ControlPlaneKubeletLocalMode) {
		if err := UpdateKubeletLocalMode(cfg, dryRun); err != nil {
			return errors.Wrap(err, "failed to update kubelet local mode")
		}
	}

	return errorsutil.NewAggregate(errs)
}

// PerformAddonsUpgrade performs the upgrade of the coredns and kube-proxy addons.
func PerformAddonsUpgrade(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, patchesDir string, out io.Writer) error {
	unupgradedControlPlanes, err := unupgradedControlPlaneInstances(client, cfg.NodeRegistration.Name)
	if err != nil {
		return errors.Wrapf(err, "failed to determine whether all the control plane instances have been upgraded")
	}
	if len(unupgradedControlPlanes) > 0 {
		fmt.Fprintf(out, "[upgrade/addons] skip upgrade addons because control plane instances %v have not been upgraded\n", unupgradedControlPlanes)
		return nil
	}

	var errs []error

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
		if err := dns.EnsureDNSAddon(&cfg.ClusterConfiguration, client, patchesDir, out, false); err != nil {
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

// unupgradedControlPlaneInstances returns a list of control plane instances that have not yet been upgraded.
//
// NB. This function can only be called after the current control plane instance has been upgraded already.
// Because it determines whether the other control plane instances have been upgraded by checking whether
// the kube-apiserver image of other control plane instance is the same as that of this instance.
func unupgradedControlPlaneInstances(client clientset.Interface, nodeName string) ([]string, error) {
	selector := labels.SelectorFromSet(labels.Set(map[string]string{
		"component": kubeadmconstants.KubeAPIServer,
	}))
	pods, err := client.CoreV1().Pods(metav1.NamespaceSystem).List(context.TODO(), metav1.ListOptions{
		LabelSelector: selector.String(),
	})
	if err != nil {
		return nil, errors.Wrap(err, "failed to list kube-apiserver Pod from cluster")
	}
	if len(pods.Items) == 0 {
		return nil, errors.Errorf("cannot find kube-apiserver Pod by label selector: %v", selector.String())
	}

	nodeImageMap := map[string]string{}

	for _, pod := range pods.Items {
		found := false
		for _, c := range pod.Spec.Containers {
			if c.Name == kubeadmconstants.KubeAPIServer {
				nodeImageMap[pod.Spec.NodeName] = c.Image
				found = true
				break
			}
		}
		if !found {
			return nil, errors.Errorf("cannot find container by name %q for Pod %v", kubeadmconstants.KubeAPIServer, klog.KObj(&pod))
		}
	}

	upgradedImage, ok := nodeImageMap[nodeName]
	if !ok {
		return nil, errors.Errorf("cannot find kube-apiserver image for current control plane instance %v", nodeName)
	}

	unupgradedNodes := sets.New[string]()
	for node, image := range nodeImageMap {
		if image != upgradedImage {
			unupgradedNodes.Insert(node)
		}
	}

	if len(unupgradedNodes) > 0 {
		return sets.List(unupgradedNodes), nil
	}

	return nil, nil
}

// WriteKubeletConfigFiles writes the kubelet config file to disk, but first creates a backup of any existing one.
func WriteKubeletConfigFiles(cfg *kubeadmapi.InitConfiguration, patchesDir string, dryRun bool, out io.Writer) error {
	// Set up the kubelet directory to use. If dry-running, this will return a fake directory
	kubeletDir, err := GetKubeletDir(dryRun)
	if err != nil {
		// The error here should never occur in reality, would only be thrown if /tmp doesn't exist on the machine.
		return err
	}

	// Create a copy of the kubelet config file in the /etc/kubernetes/tmp/ folder.
	backupDir, err := kubeadmconstants.CreateTempDirForKubeadm(kubeadmconstants.KubernetesDir, "kubeadm-kubelet-config")
	if err != nil {
		return err
	}
	src := filepath.Join(kubeletDir, kubeadmconstants.KubeletConfigurationFileName)
	dest := filepath.Join(backupDir, kubeadmconstants.KubeletConfigurationFileName)

	if !dryRun {
		fmt.Printf("[upgrade] Backing up kubelet config file to %s\n", dest)
		err := kubeadmutil.CopyFile(src, dest)
		if err != nil {
			return errors.Wrap(err, "error backing up the kubelet config file")
		}
	} else {
		fmt.Printf("[dryrun] Would back up kubelet config file to %s\n", dest)
	}

	errs := []error{}
	// Write the configuration for the kubelet down to disk so the upgraded kubelet can start with fresh config
	if err := kubeletphase.WriteConfigToDisk(&cfg.ClusterConfiguration, kubeletDir, patchesDir, out); err != nil {
		errs = append(errs, errors.Wrap(err, "error writing kubelet configuration to file"))
	}

	if dryRun { // Print what contents would be written
		err := dryrunutil.PrintDryRunFile(kubeadmconstants.KubeletConfigurationFileName, kubeletDir, kubeadmconstants.KubeletRunDirectory, os.Stdout)
		if err != nil {
			errs = append(errs, errors.Wrap(err, "error printing files on dryrun"))
		}
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

// UpdateKubeletLocalMode changes the Server URL in the kubelets kubeconfig to the local API endpoint if it is currently
// set to the ControlPlaneEndpoint.
// TODO: remove this function once kubeletKubeConfigFilePath goes GA and is hardcoded to enabled by default:
// https://github.com/kubernetes/kubeadm/issues/2271
func UpdateKubeletLocalMode(cfg *kubeadmapi.InitConfiguration, dryRun bool) error {
	kubeletKubeConfigFilePath := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.KubeletKubeConfigFileName)

	if _, err := os.Stat(kubeletKubeConfigFilePath); err != nil {
		if os.IsNotExist(err) {
			klog.V(2).Infof("Could not mutate the Server URL in %s: %v", kubeletKubeConfigFilePath, err)
			return nil
		}
		return err
	}

	config, err := clientcmd.LoadFromFile(kubeletKubeConfigFilePath)
	if err != nil {
		return err
	}

	configContext, ok := config.Contexts[config.CurrentContext]
	if !ok {
		return errors.Errorf("cannot find cluster for active context in kubeconfig %q", kubeletKubeConfigFilePath)
	}

	localAPIEndpoint, err := kubeadmutil.GetLocalAPIEndpoint(&cfg.LocalAPIEndpoint)
	if err != nil {
		return err
	}

	controlPlaneAPIEndpoint, err := kubeadmutil.GetControlPlaneEndpoint(cfg.ControlPlaneEndpoint, &cfg.LocalAPIEndpoint)
	if err != nil {
		return err
	}

	// Skip changing kubeconfig file if Server does not match the ControlPlaneEndpoint.
	if config.Clusters[configContext.Cluster].Server != controlPlaneAPIEndpoint || controlPlaneAPIEndpoint == localAPIEndpoint {
		klog.V(2).Infof("Skipping update of the Server URL in %s, because it's already not equal to %q or already matches the localAPIEndpoint", kubeletKubeConfigFilePath, cfg.ControlPlaneEndpoint)
		return nil
	}

	if dryRun {
		fmt.Printf("[dryrun] Would change the Server URL from %q to %q in %s and try to restart kubelet\n", config.Clusters[configContext.Cluster].Server, localAPIEndpoint, kubeletKubeConfigFilePath)
		return nil
	}

	klog.V(1).Infof("Changing the Server URL from %q to %q in %s", config.Clusters[configContext.Cluster].Server, localAPIEndpoint, kubeletKubeConfigFilePath)
	config.Clusters[configContext.Cluster].Server = localAPIEndpoint

	if err := clientcmd.WriteToFile(*config, kubeletKubeConfigFilePath); err != nil {
		return err
	}

	kubeletphase.TryRestartKubelet()

	return nil
}
