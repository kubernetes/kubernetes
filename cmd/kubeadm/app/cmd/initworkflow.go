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

package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/util/phases"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
)

// phasesUsingKubernetesVersion contains a list of kubeadm phases that will use the kubernetes Version
var phasesUsingKubernetesVersion = []string{
	"controlplane",  // because the control plane images to use depends on the KubernetesVersion
	"etcd",          // because the etcd image to use depends on the KubernetesVersion
	"addons",        // because the addons images to use depends on the KubernetesVersion
	"upload-config", // because in this case it is not allowed to change the KubernetesVersion
}

// initWorkflow defines the main init workflow as a sequence of ordered phases
func (c *initContext) initWorkflow() phases.PhaseWorkflow {
	return phases.PhaseWorkflow{
		{ // prints init workflow start message (version info)
			Use:    "init-start",
			Hidden: true, // this phase can't be invoked directly; it is executed only as part of the full init workflow
			Run:    c.runInitStart,
		},
		{ // executes preflight checks
			Use:   "preflight",
			Short: "Run master pre-flight checks",
			Run:   c.runMasterPreflight,
		},
		{ // generates certs
			Use:        "certs",
			Aliases:    []string{"certificates"},
			Short:      "Generates all PKI assets necessary to establish the control plane",
			Phases:     c.certsWorkflow(),              // this is complex workflow defined as a set of nested phases
			WorkflowIf: c.workflowIfNotUsingExternalCA, // within the init workflow, certs generation is executed only if not using an External CA (direct execution is always allowed)
		},
		{ // generate kubeconfig files
			Use:        "kubeconfig",
			Short:      "Generates all kubeconfig files necessary to establish the control plane and the admin kubeconfig file",
			Phases:     c.kubeconfigWorkflow(),         // this is complex workflow defined as a set of nested phases
			WorkflowIf: c.workflowIfNotUsingExternalCA, // within the init workflow, certs generation is executed only if not using an External CA (direct execution is always allowed)
		},
		{ // creates the audit-policy if not exists
			Use:        "audit-policy",
			Short:      "Generates the audit-policy to use in the API server configuration (featureGated)",
			Run:        c.runAuditPolicy,
			WorkflowIf: c.workflowIfFeatureGateFor(features.Auditing), // within the init workflow, audit policy generation is executed only if the corresponding featureGates is true (direct execution is always allowed)
		},
		{ // creates controlplane manifests
			Use:    "controlplane",
			Short:  "Generates all static Pod manifest files necessary to establish the control plane",
			Phases: c.controlplaneWorkflow(), // this is complex workflow defined as a set of nested phases
		},
		{ // creates etcd manifests
			Use:   "etcd",
			Short: "Generates the static Pod manifest file for a local, single-node etcd instance",
			Run:   c.runEtcd,
		},
		{ // prints generated files when dry running
			Use:    "printFiles",
			Hidden: true, // this phase can't be invoked directly; it is executed only as part of the full init workflow
			Run:    c.runPrintFiles,
		},
		{ // initializes the kubelet config
			Use:        "init-KubeletConfig",
			Short:      "(featureGated) Writes base configuration of kubelets to disk",
			Run:        c.runInitKubeletConfig,
			WorkflowIf: c.workflowIfFeatureGateFor(features.DynamicKubeletConfig), // within the init workflow, kubelet config is executed only if the corresponding featureGates is true (direct execution is always allowed)
		},
		{ // wait for the controlplane/kubelet to start
			Use:    "init-wait",
			Hidden: true, // this phase can't be invoked directly; it is executed only as part of the full init workflow
			Run:    c.runInitWait,
		},
		{ // binds the kubelet config to the node
			Use:        "upload-KubeletConfig",
			Short:      "(featureGated) Uploads kubelet config to a configMap and use it as a configSource for this node",
			Run:        c.runUploadKubeletConfig,
			WorkflowIf: c.workflowIfFeatureGateFor(features.DynamicKubeletConfig), // within the init workflow, kubelet config is executed only if the corresponding featureGates is true (direct execution is always allowed)
		},
		{ // uploads kubeadm-config ConfigMap
			Use:   "upload-config",
			Short: "Uploads the currently used configuration for kubeadm to a ConfigMap",
			Run:   c.runUploadConfig,
		},
		{ // marks the master node master
			Use:   "mark-master",
			Short: "Marks a node as master",
			Run:   c.runMarkMaster,
		},
		{ // setups the bootstrap-token config
			Use:    "bootstrap-token",
			Short:  "Makes all the bootstrap token configurations and creates an initial token",
			Phases: c.bootstraptokenWorkflow(), // this is complex workflow defined as a set of nested phases
		},
		{ // deploy required addons
			Use:    "addons",
			Short:  "Installs required addons for passing Conformance tests",
			Phases: c.addonWorkflow(), // this is complex workflow defined as a set of nested phases
		},
		{ // mutates the static-pod based control plane manifests into a selfhosted control plane
			Use:        "selfhosting",
			Aliases:    []string{"self-hosting"},
			Short:      "Makes a kubeadm cluster self-hosted (featureGated)",
			Run:        c.runSelfhosting,
			WorkflowIf: c.workflowIfFeatureGateFor(features.SelfHosting), // within the init workflow, self hosting is executed only if the corresponding featureGates is true (direct execution is always allowed)
		},
		{ // prints init workflow complete message
			Use:    "init-completed",
			Hidden: true, // this phase can't be invoked directly; it is executed only as part of the full init workflow
			Run:    c.runInitComplete,
		},
	}
}

// initWorkflow defines the certs nested workflow
func (c *initContext) certsWorkflow() phases.PhaseWorkflow {
	return phases.PhaseWorkflow{
		{
			Use:   "ca",
			Short: "Generates a self-signed kubernetes CA to provision identities for components of the cluster",
			Run:   c.buildRunCertsFor(certs.CreateCACertAndKeyFiles),
		},
		{
			Use:   "apiserver",
			Short: "Generates an API server serving certificate and key",
			Run:   c.buildRunCertsFor(certs.CreateAPIServerCertAndKeyFiles),
		},
		{
			Use:   "apiserver-kubelet-client",
			Short: "Generates a client certificate for the API server to connect to the kubelets securely",
			Run:   c.buildRunCertsFor(certs.CreateAPIServerKubeletClientCertAndKeyFiles),
		},
		{
			Use:   "etcd-ca",
			Short: "Generates a self-signed CA to provision identities for etcd",
			Run:   c.buildRunCertsFor(certs.CreateEtcdCACertAndKeyFiles),
		},
		{
			Use:   "etcd-server",
			Short: "Generates an etcd serving certificate and key",
			Run:   c.buildRunCertsFor(certs.CreateEtcdServerCertAndKeyFiles),
		},
		{
			Use:   "etcd-peer",
			Short: "Generates an etcd peer certificate and key",
			Run:   c.buildRunCertsFor(certs.CreateEtcdPeerCertAndKeyFiles),
		},
		{
			Use:   "etcd-healthcheck-client",
			Short: "Generates a client certificate for liveness probes to healthcheck etcd",
			Run:   c.buildRunCertsFor(certs.CreateEtcdHealthcheckClientCertAndKeyFiles),
		},
		{
			Use:   "apiserver-etcd-client",
			Short: "Generates a client certificate for the API server to connect to etcd securely",
			Run:   c.buildRunCertsFor(certs.CreateAPIServerEtcdClientCertAndKeyFiles),
		},
		{
			Use:   "sa",
			Short: "Generates a private key for signing service account tokens along with its public key",
			Run:   c.buildRunCertsFor(certs.CreateServiceAccountKeyAndPublicKeyFiles),
		},
		{
			Use:   "front-proxy-ca",
			Short: "Generates a front proxy CA certificate and key for a Kubernetes cluster",
			Run:   c.buildRunCertsFor(certs.CreateFrontProxyCACertAndKeyFiles),
		},
		{
			Use:   "front-proxy-client",
			Short: "Generates a front proxy CA client certificate and key for a Kubernetes cluster",
			Run:   c.buildRunCertsFor(certs.CreateFrontProxyClientCertAndKeyFiles),
		},
	}
}

// kubeconfigWorkflow defines the kubeconfig nested workflow
func (c *initContext) kubeconfigWorkflow() phases.PhaseWorkflow {
	return phases.PhaseWorkflow{
		{
			Use:   "admin",
			Short: "Generates a kubeconfig file for the admin to use and for kubeadm itself",
			Run:   c.buildRunKubeconfigFor(kubeconfig.CreateAdminKubeConfigFile),
		},
		{
			Use:   "kubelet",
			Short: "Generates a kubeconfig file for the kubelet to use. Please note that this should be used *only* for bootstrapping purposes.",
			Run:   c.buildRunKubeconfigFor(kubeconfig.CreateKubeletKubeConfigFile),
		},
		{
			Use:   "controller-manager",
			Short: "Generates a kubeconfig file for the controller manager to use",
			Run:   c.buildRunKubeconfigFor(kubeconfig.CreateControllerManagerKubeConfigFile),
		},
		{
			Use:   "scheduler",
			Short: "Generates a kubeconfig file for the scheduler to use",
			Run:   c.buildRunKubeconfigFor(kubeconfig.CreateSchedulerKubeConfigFile),
		},
	}
}

// initWorkflow defines the control plane workflow
func (c *initContext) controlplaneWorkflow() phases.PhaseWorkflow {
	return phases.PhaseWorkflow{
		{
			Use:   "apiserver",
			Short: "Generates the API server static Pod manifest.",
			Run:   c.buildRunControlPlaneFor(controlplane.CreateAPIServerStaticPodManifestFile),
		},
		{
			Use:   "controller-manager",
			Short: "Generates the controller-manager static Pod manifest.",
			Run:   c.buildRunControlPlaneFor(controlplane.CreateControllerManagerStaticPodManifestFile),
		},
		{
			Use:   "scheduler",
			Short: "Generates the scheduler static Pod manifest.",
			Run:   c.buildRunControlPlaneFor(controlplane.CreateSchedulerStaticPodManifestFile),
		},
	}
}

// initWorkflow defines the bootstrap token workflow
func (c *initContext) bootstraptokenWorkflow() phases.PhaseWorkflow {
	return phases.PhaseWorkflow{
		{
			Use:   "token",
			Short: "Creates an initial bootstrap token to be used for node joining",
			Run:   c.runToken,
		},
		{
			Use:     "cluster-info",
			Short:   "Uploads the cluster-info ConfigMap from the given kubeconfig file",
			Aliases: []string{"clusterinfo"},
			Run:     c.runClusterInfo,
		},
		{
			Use:   "allow-post-csr",
			Short: "Configures RBAC to allow node bootstrap tokens to post CSR in order for nodes to get long term certificate credentials",
			Run:   c.runAllowPostCSR,
		},
		{
			Use:   "allow-auto-approve",
			Short: "Configures RBAC rules to allow the CSR approver controller automatically approve CSR from a node bootstrap token",
			Run:   c.runAllowAutoApproveCSR,
		},
	}
}

// initWorkflow defines the addon workflow
func (c *initContext) addonWorkflow() phases.PhaseWorkflow {
	return phases.PhaseWorkflow{
		{
			Use:   "dns",
			Short: "Installs the dns addon to a Kubernetes cluster",
			Run:   c.buildRunAddonFor(dns.EnsureDNSAddon),
		},
		{
			Use:   "proxy",
			Short: "Installs the proxy addon to a Kubernetes cluster",
			Run:   c.buildRunAddonFor(proxy.EnsureProxyAddon),
		},
	}
}

// workflowIfNotUsingExternalCA checks there isn't an external CA.
// usingExternalCAEvaluated and usingExternalCA flags are used in order to avoid to execute the check/to give a warning many times
func (c *initContext) workflowIfNotUsingExternalCA(cmd *cobra.Command, args []string) (bool, error) {
	if !c.usingExternalCAEvaluated {
		c.usingExternalCAEvaluated = true
		c.usingExternalCA, _ = certs.UsingExternalCA(c.MasterConfiguration())
		if c.usingExternalCA {
			fmt.Println("[externalca] The file 'ca.key' was not found, yet all other certificates are present. Using external CA mode - certificates or kubeconfig will not be generated.")
		}
	}
	return !c.usingExternalCA, nil
}

// workflowIfFeatureGateFor returns true if the given feature-gate is enabled
func (c *initContext) workflowIfFeatureGateFor(featureName string) func(cmd *cobra.Command, args []string) (bool, error) {
	return func(cmd *cobra.Command, args []string) (bool, error) {
		return features.Enabled(c.MasterConfiguration().FeatureGates, featureName), nil
	}
}
