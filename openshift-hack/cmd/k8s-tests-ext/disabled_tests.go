package main

import (
	et "github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"
	"k8s.io/apimachinery/pkg/util/sets"
)

// filterOutDisabledSpecs returns the specs with those that are disabled removed from the list
func filterOutDisabledSpecs(specs et.ExtensionTestSpecs) et.ExtensionTestSpecs {
	var disabledByReason = map[string][]string{
		"Alpha": { // alpha features that are not gated
			"[Feature:StorageVersionAPI]",
			"[Feature:InPlacePodVerticalScaling]",
			"[Feature:ServiceCIDRs]",
			"[Feature:ClusterTrustBundle]",
			"[Feature:SELinuxMount]",
			"[FeatureGate:SELinuxMount]",
			"[Feature:UserNamespacesPodSecurityStandards]",
			"[Feature:UserNamespacesSupport]", // disabled Beta
			"[Feature:DynamicResourceAllocation]",
			"[Feature:VolumeAttributesClass]", // disabled Beta
			"[sig-cli] Kubectl client Kubectl prune with applyset should apply and prune objects", // Alpha feature since k8s 1.27
			// 4.19
			"[Feature:PodLevelResources]",
			"[Feature:SchedulerAsyncPreemption]",
			"[Feature:RelaxedDNSSearchValidation]",
			"[Feature:PodLogsQuerySplitStreams]",
			"[Feature:PodLifecycleSleepActionAllowZero]",
			"[Feature:volumegroupsnapshot]",
			"[Feature:OrderedNamespaceDeletion]", // disabled Beta
		},
		// tests for features that are not implemented in openshift
		"Unimplemented": {
			"Monitoring",                               // Not installed, should be
			"Cluster level logging",                    // Not installed yet
			"Kibana",                                   // Not installed
			"Ubernetes",                                // Can't set zone labels today
			"kube-ui",                                  // Not installed by default
			"Kubernetes Dashboard",                     // Not installed by default (also probably slow image pull)
			"should proxy to cadvisor",                 // we don't expose cAdvisor port directly for security reasons
			"[Feature:BootstrapTokens]",                // we don't serve cluster-info configmap
			"[Feature:KubeProxyDaemonSetMigration]",    // upgrades are run separately
			"[Feature:BoundServiceAccountTokenVolume]", // upgrades are run separately
			"[Feature:StatefulUpgrade]",                // upgrades are run separately
		},
		// tests that rely on special configuration that we do not yet support
		"SpecialConfig": {
			// GPU node needs to be available
			"[Feature:GPUDevicePlugin]",
			"[sig-scheduling] GPUDevicePluginAcrossRecreate [Feature:Recreate]",

			"[Feature:LocalStorageCapacityIsolation]", // relies on a separate daemonset?
			"[sig-cloud-provider-gcp]",                // these test require a different configuration - note that GCE tests from the sig-cluster-lifecycle were moved to the sig-cloud-provider-gcpcluster lifecycle see https://github.com/kubernetes/kubernetes/commit/0b3d50b6dccdc4bbd0b3e411c648b092477d79ac#diff-3b1910d08fb8fd8b32956b5e264f87cb

			"kube-dns-autoscaler", // Don't run kube-dns
			"should check if Kubernetes master services is included in cluster-info", // Don't run kube-dns
			"DNS configMap", // this tests dns federation configuration via configmap, which we don't support yet

			"NodeProblemDetector",                   // requires a non-master node to run on
			"Advanced Audit should audit API calls", // expects to be able to call /logs

			"Firewall rule should have correct firewall rules for e2e cluster", // Upstream-install specific

			// https://bugzilla.redhat.com/show_bug.cgi?id=2079958
			"[sig-network] [Feature:Topology Hints] should distribute endpoints evenly",

			// Tests require SSH configuration and is part of the parallel suite, which does not create the bastion
			// host. Enabling the test would result in the  bastion being created for every parallel test execution.
			// Given that we have existing oc and WMCO tests that cover this functionality, we can safely disable it.
			"[Feature:NodeLogQuery]",
		},
		// tests that are known broken and need to be fixed upstream or in openshift
		// always add an issue here
		"Broken": {
			"mount an API token into pods",                              // We add 6 secrets, not 1
			"ServiceAccounts should ensure a single API token exists",   // We create lots of secrets
			"unchanging, static URL paths for kubernetes api services",  // the test needs to exclude URLs that are not part of conformance (/logs)
			"Services should be able to up and down services",           // we don't have wget installed on nodes
			"KubeProxy should set TCP CLOSE_WAIT timeout",               // the test require communication to port 11302 in the cluster nodes
			"should check kube-proxy urls",                              // previously this test was skipped b/c we reported -1 as the number of nodes, now we report proper number and test fails
			"SSH",                                                       // TRIAGE
			"should implement service.kubernetes.io/service-proxy-name", // this is an optional test that requires SSH. sig-network
			"recreate nodes and ensure they function upon restart",      // https://bugzilla.redhat.com/show_bug.cgi?id=1756428
			"[Driver: iscsi]",                                           // https://bugzilla.redhat.com/show_bug.cgi?id=1711627

			"RuntimeClass should reject",

			"Services should implement service.kubernetes.io/headless",                    // requires SSH access to function, needs to be refactored
			"ClusterDns [Feature:Example] should create pod that uses dns",                // doesn't use bindata, not part of kube test binary
			"Simple pod should return command exit codes should handle in-cluster config", // kubectl cp doesn't work or is not preserving executable bit, we have this test already

			// TODO(node): configure the cri handler for the runtime class to make this work
			"should run a Pod requesting a RuntimeClass with a configured handler",
			"should reject a Pod requesting a RuntimeClass with conflicting node selector",
			"should run a Pod requesting a RuntimeClass with scheduling",

			// A fix is in progress: https://github.com/openshift/origin/pull/24709
			"Multi-AZ Clusters should spread the pods of a replication controller across zones",

			// Upstream assumes all control plane pods are in kube-system namespace and we should revert the change
			// https://github.com/kubernetes/kubernetes/commit/176c8e219f4c7b4c15d34b92c50bfa5ba02b3aba#diff-28a3131f96324063dd53e17270d435a3b0b3bd8f806ee0e33295929570eab209R78
			"MetricsGrabber should grab all metrics from a Kubelet",
			"MetricsGrabber should grab all metrics from API server",
			"MetricsGrabber should grab all metrics from a ControllerManager",
			"MetricsGrabber should grab all metrics from a Scheduler",

			// https://bugzilla.redhat.com/show_bug.cgi?id=1906808
			"ServiceAccounts should support OIDC discovery of service account issuer",

			// NFS umount is broken in kernels 5.7+
			// https://bugzilla.redhat.com/show_bug.cgi?id=1854379
			"[sig-storage] In-tree Volumes [Driver: nfs] [Testpattern: Dynamic PV (default fs)] subPath should be able to unmount after the subpath directory is deleted",

			// https://bugzilla.redhat.com/show_bug.cgi?id=1986306
			"[sig-cli] Kubectl client kubectl wait should ignore not found error with --for=delete",

			// https://bugzilla.redhat.com/show_bug.cgi?id=1980141
			"Netpol NetworkPolicy between server and client should enforce policy to allow traffic only from a pod in a different namespace based on PodSelector and NamespaceSelector",
			"Netpol NetworkPolicy between server and client should enforce policy to allow traffic from pods within server namespace based on PodSelector",
			"Netpol NetworkPolicy between server and client should enforce policy based on NamespaceSelector with MatchExpressions",
			"Netpol NetworkPolicy between server and client should enforce policy based on PodSelector with MatchExpressions",
			"Netpol NetworkPolicy between server and client should enforce policy based on PodSelector or NamespaceSelector",
			"Netpol NetworkPolicy between server and client should deny ingress from pods on other namespaces",
			"Netpol NetworkPolicy between server and client should enforce updated policy",
			"Netpol NetworkPolicy between server and client should enforce multiple, stacked policies with overlapping podSelectors",
			"Netpol NetworkPolicy between server and client should enforce policy based on any PodSelectors",
			"Netpol NetworkPolicy between server and client should enforce policy to allow traffic only from a different namespace, based on NamespaceSelector",
			"Netpol [LinuxOnly] NetworkPolicy between server and client using UDP should support a 'default-deny-ingress' policy",
			"Netpol [LinuxOnly] NetworkPolicy between server and client using UDP should enforce policy based on Ports",
			"Netpol [LinuxOnly] NetworkPolicy between server and client using UDP should enforce policy to allow traffic only from a pod in a different namespace based on PodSelector and NamespaceSelector",

			"Topology Hints should distribute endpoints evenly",

			// https://bugzilla.redhat.com/show_bug.cgi?id=1908645
			"[sig-network] Networking Granular Checks: Services should function for service endpoints using hostNetwork",
			"[sig-network] Networking Granular Checks: Services should function for pod-Service(hostNetwork)",

			// https://bugzilla.redhat.com/show_bug.cgi?id=1952460
			"[sig-network] Firewall rule control plane should not expose well-known ports",

			// https://bugzilla.redhat.com/show_bug.cgi?id=1988272
			"[sig-network] Networking should provide Internet connection for containers [Feature:Networking-IPv6]",
			"[sig-network] Networking should provider Internet connection for containers using DNS",

			// https://bugzilla.redhat.com/show_bug.cgi?id=1957894
			"[sig-node] Container Runtime blackbox test when running a container with a new image should be able to pull from private registry with secret",

			// https://bugzilla.redhat.com/show_bug.cgi?id=1952457
			"[sig-node] crictl should be able to run crictl on the node",

			// https://bugzilla.redhat.com/show_bug.cgi?id=1953478
			"[sig-storage] Dynamic Provisioning Invalid AWS KMS key should report an error and create no PV",

			// https://issues.redhat.com/browse/OCPBUGS-34577
			"[sig-storage] Multi-AZ Cluster Volumes should schedule pods in the same zones as statically provisioned PVs",

			// https://issues.redhat.com/browse/OCPBUGS-34594
			"[sig-node] [Feature:PodLifecycleSleepAction] when create a pod with lifecycle hook using sleep action valid prestop hook using sleep action",

			// https://issues.redhat.com/browse/OCPBUGS-38839
			"[sig-network] [Feature:Traffic Distribution] when Service has trafficDistribution=PreferClose should route traffic to an endpoint that is close to the client",
		},
		// tests that need to be temporarily disabled while the rebase is in progress.
		"RebaseInProgress": {
			// https://issues.redhat.com/browse/OCPBUGS-7297
			"DNS HostNetwork should resolve DNS of partial qualified names for services on hostNetwork pods with dnsPolicy",

			// https://issues.redhat.com/browse/OCPBUGS-45275
			"[sig-network] Connectivity Pod Lifecycle should be able to connect to other Pod from a terminating Pod",

			// https://issues.redhat.com/browse/OCPBUGS-17194
			"[sig-node] ImageCredentialProvider [Feature:KubeletCredentialProviders] should be able to create pod with image credentials fetched from external credential provider",

			// https://issues.redhat.com/browse/OCPBUGS-45273
			"[sig-network] Services should implement NodePort and HealthCheckNodePort correctly when ExternalTrafficPolicy changes",
		},
		// tests that may work, but we don't support them
		"Unsupported": {
			"[Driver: rbd]",             // OpenShift 4.x does not support Ceph RBD (use CSI instead)
			"[Driver: ceph]",            // OpenShift 4.x does not support CephFS (use CSI instead)
			"[Driver: gluster]",         // OpenShift 4.x does not support Gluster
			"Volumes GlusterFS",         // OpenShift 4.x does not support Gluster
			"GlusterDynamicProvisioner", // OpenShift 4.x does not support Gluster

			// Also, our CI doesn't support topology, so disable those tests
			"[sig-storage] In-tree Volumes [Driver: vsphere] [Testpattern: Dynamic PV (delayed binding)] topology should fail to schedule a pod which has topologies that conflict with AllowedTopologies",
			"[sig-storage] In-tree Volumes [Driver: vsphere] [Testpattern: Dynamic PV (delayed binding)] topology should provision a volume and schedule a pod with AllowedTopologies",
			"[sig-storage] In-tree Volumes [Driver: vsphere] [Testpattern: Dynamic PV (immediate binding)] topology should fail to schedule a pod which has topologies that conflict with AllowedTopologies",
			"[sig-storage] In-tree Volumes [Driver: vsphere] [Testpattern: Dynamic PV (immediate binding)] topology should provision a volume and schedule a pod with AllowedTopologies",
		},
	}

	var disabledSpecs et.ExtensionTestSpecs
	for _, disabledList := range disabledByReason {
		var selectFunctions []et.SelectFunction
		for _, disabledName := range disabledList {
			selectFunctions = append(selectFunctions, et.NameContains(disabledName))
		}

		disabledSpecs = append(disabledSpecs, specs.SelectAny(selectFunctions)...)
	}

	disabledNames := sets.New[string]()
	for _, disabledSpec := range disabledSpecs {
		disabledNames.Insert(disabledSpec.Name)
	}

	enabledSpecs := specs[:0]
	for _, spec := range specs {
		if !disabledNames.Has(spec.Name) {
			enabledSpecs = append(enabledSpecs, spec)
		}
	}

	return enabledSpecs
}
