package annotate

import (
	// ensure all the ginkgo tests are loaded
	_ "k8s.io/kubernetes/openshift-hack/e2e"
)

var (
	TestMaps = map[string][]string{
		// alpha features that are not gated
		"[Disabled:Alpha]": {
			`\[Feature:StorageVersionAPI\]`,
			`\[Feature:InPlacePodVerticalScaling\]`,
			`\[Feature:RecoverVolumeExpansionFailure\]`,
			`\[Feature:WatchList\]`,
			`\[Feature:ServiceCIDRs\]`,
			`\[Feature:ClusterTrustBundle\]`,
			`\[Feature:SELinuxMount\]`,
			`\[FeatureGate:SELinuxMount\]`,
			`\[Feature:RelaxedEnvironmentVariableValidation\]`,
			`\[Feature:UserNamespacesPodSecurityStandards\]`,
			`\[Feature:Traffic Distribution\]`,
			`\[Feature:UserNamespacesSupport\]`,
		},
		// tests for features that are not implemented in openshift
		"[Disabled:Unimplemented]": {
			`Monitoring`,               // Not installed, should be
			`Cluster level logging`,    // Not installed yet
			`Kibana`,                   // Not installed
			`Ubernetes`,                // Can't set zone labels today
			`kube-ui`,                  // Not installed by default
			`Kubernetes Dashboard`,     // Not installed by default (also probably slow image pull)
			`should proxy to cadvisor`, // we don't expose cAdvisor port directly for security reasons
		},
		// tests that rely on special configuration that we do not yet support
		"[Disabled:SpecialConfig]": {
			// GPU node needs to be available
			`\[Feature:GPUDevicePlugin\]`,
			`\[sig-scheduling\] GPUDevicePluginAcrossRecreate \[Feature:Recreate\]`,

			`\[Feature:LocalStorageCapacityIsolation\]`, // relies on a separate daemonset?
			`\[sig-cloud-provider-gcp\]`,                // these test require a different configuration - note that GCE tests from the sig-cluster-lifecycle were moved to the sig-cloud-provider-gcpcluster lifecycle see https://github.com/kubernetes/kubernetes/commit/0b3d50b6dccdc4bbd0b3e411c648b092477d79ac#diff-3b1910d08fb8fd8b32956b5e264f87cb

			`kube-dns-autoscaler`, // Don't run kube-dns
			`should check if Kubernetes master services is included in cluster-info`, // Don't run kube-dns
			`DNS configMap`, // this tests dns federation configuration via configmap, which we don't support yet

			`NodeProblemDetector`,                   // requires a non-master node to run on
			`Advanced Audit should audit API calls`, // expects to be able to call /logs

			`Firewall rule should have correct firewall rules for e2e cluster`, // Upstream-install specific

			// https://bugzilla.redhat.com/show_bug.cgi?id=2079958
			`\[sig-network\] \[Feature:Topology Hints\] should distribute endpoints evenly`,

			// Tests require SSH configuration and is part of the parallel suite, which does not create the bastion
			// host. Enabling the test would result in the  bastion being created for every parallel test execution.
			// Given that we have existing oc and WMCO tests that cover this functionality, we can safely disable it.
			`\[Feature:NodeLogQuery\]`,
		},
		// tests that are known broken and need to be fixed upstream or in openshift
		// always add an issue here
		"[Disabled:Broken]": {
			`mount an API token into pods`,                              // We add 6 secrets, not 1
			`ServiceAccounts should ensure a single API token exists`,   // We create lots of secrets
			`unchanging, static URL paths for kubernetes api services`,  // the test needs to exclude URLs that are not part of conformance (/logs)
			`Services should be able to up and down services`,           // we don't have wget installed on nodes
			`KubeProxy should set TCP CLOSE_WAIT timeout`,               // the test require communication to port 11302 in the cluster nodes
			`should check kube-proxy urls`,                              // previously this test was skipped b/c we reported -1 as the number of nodes, now we report proper number and test fails
			`SSH`,                                                       // TRIAGE
			`should implement service.kubernetes.io/service-proxy-name`, // this is an optional test that requires SSH. sig-network
			`recreate nodes and ensure they function upon restart`,      // https://bugzilla.redhat.com/show_bug.cgi?id=1756428
			`\[Driver: iscsi\]`,                                         // https://bugzilla.redhat.com/show_bug.cgi?id=1711627

			"RuntimeClass should reject",

			`Services should implement service.kubernetes.io/headless`,                    // requires SSH access to function, needs to be refactored
			`ClusterDns \[Feature:Example\] should create pod that uses dns`,              // doesn't use bindata, not part of kube test binary
			`Simple pod should return command exit codes should handle in-cluster config`, // kubectl cp doesn't work or is not preserving executable bit, we have this test already

			// TODO(node): configure the cri handler for the runtime class to make this work
			"should run a Pod requesting a RuntimeClass with a configured handler",
			"should reject a Pod requesting a RuntimeClass with conflicting node selector",
			"should run a Pod requesting a RuntimeClass with scheduling",

			// A fix is in progress: https://github.com/openshift/origin/pull/24709
			`Multi-AZ Clusters should spread the pods of a replication controller across zones`,

			// Upstream assumes all control plane pods are in kube-system namespace and we should revert the change
			// https://github.com/kubernetes/kubernetes/commit/176c8e219f4c7b4c15d34b92c50bfa5ba02b3aba#diff-28a3131f96324063dd53e17270d435a3b0b3bd8f806ee0e33295929570eab209R78
			"MetricsGrabber should grab all metrics from a Kubelet",
			"MetricsGrabber should grab all metrics from API server",
			"MetricsGrabber should grab all metrics from a ControllerManager",
			"MetricsGrabber should grab all metrics from a Scheduler",

			// https://bugzilla.redhat.com/show_bug.cgi?id=1906808
			`ServiceAccounts should support OIDC discovery of service account issuer`,

			// NFS umount is broken in kernels 5.7+
			// https://bugzilla.redhat.com/show_bug.cgi?id=1854379
			`\[sig-storage\].*\[Driver: nfs\] \[Testpattern: Dynamic PV \(default fs\)\].*subPath should be able to unmount after the subpath directory is deleted`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1986306
			`\[sig-cli\] Kubectl client kubectl wait should ignore not found error with --for=delete`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1980141
			`Netpol NetworkPolicy between server and client should enforce policy to allow traffic only from a pod in a different namespace based on PodSelector and NamespaceSelector`,
			`Netpol NetworkPolicy between server and client should enforce policy to allow traffic from pods within server namespace based on PodSelector`,
			`Netpol NetworkPolicy between server and client should enforce policy based on NamespaceSelector with MatchExpressions`,
			`Netpol NetworkPolicy between server and client should enforce policy based on PodSelector with MatchExpressions`,
			`Netpol NetworkPolicy between server and client should enforce policy based on PodSelector or NamespaceSelector`,
			`Netpol NetworkPolicy between server and client should deny ingress from pods on other namespaces`,
			`Netpol NetworkPolicy between server and client should enforce updated policy`,
			`Netpol NetworkPolicy between server and client should enforce multiple, stacked policies with overlapping podSelectors`,
			`Netpol NetworkPolicy between server and client should enforce policy based on any PodSelectors`,
			`Netpol NetworkPolicy between server and client should enforce policy to allow traffic only from a different namespace, based on NamespaceSelector`,
			`Netpol \[LinuxOnly\] NetworkPolicy between server and client using UDP should support a 'default-deny-ingress' policy`,
			`Netpol \[LinuxOnly\] NetworkPolicy between server and client using UDP should enforce policy based on Ports`,
			`Netpol \[LinuxOnly\] NetworkPolicy between server and client using UDP should enforce policy to allow traffic only from a pod in a different namespace based on PodSelector and NamespaceSelector`,

			`Topology Hints should distribute endpoints evenly`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1908645
			`\[sig-network\] Networking Granular Checks: Services should function for service endpoints using hostNetwork`,
			`\[sig-network\] Networking Granular Checks: Services should function for pod-Service\(hostNetwork\)`,

			// https://issues.redhat.com/browse/OCPBUGS-7125
			`\[sig-network\] LoadBalancers should be able to preserve UDP traffic when server pod cycles for a LoadBalancer service on different nodes`,
			`\[sig-network\] LoadBalancers should be able to preserve UDP traffic when server pod cycles for a LoadBalancer service on the same nodes`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1952460
			`\[sig-network\] Firewall rule control plane should not expose well-known ports`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1988272
			`\[sig-network\] Networking should provide Internet connection for containers \[Feature:Networking-IPv6\]`,
			`\[sig-network\] Networking should provider Internet connection for containers using DNS`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1957894
			`\[sig-node\] Container Runtime blackbox test when running a container with a new image should be able to pull from private registry with secret`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1952457
			`\[sig-node\] crictl should be able to run crictl on the node`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1953478
			`\[sig-storage\] Dynamic Provisioning Invalid AWS KMS key should report an error and create no PV`,
		},
		// tests that need to be temporarily disabled while the rebase is in progress.
		"[Disabled:RebaseInProgress]": {
			// https://issues.redhat.com/browse/OCPBUGS-7297
			`DNS HostNetwork should resolve DNS of partial qualified names for services on hostNetwork pods with dnsPolicy`,
			`\[sig-network\] Connectivity Pod Lifecycle should be able to connect to other Pod from a terminating Pod`, // TODO(network): simple test in k8s 1.27, needs investigation
			`\[sig-cli\] Kubectl client Kubectl prune with applyset should apply and prune objects`,                    // TODO(workloads): alpha feature in k8s 1.27. It's failing with `error: unknown flag: --applyset`. Needs investigation

			// https://issues.redhat.com/browse/OCPBUGS-17194
			`\[sig-node\] ImageCredentialProvider \[Feature:KubeletCredentialProviders\] should be able to create pod with image credentials fetched from external credential provider`,
		},
		// tests that may work, but we don't support them
		"[Disabled:Unsupported]": {
			`\[Driver: rbd\]`,           // OpenShift 4.x does not support Ceph RBD (use CSI instead)
			`\[Driver: ceph\]`,          // OpenShift 4.x does not support CephFS (use CSI instead)
			`\[Driver: gluster\]`,       // OpenShift 4.x does not support Gluster
			`Volumes GlusterFS`,         // OpenShift 4.x does not support Gluster
			`GlusterDynamicProvisioner`, // OpenShift 4.x does not support Gluster

			// Skip vSphere-specific storage tests. The standard in-tree storage tests for vSphere
			// (prefixed with `In-tree Volumes [Driver: vsphere]`) are enough for testing this plugin.
			// https://bugzilla.redhat.com/show_bug.cgi?id=2019115
			`\[sig-storage\].*\[Feature:vsphere\]`,
			// Also, our CI doesn't support topology, so disable those tests
			`\[sig-storage\] In-tree Volumes \[Driver: vsphere\] \[Testpattern: Dynamic PV \(delayed binding\)\] topology should fail to schedule a pod which has topologies that conflict with AllowedTopologies`,
			`\[sig-storage\] In-tree Volumes \[Driver: vsphere\] \[Testpattern: Dynamic PV \(delayed binding\)\] topology should provision a volume and schedule a pod with AllowedTopologies`,
			`\[sig-storage\] In-tree Volumes \[Driver: vsphere\] \[Testpattern: Dynamic PV \(immediate binding\)\] topology should fail to schedule a pod which has topologies that conflict with AllowedTopologies`,
			`\[sig-storage\] In-tree Volumes \[Driver: vsphere\] \[Testpattern: Dynamic PV \(immediate binding\)\] topology should provision a volume and schedule a pod with AllowedTopologies`,
		},
		// tests too slow to be part of conformance
		"[Slow]": {
			`\[sig-scalability\]`,                          // disable from the default set for now
			`should create and stop a working application`, // Inordinately slow tests

			`\[Feature:PerformanceDNS\]`, // very slow

			`validates that there exists conflict between pods with same hostPort and protocol but one using 0\.0\.0\.0 hostIP`, // 5m, really?
		},
		// tests that are known flaky
		"[Flaky]": {
			`Job should run a job to completion when tasks sometimes fail and are not locally restarted`, // seems flaky, also may require too many resources
			// TODO(node): test works when run alone, but not in the suite in CI
			`\[Feature:HPA\] Horizontal pod autoscaling \(scale resource: CPU\) \[sig-autoscaling\] ReplicationController light Should scale from 1 pod to 2 pods`,
		},
		// tests that must be run without competition
		"[Serial]": {
			`\[Disruptive\]`,
			`\[Feature:Performance\]`, // requires isolation

			`Service endpoints latency`, // requires low latency
			`Clean up pods on node`,     // schedules up to max pods per node
			`DynamicProvisioner should test that deleting a claim before the volume is provisioned deletes the volume`, // test is very disruptive to other tests

			`Should be able to support the 1\.7 Sample API Server using the current Aggregator`, // down apiservices break other clients today https://bugzilla.redhat.com/show_bug.cgi?id=1623195

			`\[Feature:HPA\] Horizontal pod autoscaling \(scale resource: CPU\) \[sig-autoscaling\] ReplicationController light Should scale from 1 pod to 2 pods`,

			`should prevent Ingress creation if more than 1 IngressClass marked as default`, // https://bugzilla.redhat.com/show_bug.cgi?id=1822286

			`\[sig-network\] IngressClass \[Feature:Ingress\] should set default value on new IngressClass`, //https://bugzilla.redhat.com/show_bug.cgi?id=1833583
		},
		// Tests that don't pass on disconnected, either due to requiring
		// internet access for GitHub (e.g. many of the s2i builds), or
		// because of pullthrough not supporting ICSP (https://bugzilla.redhat.com/show_bug.cgi?id=1918376)
		"[Skipped:Disconnected]": {
			// Internet access required
			`\[sig-network\] Networking should provide Internet connection for containers`,
		},
		"[Skipped:azure]": {
			"Networking should provide Internet connection for containers", // Azure does not allow ICMP traffic to internet.
			// Azure CSI migration changed how we treat regions without zones.
			// See https://bugzilla.redhat.com/bugzilla/show_bug.cgi?id=2066865
			`\[sig-storage\] In-tree Volumes \[Driver: azure-disk\] \[Testpattern: Dynamic PV \(immediate binding\)\] topology should provision a volume and schedule a pod with AllowedTopologies`,
			`\[sig-storage\] In-tree Volumes \[Driver: azure-disk\] \[Testpattern: Dynamic PV \(delayed binding\)\] topology should provision a volume and schedule a pod with AllowedTopologies`,
		},
		"[Skipped:gce]": {
			// Requires creation of a different compute instance in a different zone and is not compatible with volumeBindingMode of WaitForFirstConsumer which we use in 4.x
			`\[sig-storage\] Multi-AZ Cluster Volumes should only be allowed to provision PDs in zones where nodes exist`,

			// The following tests try to ssh directly to a node. None of our nodes have external IPs
			`\[k8s.io\] \[sig-node\] crictl should be able to run crictl on the node`,
			`\[sig-storage\] Flexvolumes should be mountable`,
			`\[sig-storage\] Detaching volumes should not work when mount is in progress`,

			// We are using openshift-sdn to conceal metadata
			`\[sig-auth\] Metadata Concealment should run a check-metadata-concealment job to completion`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1740959
			`\[sig-api-machinery\] AdmissionWebhook should be able to deny pod and configmap creation`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1745720
			`\[sig-storage\] CSI Volumes \[Driver: pd.csi.storage.gke.io\]`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1749882
			`\[sig-storage\] CSI Volumes CSI Topology test using GCE PD driver \[Serial\]`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1751367
			`gce-localssd-scsi-fs`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1750851
			// should be serial if/when it's re-enabled
			`\[HPA\] Horizontal pod autoscaling \(scale resource: Custom Metrics from Stackdriver\)`,
			`\[Feature:CustomMetricsAutoscaling\]`,
		},
		"[sig-node]": {
			`\[NodeConformance\]`,
			`NodeLease`,
			`lease API`,
			`\[NodeFeature`,
			`\[NodeAlphaFeature`,
			`Probing container`,
			`Security Context When creating a`,
			`Downward API should create a pod that prints his name and namespace`,
			`Liveness liveness pods should be automatically restarted`,
			`Secret should create a pod that reads a secret`,
			`Pods should delete a collection of pods`,
			`Pods should run through the lifecycle of Pods and PodStatus`,
		},
		"[sig-cluster-lifecycle]": {
			`Feature:ClusterAutoscalerScalability`,
			`recreate nodes and ensure they function`,
		},
		"[sig-arch]": {
			// not run, assigned to arch as catch-all
			`\[Feature:GKELocalSSD\]`,
			`\[Feature:GKENodePool\]`,
		},
		// Tests that don't pass under openshift-sdn.
		// These are skipped explicitly by openshift-hack/test-kubernetes-e2e.sh,
		// but will also be skipped by openshift-tests in jobs that use openshift-sdn.
		"[Skipped:Network/OpenShiftSDN]": {
			`NetworkPolicy.*IPBlock`,    // feature is not supported by openshift-sdn
			`NetworkPolicy.*[Ee]gress`,  // feature is not supported by openshift-sdn
			`NetworkPolicy.*named port`, // feature is not supported by openshift-sdn

			`NetworkPolicy between server and client should support a 'default-deny-all' policy`,            // uses egress feature
			`NetworkPolicy between server and client should stop enforcing policies after they are deleted`, // uses egress feature
		},

		// These tests are skipped when openshift-tests needs to use a proxy to reach the
		// cluster -- either because the test won't work while proxied, or because the test
		// itself is testing a functionality using it's own proxy.
		"[Skipped:Proxy]": {
			// These tests setup their own proxy, which won't work when we need to access the
			// cluster through a proxy.
			`\[sig-cli\] Kubectl client Simple pod should support exec through an HTTP proxy`,
			`\[sig-cli\] Kubectl client Simple pod should support exec through kubectl proxy`,

			// Kube currently uses the x/net/websockets pkg, which doesn't work with proxies.
			// See: https://github.com/kubernetes/kubernetes/pull/103595
			`\[sig-node\] Pods should support retrieving logs from the container over websockets`,
			`\[sig-cli\] Kubectl Port forwarding With a server listening on localhost should support forwarding over websockets`,
			`\[sig-cli\] Kubectl Port forwarding With a server listening on 0.0.0.0 should support forwarding over websockets`,
			`\[sig-node\] Pods should support remote command execution over websockets`,

			// These tests are flacky and require internet access
			// See https://bugzilla.redhat.com/show_bug.cgi?id=2019375
			`\[sig-network\] DNS should resolve DNS of partial qualified names for services`,
			`\[sig-network\] DNS should provide DNS for the cluster`,
			// This test does not work when using in-proxy cluster, see https://bugzilla.redhat.com/show_bug.cgi?id=2084560
			`\[sig-network\] Networking should provide Internet connection for containers`,
		},

		"[Skipped:SingleReplicaTopology]": {
			`\[sig-apps\] Daemon set \[Serial\] should rollback without unnecessary restarts \[Conformance\]`,
			`\[sig-node\] NoExecuteTaintManager Single Pod \[Serial\] doesn't evict pod with tolerations from tainted nodes`,
			`\[sig-node\] NoExecuteTaintManager Single Pod \[Serial\] eventually evict pod with finite tolerations from tainted nodes`,
			`\[sig-node\] NoExecuteTaintManager Single Pod \[Serial\] evicts pods from tainted nodes`,
			`\[sig-node\] NoExecuteTaintManager Single Pod \[Serial\] removing taint cancels eviction \[Disruptive\] \[Conformance\]`,
			`\[sig-node\] NoExecuteTaintManager Single Pod \[Serial\] pods evicted from tainted nodes have pod disruption condition`,
			`\[sig-node\] NoExecuteTaintManager Multiple Pods \[Serial\] evicts pods with minTolerationSeconds \[Disruptive\] \[Conformance\]`,
			`\[sig-node\] NoExecuteTaintManager Multiple Pods \[Serial\] only evicts pods without tolerations from tainted nodes`,
			`\[sig-cli\] Kubectl client Kubectl taint \[Serial\] should remove all the taints with the same key off a node`,
			`\[sig-network\] LoadBalancers should be able to preserve UDP traffic when server pod cycles for a LoadBalancer service on different nodes`,
			`\[sig-network\] LoadBalancers should be able to preserve UDP traffic when server pod cycles for a LoadBalancer service on the same nodes`,
		},

		// Tests which can't be run/don't make sense to run against a cluster with all optional capabilities disabled
		"[Skipped:NoOptionalCapabilities]": {
			// Requires CSISnapshot capability
			`\[Feature:VolumeSnapshotDataSource\]`,
			// Requires Storage capability
			`\[Driver: aws\]`,
			`\[Feature:StorageProvider\]`,
		},

		// tests that don't pass under openshift-sdn multitenant mode
		"[Skipped:Network/OpenShiftSDN/Multitenant]": {
			`\[Feature:NetworkPolicy\]`, // not compatible with multitenant mode
		},
		// tests that don't pass under OVN Kubernetes
		"[Skipped:Network/OVNKubernetes]": {
			// ovn-kubernetes does not support named ports
			`NetworkPolicy.*named port`,
		},

		"[Skipped:ibmroks]": {
			// Calico is allowing the request to timeout instead of returning 'REFUSED'
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825021 - ROKS: calico SDN results in a request timeout when accessing services with no endpoints
			`\[sig-network\] Services should be rejected when no endpoints exist`,

			// Nodes in ROKS have access to secrets in the cluster to handle encryption
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825013 - ROKS: worker nodes have access to secrets in the cluster
			`\[sig-auth\] \[Feature:NodeAuthorizer\] Getting a non-existent configmap should exit with the Forbidden error, not a NotFound error`,
			`\[sig-auth\] \[Feature:NodeAuthorizer\] Getting a non-existent secret should exit with the Forbidden error, not a NotFound error`,
			`\[sig-auth\] \[Feature:NodeAuthorizer\] Getting a secret for a workload the node has access to should succeed`,
			`\[sig-auth\] \[Feature:NodeAuthorizer\] Getting an existing configmap should exit with the Forbidden error`,
			`\[sig-auth\] \[Feature:NodeAuthorizer\] Getting an existing secret should exit with the Forbidden error`,

			// Access to node external address is blocked from pods within a ROKS cluster by Calico
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825016 - e2e: NodeAuthenticator tests use both external and internal addresses for node
			`\[sig-auth\] \[Feature:NodeAuthenticator\] The kubelet's main port 10250 should reject requests with no credentials`,
			`\[sig-auth\] \[Feature:NodeAuthenticator\] The kubelet can delegate ServiceAccount tokens to the API server`,

			// Mode returned by RHEL7 worker contains an extra character not expected by the test: dgtrwx vs dtrwx
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825024 - e2e: Failing test - HostPath should give a volume the correct mode
			`\[sig-storage\] HostPath should give a volume the correct mode`,
		},
	}

	ExcludedTests = []string{
		`\[Disabled:`,
		`\[Disruptive\]`,
		`\[Skipped\]`,
		`\[Slow\]`,
		`\[Flaky\]`,
		`\[Local\]`,
	}
)
