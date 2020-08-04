package main

import (
	// ensure all the ginkgo tests are loaded
	_ "github.com/openshift/origin/test/extended"
)

var (
	testMaps = map[string][]string{
		// tests that require a local host
		"[Local]": {
			// Doesn't work on scaled up clusters
			`\[Feature:ImagePrune\]`,
		},
		// alpha features that are not gated
		"[Disabled:Alpha]": {
			// ALPHA features in 1.19, disabled by default.
			// !!! Review their status as part of the 1.20 rebase.
			`\[Feature:CSIStorageCapacity\]`,
			`\[Feature:IPv6DualStack.*\]`,
			`\[Feature:ServiceAccountIssuerDiscovery\]`,
			`\[Feature:SetHostnameAsFQDN\]`,
			`\[Feature:TTLAfterFinished\]`,

			// BETA features in 1.19, enabled by default
			// Their enablement is tracked via bz's targeted at 4.6.
			`\[Feature:ExpandCSIVolumes\]`, // https://bugzilla.redhat.com/show_bug.cgi?id=1861218
			`\[Feature:SCTPConnectivity\]`, // https://bugzilla.redhat.com/show_bug.cgi?id=1861606
		},
		// tests for features that are not implemented in openshift
		"[Disabled:Unimplemented]": {
			`\[Feature:Networking-IPv6\]`, // openshift-sdn doesn't support yet
			`Monitoring`,                  // Not installed, should be
			`Cluster level logging`,       // Not installed yet
			`Kibana`,                      // Not installed
			`Ubernetes`,                   // Can't set zone labels today
			`kube-ui`,                     // Not installed by default
			`Kubernetes Dashboard`,        // Not installed by default (also probably slow image pull)

			`NetworkPolicy.*egress`,     // not supported
			`NetworkPolicy.*named port`, // not yet implemented
			`enforce egress policy`,     // not support
			`should proxy to cadvisor`,  // we don't expose cAdvisor port directly for security reasons

			`NetworkPolicy.*IPBlock`,          // not supported
			`NetworkPolicy.*Egress`,           // not supported
			`NetworkPolicy.*default-deny-all`, // not supported
		},
		// tests that rely on special configuration that we do not yet support
		"[Disabled:SpecialConfig]": {
			// GPU node needs to be available
			`\[Feature:GPUDevicePlugin\]`,
			`\[sig-scheduling\] GPUDevicePluginAcrossRecreate \[Feature:Recreate\]`,

			`\[Feature:ImageQuota\]`,                    // Quota isn't turned on by default, we should do that and then reenable these tests
			`\[Feature:Audit\]`,                         // Needs special configuration
			`\[Feature:LocalStorageCapacityIsolation\]`, // relies on a separate daemonset?
			`\[sig-cloud-provider-gcp\]`,                // these test require a different configuration - note that GCE tests from the sig-cluster-lifecycle were moved to the sig-cloud-provider-gcpcluster lifecycle see https://github.com/kubernetes/kubernetes/commit/0b3d50b6dccdc4bbd0b3e411c648b092477d79ac#diff-3b1910d08fb8fd8b32956b5e264f87cb

			`kube-dns-autoscaler`, // Don't run kube-dns
			`should check if Kubernetes master services is included in cluster-info`, // Don't run kube-dns
			`DNS configMap`, // this tests dns federation configuration via configmap, which we don't support yet

			`NodeProblemDetector`,                   // requires a non-master node to run on
			`Advanced Audit should audit API calls`, // expects to be able to call /logs

			`Firewall rule should have correct firewall rules for e2e cluster`, // Upstream-install specific
		},
		// tests that are known broken and need to be fixed upstream or in openshift
		// always add an issue here
		"[Disabled:Broken]": {
			`mount an API token into pods`,                                // We add 6 secrets, not 1
			`ServiceAccounts should ensure a single API token exists`,     // We create lots of secrets
			`unchanging, static URL paths for kubernetes api services`,    // the test needs to exclude URLs that are not part of conformance (/logs)
			`Services should be able to up and down services`,             // we don't have wget installed on nodes
			`Network should set TCP CLOSE_WAIT timeout`,                   // possibly some difference between ubuntu and fedora
			`\[NodeFeature:Sysctls\]`,                                     // needs SCC support
			`should check kube-proxy urls`,                                // previously this test was skipped b/c we reported -1 as the number of nodes, now we report proper number and test fails
			`SSH`,                                                         // TRIAGE
			`should implement service.kubernetes.io/service-proxy-name`,   // this is an optional test that requires SSH. sig-network
			`should idle the service and DeploymentConfig properly`,       // idling with a single service and DeploymentConfig
			`should answer endpoint and wildcard queries for the cluster`, // currently not supported by dns operator https://github.com/openshift/cluster-dns-operator/issues/43
			`should allow ingress access on one named port`,               // https://bugzilla.redhat.com/show_bug.cgi?id=1711602
			`recreate nodes and ensure they function upon restart`,        // https://bugzilla.redhat.com/show_bug.cgi?id=1756428
			`\[Driver: iscsi\]`,                                           // https://bugzilla.redhat.com/show_bug.cgi?id=1711627

			"RuntimeClass should reject",

			`Services should implement service.kubernetes.io/headless`,       // requires SSH access to function, needs to be refactored
			`ClusterDns \[Feature:Example\] should create pod that uses dns`, // doesn't use bindata, not part of kube test binary
			`Simple pod should handle in-cluster config`,                     // kubectl cp doesn't work or is not preserving executable bit, we have this test already

			// TODO(node): configure the cri handler for the runtime class to make this work
			"should run a Pod requesting a RuntimeClass with a configured handler",
			"should reject a Pod requesting a RuntimeClass with conflicting node selector",
			"should run a Pod requesting a RuntimeClass with scheduling",

			// A fix is in progress: https://github.com/openshift/origin/pull/24709
			`Multi-AZ Clusters should spread the pods of a replication controller across zones`,

			`Cluster should restore itself after quorum loss`, // https://bugzilla.redhat.com/show_bug.cgi?id=1837157

			// Fix is in progress upstream and tracked via https://bugzilla.redhat.com/show_bug.cgi?id=1861215
			`Secret should create a pod that reads a secret`,

			// disabled until oc and origin are on k8s 1.19 - workloads team
			`should return command exit codes`,

			// Disabled as per networking team. Follow-up tracked via https://bugzilla.redhat.com/show_bug.cgi?id=1861214
			`EndpointSliceMirroring should mirror a custom Endpoints resource through create update and delete`,

			// Test passes but container it uses exits with non-zero.
			// https://bugzilla.redhat.com/show_bug.cgi?id=1861526
			`ServiceAccounts should set ownership and permission when RunAsUser or FsGroup is present`,

			// Perma-fail
			// https://bugzilla.redhat.com/show_bug.cgi?id=1862322
			`an end user can use OLM can subscribe to the operator`,
		},
		// tests that may work, but we don't support them
		"[Disabled:Unsupported]": {
			`\[Driver: rbd\]`,  // OpenShift 4.x does not support Ceph RBD (use CSI instead)
			`\[Driver: ceph\]`, // OpenShift 4.x does not support CephFS (use CSI instead)
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
			`openshift mongodb replication creating from a template`,                                     // flaking on deployment

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

			`Multi-AZ Clusters should spread the pods of a service across zones`, // spreading is a priority, not a predicate, and if the node is temporarily full the priority will be ignored

			`Should be able to support the 1\.7 Sample API Server using the current Aggregator`, // down apiservices break other clients today https://bugzilla.redhat.com/show_bug.cgi?id=1623195

			`\[Feature:HPA\] Horizontal pod autoscaling \(scale resource: CPU\) \[sig-autoscaling\] ReplicationController light Should scale from 1 pod to 2 pods`,

			`should prevent Ingress creation if more than 1 IngressClass marked as default`, // https://bugzilla.redhat.com/show_bug.cgi?id=1822286

			`\[sig-network\] IngressClass \[Feature:Ingress\] should set default value on new IngressClass`, //https://bugzilla.redhat.com/show_bug.cgi?id=1833583
		},
		"[Skipped:azure]": {
			"Networking should provide Internet connection for containers", // Azure does not allow ICMP traffic to internet.

			// openshift-tests cannot access Azure API to create in-line or pre-provisioned volumes, https://bugzilla.redhat.com/show_bug.cgi?id=1723603
			`\[sig-storage\] In-tree Volumes \[Driver: azure\] \[Testpattern: Inline-volume`,
			`\[sig-storage\] In-tree Volumes \[Driver: azure\] \[Testpattern: Pre-provisioned PV`,
		},
		"[Skipped:ovirt]": {
			// https://bugzilla.redhat.com/show_bug.cgi?id=1838751
			`\[sig-network\] Networking Granular Checks: Services should function for endpoint-Service`,
			`\[sig-network\] Networking Granular Checks: Services should function for node-Service`,
			`\[sig-network\] Networking Granular Checks: Services should function for pod-Service`,
		},
		"[Skipped:gce]": {
			// Requires creation of a different compute instance in a different zone and is not compatible with volumeBindingMode of WaitForFirstConsumer which we use in 4.x
			`\[sig-scheduling\] Multi-AZ Cluster Volumes \[sig-storage\] should only be allowed to provision PDs in zones where nodes exist`,

			// The following tests try to ssh directly to a node. None of our nodes have external IPs
			`\[k8s.io\] \[sig-node\] crictl should be able to run crictl on the node`,
			`\[sig-storage\] Flexvolumes should be mountable`,
			`\[sig-storage\] Detaching volumes should not work when mount is in progress`,

			// We are using openshift-sdn to conceal metadata
			`\[sig-auth\] Metadata Concealment should run a check-metadata-concealment job to completion`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1740959
			`\[sig-api-machinery\] AdmissionWebhook should be able to deny pod and configmap creation`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1745720
			`\[sig-storage\] CSI Volumes \[Driver: pd.csi.storage.gke.io\]\[Serial\]`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1749882
			`\[sig-storage\] CSI Volumes CSI Topology test using GCE PD driver \[Serial\]`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1751367
			`gce-localssd-scsi-fs`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1750851
			// should be serial if/when it's re-enabled
			`\[HPA\] Horizontal pod autoscaling \(scale resource: Custom Metrics from Stackdriver\)`,
		},
		// tests that don't pass under openshift-sdn but that are expected to pass
		// with other network plugins (particularly ovn-kubernetes)
		"[Skipped:Network/OpenShiftSDN]": {
			`NetworkPolicy between server and client should allow egress access on one named port`, // not yet implemented
		},
		// tests that don't pass under openshift-sdn multitenant mode
		"[Skipped:Network/OpenShiftSDN/Multitenant]": {
			`\[Feature:NetworkPolicy\]`, // not compatible with multitenant mode
			`\[sig-network\] Services should preserve source pod IP for traffic thru service cluster IP`, // known bug, not planned to be fixed
		},
		// tests that don't pass under OVN Kubernetes
		"[Skipped:Network/OVNKubernetes]": {
			// https://jira.coreos.com/browse/SDN-510: OVN-K doesn't support session affinity
			`\[sig-network\] Networking Granular Checks: Services should function for client IP based session affinity: http`,
			`\[sig-network\] Networking Granular Checks: Services should function for client IP based session affinity: udp`,
			`\[sig-network\] Services should be able to switch session affinity for NodePort service`,
			`\[sig-network\] Services should be able to switch session affinity for service with type clusterIP`,
			`\[sig-network\] Services should have session affinity work for NodePort service`,
			`\[sig-network\] Services should have session affinity work for service with type clusterIP`,
			`\[sig-network\] Services should have session affinity timeout work for NodePort service`,
			`\[sig-network\] Services should have session affinity timeout work for service with type clusterIP`,
			// https://github.com/kubernetes/kubernetes/pull/93597: upstream is flaky
			`\[sig-network\] Conntrack should be able to preserve UDP traffic when server pod cycles for a NodePort service`,
		},
		"[Skipped:ibmcloud]": {
			// skip Gluster tests (not supported on ROKS worker nodes)
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825009 - e2e: skip Glusterfs-related tests upstream for rhel7 worker nodes
			`\[Driver: gluster\]`,
			`GlusterFS`,
			`GlusterDynamicProvisioner`,

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

			// The cluster-network-operator creates the kube-proxy daemonset pods without mem/cpu requests,
			// resulting in a qosClass of BestEffort
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825019 - kube-proxy deployment does not include any memory/cpu requests
			`\[Feature:Platform\] Managed cluster should ensure control plane pods do not run in best-effort QoS`,

			// Calico is allowing the request to timeout instead of returning 'REFUSED'
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825021 - ROKS: calico SDN results in a request timeout when accessing services with no endpoints
			`\[sig-network\] Services should be rejected when no endpoints exist`,

			// Mode returned by RHEL7 worker contains an extra character not expected by the test: dgtrwx vs dtrwx
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825024 - e2e: Failing test - HostPath should give a volume the correct mode
			`\[sig-storage\] HostPath should give a volume the correct mode`,

			// Currently ibm-master-proxy-static and imbcloud-block-storage-plugin tolerate all taints
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825027
			`\[Feature:Platform\] Managed cluster should ensure control plane operators do not make themselves unevictable`,

			// Prometheus is not reporting router metrics
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825029
			`\[Feature:Prometheus\]\[Conformance\] Prometheus when installed on the cluster should provide ingress metrics`,
			`\[Conformance\]\[Area:Networking\]\[Feature:Router\] The HAProxy router should enable openshift-monitoring to pull metrics`,

			// Test does not allow enough time for the pods to be created before deleting them
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825372
			`Pod Container Status should never report success for a pending container`,
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
	}

	// labelExcludes temporarily block tests out of a specific suite
	labelExcludes = map[string][]string{}

	excludedTests = []string{
		`\[Disabled:`,
		`\[Disruptive\]`,
		`\[Skipped\]`,
		`\[Slow\]`,
		`\[Flaky\]`,
		`\[Local\]`,
	}
)
