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
			`\[Feature:StatefulSetMinReadySeconds\]`,
			`\[Feature:PodSecurityPolicy\]`,
			`\[Feature:StatefulSetAutoDeletePVC\]`,
			`\[Feature:CustomResourceValidationExpressions\]`,
			`\[Feature:GRPCContainerProbe\]`,
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

			`Services should implement service.kubernetes.io/headless`,       // requires SSH access to function, needs to be refactored
			`ClusterDns \[Feature:Example\] should create pod that uses dns`, // doesn't use bindata, not part of kube test binary
			`Simple pod should handle in-cluster config`,                     // kubectl cp doesn't work or is not preserving executable bit, we have this test already

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

			// https://bugzilla.redhat.com/show_bug.cgi?id=1945329
			`should drop INVALID conntrack entries`,

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
		},
		// tests that may work, but we don't support them
		"[Disabled:Unsupported]": {
			`\[Driver: rbd\]`,               // OpenShift 4.x does not support Ceph RBD (use CSI instead)
			`\[Driver: ceph\]`,              // OpenShift 4.x does not support CephFS (use CSI instead)
			`\[Driver: gluster\]`,           // OpenShift 4.x does not support Gluster
			`Volumes GlusterFS`,             // OpenShift 4.x does not support Gluster
			`GlusterDynamicProvisioner`,     // OpenShift 4.x does not support Gluster
			`\[Feature:PodSecurityPolicy\]`, // OpenShift 4.x does not enable PSP by default
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
		"[Skipped:azure]": {
			"Networking should provide Internet connection for containers", // Azure does not allow ICMP traffic to internet.
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
			`\[sig-storage\] CSI Volumes \[Driver: pd.csi.storage.gke.io\]\[Serial\]`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1749882
			`\[sig-storage\] CSI Volumes CSI Topology test using GCE PD driver \[Serial\]`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1751367
			`gce-localssd-scsi-fs`,

			// https://bugzilla.redhat.com/show_bug.cgi?id=1750851
			// should be serial if/when it's re-enabled
			`\[HPA\] Horizontal pod autoscaling \(scale resource: Custom Metrics from Stackdriver\)`,
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
	}

	// labelExcludes temporarily block tests out of a specific suite
	LabelExcludes = map[string][]string{}

	ExcludedTests = []string{
		`\[Disabled:`,
		`\[Disruptive\]`,
		`\[Skipped\]`,
		`\[Slow\]`,
		`\[Flaky\]`,
		`\[Local\]`,
	}
)
