package main

import et "github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"

// addEnvironmentSelectors adds the environmentSelector field to appropriate specs to facilitate including or excluding
// them based on attributes of the cluster they are running on
func addEnvironmentSelectors(specs et.ExtensionTestSpecs) {
	filterByPlatform(specs)
	filterByExternalConnectivity(specs)
	filterByTopology(specs)
	filterByNoOptionalCapabilities(specs)
	filterByNetwork(specs)

	// LoadBalancer tests in 1.31 require explicit platform-specific skips
	// https://issues.redhat.com/browse/OCPBUGS-38840
	specs.SelectAny([]et.SelectFunction{ // Since these must use "NameContainsAll" they cannot be included in filterByPlatform
		et.NameContainsAll("[sig-network] LoadBalancers [Feature:LoadBalancer]", "UDP"),
		et.NameContainsAll("[sig-network] LoadBalancers [Feature:LoadBalancer]", "session affinity"),
	}).Exclude(et.PlatformEquals("aws"))

	specs.SelectAny([]et.SelectFunction{ // Since these must use "NameContainsAll" they cannot be included in filterByNetwork
		et.NameContainsAll("NetworkPolicy", "named port"),
	}).Exclude(et.NetworkEquals("OVNKubernetes"))
}

// filterByPlatform is a helper function to do, simple, "NameContains" filtering on tests by platform
func filterByPlatform(specs et.ExtensionTestSpecs) {
	var platformExclusions = map[string][]string{
		"alibabacloud": {
			// LoadBalancer tests in 1.31 require explicit platform-specific skips
			// https://issues.redhat.com/browse/OCPBUGS-38840
			"[Feature:LoadBalancer]",
		},
		"azure": {
			"Networking should provide Internet connection for containers", // Azure does not allow ICMP traffic to internet.
			// Azure CSI migration changed how we treat regions without zones.
			// See https://bugzilla.redhat.com/bugzilla/show_bug.cgi?id=2066865
			"[sig-storage] In-tree Volumes [Driver: azure-disk] [Testpattern: Dynamic PV (immediate binding)] topology should provision a volume and schedule a pod with AllowedTopologies",
			"[sig-storage] In-tree Volumes [Driver: azure-disk] [Testpattern: Dynamic PV (delayed binding)] topology should provision a volume and schedule a pod with AllowedTopologies",
		},
		"baremetal": {
			// LoadBalancer tests in 1.31 require explicit platform-specific skips
			// https://issues.redhat.com/browse/OCPBUGS-38840
			"[Feature:LoadBalancer]",
		},
		"gce": {
			// Requires creation of a different compute instance in a different zone and is not compatible with volumeBindingMode of WaitForFirstConsumer which we use in 4.x
			"[sig-storage] Multi-AZ Cluster Volumes should only be allowed to provision PDs in zones where nodes exist",
			// The following tests try to ssh directly to a node. None of our nodes have external IPs
			"[k8s.io] [sig-node] crictl should be able to run crictl on the node",
			"[sig-storage] Flexvolumes should be mountable",
			"[sig-storage] Detaching volumes should not work when mount is in progress",
			// We are using ovn-kubernetes to conceal metadata
			"[sig-auth] Metadata Concealment should run a check-metadata-concealment job to completion",
			// https://bugzilla.redhat.com/show_bug.cgi?id=1740959
			"[sig-api-machinery] AdmissionWebhook should be able to deny pod and configmap creation",
			// https://bugzilla.redhat.com/show_bug.cgi?id=1745720
			"[sig-storage] CSI Volumes [Driver: pd.csi.storage.gke.io]",
			// https://bugzilla.redhat.com/show_bug.cgi?id=1749882
			"[sig-storage] CSI Volumes CSI Topology test using GCE PD driver [Serial]",
			// https://bugzilla.redhat.com/show_bug.cgi?id=1751367
			"gce-localssd-scsi-fs",
			// https://bugzilla.redhat.com/show_bug.cgi?id=1750851
			// should be serial if/when it's re-enabled
			"[HPA] Horizontal pod autoscaling (scale resource: Custom Metrics from Stackdriver)",
			"[Feature:CustomMetricsAutoscaling]",
		},
		"ibmcloud": {
			// LoadBalancer tests in 1.31 require explicit platform-specific skips
			// https://issues.redhat.com/browse/OCPBUGS-38840
			"[Feature:LoadBalancer]",
		},
		"ibmroks": {
			// Calico is allowing the request to timeout instead of returning 'REFUSED'
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825021 - ROKS: calico SDN results in a request timeout when accessing services with no endpoints
			"[sig-network] Services should be rejected when no endpoints exist",
			// Nodes in ROKS have access to secrets in the cluster to handle encryption
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825013 - ROKS: worker nodes have access to secrets in the cluster
			"[sig-auth] [Feature:NodeAuthorizer] Getting a non-existent configmap should exit with the Forbidden error, not a NotFound error",
			"[sig-auth] [Feature:NodeAuthorizer] Getting a non-existent secret should exit with the Forbidden error, not a NotFound error",
			"[sig-auth] [Feature:NodeAuthorizer] Getting a secret for a workload the node has access to should succeed",
			"[sig-auth] [Feature:NodeAuthorizer] Getting an existing configmap should exit with the Forbidden error",
			"[sig-auth] [Feature:NodeAuthorizer] Getting an existing secret should exit with the Forbidden error",
			// Access to node external address is blocked from pods within a ROKS cluster by Calico
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825016 - e2e: NodeAuthenticator tests use both external and internal addresses for node
			"[sig-auth] [Feature:NodeAuthenticator] The kubelet's main port 10250 should reject requests with no credentials",
			"[sig-auth] [Feature:NodeAuthenticator] The kubelet can delegate ServiceAccount tokens to the API server",
			// Mode returned by RHEL7 worker contains an extra character not expected by the test: dgtrwx vs dtrwx
			// https://bugzilla.redhat.com/show_bug.cgi?id=1825024 - e2e: Failing test - HostPath should give a volume the correct mode
			"[sig-storage] HostPath should give a volume the correct mode",
		},
		"kubevirt": {
			// LoadBalancer tests in 1.31 require explicit platform-specific skips
			// https://issues.redhat.com/browse/OCPBUGS-38840
			"[Feature:LoadBalancer]",
		},
		"nutanix": {
			// LoadBalancer tests in 1.31 require explicit platform-specific skips
			// https://issues.redhat.com/browse/OCPBUGS-38840
			"[Feature:LoadBalancer]",
		},
		"openstack": {
			// LoadBalancer tests in 1.31 require explicit platform-specific skips
			// https://issues.redhat.com/browse/OCPBUGS-38840
			"[Feature:LoadBalancer]",
		},
		"ovirt": {
			// LoadBalancer tests in 1.31 require explicit platform-specific skips
			// https://issues.redhat.com/browse/OCPBUGS-38840
			"[Feature:LoadBalancer]",
		},
		"vsphere": {
			// LoadBalancer tests in 1.31 require explicit platform-specific skips
			// https://issues.redhat.com/browse/OCPBUGS-38840
			"[Feature:LoadBalancer]",
		},
		"external": {
			// LoadBalancer tests in 1.31 require explicit platform-specific skips
			// https://issues.redhat.com/browse/OCPBUGS-53249
			"[sig-network] LoadBalancers [Feature:LoadBalancer] should be able to preserve UDP traffic when server pod cycles for a LoadBalancer service on",
		},
	}

	for platform, exclusions := range platformExclusions {
		var selectFunctions []et.SelectFunction
		for _, exclusion := range exclusions {
			selectFunctions = append(selectFunctions, et.NameContains(exclusion))
		}

		specs.SelectAny(selectFunctions).Exclude(et.PlatformEquals(platform))
	}
}

// filterByExternalConnectivity is a helper function to do, simple, "NameContains" filtering on tests by external connectivity
func filterByExternalConnectivity(specs et.ExtensionTestSpecs) {
	var externalConnectivityExclusions = map[string][]string{
		// Tests that don't pass on disconnected, either due to requiring
		// internet access for GitHub (e.g. many of the s2i builds), or
		// because of pullthrough not supporting ICSP (https://bugzilla.redhat.com/show_bug.cgi?id=1918376)
		"Disconnected": {
			"[sig-network] Networking should provide Internet connection for containers",
		},
		// These tests are skipped when openshift-tests needs to use a proxy to reach the
		// cluster -- either because the test won't work while proxied, or because the test
		// itself is testing a functionality using it's own proxy.
		"Proxy": {
			// These tests setup their own proxy, which won't work when we need to access the
			// cluster through a proxy.
			"[sig-cli] Kubectl client Simple pod should support exec through an HTTP proxy",
			"[sig-cli] Kubectl client Simple pod should support exec through kubectl proxy",
			// Kube currently uses the x/net/websockets pkg, which doesn't work with proxies.
			// See: https://github.com/kubernetes/kubernetes/pull/103595
			"[sig-node] Pods should support retrieving logs from the container over websockets",
			"[sig-cli] Kubectl Port forwarding With a server listening on localhost should support forwarding over websockets",
			"[sig-cli] Kubectl Port forwarding With a server listening on 0.0.0.0 should support forwarding over websockets",
			"[sig-node] Pods should support remote command execution over websockets",
			// These tests are flacky and require internet access
			// See https://bugzilla.redhat.com/show_bug.cgi?id=2019375
			"[sig-network] DNS should resolve DNS of partial qualified names for services",
			"[sig-network] DNS should provide DNS for the cluster",
			// This test does not work when using in-proxy cluster, see https://bugzilla.redhat.com/show_bug.cgi?id=2084560
			"[sig-network] Networking should provide Internet connection for containers",
		},
	}

	for externalConnectivity, exclusions := range externalConnectivityExclusions {
		var selectFunctions []et.SelectFunction
		for _, exclusion := range exclusions {
			selectFunctions = append(selectFunctions, et.NameContains(exclusion))
		}

		specs.SelectAny(selectFunctions).Exclude(et.ExternalConnectivityEquals(externalConnectivity))
	}
}

// filterByTopology is a helper function to do, simple, "NameContains" filtering on tests by topology
func filterByTopology(specs et.ExtensionTestSpecs) {
	var topologyExclusions = map[string][]string{
		"SingleReplicaTopology": {
			"[sig-apps] Daemon set [Serial] should rollback without unnecessary restarts [Conformance]",
			"[sig-node] NoExecuteTaintManager Single Pod [Serial] doesn't evict pod with tolerations from tainted nodes",
			"[sig-node] NoExecuteTaintManager Single Pod [Serial] eventually evict pod with finite tolerations from tainted nodes",
			"[sig-node] NoExecuteTaintManager Single Pod [Serial] evicts pods from tainted nodes",
			"[sig-node] NoExecuteTaintManager Single Pod [Serial] removing taint cancels eviction [Disruptive] [Conformance]",
			"[sig-node] NoExecuteTaintManager Single Pod [Serial] pods evicted from tainted nodes have pod disruption condition",
			"[sig-node] NoExecuteTaintManager Multiple Pods [Serial] evicts pods with minTolerationSeconds [Disruptive] [Conformance]",
			"[sig-node] NoExecuteTaintManager Multiple Pods [Serial] only evicts pods without tolerations from tainted nodes",
			"[sig-cli] Kubectl client Kubectl taint [Serial] should remove all the taints with the same key off a node",
			"[sig-network] LoadBalancers should be able to preserve UDP traffic when server pod cycles for a LoadBalancer service on different nodes",
			"[sig-network] LoadBalancers should be able to preserve UDP traffic when server pod cycles for a LoadBalancer service on the same nodes",
			"[sig-architecture] Conformance Tests should have at least two untainted nodes",
		},
	}

	for topology, exclusions := range topologyExclusions {
		var selectFunctions []et.SelectFunction
		for _, exclusion := range exclusions {
			selectFunctions = append(selectFunctions, et.NameContains(exclusion))
		}

		specs.SelectAny(selectFunctions).Exclude(et.TopologyEquals(topology))
	}
}

// filterByNoOptionalCapabilities is a helper function to facilitate adding environment selectors for tests which can't
// be run/don't make sense to run against a cluster with all optional capabilities disabled
func filterByNoOptionalCapabilities(specs et.ExtensionTestSpecs) {
	var exclusions = []string{
		// Requires CSISnapshot capability
		"[Feature:VolumeSnapshotDataSource]",
		// Requires Storage capability
		"[Driver: aws]",
		"[Feature:StorageProvider]",
	}

	var selectFunctions []et.SelectFunction
	for _, exclusion := range exclusions {
		selectFunctions = append(selectFunctions, et.NameContains(exclusion))
	}
	specs.SelectAny(selectFunctions).Exclude(et.NoOptionalCapabilitiesExist())
}

// filterByNetwork is a helper function to do, simple, "NameContains" filtering on tests by network
func filterByNetwork(specs et.ExtensionTestSpecs) {
	var networkExclusions = map[string][]string{}

	for network, exclusions := range networkExclusions {
		var selectFunctions []et.SelectFunction
		for _, exclusion := range exclusions {
			selectFunctions = append(selectFunctions, et.NameContains(exclusion))
		}

		specs.SelectAny(selectFunctions).Exclude(et.NetworkEquals(network))
	}
}
