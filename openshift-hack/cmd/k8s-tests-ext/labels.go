package main

import (
	et "github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"
)

func addLabelsToSpecs(specs et.ExtensionTestSpecs) {
	var namesByLabel = map[string][]string{
		// tests too slow to be part of conformance
		"[Slow]": {
			"[sig-scalability]",                            // disable from the default set for now
			"should create and stop a working application", // Inordinately slow tests

			"[Feature:PerformanceDNS]", // very slow

			"validates that there exists conflict between pods with same hostPort and protocol but one using 0.0.0.0 hostIP", // 5m, really?
		},
		// tests that are known flaky
		"[Flaky]": {
			"Job should run a job to completion when tasks sometimes fail and are not locally restarted", // seems flaky, also may require too many resources
			// TODO(node): test works when run alone, but not in the suite in CI
			"[Feature:HPA] Horizontal pod autoscaling (scale resource: CPU) [sig-autoscaling] ReplicationController light Should scale from 1 pod to 2 pods",
		},
		// tests that must be run without competition
		"[Serial]": {
			"[Disruptive]",
			"[Feature:Performance]", // requires isolation

			"Service endpoints latency", // requires low latency
			"Clean up pods on node",     // schedules up to max pods per node
			"DynamicProvisioner should test that deleting a claim before the volume is provisioned deletes the volume", // test is very disruptive to other tests

			"Should be able to support the 1.7 Sample API Server using the current Aggregator", // down apiservices break other clients today https://bugzilla.redhat.com/show_bug.cgi?id=1623195

			"[Feature:HPA] Horizontal pod autoscaling (scale resource: CPU) [sig-autoscaling] ReplicationController light Should scale from 1 pod to 2 pods",

			"should prevent Ingress creation if more than 1 IngressClass marked as default", // https://bugzilla.redhat.com/show_bug.cgi?id=1822286

			"[sig-network] IngressClass [Feature:Ingress] should set default value on new IngressClass", //https://bugzilla.redhat.com/show_bug.cgi?id=1833583
		},
	}

	for label, names := range namesByLabel {
		var selectFunctions []et.SelectFunction
		for _, name := range names {
			selectFunctions = append(selectFunctions, et.NameContains(name))
		}

		//TODO: once annotation logic has been removed, it might also be necessary to annotate the test name with the label as well
		specs.SelectAny(selectFunctions).AddLabel(label)
	}
}
