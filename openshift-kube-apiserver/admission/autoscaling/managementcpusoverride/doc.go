package managementcpusoverride

// The ManagementCPUOverride admission plugin replaces pod container CPU requests with a new management resource.
// It applies to all pods that:
// 1. are in an allowed namespace
// 2. and have the workload annotation.
//
// It also sets the new management resource request and limit and  set resource annotation that CRI-O can
// recognize and apply the relevant changes.
// For more information, see - https://github.com/openshift/enhancements/pull/703
//
// Conditions for CPUs requests deletion:
// 1. The namespace should have allowed annotation "workload.openshift.io/allowed": "management"
// 2. The pod should have management annotation: "workload.openshift.io/management": "{"effect": "PreferredDuringScheduling"}"
// 3. All nodes under the cluster should have new management resource - "management.workload.openshift.io/cores"
// 4. The CPU request deletion will not change the pod QoS class
