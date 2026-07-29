// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

// fieldSortOrder contains the relative ordering of fields when formatting an
// object.
var fieldSortOrder = []string{
	// top-level metadata
	"name", "generateName", "namespace", "clusterName",
	"apiVersion", "kind", "metadata", "type",
	"labels", "annotations",
	"spec", "status",

	// secret and configmap
	"stringData", "data", "binaryData",

	// cronjobspec,  daemonsetspec, deploymentspec, statefulsetspec,
	// jobspec fields
	"parallelism", "completions", "activeDeadlineSeconds", "backoffLimit",
	"replicas", "selector", "manualSelector", "template",
	"ttlSecondsAfterFinished", "volumeClaimTemplates", "service", "serviceName",
	"podManagementPolicy", "updateStrategy", "strategy", "minReadySeconds",
	"revision", "revisionHistoryLimit", "paused", "progressDeadlineSeconds",

	// podspec
	// podspec scalars
	"restartPolicy", "terminationGracePeriodSeconds",
	"activeDeadlineSeconds", "dnsPolicy", "serviceAccountName",
	"serviceAccount", "automountServiceAccountToken", "nodeName",
	"hostNetwork", "hostPID", "hostIPC", "shareProcessNamespace", "hostname",
	"subdomain", "schedulerName", "priorityClassName", "priority",
	"runtimeClassName", "enableServiceLinks",

	// podspec lists and maps
	"nodeSelector", "hostAliases",

	// podspec objects
	"initContainers", "containers", "volumes", "securityContext",
	"imagePullSecrets", "affinity", "tolerations", "dnsConfig",
	"readinessGates",

	// containers
	"image", "command", "args", "workingDir", "ports", "envFrom", "env",
	"resources", "volumeMounts", "volumeDevices", "livenessProbe",
	"readinessProbe", "lifecycle", "terminationMessagePath",
	"terminationMessagePolicy", "imagePullPolicy", "securityContext",
	"stdin", "stdinOnce", "tty",

	// service
	"clusterIP", "externalIPs", "loadBalancerIP", "loadBalancerSourceRanges",
	"externalName", "externalTrafficPolicy", "sessionAffinity",

	// ports
	"protocol", "port", "targetPort", "hostPort", "containerPort", "hostIP",

	// volumemount
	"readOnly", "mountPath", "subPath", "subPathExpr", "mountPropagation",

	// envvar + envvarsource
	"value", "valueFrom", "fieldRef", "resourceFieldRef", "configMapKeyRef",
	"secretKeyRef", "prefix", "configMapRef", "secretRef",
}

type set map[string]interface{}

func newSet(values ...string) set {
	m := map[string]interface{}{}
	for _, value := range values {
		m[value] = nil
	}
	return m
}

func (s set) Has(key string) bool {
	_, found := s[key]
	return found
}

// WhitelistedListSortKinds contains the set of kinds that are whitelisted
// for sorting list field elements
var WhitelistedListSortKinds = newSet(
	"CronJob", "DaemonSet", "Deployment", "Job", "ReplicaSet", "StatefulSet",
	"ValidatingWebhookConfiguration")

// WhitelistedListSortApis contains the set of apis that are whitelisted for
// sorting list field elements
var WhitelistedListSortApis = newSet(
	"apps/v1", "apps/v1beta1", "apps/v1beta2", "batch/v1", "batch/v1beta1",
	"extensions/v1beta1", "v1", "admissionregistration.k8s.io/v1")

// WhitelistedListSortFields contains json paths to list fields that should
// be sorted, and the field they should be sorted by
var WhitelistedListSortFields = map[string]string{
	".spec.template.spec.containers": "name",
	".webhooks.rules.operations":     "",
}

// FieldOrder indexes fields and maps them to relative precedence
var FieldOrder = func() map[string]int {
	// create an index of field orderings
	fo := map[string]int{}
	for i, f := range fieldSortOrder {
		fo[f] = i + 1
	}
	return fo
}()
