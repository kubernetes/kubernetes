/*
Copyright 2016 The Kubernetes Authors.

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

package bootstrappolicy

import (
	"strings"

	"k8s.io/klog"

	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
	"k8s.io/kubernetes/pkg/features"
)

const saRolePrefix = "system:controller:"

func addControllerRole(controllerRoles *[]rbacv1.ClusterRole, controllerRoleBindings *[]rbacv1.ClusterRoleBinding, role rbacv1.ClusterRole) {
	if !strings.HasPrefix(role.Name, saRolePrefix) {
		klog.Fatalf(`role %q must start with %q`, role.Name, saRolePrefix)
	}

	for _, existingRole := range *controllerRoles {
		if role.Name == existingRole.Name {
			klog.Fatalf("role %q was already registered", role.Name)
		}
	}

	*controllerRoles = append(*controllerRoles, role)
	addClusterRoleLabel(*controllerRoles)

	*controllerRoleBindings = append(*controllerRoleBindings,
		rbacv1helpers.NewClusterBinding(role.Name).SAs("kube-system", role.Name[len(saRolePrefix):]).BindingOrDie())
	addClusterRoleBindingLabel(*controllerRoleBindings)
}

func eventsRule() rbacv1.PolicyRule {
	return rbacv1helpers.NewRule("create", "update", "patch").Groups(legacyGroup).Resources("events").RuleOrDie()
}

func buildControllerRoles() ([]rbacv1.ClusterRole, []rbacv1.ClusterRoleBinding) {
	// controllerRoles is a slice of roles used for controllers
	controllerRoles := []rbacv1.ClusterRole{}
	// controllerRoleBindings is a slice of roles used for controllers
	controllerRoleBindings := []rbacv1.ClusterRoleBinding{}

	addControllerRole(&controllerRoles, &controllerRoleBindings, func() rbacv1.ClusterRole {
		role := rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "attachdetach-controller"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("list", "watch").Groups(legacyGroup).Resources("persistentvolumes", "persistentvolumeclaims").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
				rbacv1helpers.NewRule("patch", "update").Groups(legacyGroup).Resources("nodes/status").RuleOrDie(),
				rbacv1helpers.NewRule("list", "watch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
				eventsRule(),
			},
		}

		if utilfeature.DefaultFeatureGate.Enabled(features.CSIPersistentVolume) {
			role.Rules = append(role.Rules, rbacv1helpers.NewRule("get", "create", "delete", "list", "watch").Groups(storageGroup).Resources("volumeattachments").RuleOrDie())
			if utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
				role.Rules = append(role.Rules, rbacv1helpers.NewRule("get", "watch", "list").Groups("csi.storage.k8s.io").Resources("csidrivers").RuleOrDie())
			}
		}

		return role
	}())

	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "clusterrole-aggregation-controller"},
		Rules: []rbacv1.PolicyRule{
			// this controller must have full permissions to allow it to mutate any role in any way
			rbacv1helpers.NewRule("*").Groups("*").Resources("*").RuleOrDie(),
			rbacv1helpers.NewRule("*").URLs("*").RuleOrDie(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "cronjob-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "update").Groups(batchGroup).Resources("cronjobs").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "watch", "create", "update", "delete", "patch").Groups(batchGroup).Resources("jobs").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(batchGroup).Resources("cronjobs/status").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(batchGroup).Resources("cronjobs/finalizers").RuleOrDie(),
			rbacv1helpers.NewRule("list", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "daemon-set-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups(extensionsGroup, appsGroup).Resources("daemonsets").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(extensionsGroup, appsGroup).Resources("daemonsets/status").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(extensionsGroup, appsGroup).Resources("daemonsets/finalizers").RuleOrDie(),
			rbacv1helpers.NewRule("list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			rbacv1helpers.NewRule("list", "watch", "create", "delete", "patch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			rbacv1helpers.NewRule("create").Groups(legacyGroup).Resources("pods/binding").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "watch", "create", "delete", "update", "patch").Groups(appsGroup).Resources("controllerrevisions").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "deployment-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "update").Groups(extensionsGroup, appsGroup).Resources("deployments").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(extensionsGroup, appsGroup).Resources("deployments/status").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(extensionsGroup, appsGroup).Resources("deployments/finalizers").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "watch", "create", "update", "patch", "delete").Groups(appsGroup, extensionsGroup).Resources("replicasets").RuleOrDie(),
			// TODO: remove "update" once
			// https://github.com/kubernetes/kubernetes/issues/36897 is resolved.
			rbacv1helpers.NewRule("get", "list", "watch", "update").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "disruption-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups(extensionsGroup, appsGroup).Resources("deployments").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "watch").Groups(appsGroup, extensionsGroup).Resources("replicasets").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("replicationcontrollers").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "watch").Groups(policyGroup).Resources("poddisruptionbudgets").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "watch").Groups(appsGroup).Resources("statefulsets").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(policyGroup).Resources("poddisruptionbudgets/status").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "endpoint-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("services", "pods").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "create", "update", "delete").Groups(legacyGroup).Resources("endpoints").RuleOrDie(),
			rbacv1helpers.NewRule("create").Groups(legacyGroup).Resources("endpoints/restricted").RuleOrDie(),
			eventsRule(),
		},
	})

	if utilfeature.DefaultFeatureGate.Enabled(features.ExpandPersistentVolumes) {
		addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "expand-controller"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("get", "list", "watch", "update", "patch").Groups(legacyGroup).Resources("persistentvolumes").RuleOrDie(),
				rbacv1helpers.NewRule("update", "patch").Groups(legacyGroup).Resources("persistentvolumeclaims/status").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("persistentvolumeclaims").RuleOrDie(),
				// glusterfs
				rbacv1helpers.NewRule("get", "list", "watch").Groups(storageGroup).Resources("storageclasses").RuleOrDie(),
				rbacv1helpers.NewRule("get").Groups(legacyGroup).Resources("services", "endpoints").RuleOrDie(),
				rbacv1helpers.NewRule("get").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
				eventsRule(),
			},
		})
	}

	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "generic-garbage-collector"},
		Rules: []rbacv1.PolicyRule{
			// the GC controller needs to run list/watches, selective gets, and updates against any resource
			rbacv1helpers.NewRule("get", "list", "watch", "patch", "update", "delete").Groups("*").Resources("*").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "horizontal-pod-autoscaler"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups(autoscalingGroup).Resources("horizontalpodautoscalers").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(autoscalingGroup).Resources("horizontalpodautoscalers/status").RuleOrDie(),
			rbacv1helpers.NewRule("get", "update").Groups("*").Resources("*/scale").RuleOrDie(),
			rbacv1helpers.NewRule("list").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			// TODO: restrict this to the appropriate namespace
			rbacv1helpers.NewRule("get").Groups(legacyGroup).Resources("services/proxy").Names("https:heapster:", "http:heapster:").RuleOrDie(),
			// allow listing resource metrics and custom metrics
			rbacv1helpers.NewRule("list").Groups(resMetricsGroup).Resources("pods").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list").Groups(customMetricsGroup).Resources("*").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "job-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "update").Groups(batchGroup).Resources("jobs").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(batchGroup).Resources("jobs/status").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(batchGroup).Resources("jobs/finalizers").RuleOrDie(),
			rbacv1helpers.NewRule("list", "watch", "create", "delete", "patch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "namespace-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "delete").Groups(legacyGroup).Resources("namespaces").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("namespaces/finalize", "namespaces/status").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "delete", "deletecollection").Groups("*").Resources("*").RuleOrDie(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "node-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "update", "delete", "patch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			rbacv1helpers.NewRule("patch", "update").Groups(legacyGroup).Resources("nodes/status").RuleOrDie(),
			// used for pod eviction
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("pods/status").RuleOrDie(),
			rbacv1helpers.NewRule("list", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "persistent-volume-binder"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "update", "create", "delete").Groups(legacyGroup).Resources("persistentvolumes").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("persistentvolumes/status").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "watch", "update").Groups(legacyGroup).Resources("persistentvolumeclaims").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("persistentvolumeclaims/status").RuleOrDie(),
			rbacv1helpers.NewRule("list", "watch", "get", "create", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),

			// glusterfs
			rbacv1helpers.NewRule("get", "list", "watch").Groups(storageGroup).Resources("storageclasses").RuleOrDie(),
			rbacv1helpers.NewRule("get", "create", "delete").Groups(legacyGroup).Resources("services", "endpoints").RuleOrDie(),
			rbacv1helpers.NewRule("get").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
			// openstack
			rbacv1helpers.NewRule("get", "list").Groups(legacyGroup).Resources("nodes").RuleOrDie(),

			// recyclerClient.WatchPod
			rbacv1helpers.NewRule("watch").Groups(legacyGroup).Resources("events").RuleOrDie(),

			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "pod-garbage-collector"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("list", "watch", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			rbacv1helpers.NewRule("list").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "replicaset-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "update").Groups(appsGroup, extensionsGroup).Resources("replicasets").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(appsGroup, extensionsGroup).Resources("replicasets/status").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(appsGroup, extensionsGroup).Resources("replicasets/finalizers").RuleOrDie(),
			rbacv1helpers.NewRule("list", "watch", "patch", "create", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "replication-controller"},
		Rules: []rbacv1.PolicyRule{
			// 1.0 controllers needed get, update, so without these old controllers break on new servers
			rbacv1helpers.NewRule("get", "list", "watch", "update").Groups(legacyGroup).Resources("replicationcontrollers").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("replicationcontrollers/status").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("replicationcontrollers/finalizers").RuleOrDie(),
			rbacv1helpers.NewRule("list", "watch", "patch", "create", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "resourcequota-controller"},
		Rules: []rbacv1.PolicyRule{
			// quota can count quota on anything for reconciliation, so it needs full viewing powers
			rbacv1helpers.NewRule("list", "watch").Groups("*").Resources("*").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("resourcequotas/status").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "route-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			rbacv1helpers.NewRule("patch").Groups(legacyGroup).Resources("nodes/status").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "service-account-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("create").Groups(legacyGroup).Resources("serviceaccounts").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "service-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("services").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("services/status").RuleOrDie(),
			rbacv1helpers.NewRule("list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "statefulset-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("list", "watch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "watch").Groups(appsGroup).Resources("statefulsets").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(appsGroup).Resources("statefulsets/status").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(appsGroup).Resources("statefulsets/finalizers").RuleOrDie(),
			rbacv1helpers.NewRule("get", "create", "delete", "update", "patch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			rbacv1helpers.NewRule("get", "create", "delete", "update", "patch", "list", "watch").Groups(appsGroup).Resources("controllerrevisions").RuleOrDie(),
			rbacv1helpers.NewRule("get", "create").Groups(legacyGroup).Resources("persistentvolumeclaims").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "ttl-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("update", "patch", "list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "certificate-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "delete").Groups(certificatesGroup).Resources("certificatesigningrequests").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(certificatesGroup).Resources("certificatesigningrequests/status", "certificatesigningrequests/approval").RuleOrDie(),
			rbacv1helpers.NewRule("create").Groups(authorizationGroup).Resources("subjectaccessreviews").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "pvc-protection-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "update").Groups(legacyGroup).Resources("persistentvolumeclaims").RuleOrDie(),
			rbacv1helpers.NewRule("list", "watch", "get").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "pv-protection-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "update").Groups(legacyGroup).Resources("persistentvolumes").RuleOrDie(),
			eventsRule(),
		},
	})
	if utilfeature.DefaultFeatureGate.Enabled(features.TTLAfterFinished) {
		addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "ttl-after-finished-controller"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("get", "list", "watch", "delete").Groups(batchGroup).Resources("jobs").RuleOrDie(),
				eventsRule(),
			},
		})
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.BoundServiceAccountTokenVolume) {
		addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "root-ca-cert-publisher"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("create", "update").Groups(legacyGroup).Resources("configmaps").RuleOrDie(),
				eventsRule(),
			},
		})
	}

	return controllerRoles, controllerRoleBindings
}

// ControllerRoles returns the cluster roles used by controllers
func ControllerRoles() []rbacv1.ClusterRole {
	controllerRoles, _ := buildControllerRoles()
	return controllerRoles
}

// ControllerRoleBindings returns the role bindings used by controllers
func ControllerRoleBindings() []rbacv1.ClusterRoleBinding {
	_, controllerRoleBindings := buildControllerRoles()
	return controllerRoleBindings
}
