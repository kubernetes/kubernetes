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

	"github.com/golang/glog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	rbac "k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/features"
)

const saRolePrefix = "system:controller:"

func addControllerRole(controllerRoles *[]rbac.ClusterRole, controllerRoleBindings *[]rbac.ClusterRoleBinding, role rbac.ClusterRole) {
	if !strings.HasPrefix(role.Name, saRolePrefix) {
		glog.Fatalf(`role %q must start with %q`, role.Name, saRolePrefix)
	}

	for _, existingRole := range *controllerRoles {
		if role.Name == existingRole.Name {
			glog.Fatalf("role %q was already registered", role.Name)
		}
	}

	*controllerRoles = append(*controllerRoles, role)
	addClusterRoleLabel(*controllerRoles)

	*controllerRoleBindings = append(*controllerRoleBindings,
		rbac.NewClusterBinding(role.Name).SAs("kube-system", role.Name[len(saRolePrefix):]).BindingOrDie())
	addClusterRoleBindingLabel(*controllerRoleBindings)
}

func eventsRule() rbac.PolicyRule {
	return rbac.NewRule("create", "update", "patch").Groups(legacyGroup).Resources("events").RuleOrDie()
}

func buildControllerRoles() ([]rbac.ClusterRole, []rbac.ClusterRoleBinding) {
	// controllerRoles is a slice of roles used for controllers
	controllerRoles := []rbac.ClusterRole{}
	// controllerRoleBindings is a slice of roles used for controllers
	controllerRoleBindings := []rbac.ClusterRoleBinding{}

	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "attachdetach-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("list", "watch").Groups(legacyGroup).Resources("persistentvolumes", "persistentvolumeclaims").RuleOrDie(),
			rbac.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			rbac.NewRule("patch", "update").Groups(legacyGroup).Resources("nodes/status").RuleOrDie(),
			rbac.NewRule("list", "watch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "cronjob-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch", "update").Groups(batchGroup).Resources("cronjobs").RuleOrDie(),
			rbac.NewRule("get", "list", "watch", "create", "update", "delete", "patch").Groups(batchGroup).Resources("jobs").RuleOrDie(),
			rbac.NewRule("update").Groups(batchGroup).Resources("cronjobs/status").RuleOrDie(),
			rbac.NewRule("update").Groups(batchGroup).Resources("cronjobs/finalizers").RuleOrDie(),
			rbac.NewRule("list", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "daemon-set-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch").Groups(extensionsGroup, appsGroup).Resources("daemonsets").RuleOrDie(),
			rbac.NewRule("update").Groups(extensionsGroup, appsGroup).Resources("daemonsets/status").RuleOrDie(),
			rbac.NewRule("update").Groups(extensionsGroup, appsGroup).Resources("daemonsets/finalizers").RuleOrDie(),
			rbac.NewRule("list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			rbac.NewRule("list", "watch", "create", "delete", "patch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			rbac.NewRule("create").Groups(legacyGroup).Resources("pods/binding").RuleOrDie(),
			rbac.NewRule("list", "watch", "create", "delete", "update", "patch").Groups(appsGroup).Resources("controllerrevisions").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "deployment-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch", "update").Groups(extensionsGroup, appsGroup).Resources("deployments").RuleOrDie(),
			rbac.NewRule("update").Groups(extensionsGroup, appsGroup).Resources("deployments/status").RuleOrDie(),
			rbac.NewRule("update").Groups(extensionsGroup, appsGroup).Resources("deployments/finalizers").RuleOrDie(),
			rbac.NewRule("get", "list", "watch", "create", "update", "patch", "delete").Groups(appsGroup, extensionsGroup).Resources("replicasets").RuleOrDie(),
			// TODO: remove "update" once
			// https://github.com/kubernetes/kubernetes/issues/36897 is resolved.
			rbac.NewRule("get", "list", "watch", "update").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "disruption-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch").Groups(extensionsGroup, appsGroup).Resources("deployments").RuleOrDie(),
			rbac.NewRule("get", "list", "watch").Groups(appsGroup, extensionsGroup).Resources("replicasets").RuleOrDie(),
			rbac.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("replicationcontrollers").RuleOrDie(),
			rbac.NewRule("get", "list", "watch").Groups(policyGroup).Resources("poddisruptionbudgets").RuleOrDie(),
			rbac.NewRule("get", "list", "watch").Groups(appsGroup).Resources("statefulsets").RuleOrDie(),
			rbac.NewRule("update").Groups(policyGroup).Resources("poddisruptionbudgets/status").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "endpoint-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("services", "pods").RuleOrDie(),
			rbac.NewRule("get", "list", "create", "update", "delete").Groups(legacyGroup).Resources("endpoints").RuleOrDie(),
			rbac.NewRule("create").Groups(legacyGroup).Resources("endpoints/restricted").RuleOrDie(),
			eventsRule(),
		},
	})

	if utilfeature.DefaultFeatureGate.Enabled(features.ExpandPersistentVolumes) {
		addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "expand-controller"},
			Rules: []rbac.PolicyRule{
				rbac.NewRule("get", "list", "watch", "update", "patch").Groups(legacyGroup).Resources("persistentvolumes").RuleOrDie(),
				rbac.NewRule("update", "patch").Groups(legacyGroup).Resources("persistentvolumeclaims/status").RuleOrDie(),
				rbac.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("persistentvolumeclaims").RuleOrDie(),
				// glusterfs
				rbac.NewRule("get", "list", "watch").Groups(storageGroup).Resources("storageclasses").RuleOrDie(),
				rbac.NewRule("get").Groups(legacyGroup).Resources("services", "endpoints").RuleOrDie(),
				rbac.NewRule("get").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
				eventsRule(),
			},
		})
	}

	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "generic-garbage-collector"},
		Rules: []rbac.PolicyRule{
			// the GC controller needs to run list/watches, selective gets, and updates against any resource
			rbac.NewRule("get", "list", "watch", "patch", "update", "delete").Groups("*").Resources("*").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "horizontal-pod-autoscaler"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch").Groups(autoscalingGroup).Resources("horizontalpodautoscalers").RuleOrDie(),
			rbac.NewRule("update").Groups(autoscalingGroup).Resources("horizontalpodautoscalers/status").RuleOrDie(),
			rbac.NewRule("get", "update").Groups("*").Resources("*/scale").RuleOrDie(),
			rbac.NewRule("list").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			// TODO: restrict this to the appropriate namespace
			rbac.NewRule("get").Groups(legacyGroup).Resources("services/proxy").Names("https:heapster:", "http:heapster:").RuleOrDie(),
			// allow listing resource metrics and custom metrics
			rbac.NewRule("list").Groups(resMetricsGroup).Resources("pods").RuleOrDie(),
			rbac.NewRule("list").Groups(customMetricsGroup).Resources("*").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "job-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch", "update").Groups(batchGroup).Resources("jobs").RuleOrDie(),
			rbac.NewRule("update").Groups(batchGroup).Resources("jobs/status").RuleOrDie(),
			rbac.NewRule("update").Groups(batchGroup).Resources("jobs/finalizers").RuleOrDie(),
			rbac.NewRule("list", "watch", "create", "delete", "patch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "namespace-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch", "delete").Groups(legacyGroup).Resources("namespaces").RuleOrDie(),
			rbac.NewRule("update").Groups(legacyGroup).Resources("namespaces/finalize", "namespaces/status").RuleOrDie(),
			rbac.NewRule("get", "list", "delete", "deletecollection").Groups("*").Resources("*").RuleOrDie(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "node-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "update", "delete", "patch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			rbac.NewRule("patch", "update").Groups(legacyGroup).Resources("nodes/status").RuleOrDie(),
			// used for pod eviction
			rbac.NewRule("update").Groups(legacyGroup).Resources("pods/status").RuleOrDie(),
			rbac.NewRule("list", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "persistent-volume-binder"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch", "update", "create", "delete").Groups(legacyGroup).Resources("persistentvolumes").RuleOrDie(),
			rbac.NewRule("update").Groups(legacyGroup).Resources("persistentvolumes/status").RuleOrDie(),
			rbac.NewRule("get", "list", "watch", "update").Groups(legacyGroup).Resources("persistentvolumeclaims").RuleOrDie(),
			rbac.NewRule("update").Groups(legacyGroup).Resources("persistentvolumeclaims/status").RuleOrDie(),
			rbac.NewRule("list", "watch", "get", "create", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),

			// glusterfs
			rbac.NewRule("get", "list", "watch").Groups(storageGroup).Resources("storageclasses").RuleOrDie(),
			rbac.NewRule("get", "create", "delete").Groups(legacyGroup).Resources("services", "endpoints").RuleOrDie(),
			rbac.NewRule("get").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
			// openstack
			rbac.NewRule("get", "list").Groups(legacyGroup).Resources("nodes").RuleOrDie(),

			// recyclerClient.WatchPod
			rbac.NewRule("watch").Groups(legacyGroup).Resources("events").RuleOrDie(),

			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "pod-garbage-collector"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("list", "watch", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			rbac.NewRule("list").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "replicaset-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch", "update").Groups(appsGroup, extensionsGroup).Resources("replicasets").RuleOrDie(),
			rbac.NewRule("update").Groups(appsGroup, extensionsGroup).Resources("replicasets/status").RuleOrDie(),
			rbac.NewRule("update").Groups(appsGroup, extensionsGroup).Resources("replicasets/finalizers").RuleOrDie(),
			rbac.NewRule("list", "watch", "patch", "create", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "replication-controller"},
		Rules: []rbac.PolicyRule{
			// 1.0 controllers needed get, update, so without these old controllers break on new servers
			rbac.NewRule("get", "list", "watch", "update").Groups(legacyGroup).Resources("replicationcontrollers").RuleOrDie(),
			rbac.NewRule("update").Groups(legacyGroup).Resources("replicationcontrollers/status").RuleOrDie(),
			rbac.NewRule("update").Groups(legacyGroup).Resources("replicationcontrollers/finalizers").RuleOrDie(),
			rbac.NewRule("list", "watch", "patch", "create", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "resourcequota-controller"},
		Rules: []rbac.PolicyRule{
			// quota can count quota on anything for reconcilation, so it needs full viewing powers
			rbac.NewRule("list", "watch").Groups("*").Resources("*").RuleOrDie(),
			rbac.NewRule("update").Groups(legacyGroup).Resources("resourcequotas/status").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "route-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			rbac.NewRule("patch").Groups(legacyGroup).Resources("nodes/status").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "service-account-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("create").Groups(legacyGroup).Resources("serviceaccounts").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "service-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("services").RuleOrDie(),
			rbac.NewRule("update").Groups(legacyGroup).Resources("services/status").RuleOrDie(),
			rbac.NewRule("list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "statefulset-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("list", "watch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			rbac.NewRule("get", "list", "watch").Groups(appsGroup).Resources("statefulsets").RuleOrDie(),
			rbac.NewRule("update").Groups(appsGroup).Resources("statefulsets/status").RuleOrDie(),
			rbac.NewRule("update").Groups(appsGroup).Resources("statefulsets/finalizers").RuleOrDie(),
			rbac.NewRule("get", "create", "delete", "update", "patch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			rbac.NewRule("get", "create", "delete", "update", "patch", "list", "watch").Groups(appsGroup).Resources("controllerrevisions").RuleOrDie(),
			rbac.NewRule("get", "create").Groups(legacyGroup).Resources("persistentvolumeclaims").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "ttl-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("update", "patch", "list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbac.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "certificate-controller"},
		Rules: []rbac.PolicyRule{
			rbac.NewRule("get", "list", "watch").Groups(certificatesGroup).Resources("certificatesigningrequests").RuleOrDie(),
			rbac.NewRule("update").Groups(certificatesGroup).Resources("certificatesigningrequests/status", "certificatesigningrequests/approval").RuleOrDie(),
			rbac.NewRule("create").Groups(authorizationGroup).Resources("subjectaccessreviews").RuleOrDie(),
			eventsRule(),
		},
	})

	return controllerRoles, controllerRoleBindings
}

// ControllerRoles returns the cluster roles used by controllers
func ControllerRoles() []rbac.ClusterRole {
	controllerRoles, _ := buildControllerRoles()
	return controllerRoles
}

// ControllerRoleBindings returns the role bindings used by controllers
func ControllerRoleBindings() []rbac.ClusterRoleBinding {
	_, controllerRoleBindings := buildControllerRoles()
	return controllerRoleBindings
}
