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

	"k8s.io/klog/v2"

	capi "k8s.io/api/certificates/v1beta1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	rbacv1helpers "k8s.io/kubernetes/pkg/apis/rbac/v1"
	"k8s.io/kubernetes/pkg/controlplane/controller/legacytokentracking"
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
	return rbacv1helpers.NewRule("create", "update", "patch").Groups(legacyGroup, eventsGroup).Resources("events").RuleOrDie()
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
				rbacv1helpers.NewRule("get", "create", "delete", "list", "watch").Groups(storageGroup).Resources("volumeattachments").RuleOrDie(),
			},
		}

		role.Rules = append(role.Rules, rbacv1helpers.NewRule("get", "watch", "list").Groups("storage.k8s.io").Resources("csidrivers").RuleOrDie())
		role.Rules = append(role.Rules, rbacv1helpers.NewRule("get", "watch", "list").Groups("storage.k8s.io").Resources("csinodes").RuleOrDie())

		return role
	}())

	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "clusterrole-aggregation-controller"},
		Rules: []rbacv1.PolicyRule{
			// this controller must have full permissions on clusterroles to allow it to mutate them in any way
			rbacv1helpers.NewRule("escalate", "get", "list", "watch", "update", "patch").Groups(rbacGroup).Resources("clusterroles").RuleOrDie(),
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
	addControllerRole(&controllerRoles, &controllerRoleBindings, func() rbacv1.ClusterRole {
		role := rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "disruption-controller"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("get", "list", "watch").Groups(extensionsGroup, appsGroup).Resources("deployments").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list", "watch").Groups(appsGroup, extensionsGroup).Resources("replicasets").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("replicationcontrollers").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list", "watch").Groups(policyGroup).Resources("poddisruptionbudgets").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list", "watch").Groups(appsGroup).Resources("statefulsets").RuleOrDie(),
				rbacv1helpers.NewRule("update").Groups(policyGroup).Resources("poddisruptionbudgets/status").RuleOrDie(),
				rbacv1helpers.NewRule("patch", "update").Groups(legacyGroup).Resources("pods/status").RuleOrDie(),
				rbacv1helpers.NewRule("get").Groups("*").Resources("*/scale").RuleOrDie(),
				eventsRule(),
			},
		}
		return role
	}())
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "endpoint-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("services", "pods").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "create", "update", "delete").Groups(legacyGroup).Resources("endpoints").RuleOrDie(),
			rbacv1helpers.NewRule("create").Groups(legacyGroup).Resources("endpoints/restricted").RuleOrDie(),
			eventsRule(),
		},
	})

	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "endpointslice-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("services", "pods", "nodes").RuleOrDie(),
			// The controller needs to be able to set a service's finalizers to be able to create an EndpointSlice
			// resource that is owned by the service and sets blockOwnerDeletion=true in its ownerRef.
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("services/finalizers").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "create", "update", "delete").Groups(discoveryGroup).Resources("endpointslices").RuleOrDie(),
			eventsRule(),
		},
	})

	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "endpointslicemirroring-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("services", "endpoints").RuleOrDie(),
			// The controller needs to be able to set a service's finalizers to be able to create an EndpointSlice
			// resource that is owned by the service and sets blockOwnerDeletion=true in its ownerRef.
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("services/finalizers").RuleOrDie(),
			// The controller needs to be able to set a service's finalizers to be able to create an EndpointSlice
			// resource that is owned by the endpoint and sets blockOwnerDeletion=true in its ownerRef.
			// see https://github.com/openshift/kubernetes/blob/8691466059314c3f7d6dcffcbb76d14596ca716c/pkg/controller/endpointslicemirroring/utils.go#L87-L88
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("endpoints/finalizers").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "create", "update", "delete").Groups(discoveryGroup).Resources("endpointslices").RuleOrDie(),
			eventsRule(),
		},
	})

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

	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "ephemeral-volume-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
			rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("pods/finalizers").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list", "watch", "create").Groups(legacyGroup).Resources("persistentvolumeclaims").RuleOrDie(),
			eventsRule(),
		},
	})

	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
		addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "resource-claim-controller"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("get", "list", "watch").Groups(legacyGroup).Resources("pods").RuleOrDie(),
				rbacv1helpers.NewRule("update").Groups(legacyGroup).Resources("pods/finalizers").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list", "watch", "create", "delete").Groups(resourceGroup).Resources("resourceclaims").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list", "watch", "create", "update", "patch").Groups(resourceGroup).Resources("podschedulingcontexts").RuleOrDie(),
				rbacv1helpers.NewRule("update", "patch").Groups(resourceGroup).Resources("resourceclaims", "resourceclaims/status").RuleOrDie(),
				rbacv1helpers.NewRule("update", "patch").Groups(legacyGroup).Resources("pods/status").RuleOrDie(),
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
			// allow listing resource, custom, and external metrics
			rbacv1helpers.NewRule("list").Groups(resMetricsGroup).Resources("pods").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list").Groups(customMetricsGroup).Resources("*").RuleOrDie(),
			rbacv1helpers.NewRule("get", "list").Groups(externalMetricsGroup).Resources("*").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "job-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "update", "patch").Groups(batchGroup).Resources("jobs").RuleOrDie(),
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
	addControllerRole(&controllerRoles, &controllerRoleBindings, func() rbacv1.ClusterRole {
		role := rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "node-controller"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("get", "list", "update", "delete", "patch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
				rbacv1helpers.NewRule("patch", "update").Groups(legacyGroup).Resources("nodes/status").RuleOrDie(),
				// used for pod deletion
				rbacv1helpers.NewRule("patch", "update").Groups(legacyGroup).Resources("pods/status").RuleOrDie(),
				rbacv1helpers.NewRule("list", "get", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
				eventsRule(),
			},
		}
		return role
	}())
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
			rbacv1helpers.NewRule("get", "create", "update", "delete").Groups(legacyGroup).Resources("endpoints").RuleOrDie(),
			rbacv1helpers.NewRule("get", "create", "delete").Groups(legacyGroup).Resources("services").RuleOrDie(),
			rbacv1helpers.NewRule("get").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
			// openstack
			rbacv1helpers.NewRule("get", "list").Groups(legacyGroup).Resources("nodes").RuleOrDie(),

			// recyclerClient.WatchPod
			rbacv1helpers.NewRule("watch").Groups(legacyGroup).Resources("events").RuleOrDie(),

			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, func() rbacv1.ClusterRole {
		role := rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "pod-garbage-collector"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("list", "watch", "delete").Groups(legacyGroup).Resources("pods").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
				rbacv1helpers.NewRule("patch").Groups(legacyGroup).Resources("pods/status").RuleOrDie(),
			},
		}
		return role
	}())
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
			rbacv1helpers.NewRule("patch", "update").Groups(legacyGroup).Resources("services/status").RuleOrDie(),
			rbacv1helpers.NewRule("list", "watch").Groups(legacyGroup).Resources("nodes").RuleOrDie(),
			eventsRule(),
		},
	})
	if utilfeature.DefaultFeatureGate.Enabled(features.MultiCIDRServiceAllocator) {
		addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "service-cidrs-controller"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("get", "list", "watch", "patch", "update").Groups(networkingGroup).Resources("servicecidrs").RuleOrDie(),
				rbacv1helpers.NewRule("patch", "update").Groups(networkingGroup).Resources("servicecidrs/finalizers").RuleOrDie(),
				rbacv1helpers.NewRule("patch", "update").Groups(networkingGroup).Resources("servicecidrs/status").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list", "watch").Groups(networkingGroup).Resources("ipaddresses").RuleOrDie(),
				eventsRule(),
			},
		})
	}
	addControllerRole(&controllerRoles, &controllerRoleBindings, func() rbacv1.ClusterRole {
		role := rbacv1.ClusterRole{
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
		}

		if utilfeature.DefaultFeatureGate.Enabled(features.StatefulSetAutoDeletePVC) {
			role.Rules = append(role.Rules, rbacv1helpers.NewRule("update", "delete").Groups(legacyGroup).Resources("persistentvolumeclaims").RuleOrDie())
		}

		return role
	}())
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
			rbacv1helpers.NewRule("approve").Groups(certificatesGroup).Resources("signers").Names(capi.KubeAPIServerClientKubeletSignerName).RuleOrDie(),
			rbacv1helpers.NewRule("sign").Groups(certificatesGroup).Resources("signers").Names(
				capi.LegacyUnknownSignerName,
				capi.KubeAPIServerClientSignerName,
				capi.KubeAPIServerClientKubeletSignerName,
				capi.KubeletServingSignerName,
			).RuleOrDie(),
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
	if utilfeature.DefaultFeatureGate.Enabled(features.VolumeAttributesClass) {
		addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "volumeattributesclass-protection-controller"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("list", "watch", "get").Groups(legacyGroup).Resources("persistentvolumeclaims").RuleOrDie(),
				rbacv1helpers.NewRule("list", "watch", "get").Groups(legacyGroup).Resources("persistentvolumes").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list", "watch", "update").Groups(storageGroup).Resources("volumeattributesclasses").RuleOrDie(),
				eventsRule(),
			},
		})
	}

	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "ttl-after-finished-controller"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get", "list", "watch", "delete").Groups(batchGroup).Resources("jobs").RuleOrDie(),
			eventsRule(),
		},
	})
	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "root-ca-cert-publisher"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("create", "update").Groups(legacyGroup).Resources("configmaps").RuleOrDie(),
			eventsRule(),
		},
	})
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ValidatingAdmissionPolicy) {
		addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "validatingadmissionpolicy-status-controller"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("get", "list", "watch").Groups(admissionRegistrationGroup).
					Resources("validatingadmissionpolicies").RuleOrDie(),
				rbacv1helpers.NewRule("get", "patch", "update").Groups(admissionRegistrationGroup).
					Resources("validatingadmissionpolicies/status").RuleOrDie(),
				eventsRule(),
			},
		})
	}
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.StorageVersionAPI) &&
		utilfeature.DefaultFeatureGate.Enabled(genericfeatures.APIServerIdentity) {
		addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "storage-version-garbage-collector"},
			Rules: []rbacv1.PolicyRule{
				rbacv1helpers.NewRule("get", "list", "watch").Groups(coordinationGroup).Resources("leases").RuleOrDie(),
				rbacv1helpers.NewRule("get", "list", "watch", "patch", "update", "delete").Groups(internalAPIServerGroup).
					Resources("storageversions").RuleOrDie(),
				rbacv1helpers.NewRule("get", "patch", "update").Groups(internalAPIServerGroup).
					Resources("storageversions/status").RuleOrDie(),
			},
		})
	}

	addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: saRolePrefix + "legacy-service-account-token-cleaner"},
		Rules: []rbacv1.PolicyRule{
			rbacv1helpers.NewRule("get").Groups(legacyGroup).Resources("configmaps").Names(legacytokentracking.ConfigMapName).RuleOrDie(),
			rbacv1helpers.NewRule("patch", "delete").Groups(legacyGroup).Resources("secrets").RuleOrDie(),
		},
	})

	if utilfeature.DefaultFeatureGate.Enabled(features.StorageVersionMigrator) {
		addControllerRole(&controllerRoles, &controllerRoleBindings, rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{
				Name: saRolePrefix + "storage-version-migrator-controller",
			},
			Rules: []rbacv1.PolicyRule{
				// need list to get current RV for any resource
				// need patch for SSA of any resource
				// need create because SSA of a deleted resource will be interpreted as a create request, these always fail with a conflict error because UID is set
				rbacv1helpers.NewRule("list", "create", "patch").Groups("*").Resources("*").RuleOrDie(),
				rbacv1helpers.NewRule("update").Groups(storageVersionMigrationGroup).Resources("storageversionmigrations/status").RuleOrDie(),
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
