/*
Copyright 2018 The Kubernetes Authors.

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

package etcd

import (
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/pointer"
)

// GetEtcdStorageData returns etcd data for all persisted objects.
// It is exported so that it can be reused across multiple tests.
// It returns a new map on every invocation to prevent different tests from mutating shared state.
func GetEtcdStorageData() map[schema.GroupVersionResource]StorageData {
	return GetEtcdStorageDataForNamespace("etcdstoragepathtestnamespace")
}

// GetEtcdStorageDataForNamespace returns etcd data for all persisted objects.
// It is exported so that it can be reused across multiple tests.
// It returns a new map on every invocation to prevent different tests from mutating shared state.
// Namespaced objects keys are computed for the specified namespace.
func GetEtcdStorageDataForNamespace(namespace string) map[schema.GroupVersionResource]StorageData {
	etcdStorageData := map[schema.GroupVersionResource]StorageData{
		// k8s.io/kubernetes/pkg/api/v1
		gvr("", "v1", "configmaps"): {
			Stub:             `{"data": {"foo": "bar"}, "metadata": {"name": "cm1"}}`,
			ExpectedEtcdPath: "/registry/configmaps/" + namespace + "/cm1",
		},
		gvr("", "v1", "services"): {
			Stub:             `{"metadata": {"name": "service1"}, "spec": {"externalName": "service1name", "ports": [{"port": 10000, "targetPort": 11000}], "selector": {"test": "data"}}}`,
			ExpectedEtcdPath: "/registry/services/specs/" + namespace + "/service1",
		},
		gvr("", "v1", "podtemplates"): {
			Stub:             `{"metadata": {"name": "pt1name"}, "template": {"metadata": {"labels": {"pt": "01"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container9"}]}}}`,
			ExpectedEtcdPath: "/registry/podtemplates/" + namespace + "/pt1name",
		},
		gvr("", "v1", "pods"): {
			Stub:             `{"metadata": {"name": "pod1"}, "spec": {"containers": [{"image": "fedora:latest", "name": "container7", "resources": {"limits": {"cpu": "1M"}, "requests": {"cpu": "1M"}}}]}}`,
			ExpectedEtcdPath: "/registry/pods/" + namespace + "/pod1",
		},
		gvr("", "v1", "endpoints"): {
			Stub:             `{"metadata": {"name": "ep1name"}, "subsets": [{"addresses": [{"hostname": "bar-001", "ip": "192.168.3.1"}], "ports": [{"port": 8000}]}]}`,
			ExpectedEtcdPath: "/registry/services/endpoints/" + namespace + "/ep1name",
		},
		gvr("", "v1", "resourcequotas"): {
			Stub:             `{"metadata": {"name": "rq1name"}, "spec": {"hard": {"cpu": "5M"}}}`,
			ExpectedEtcdPath: "/registry/resourcequotas/" + namespace + "/rq1name",
		},
		gvr("", "v1", "limitranges"): {
			Stub:             `{"metadata": {"name": "lr1name"}, "spec": {"limits": [{"type": "Pod"}]}}`,
			ExpectedEtcdPath: "/registry/limitranges/" + namespace + "/lr1name",
		},
		gvr("", "v1", "namespaces"): {
			Stub:             `{"metadata": {"name": "namespace1"}, "spec": {"finalizers": ["kubernetes"]}}`,
			ExpectedEtcdPath: "/registry/namespaces/namespace1",
		},
		gvr("", "v1", "nodes"): {
			Stub:             `{"metadata": {"name": "node1"}, "spec": {"unschedulable": true}}`,
			ExpectedEtcdPath: "/registry/minions/node1",
		},
		gvr("", "v1", "persistentvolumes"): {
			Stub:             `{"metadata": {"name": "pv1name"}, "spec": {"accessModes": ["ReadWriteOnce"], "capacity": {"storage": "3M"}, "hostPath": {"path": "/tmp/test/"}}}`,
			ExpectedEtcdPath: "/registry/persistentvolumes/pv1name",
		},
		gvr("", "v1", "events"): {
			Stub:             `{"involvedObject": {"namespace": "` + namespace + `"}, "message": "some data here", "metadata": {"name": "event1"}}`,
			ExpectedEtcdPath: "/registry/events/" + namespace + "/event1",
		},
		gvr("", "v1", "persistentvolumeclaims"): {
			Stub:             `{"metadata": {"name": "pvc1"}, "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"limits": {"storage": "1M"}, "requests": {"storage": "2M"}}, "selector": {"matchLabels": {"pvc": "stuff"}}}}`,
			ExpectedEtcdPath: "/registry/persistentvolumeclaims/" + namespace + "/pvc1",
		},
		gvr("", "v1", "serviceaccounts"): {
			Stub:             `{"metadata": {"name": "sa1name"}, "secrets": [{"name": "secret00"}]}`,
			ExpectedEtcdPath: "/registry/serviceaccounts/" + namespace + "/sa1name",
		},
		gvr("", "v1", "secrets"): {
			Stub:             `{"data": {"key": "ZGF0YSBmaWxl"}, "metadata": {"name": "secret1"}}`,
			ExpectedEtcdPath: "/registry/secrets/" + namespace + "/secret1",
		},
		gvr("", "v1", "replicationcontrollers"): {
			Stub:             `{"metadata": {"name": "rc1"}, "spec": {"selector": {"new": "stuff"}, "template": {"metadata": {"labels": {"new": "stuff"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container8"}]}}}}`,
			ExpectedEtcdPath: "/registry/controllers/" + namespace + "/rc1",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/apps/v1
		gvr("apps", "v1", "daemonsets"): {
			Stub:             `{"metadata": {"name": "ds6"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
			ExpectedEtcdPath: "/registry/daemonsets/" + namespace + "/ds6",
		},
		gvr("apps", "v1", "deployments"): {
			Stub:             `{"metadata": {"name": "deployment4"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
			ExpectedEtcdPath: "/registry/deployments/" + namespace + "/deployment4",
		},
		gvr("apps", "v1", "statefulsets"): {
			Stub:             `{"metadata": {"name": "ss3"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}}}}`,
			ExpectedEtcdPath: "/registry/statefulsets/" + namespace + "/ss3",
		},
		gvr("apps", "v1", "replicasets"): {
			Stub:             `{"metadata": {"name": "rs3"}, "spec": {"selector": {"matchLabels": {"g": "h"}}, "template": {"metadata": {"labels": {"g": "h"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container4"}]}}}}`,
			ExpectedEtcdPath: "/registry/replicasets/" + namespace + "/rs3",
		},
		gvr("apps", "v1", "controllerrevisions"): {
			Stub:             `{"metadata":{"name":"crs3"},"data":{"name":"abc","namespace":"default","creationTimestamp":null,"Spec":{"Replicas":0,"Selector":{"matchLabels":{"foo":"bar"}},"Template":{"creationTimestamp":null,"labels":{"foo":"bar"},"Spec":{"Volumes":null,"InitContainers":null,"Containers":null,"RestartPolicy":"Always","TerminationGracePeriodSeconds":null,"ActiveDeadlineSeconds":null,"DNSPolicy":"ClusterFirst","NodeSelector":null,"ServiceAccountName":"","AutomountServiceAccountToken":null,"NodeName":"","SecurityContext":null,"ImagePullSecrets":null,"Hostname":"","Subdomain":"","Affinity":null,"SchedulerName":"","Tolerations":null,"HostAliases":null}},"VolumeClaimTemplates":null,"ServiceName":""},"Status":{"ObservedGeneration":null,"Replicas":0}},"revision":0}`,
			ExpectedEtcdPath: "/registry/controllerrevisions/" + namespace + "/crs3",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/autoscaling/v1
		gvr("autoscaling", "v1", "horizontalpodautoscalers"): {
			Stub:             `{"metadata": {"name": "hpa2"}, "spec": {"maxReplicas": 3, "scaleTargetRef": {"kind": "something", "name": "cross"}}}`,
			ExpectedEtcdPath: "/registry/horizontalpodautoscalers/" + namespace + "/hpa2",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/autoscaling/v2beta1
		gvr("autoscaling", "v2beta1", "horizontalpodautoscalers"): {
			Stub:             `{"metadata": {"name": "hpa1"}, "spec": {"maxReplicas": 3, "scaleTargetRef": {"kind": "something", "name": "cross"}}}`,
			ExpectedEtcdPath: "/registry/horizontalpodautoscalers/" + namespace + "/hpa1",
			ExpectedGVK:      gvkP("autoscaling", "v1", "HorizontalPodAutoscaler"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/autoscaling/v2beta2
		gvr("autoscaling", "v2beta2", "horizontalpodautoscalers"): {
			Stub:             `{"metadata": {"name": "hpa3"}, "spec": {"maxReplicas": 3, "scaleTargetRef": {"kind": "something", "name": "cross"}}}`,
			ExpectedEtcdPath: "/registry/horizontalpodautoscalers/" + namespace + "/hpa3",
			ExpectedGVK:      gvkP("autoscaling", "v1", "HorizontalPodAutoscaler"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/batch/v1
		gvr("batch", "v1", "jobs"): {
			Stub:             `{"metadata": {"name": "job1"}, "spec": {"manualSelector": true, "selector": {"matchLabels": {"controller-uid": "uid1"}}, "template": {"metadata": {"labels": {"controller-uid": "uid1"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container1"}], "dnsPolicy": "ClusterFirst", "restartPolicy": "Never"}}}}`,
			ExpectedEtcdPath: "/registry/jobs/" + namespace + "/job1",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/batch/v1beta1
		gvr("batch", "v1beta1", "cronjobs"): {
			Stub:             `{"metadata": {"name": "cjv1beta1"}, "spec": {"jobTemplate": {"spec": {"template": {"metadata": {"labels": {"controller-uid": "uid0"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container0"}], "dnsPolicy": "ClusterFirst", "restartPolicy": "Never"}}}}, "schedule": "* * * * *"}}`,
			ExpectedEtcdPath: "/registry/cronjobs/" + namespace + "/cjv1beta1",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/batch/v2alpha1
		gvr("batch", "v2alpha1", "cronjobs"): {
			Stub:             `{"metadata": {"name": "cjv2alpha1"}, "spec": {"jobTemplate": {"spec": {"template": {"metadata": {"labels": {"controller-uid": "uid0"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container0"}], "dnsPolicy": "ClusterFirst", "restartPolicy": "Never"}}}}, "schedule": "* * * * *"}}`,
			ExpectedEtcdPath: "/registry/cronjobs/" + namespace + "/cjv2alpha1",
			ExpectedGVK:      gvkP("batch", "v1beta1", "CronJob"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/certificates/v1beta1
		gvr("certificates.k8s.io", "v1beta1", "certificatesigningrequests"): {
			Stub:             `{"metadata": {"name": "csr1"}, "spec": {"request": "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0KTUlJQnlqQ0NBVE1DQVFBd2dZa3hDekFKQmdOVkJBWVRBbFZUTVJNd0VRWURWUVFJRXdwRFlXeHBabTl5Ym1saApNUll3RkFZRFZRUUhFdzFOYjNWdWRHRnBiaUJXYVdWM01STXdFUVlEVlFRS0V3cEhiMjluYkdVZ1NXNWpNUjh3CkhRWURWUVFMRXhaSmJtWnZjbTFoZEdsdmJpQlVaV05vYm05c2IyZDVNUmN3RlFZRFZRUURFdzUzZDNjdVoyOXYKWjJ4bExtTnZiVENCbnpBTkJna3Foa2lHOXcwQkFRRUZBQU9CalFBd2dZa0NnWUVBcFp0WUpDSEo0VnBWWEhmVgpJbHN0UVRsTzRxQzAzaGpYK1prUHl2ZFlkMVE0K3FiQWVUd1htQ1VLWUhUaFZSZDVhWFNxbFB6eUlCd2llTVpyCldGbFJRZGRaMUl6WEFsVlJEV3dBbzYwS2VjcWVBWG5uVUsrNWZYb1RJL1VnV3NocmU4dEoreC9UTUhhUUtSL0oKY0lXUGhxYVFoc0p1elpidkFkR0E4MEJMeGRNQ0F3RUFBYUFBTUEwR0NTcUdTSWIzRFFFQkJRVUFBNEdCQUlobAo0UHZGcStlN2lwQVJnSTVaTStHWng2bXBDejQ0RFRvMEprd2ZSRGYrQnRyc2FDMHE2OGVUZjJYaFlPc3E0ZmtIClEwdUEwYVZvZzNmNWlKeENhM0hwNWd4YkpRNnpWNmtKMFRFc3VhYU9oRWtvOXNkcENvUE9uUkJtMmkvWFJEMkQKNmlOaDhmOHowU2hHc0ZxakRnRkh5RjNvK2xVeWorVUM2SDFRVzdibgotLS0tLUVORCBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0="}}`,
			ExpectedEtcdPath: "/registry/certificatesigningrequests/csr1",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/coordination/v1
		gvr("coordination.k8s.io", "v1", "leases"): {
			Stub:             `{"metadata": {"name": "leasev1"}, "spec": {"holderIdentity": "holder", "leaseDurationSeconds": 5}}`,
			ExpectedEtcdPath: "/registry/leases/" + namespace + "/leasev1",
			ExpectedGVK:      gvkP("coordination.k8s.io", "v1beta1", "Lease"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/coordination/v1beta1
		gvr("coordination.k8s.io", "v1beta1", "leases"): {
			Stub:             `{"metadata": {"name": "leasev1beta1"}, "spec": {"holderIdentity": "holder", "leaseDurationSeconds": 5}}`,
			ExpectedEtcdPath: "/registry/leases/" + namespace + "/leasev1beta1",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/discovery/v1beta1
		gvr("discovery.k8s.io", "v1beta1", "endpointslices"): {
			Stub:             `{"metadata": {"name": "slicev1beta1"}, "addressType": "IPv4", "protocol": "TCP", "ports": [], "endpoints": []}`,
			ExpectedEtcdPath: "/registry/endpointslices/" + namespace + "/slicev1beta1",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/events/v1beta1
		gvr("events.k8s.io", "v1beta1", "events"): {
			Stub:             `{"metadata": {"name": "event2"}, "regarding": {"namespace": "` + namespace + `"}, "note": "some data here", "eventTime": "2017-08-09T15:04:05.000000Z", "reportingInstance": "node-xyz", "reportingController": "k8s.io/my-controller", "action": "DidNothing", "reason": "Laziness"}`,
			ExpectedEtcdPath: "/registry/events/" + namespace + "/event2",
			ExpectedGVK:      gvkP("", "v1", "Event"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/extensions/v1beta1
		gvr("extensions", "v1beta1", "ingresses"): {
			Stub:             `{"metadata": {"name": "ingress1"}, "spec": {"backend": {"serviceName": "service", "servicePort": 5000}}}`,
			ExpectedEtcdPath: "/registry/ingress/" + namespace + "/ingress1",
			ExpectedGVK:      gvkP("networking.k8s.io", "v1beta1", "Ingress"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/networking/v1beta1
		gvr("networking.k8s.io", "v1beta1", "ingresses"): {
			Stub:             `{"metadata": {"name": "ingress2"}, "spec": {"backend": {"serviceName": "service", "servicePort": 5000}}}`,
			ExpectedEtcdPath: "/registry/ingress/" + namespace + "/ingress2",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/networking/v1
		gvr("networking.k8s.io", "v1", "networkpolicies"): {
			Stub:             `{"metadata": {"name": "np2"}, "spec": {"podSelector": {"matchLabels": {"e": "f"}}}}`,
			ExpectedEtcdPath: "/registry/networkpolicies/" + namespace + "/np2",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/policy/v1beta1
		gvr("policy", "v1beta1", "poddisruptionbudgets"): {
			Stub:             `{"metadata": {"name": "pdb1"}, "spec": {"selector": {"matchLabels": {"anokkey": "anokvalue"}}}}`,
			ExpectedEtcdPath: "/registry/poddisruptionbudgets/" + namespace + "/pdb1",
		},
		gvr("policy", "v1beta1", "podsecuritypolicies"): {
			Stub:             `{"metadata": {"name": "psp2"}, "spec": {"fsGroup": {"rule": "RunAsAny"}, "privileged": true, "runAsUser": {"rule": "RunAsAny"}, "seLinux": {"rule": "MustRunAs"}, "supplementalGroups": {"rule": "RunAsAny"}}}`,
			ExpectedEtcdPath: "/registry/podsecuritypolicy/psp2",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/storage/v1alpha1
		gvr("storage.k8s.io", "v1alpha1", "volumeattachments"): {
			Stub:             `{"metadata": {"name": "va1"}, "spec": {"attacher": "gce", "nodeName": "localhost", "source": {"persistentVolumeName": "pv1"}}}`,
			ExpectedEtcdPath: "/registry/volumeattachments/va1",
			ExpectedGVK:      gvkP("storage.k8s.io", "v1", "VolumeAttachment"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/flowcontrol/v1alpha1
		gvr("flowcontrol.apiserver.k8s.io", "v1alpha1", "flowschemas"): {
			Stub:             `{"metadata": {"name": "va1"}, "spec": {"priorityLevelConfiguration": {"name": "name1"}}}`,
			ExpectedEtcdPath: "/registry/flowschemas/va1",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/flowcontrol/v1alpha1
		gvr("flowcontrol.apiserver.k8s.io", "v1alpha1", "prioritylevelconfigurations"): {
			Stub:             `{"metadata": {"name": "conf1"}, "spec": {"type": "Limited", "limited": {"assuredConcurrencyShares":3, "limitResponse": {"type": "Reject"}}}}`,
			ExpectedEtcdPath: "/registry/prioritylevelconfigurations/conf1",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/storage/v1beta1
		gvr("storage.k8s.io", "v1beta1", "volumeattachments"): {
			Stub:             `{"metadata": {"name": "va2"}, "spec": {"attacher": "gce", "nodeName": "localhost", "source": {"persistentVolumeName": "pv2"}}}`,
			ExpectedEtcdPath: "/registry/volumeattachments/va2",
			ExpectedGVK:      gvkP("storage.k8s.io", "v1", "VolumeAttachment"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/storage/v1
		gvr("storage.k8s.io", "v1", "volumeattachments"): {
			Stub:             `{"metadata": {"name": "va3"}, "spec": {"attacher": "gce", "nodeName": "localhost", "source": {"persistentVolumeName": "pv3"}}}`,
			ExpectedEtcdPath: "/registry/volumeattachments/va3",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/storage/v1beta1
		gvr("storage.k8s.io", "v1beta1", "storageclasses"): {
			Stub:             `{"metadata": {"name": "sc1"}, "provisioner": "aws"}`,
			ExpectedEtcdPath: "/registry/storageclasses/sc1",
			ExpectedGVK:      gvkP("storage.k8s.io", "v1", "StorageClass"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/storage/v1
		gvr("storage.k8s.io", "v1", "storageclasses"): {
			Stub:             `{"metadata": {"name": "sc2"}, "provisioner": "aws"}`,
			ExpectedEtcdPath: "/registry/storageclasses/sc2",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/settings/v1alpha1
		gvr("settings.k8s.io", "v1alpha1", "podpresets"): {
			Stub:             `{"metadata": {"name": "podpre1"}, "spec": {"env": [{"name": "FOO"}]}}`,
			ExpectedEtcdPath: "/registry/podpresets/" + namespace + "/podpre1",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/rbac/v1alpha1
		gvr("rbac.authorization.k8s.io", "v1alpha1", "roles"): {
			Stub:             `{"metadata": {"name": "role1"}, "rules": [{"apiGroups": ["v1"], "resources": ["events"], "verbs": ["watch"]}]}`,
			ExpectedEtcdPath: "/registry/roles/" + namespace + "/role1",
			ExpectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "Role"),
		},
		gvr("rbac.authorization.k8s.io", "v1alpha1", "clusterroles"): {
			Stub:             `{"metadata": {"name": "crole1"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
			ExpectedEtcdPath: "/registry/clusterroles/crole1",
			ExpectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "ClusterRole"),
		},
		gvr("rbac.authorization.k8s.io", "v1alpha1", "rolebindings"): {
			Stub:             `{"metadata": {"name": "roleb1"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
			ExpectedEtcdPath: "/registry/rolebindings/" + namespace + "/roleb1",
			ExpectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "RoleBinding"),
		},
		gvr("rbac.authorization.k8s.io", "v1alpha1", "clusterrolebindings"): {
			Stub:             `{"metadata": {"name": "croleb1"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
			ExpectedEtcdPath: "/registry/clusterrolebindings/croleb1",
			ExpectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "ClusterRoleBinding"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/rbac/v1beta1
		gvr("rbac.authorization.k8s.io", "v1beta1", "roles"): {
			Stub:             `{"metadata": {"name": "role2"}, "rules": [{"apiGroups": ["v1"], "resources": ["events"], "verbs": ["watch"]}]}`,
			ExpectedEtcdPath: "/registry/roles/" + namespace + "/role2",
			ExpectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "Role"),
		},
		gvr("rbac.authorization.k8s.io", "v1beta1", "clusterroles"): {
			Stub:             `{"metadata": {"name": "crole2"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
			ExpectedEtcdPath: "/registry/clusterroles/crole2",
			ExpectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "ClusterRole"),
		},
		gvr("rbac.authorization.k8s.io", "v1beta1", "rolebindings"): {
			Stub:             `{"metadata": {"name": "roleb2"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
			ExpectedEtcdPath: "/registry/rolebindings/" + namespace + "/roleb2",
			ExpectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "RoleBinding"),
		},
		gvr("rbac.authorization.k8s.io", "v1beta1", "clusterrolebindings"): {
			Stub:             `{"metadata": {"name": "croleb2"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
			ExpectedEtcdPath: "/registry/clusterrolebindings/croleb2",
			ExpectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "ClusterRoleBinding"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/rbac/v1
		gvr("rbac.authorization.k8s.io", "v1", "roles"): {
			Stub:             `{"metadata": {"name": "role3"}, "rules": [{"apiGroups": ["v1"], "resources": ["events"], "verbs": ["watch"]}]}`,
			ExpectedEtcdPath: "/registry/roles/" + namespace + "/role3",
		},
		gvr("rbac.authorization.k8s.io", "v1", "clusterroles"): {
			Stub:             `{"metadata": {"name": "crole3"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
			ExpectedEtcdPath: "/registry/clusterroles/crole3",
		},
		gvr("rbac.authorization.k8s.io", "v1", "rolebindings"): {
			Stub:             `{"metadata": {"name": "roleb3"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
			ExpectedEtcdPath: "/registry/rolebindings/" + namespace + "/roleb3",
		},
		gvr("rbac.authorization.k8s.io", "v1", "clusterrolebindings"): {
			Stub:             `{"metadata": {"name": "croleb3"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
			ExpectedEtcdPath: "/registry/clusterrolebindings/croleb3",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/admissionregistration/v1
		gvr("admissionregistration.k8s.io", "v1", "validatingwebhookconfigurations"): {
			Stub:             `{"metadata":{"name":"hook2","creationTimestamp":null},"webhooks":[{"name":"externaladmissionhook.k8s.io","clientConfig":{"service":{"namespace":"ns","name":"n"},"caBundle":null},"rules":[{"operations":["CREATE"],"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore","sideEffects":"None","admissionReviewVersions":["v1beta1"]}]}`,
			ExpectedEtcdPath: "/registry/validatingwebhookconfigurations/hook2",
			ExpectedGVK:      gvkP("admissionregistration.k8s.io", "v1beta1", "ValidatingWebhookConfiguration"),
		},
		gvr("admissionregistration.k8s.io", "v1", "mutatingwebhookconfigurations"): {
			Stub:             `{"metadata":{"name":"hook2","creationTimestamp":null},"webhooks":[{"name":"externaladmissionhook.k8s.io","clientConfig":{"service":{"namespace":"ns","name":"n"},"caBundle":null},"rules":[{"operations":["CREATE"],"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore","sideEffects":"None","admissionReviewVersions":["v1beta1"]}]}`,
			ExpectedEtcdPath: "/registry/mutatingwebhookconfigurations/hook2",
			ExpectedGVK:      gvkP("admissionregistration.k8s.io", "v1beta1", "MutatingWebhookConfiguration"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/admissionregistration/v1beta1
		gvr("admissionregistration.k8s.io", "v1beta1", "validatingwebhookconfigurations"): {
			Stub:             `{"metadata":{"name":"hook1","creationTimestamp":null},"webhooks":[{"name":"externaladmissionhook.k8s.io","clientConfig":{"service":{"namespace":"ns","name":"n"},"caBundle":null},"rules":[{"operations":["CREATE"],"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore"}]}`,
			ExpectedEtcdPath: "/registry/validatingwebhookconfigurations/hook1",
		},
		gvr("admissionregistration.k8s.io", "v1beta1", "mutatingwebhookconfigurations"): {
			Stub:             `{"metadata":{"name":"hook1","creationTimestamp":null},"webhooks":[{"name":"externaladmissionhook.k8s.io","clientConfig":{"service":{"namespace":"ns","name":"n"},"caBundle":null},"rules":[{"operations":["CREATE"],"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore"}]}`,
			ExpectedEtcdPath: "/registry/mutatingwebhookconfigurations/hook1",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/scheduling/v1alpha1
		gvr("scheduling.k8s.io", "v1alpha1", "priorityclasses"): {
			Stub:             `{"metadata":{"name":"pc1"},"Value":1000}`,
			ExpectedEtcdPath: "/registry/priorityclasses/pc1",
			ExpectedGVK:      gvkP("scheduling.k8s.io", "v1", "PriorityClass"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/scheduling/v1beta1
		gvr("scheduling.k8s.io", "v1beta1", "priorityclasses"): {
			Stub:             `{"metadata":{"name":"pc2"},"Value":1000}`,
			ExpectedEtcdPath: "/registry/priorityclasses/pc2",
			ExpectedGVK:      gvkP("scheduling.k8s.io", "v1", "PriorityClass"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/scheduling/v1
		gvr("scheduling.k8s.io", "v1", "priorityclasses"): {
			Stub:             `{"metadata":{"name":"pc3"},"Value":1000}`,
			ExpectedEtcdPath: "/registry/priorityclasses/pc3",
		},
		// --

		// k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1
		// depends on aggregator using the same ungrouped RESTOptionsGetter as the kube apiserver, not SimpleRestOptionsFactory in aggregator.go
		gvr("apiregistration.k8s.io", "v1beta1", "apiservices"): {
			Stub:             `{"metadata": {"name": "as1.foo.com"}, "spec": {"group": "foo.com", "version": "as1", "groupPriorityMinimum":100, "versionPriority":10}}`,
			ExpectedEtcdPath: "/registry/apiregistration.k8s.io/apiservices/as1.foo.com",
		},
		// --

		// k8s.io/kube-aggregator/pkg/apis/apiregistration/v1
		// depends on aggregator using the same ungrouped RESTOptionsGetter as the kube apiserver, not SimpleRestOptionsFactory in aggregator.go
		gvr("apiregistration.k8s.io", "v1", "apiservices"): {
			Stub:             `{"metadata": {"name": "as2.foo.com"}, "spec": {"group": "foo.com", "version": "as2", "groupPriorityMinimum":100, "versionPriority":10}}`,
			ExpectedEtcdPath: "/registry/apiregistration.k8s.io/apiservices/as2.foo.com",
			ExpectedGVK:      gvkP("apiregistration.k8s.io", "v1beta1", "APIService"),
		},
		// --

		// k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1
		gvr("apiextensions.k8s.io", "v1", "customresourcedefinitions"): {
			Stub: `{"metadata": {"name": "openshiftwebconsoleconfigs.webconsole2.operator.openshift.io"},"spec": {` +
				`"scope": "Cluster","group": "webconsole2.operator.openshift.io",` +
				`"versions": [{"name":"v1alpha1","storage":true,"served":true,"schema":{"openAPIV3Schema":{"type":"object"}}}],` +
				`"names": {"kind": "OpenShiftWebConsoleConfig","plural": "openshiftwebconsoleconfigs","singular": "openshiftwebconsoleconfig"}}}`,
			ExpectedEtcdPath: "/registry/apiextensions.k8s.io/customresourcedefinitions/openshiftwebconsoleconfigs.webconsole2.operator.openshift.io",
			ExpectedGVK:      gvkP("apiextensions.k8s.io", "v1beta1", "CustomResourceDefinition"),
		},
		// k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1
		gvr("apiextensions.k8s.io", "v1beta1", "customresourcedefinitions"): {
			Stub:             `{"metadata": {"name": "openshiftwebconsoleconfigs.webconsole.operator.openshift.io"},"spec": {"scope": "Cluster","group": "webconsole.operator.openshift.io","version": "v1alpha1","names": {"kind": "OpenShiftWebConsoleConfig","plural": "openshiftwebconsoleconfigs","singular": "openshiftwebconsoleconfig"}}}`,
			ExpectedEtcdPath: "/registry/apiextensions.k8s.io/customresourcedefinitions/openshiftwebconsoleconfigs.webconsole.operator.openshift.io",
		},
		gvr("cr.bar.com", "v1", "foos"): {
			Stub:             `{"kind": "Foo", "apiVersion": "cr.bar.com/v1", "metadata": {"name": "cr1foo"}, "color": "blue"}`, // requires TypeMeta due to CRD scheme's UnstructuredObjectTyper
			ExpectedEtcdPath: "/registry/cr.bar.com/foos/" + namespace + "/cr1foo",
		},
		gvr("custom.fancy.com", "v2", "pants"): {
			Stub:             `{"kind": "Pant", "apiVersion": "custom.fancy.com/v2", "metadata": {"name": "cr2pant"}, "isFancy": true}`, // requires TypeMeta due to CRD scheme's UnstructuredObjectTyper
			ExpectedEtcdPath: "/registry/custom.fancy.com/pants/cr2pant",
		},
		gvr("awesome.bears.com", "v1", "pandas"): {
			Stub:             `{"kind": "Panda", "apiVersion": "awesome.bears.com/v1", "metadata": {"name": "cr3panda"}, "spec":{"replicas": 100}}`, // requires TypeMeta due to CRD scheme's UnstructuredObjectTyper
			ExpectedEtcdPath: "/registry/awesome.bears.com/pandas/cr3panda",
		},
		gvr("awesome.bears.com", "v3", "pandas"): {
			Stub:             `{"kind": "Panda", "apiVersion": "awesome.bears.com/v3", "metadata": {"name": "cr4panda"}, "spec":{"replicas": 300}}`, // requires TypeMeta due to CRD scheme's UnstructuredObjectTyper
			ExpectedEtcdPath: "/registry/awesome.bears.com/pandas/cr4panda",
			ExpectedGVK:      gvkP("awesome.bears.com", "v1", "Panda"),
		},
		gvr("random.numbers.com", "v1", "integers"): {
			Stub:             `{"kind": "Integer", "apiVersion": "random.numbers.com/v1", "metadata": {"name": "fortytwo"}, "value": 42, "garbage": "oiujnasdf"}`, // requires TypeMeta due to CRD scheme's UnstructuredObjectTyper
			ExpectedEtcdPath: "/registry/random.numbers.com/integers/fortytwo",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/auditregistration/v1alpha1
		gvr("auditregistration.k8s.io", "v1alpha1", "auditsinks"): {
			Stub:             `{"metadata":{"name":"sink1"},"spec":{"policy":{"level":"Metadata","stages":["ResponseStarted"]},"webhook":{"clientConfig":{"url":"http://localhost:4444","service":null,"caBundle":null}}}}`,
			ExpectedEtcdPath: "/registry/auditsinks/sink1",
		},
		// --

		// k8s.io/kubernetes/pkg/apis/node/v1alpha1
		gvr("node.k8s.io", "v1alpha1", "runtimeclasses"): {
			Stub:             `{"metadata": {"name": "rc1"}, "spec": {"runtimeHandler": "h1"}}`,
			ExpectedEtcdPath: "/registry/runtimeclasses/rc1",
			ExpectedGVK:      gvkP("node.k8s.io", "v1beta1", "RuntimeClass"),
		},
		// --

		// k8s.io/kubernetes/pkg/apis/node/v1beta1
		gvr("node.k8s.io", "v1beta1", "runtimeclasses"): {
			Stub:             `{"metadata": {"name": "rc2"}, "handler": "h2"}`,
			ExpectedEtcdPath: "/registry/runtimeclasses/rc2",
		},
		// --
	}

	// add csinodes if CSINodeInfo feature gate is enabled
	if utilfeature.DefaultFeatureGate.Enabled(features.CSINodeInfo) {
		// k8s.io/kubernetes/pkg/apis/storage/v1beta1
		etcdStorageData[gvr("storage.k8s.io", "v1beta1", "csinodes")] = StorageData{
			Stub:             `{"metadata": {"name": "csini1"}, "spec": {"drivers": [{"name": "test-driver", "nodeID": "localhost", "topologyKeys": ["company.com/zone1", "company.com/zone2"]}]}}`,
			ExpectedEtcdPath: "/registry/csinodes/csini1",
			ExpectedGVK:      gvkP("storage.k8s.io", "v1", "CSINode"),
		}

		// k8s.io/kubernetes/pkg/apis/storage/v1
		etcdStorageData[gvr("storage.k8s.io", "v1", "csinodes")] = StorageData{
			Stub:             `{"metadata": {"name": "csini2"}, "spec": {"drivers": [{"name": "test-driver", "nodeID": "localhost", "topologyKeys": ["company.com/zone1", "company.com/zone2"]}]}}`,
			ExpectedEtcdPath: "/registry/csinodes/csini2",
		}
	}

	// k8s.io/kubernetes/pkg/apis/storage/v1beta1
	// add csidrivers if CSIDriverRegistry feature gate is enabled
	if utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
		etcdStorageData[gvr("storage.k8s.io", "v1beta1", "csidrivers")] = StorageData{
			Stub:             `{"metadata": {"name": "csid1"}, "spec": {"attachRequired": true, "podInfoOnMount": true}}`,
			ExpectedEtcdPath: "/registry/csidrivers/csid1",
		}
	}

	return etcdStorageData
}

// StorageData contains information required to create an object and verify its storage in etcd
// It must be paired with a specific resource
type StorageData struct {
	Stub             string                   // Valid JSON stub to use during create
	Prerequisites    []Prerequisite           // Optional, ordered list of JSON objects to create before stub
	ExpectedEtcdPath string                   // Expected location of object in etcd, do not use any variables, constants, etc to derive this value - always supply the full raw string
	ExpectedGVK      *schema.GroupVersionKind // The GVK that we expect this object to be stored as - leave this nil to use the default
}

// Prerequisite contains information required to create a resource (but not verify it)
type Prerequisite struct {
	GvrData schema.GroupVersionResource
	Stub    string
}

// GetCustomResourceDefinitionData returns the resource definitions that back the custom resources
// included in GetEtcdStorageData.  They should be created using CreateTestCRDs before running any tests.
func GetCustomResourceDefinitionData() []*apiextensionsv1beta1.CustomResourceDefinition {
	return []*apiextensionsv1beta1.CustomResourceDefinition{
		// namespaced with legacy version field
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foos.cr.bar.com",
			},
			Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
				Group:   "cr.bar.com",
				Version: "v1",
				Scope:   apiextensionsv1beta1.NamespaceScoped,
				Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
					Plural: "foos",
					Kind:   "Foo",
				},
			},
		},
		// cluster scoped with legacy version field
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pants.custom.fancy.com",
			},
			Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
				Group:   "custom.fancy.com",
				Version: "v2",
				Scope:   apiextensionsv1beta1.ClusterScoped,
				Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
					Plural: "pants",
					Kind:   "Pant",
				},
			},
		},
		// cluster scoped with legacy version field and pruning.
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "integers.random.numbers.com",
			},
			Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
				Group:   "random.numbers.com",
				Version: "v1",
				Scope:   apiextensionsv1beta1.ClusterScoped,
				Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
					Plural: "integers",
					Kind:   "Integer",
				},
				Validation: &apiextensionsv1beta1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1beta1.JSONSchemaProps{
						Type: "object",
						Properties: map[string]apiextensionsv1beta1.JSONSchemaProps{
							"value": {
								Type: "number",
							},
						},
					},
				},
				PreserveUnknownFields: pointer.BoolPtr(false),
			},
		},
		// cluster scoped with versions field
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pandas.awesome.bears.com",
			},
			Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
				Group: "awesome.bears.com",
				Versions: []apiextensionsv1beta1.CustomResourceDefinitionVersion{
					{
						Name:    "v1",
						Served:  true,
						Storage: true,
					},
					{
						Name:    "v2",
						Served:  false,
						Storage: false,
					},
					{
						Name:    "v3",
						Served:  true,
						Storage: false,
					},
				},
				Scope: apiextensionsv1beta1.ClusterScoped,
				Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
					Plural: "pandas",
					Kind:   "Panda",
				},
				Subresources: &apiextensionsv1beta1.CustomResourceSubresources{
					Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
					Scale: &apiextensionsv1beta1.CustomResourceSubresourceScale{
						SpecReplicasPath:   ".spec.replicas",
						StatusReplicasPath: ".status.replicas",
						LabelSelectorPath:  func() *string { path := ".status.selector"; return &path }(),
					},
				},
			},
		},
	}
}

func gvr(g, v, r string) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: g, Version: v, Resource: r}
}

func gvkP(g, v, k string) *schema.GroupVersionKind {
	return &schema.GroupVersionKind{Group: g, Version: v, Kind: k}
}
