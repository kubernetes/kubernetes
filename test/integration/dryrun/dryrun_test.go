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

package dryrun

import (
	"encoding/json"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/features"
	genericapiserveroptions "k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
	cacheddiscovery "k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"

	// install all APIs
	_ "k8s.io/kubernetes/pkg/master" // TODO what else is needed
)

// dryrun data for all persisted objects.
var dryrunData = map[schema.GroupVersionResource]struct {
	stub string // Valid JSON stub to use during create
}{
	// k8s.io/kubernetes/pkg/api/v1
	gvr("", "v1", "configmaps"): {
		stub: `{"data": {"foo": "bar"}, "metadata": {"name": "cm1"}}`,
	},
	gvr("", "v1", "services"): {
		stub: `{"metadata": {"name": "service1"}, "spec": {"externalName": "service1name", "ports": [{"port": 10000, "targetPort": 11000}], "selector": {"test": "data"}}}`,
	},
	gvr("", "v1", "podtemplates"): {
		stub: `{"metadata": {"name": "pt1name"}, "template": {"metadata": {"labels": {"pt": "01"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container9"}]}}}`,
	},
	gvr("", "v1", "pods"): {
		stub: `{"metadata": {"name": "pod1"}, "spec": {"containers": [{"image": "fedora:latest", "name": "container7", "resources": {"limits": {"cpu": "1M"}, "requests": {"cpu": "1M"}}}]}}`,
	},
	gvr("", "v1", "endpoints"): {
		stub: `{"metadata": {"name": "ep1name"}, "subsets": [{"addresses": [{"hostname": "bar-001", "ip": "192.168.3.1"}], "ports": [{"port": 8000}]}]}`,
	},
	gvr("", "v1", "resourcequotas"): {
		stub: `{"metadata": {"name": "rq1name"}, "spec": {"hard": {"cpu": "5M"}}}`,
	},
	gvr("", "v1", "limitranges"): {
		stub: `{"metadata": {"name": "lr1name"}, "spec": {"limits": [{"type": "Pod"}]}}`,
	},
	gvr("", "v1", "namespaces"): {
		stub: `{"metadata": {"name": "namespace2"}, "spec": {"finalizers": ["kubernetes"]}}`,
	},
	gvr("", "v1", "nodes"): {
		stub: `{"metadata": {"name": "node1"}, "spec": {"unschedulable": true}}`,
	},
	gvr("", "v1", "persistentvolumes"): {
		stub: `{"metadata": {"name": "pv1name"}, "spec": {"accessModes": ["ReadWriteOnce"], "capacity": {"storage": "3M"}, "hostPath": {"path": "/tmp/test/"}}}`,
	},
	gvr("", "v1", "events"): {
		stub: `{"involvedObject": {"namespace": "dryrunnamespace"}, "message": "some data here", "metadata": {"name": "event1"}}`,
	},
	gvr("", "v1", "persistentvolumeclaims"): {
		stub: `{"metadata": {"name": "pvc1"}, "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"limits": {"storage": "1M"}, "requests": {"storage": "2M"}}, "selector": {"matchLabels": {"pvc": "stuff"}}}}`,
	},
	gvr("", "v1", "serviceaccounts"): {
		stub: `{"metadata": {"name": "sa1name"}, "secrets": [{"name": "secret00"}]}`,
	},
	gvr("", "v1", "secrets"): {
		stub: `{"data": {"key": "ZGF0YSBmaWxl"}, "metadata": {"name": "secret1"}}`,
	},
	gvr("", "v1", "replicationcontrollers"): {
		stub: `{"metadata": {"name": "rc1"}, "spec": {"selector": {"new": "stuff"}, "template": {"metadata": {"labels": {"new": "stuff"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container8"}]}}}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/apps/v1beta1
	gvr("apps", "v1beta1", "statefulsets"): {
		stub: `{"metadata": {"name": "ss1"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}}}}`,
	},
	gvr("apps", "v1beta1", "deployments"): {
		stub: `{"metadata": {"name": "deployment2"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
	},
	gvr("apps", "v1beta1", "controllerrevisions"): {
		stub: `{"metadata":{"name":"crs1"},"data":{"name":"abc","namespace":"default","creationTimestamp":null,"Spec":{"Replicas":0,"Selector":{"matchLabels":{"foo":"bar"}},"Template":{"creationTimestamp":null,"labels":{"foo":"bar"},"Spec":{"Volumes":null,"InitContainers":null,"Containers":null,"RestartPolicy":"Always","TerminationGracePeriodSeconds":null,"ActiveDeadlineSeconds":null,"DNSPolicy":"ClusterFirst","NodeSelector":null,"ServiceAccountName":"","AutomountServiceAccountToken":null,"NodeName":"","SecurityContext":null,"ImagePullSecrets":null,"Hostname":"","Subdomain":"","Affinity":null,"SchedulerName":"","Tolerations":null,"HostAliases":null}},"VolumeClaimTemplates":null,"ServiceName":""},"Status":{"ObservedGeneration":null,"Replicas":0}},"revision":0}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/apps/v1beta2
	gvr("apps", "v1beta2", "statefulsets"): {
		stub: `{"metadata": {"name": "ss2"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}}}}`,
	},
	gvr("apps", "v1beta2", "deployments"): {
		stub: `{"metadata": {"name": "deployment3"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
	},
	gvr("apps", "v1beta2", "daemonsets"): {
		stub: `{"metadata": {"name": "ds5"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
	},
	gvr("apps", "v1beta2", "replicasets"): {
		stub: `{"metadata": {"name": "rs2"}, "spec": {"selector": {"matchLabels": {"g": "h"}}, "template": {"metadata": {"labels": {"g": "h"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container4"}]}}}}`,
	},
	gvr("apps", "v1beta2", "controllerrevisions"): {
		stub: `{"metadata":{"name":"crs2"},"data":{"name":"abc","namespace":"default","creationTimestamp":null,"Spec":{"Replicas":0,"Selector":{"matchLabels":{"foo":"bar"}},"Template":{"creationTimestamp":null,"labels":{"foo":"bar"},"Spec":{"Volumes":null,"InitContainers":null,"Containers":null,"RestartPolicy":"Always","TerminationGracePeriodSeconds":null,"ActiveDeadlineSeconds":null,"DNSPolicy":"ClusterFirst","NodeSelector":null,"ServiceAccountName":"","AutomountServiceAccountToken":null,"NodeName":"","SecurityContext":null,"ImagePullSecrets":null,"Hostname":"","Subdomain":"","Affinity":null,"SchedulerName":"","Tolerations":null,"HostAliases":null}},"VolumeClaimTemplates":null,"ServiceName":""},"Status":{"ObservedGeneration":null,"Replicas":0}},"revision":0}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/apps/v1
	gvr("apps", "v1", "daemonsets"): {
		stub: `{"metadata": {"name": "ds6"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
	},
	gvr("apps", "v1", "deployments"): {
		stub: `{"metadata": {"name": "deployment4"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
	},
	gvr("apps", "v1", "statefulsets"): {
		stub: `{"metadata": {"name": "ss3"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}}}}`,
	},
	gvr("apps", "v1", "replicasets"): {
		stub: `{"metadata": {"name": "rs3"}, "spec": {"selector": {"matchLabels": {"g": "h"}}, "template": {"metadata": {"labels": {"g": "h"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container4"}]}}}}`,
	},
	gvr("apps", "v1", "controllerrevisions"): {
		stub: `{"metadata":{"name":"crs3"},"data":{"name":"abc","namespace":"default","creationTimestamp":null,"Spec":{"Replicas":0,"Selector":{"matchLabels":{"foo":"bar"}},"Template":{"creationTimestamp":null,"labels":{"foo":"bar"},"Spec":{"Volumes":null,"InitContainers":null,"Containers":null,"RestartPolicy":"Always","TerminationGracePeriodSeconds":null,"ActiveDeadlineSeconds":null,"DNSPolicy":"ClusterFirst","NodeSelector":null,"ServiceAccountName":"","AutomountServiceAccountToken":null,"NodeName":"","SecurityContext":null,"ImagePullSecrets":null,"Hostname":"","Subdomain":"","Affinity":null,"SchedulerName":"","Tolerations":null,"HostAliases":null}},"VolumeClaimTemplates":null,"ServiceName":""},"Status":{"ObservedGeneration":null,"Replicas":0}},"revision":0}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/autoscaling/v1
	gvr("autoscaling", "v1", "horizontalpodautoscalers"): {
		stub: `{"metadata": {"name": "hpa2"}, "spec": {"maxReplicas": 3, "scaleTargetRef": {"kind": "something", "name": "cross"}}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/autoscaling/v2beta1
	gvr("autoscaling", "v2beta1", "horizontalpodautoscalers"): {
		stub: `{"metadata": {"name": "hpa1"}, "spec": {"maxReplicas": 3, "scaleTargetRef": {"kind": "something", "name": "cross"}}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/autoscaling/v2beta2
	gvr("autoscaling", "v2beta2", "horizontalpodautoscalers"): {
		stub: `{"metadata": {"name": "hpa3"}, "spec": {"maxReplicas": 3, "scaleTargetRef": {"kind": "something", "name": "cross"}}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/batch/v1
	gvr("batch", "v1", "jobs"): {
		stub: `{"metadata": {"name": "job1"}, "spec": {"manualSelector": true, "selector": {"matchLabels": {"controller-uid": "uid1"}}, "template": {"metadata": {"labels": {"controller-uid": "uid1"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container1"}], "dnsPolicy": "ClusterFirst", "restartPolicy": "Never"}}}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/batch/v1beta1
	gvr("batch", "v1beta1", "cronjobs"): {
		stub: `{"metadata": {"name": "cjv1beta1"}, "spec": {"jobTemplate": {"spec": {"template": {"metadata": {"labels": {"controller-uid": "uid0"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container0"}], "dnsPolicy": "ClusterFirst", "restartPolicy": "Never"}}}}, "schedule": "* * * * *"}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/batch/v2alpha1
	gvr("batch", "v2alpha1", "cronjobs"): {
		stub: `{"metadata": {"name": "cjv2alpha1"}, "spec": {"jobTemplate": {"spec": {"template": {"metadata": {"labels": {"controller-uid": "uid0"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container0"}], "dnsPolicy": "ClusterFirst", "restartPolicy": "Never"}}}}, "schedule": "* * * * *"}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/certificates/v1beta1
	gvr("certificates.k8s.io", "v1beta1", "certificatesigningrequests"): {
		stub: `{"metadata": {"name": "csr1"}, "spec": {"request": "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0KTUlJQnlqQ0NBVE1DQVFBd2dZa3hDekFKQmdOVkJBWVRBbFZUTVJNd0VRWURWUVFJRXdwRFlXeHBabTl5Ym1saApNUll3RkFZRFZRUUhFdzFOYjNWdWRHRnBiaUJXYVdWM01STXdFUVlEVlFRS0V3cEhiMjluYkdVZ1NXNWpNUjh3CkhRWURWUVFMRXhaSmJtWnZjbTFoZEdsdmJpQlVaV05vYm05c2IyZDVNUmN3RlFZRFZRUURFdzUzZDNjdVoyOXYKWjJ4bExtTnZiVENCbnpBTkJna3Foa2lHOXcwQkFRRUZBQU9CalFBd2dZa0NnWUVBcFp0WUpDSEo0VnBWWEhmVgpJbHN0UVRsTzRxQzAzaGpYK1prUHl2ZFlkMVE0K3FiQWVUd1htQ1VLWUhUaFZSZDVhWFNxbFB6eUlCd2llTVpyCldGbFJRZGRaMUl6WEFsVlJEV3dBbzYwS2VjcWVBWG5uVUsrNWZYb1RJL1VnV3NocmU4dEoreC9UTUhhUUtSL0oKY0lXUGhxYVFoc0p1elpidkFkR0E4MEJMeGRNQ0F3RUFBYUFBTUEwR0NTcUdTSWIzRFFFQkJRVUFBNEdCQUlobAo0UHZGcStlN2lwQVJnSTVaTStHWng2bXBDejQ0RFRvMEprd2ZSRGYrQnRyc2FDMHE2OGVUZjJYaFlPc3E0ZmtIClEwdUEwYVZvZzNmNWlKeENhM0hwNWd4YkpRNnpWNmtKMFRFc3VhYU9oRWtvOXNkcENvUE9uUkJtMmkvWFJEMkQKNmlOaDhmOHowU2hHc0ZxakRnRkh5RjNvK2xVeWorVUM2SDFRVzdibgotLS0tLUVORCBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0="}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/coordination/v1beta1
	gvr("coordination.k8s.io", "v1beta1", "leases"): {
		stub: `{"metadata": {"name": "lease1"}, "spec": {"holderIdentity": "holder", "leaseDurationSeconds": 5}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/events/v1beta1
	gvr("events.k8s.io", "v1beta1", "events"): {
		stub: `{"metadata": {"name": "event2"}, "regarding": {"namespace": "dryrunnamespace"}, "note": "some data here", "eventTime": "2017-08-09T15:04:05.000000Z", "reportingInstance": "node-xyz", "reportingController": "k8s.io/my-controller", "action": "DidNothing", "reason": "Laziness"}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/extensions/v1beta1
	gvr("extensions", "v1beta1", "daemonsets"): {
		stub: `{"metadata": {"name": "ds1"}, "spec": {"selector": {"matchLabels": {"u": "t"}}, "template": {"metadata": {"labels": {"u": "t"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container5"}]}}}}`,
	},
	gvr("extensions", "v1beta1", "podsecuritypolicies"): {
		stub: `{"metadata": {"name": "psp1"}, "spec": {"fsGroup": {"rule": "RunAsAny"}, "privileged": true, "runAsUser": {"rule": "RunAsAny"}, "seLinux": {"rule": "MustRunAs"}, "supplementalGroups": {"rule": "RunAsAny"}}}`,
	},
	gvr("extensions", "v1beta1", "ingresses"): {
		stub: `{"metadata": {"name": "ingress1"}, "spec": {"backend": {"serviceName": "service", "servicePort": 5000}}}`,
	},
	gvr("extensions", "v1beta1", "networkpolicies"): {
		stub: `{"metadata": {"name": "np1"}, "spec": {"podSelector": {"matchLabels": {"e": "f"}}}}`,
	},
	gvr("extensions", "v1beta1", "deployments"): {
		stub: `{"metadata": {"name": "deployment1"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
	},
	gvr("extensions", "v1beta1", "replicasets"): {
		stub: `{"metadata": {"name": "rs1"}, "spec": {"selector": {"matchLabels": {"g": "h"}}, "template": {"metadata": {"labels": {"g": "h"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container4"}]}}}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/networking/v1
	gvr("networking.k8s.io", "v1", "networkpolicies"): {
		stub: `{"metadata": {"name": "np2"}, "spec": {"podSelector": {"matchLabels": {"e": "f"}}}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/policy/v1beta1
	gvr("policy", "v1beta1", "poddisruptionbudgets"): {
		stub: `{"metadata": {"name": "pdb1"}, "spec": {"selector": {"matchLabels": {"anokkey": "anokvalue"}}}}`,
	},
	gvr("policy", "v1beta1", "podsecuritypolicies"): {
		stub: `{"metadata": {"name": "psp2"}, "spec": {"fsGroup": {"rule": "RunAsAny"}, "privileged": true, "runAsUser": {"rule": "RunAsAny"}, "seLinux": {"rule": "MustRunAs"}, "supplementalGroups": {"rule": "RunAsAny"}}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/storage/v1alpha1
	gvr("storage.k8s.io", "v1alpha1", "volumeattachments"): {
		stub: `{"metadata": {"name": "va1"}, "spec": {"attacher": "gce", "nodeName": "localhost", "source": {"persistentVolumeName": "pv1"}}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/storage/v1beta1
	gvr("storage.k8s.io", "v1beta1", "volumeattachments"): {
		stub: `{"metadata": {"name": "va2"}, "spec": {"attacher": "gce", "nodeName": "localhost", "source": {"persistentVolumeName": "pv2"}}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/storage/v1beta1
	gvr("storage.k8s.io", "v1beta1", "storageclasses"): {
		stub: `{"metadata": {"name": "sc1"}, "provisioner": "aws"}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/storage/v1
	gvr("storage.k8s.io", "v1", "storageclasses"): {
		stub: `{"metadata": {"name": "sc2"}, "provisioner": "aws"}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/settings/v1alpha1
	gvr("settings.k8s.io", "v1alpha1", "podpresets"): {
		stub: `{"metadata": {"name": "podpre1"}, "spec": {"env": [{"name": "FOO"}]}}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/rbac/v1alpha1
	gvr("rbac.authorization.k8s.io", "v1alpha1", "roles"): {
		stub: `{"metadata": {"name": "role1"}, "rules": [{"apiGroups": ["v1"], "resources": ["events"], "verbs": ["watch"]}]}`,
	},
	gvr("rbac.authorization.k8s.io", "v1alpha1", "clusterroles"): {
		stub: `{"metadata": {"name": "drcrole1"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
	},
	gvr("rbac.authorization.k8s.io", "v1alpha1", "rolebindings"): {
		stub: `{"metadata": {"name": "drroleb1"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
	},
	gvr("rbac.authorization.k8s.io", "v1alpha1", "clusterrolebindings"): {
		stub: `{"metadata": {"name": "drcroleb1"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/rbac/v1beta1
	gvr("rbac.authorization.k8s.io", "v1beta1", "roles"): {
		stub: `{"metadata": {"name": "drrole2"}, "rules": [{"apiGroups": ["v1"], "resources": ["events"], "verbs": ["watch"]}]}`,
	},
	gvr("rbac.authorization.k8s.io", "v1beta1", "clusterroles"): {
		stub: `{"metadata": {"name": "drcrole2"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
	},
	gvr("rbac.authorization.k8s.io", "v1beta1", "rolebindings"): {
		stub: `{"metadata": {"name": "drroleb2"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
	},
	gvr("rbac.authorization.k8s.io", "v1beta1", "clusterrolebindings"): {
		stub: `{"metadata": {"name": "drcroleb2"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/rbac/v1
	gvr("rbac.authorization.k8s.io", "v1", "roles"): {
		stub: `{"metadata": {"name": "drrole3"}, "rules": [{"apiGroups": ["v1"], "resources": ["events"], "verbs": ["watch"]}]}`,
	},
	gvr("rbac.authorization.k8s.io", "v1", "clusterroles"): {
		stub: `{"metadata": {"name": "drcrole3"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
	},
	gvr("rbac.authorization.k8s.io", "v1", "rolebindings"): {
		stub: `{"metadata": {"name": "drroleb3"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
	},
	gvr("rbac.authorization.k8s.io", "v1", "clusterrolebindings"): {
		stub: `{"metadata": {"name": "drcroleb3"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/admissionregistration/v1alpha1
	gvr("admissionregistration.k8s.io", "v1alpha1", "initializerconfigurations"): {
		stub: `{"metadata":{"name":"ic1"},"initializers":[{"name":"initializer.k8s.io","rules":[{"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore"}]}`,
	},
	// k8s.io/kubernetes/pkg/apis/admissionregistration/v1beta1
	gvr("admissionregistration.k8s.io", "v1beta1", "validatingwebhookconfigurations"): {
		stub: `{"metadata":{"name":"hook1","creationTimestamp":null},"webhooks":[{"name":"externaladmissionhook.k8s.io","clientConfig":{"service":{"namespace":"ns","name":"n"},"caBundle":null},"rules":[{"operations":["CREATE"],"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore"}]}`,
	},
	gvr("admissionregistration.k8s.io", "v1beta1", "mutatingwebhookconfigurations"): {
		stub: `{"metadata":{"name":"hook1","creationTimestamp":null},"webhooks":[{"name":"externaladmissionhook.k8s.io","clientConfig":{"service":{"namespace":"ns","name":"n"},"caBundle":null},"rules":[{"operations":["CREATE"],"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore"}]}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/scheduling/v1alpha1
	gvr("scheduling.k8s.io", "v1alpha1", "priorityclasses"): {
		stub: `{"metadata":{"name":"pc1"},"Value":1000}`,
	},
	// --

	// k8s.io/kubernetes/pkg/apis/scheduling/v1beta1
	gvr("scheduling.k8s.io", "v1beta1", "priorityclasses"): {
		stub: `{"metadata":{"name":"pc2"},"Value":1000}`,
	},
	// --

	// k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1
	// depends on aggregator using the same ungrouped RESTOptionsGetter as the kube apiserver, not SimpleRestOptionsFactory in aggregator.go
	gvr("apiregistration.k8s.io", "v1beta1", "apiservices"): {
		stub: `{"metadata": {"name": "dras1.foo.com"}, "spec": {"group": "foo.com", "version": "dras1", "groupPriorityMinimum":100, "versionPriority":10}}`,
	},
	// --

	// k8s.io/kube-aggregator/pkg/apis/apiregistration/v1
	// depends on aggregator using the same ungrouped RESTOptionsGetter as the kube apiserver, not SimpleRestOptionsFactory in aggregator.go
	gvr("apiregistration.k8s.io", "v1", "apiservices"): {
		stub: `{"metadata": {"name": "dras2.foo.com"}, "spec": {"group": "foo.com", "version": "dras2", "groupPriorityMinimum":100, "versionPriority":10}}`,
	},
	// --

	// k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1
	gvr("apiextensions.k8s.io", "v1beta1", "customresourcedefinitions"): {
		stub: `{"metadata": {"name": "openshiftwebconsoleconfigs.webconsole.operator.openshift.io"},"spec": {"scope": "Cluster","group": "webconsole.operator.openshift.io","version": "v1alpha1","names": {"kind": "OpenShiftWebConsoleConfig","plural": "openshiftwebconsoleconfigs","singular": "openshiftwebconsoleconfig"}}}`,
	},
	// --

}

// Only add kinds to this list when this a virtual resource with get and create verbs that doesn't actually
// store into it's kind.  We've used this downstream for mappings before.
var kindWhiteList = sets.NewString()

// namespace used for all tests, do not change this
const testNamespace = "dryrunnamespace"

func DryRunCreateTest(t *testing.T, rsc dynamic.ResourceInterface, obj *unstructured.Unstructured, gvResource schema.GroupVersionResource) {
	createdObj, err := rsc.Create(obj, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("failed to dry-run create stub for %s: %#v", gvResource, err)
	}
	if obj.GroupVersionKind() != createdObj.GroupVersionKind() {
		t.Fatalf("created object doesn't have the same gvk as original object: got %v, expected %v",
			createdObj.GroupVersionKind(),
			obj.GroupVersionKind())
	}

	if _, err := rsc.Get(obj.GetName(), metav1.GetOptions{}); !errors.IsNotFound(err) {
		t.Fatalf("object shouldn't exist: %v", err)
	}
}

func DryRunPatchTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	patch := []byte(`{"metadata":{"annotations":{"patch": "true"}}}`)
	obj, err := rsc.Patch(name, types.MergePatchType, patch, metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("failed to dry-run patch object: %v", err)
	}
	if v := obj.GetAnnotations()["patch"]; v != "true" {
		t.Fatalf("dry-run patched annotations should be returned, got: %v", obj.GetAnnotations())
	}
	obj, err = rsc.Get(obj.GetName(), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}
	if v := obj.GetAnnotations()["patch"]; v == "true" {
		t.Fatalf("dry-run patched annotations should not be persisted, got: %v", obj.GetAnnotations())
	}
}

func getReplicasOrFail(t *testing.T, obj *unstructured.Unstructured) int64 {
	t.Helper()
	replicas, found, err := unstructured.NestedInt64(obj.UnstructuredContent(), "spec", "replicas")
	if err != nil {
		t.Fatalf("failed to get int64 for replicas: %v", err)
	}
	if !found {
		t.Fatal("object doesn't have spec.replicas")
	}
	return replicas
}

func setReplicasOrFail(t *testing.T, obj *unstructured.Unstructured, replicas int64) {
	m, found, err := unstructured.NestedMap(obj.UnstructuredContent(), "spec")
	if err != nil {
		t.Fatalf("failed to get spec: %v", err)
	}
	if !found {
		t.Fatal("object doesn't have spec")
	}
	m["replicas"] = replicas
	unstructured.SetNestedMap(obj.UnstructuredContent(), m, "spec")
}

func DryRunScalePatchTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	obj, err := rsc.Get(name, metav1.GetOptions{}, "scale")
	if errors.IsNotFound(err) {
		return
	}
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}

	replicas := getReplicasOrFail(t, obj)
	patch := []byte(`{"spec":{"replicas":10}}`)
	patchedObj, err := rsc.Patch(name, types.MergePatchType, patch, metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}}, "scale")
	if err != nil {
		t.Fatalf("failed to dry-run patch object: %v", err)
	}
	if newReplicas := getReplicasOrFail(t, patchedObj); newReplicas != 10 {
		t.Fatalf("dry-run patch to replicas didn't return new value: %v", newReplicas)
	}
	persistedObj, err := rsc.Get(name, metav1.GetOptions{}, "scale")
	if err != nil {
		t.Fatalf("failed to get scale sub-resource")
	}
	if newReplicas := getReplicasOrFail(t, persistedObj); newReplicas != replicas {
		t.Fatalf("number of replicas changed, expected %v, got %v", replicas, newReplicas)
	}
}

func DryRunScaleUpdateTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	obj, err := rsc.Get(name, metav1.GetOptions{}, "scale")
	if errors.IsNotFound(err) {
		return
	}
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}

	replicas := getReplicasOrFail(t, obj)
	unstructured.SetNestedField(obj.Object, int64(10), "spec", "replicas")
	updatedObj, err := rsc.Update(obj, metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}}, "scale")
	if err != nil {
		t.Fatalf("failed to dry-run update scale sub-resource: %v", err)
	}
	if newReplicas := getReplicasOrFail(t, updatedObj); newReplicas != 10 {
		t.Fatalf("dry-run update to replicas didn't return new value: %v", newReplicas)
	}
	persistedObj, err := rsc.Get(name, metav1.GetOptions{}, "scale")
	if err != nil {
		t.Fatalf("failed to get scale sub-resource")
	}
	if newReplicas := getReplicasOrFail(t, persistedObj); newReplicas != replicas {
		t.Fatalf("number of replicas changed, expected %v, got %v", replicas, newReplicas)
	}
}

func DryRunUpdateTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	var err error
	var obj *unstructured.Unstructured
	for i := 0; i < 3; i++ {
		obj, err = rsc.Get(name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to retrieve object: %v", err)
		}
		obj.SetAnnotations(map[string]string{"update": "true"})
		obj, err = rsc.Update(obj, metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
		if err == nil || !errors.IsConflict(err) {
			break
		}
	}
	if err != nil {
		t.Fatalf("failed to dry-run update resource: %v", err)
	}
	if v := obj.GetAnnotations()["update"]; v != "true" {
		t.Fatalf("dry-run updated annotations should be returned, got: %v", obj.GetAnnotations())
	}

	obj, err = rsc.Get(obj.GetName(), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}
	if v := obj.GetAnnotations()["update"]; v == "true" {
		t.Fatalf("dry-run updated annotations should not be persisted, got: %v", obj.GetAnnotations())
	}
}

func DryRunDeleteCollectionTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	err := rsc.DeleteCollection(&metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}}, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("dry-run delete collection failed: %v", err)
	}
	obj, err := rsc.Get(name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}
	ts := obj.GetDeletionTimestamp()
	if ts != nil {
		t.Fatalf("object has a deletion timestamp after dry-run delete collection")
	}
}

func DryRunDeleteTest(t *testing.T, rsc dynamic.ResourceInterface, name string) {
	err := rsc.Delete(name, &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}})
	if err != nil {
		t.Fatalf("dry-run delete failed: %v", err)
	}
	obj, err := rsc.Get(name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get object: %v", err)
	}
	ts := obj.GetDeletionTimestamp()
	if ts != nil {
		t.Fatalf("object has a deletion timestamp after dry-run delete")
	}
}

// TestDryRun tests dry-run on all types.
func TestDryRun(t *testing.T) {
	certDir, _ := ioutil.TempDir("", "test-integration-dryrun")
	defer os.RemoveAll(certDir)

	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DryRun, true)()
	clientConfig := startRealMasterOrDie(t, certDir)
	dClient := dynamic.NewForConfigOrDie(clientConfig)
	kubeClient := clientset.NewForConfigOrDie(clientConfig)
	if _, err := kubeClient.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}); err != nil {
		t.Fatal(err)
	}

	discoveryClient := cacheddiscovery.NewMemCacheClient(kubeClient.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)
	restMapper.Reset()

	serverResources, err := kubeClient.Discovery().ServerResources()
	if err != nil {
		t.Fatal(err)
	}
	resourcesToTest := getResourcesToTest(serverResources, false, t)

	for _, resourceToTest := range resourcesToTest {
		t.Run(resourceToTest.gvr.String(), func(t *testing.T) {
			gvk := resourceToTest.gvk
			gvResource := resourceToTest.gvr
			kind := gvk.Kind

			mapping := &meta.RESTMapping{
				Resource:         resourceToTest.gvr,
				GroupVersionKind: resourceToTest.gvk,
				Scope:            meta.RESTScopeRoot,
			}
			if resourceToTest.namespaced {
				mapping.Scope = meta.RESTScopeNamespace
			}

			if kindWhiteList.Has(kind) {
				t.Skip("whitelisted")
			}

			testData, hasTest := dryrunData[gvResource]

			if !hasTest {
				t.Fatalf("no test data for %s.  Please add a test for your new type to dryrunData.", gvResource)
			}

			// we don't require GVK on the data we provide, so we fill it in here.  We could, but that seems extraneous.
			typeMetaAdder := map[string]interface{}{}
			err := json.Unmarshal([]byte(testData.stub), &typeMetaAdder)
			if err != nil {
				t.Fatalf("failed to unmarshal stub (%v): %v", testData.stub, err)
			}
			typeMetaAdder["apiVersion"] = mapping.GroupVersionKind.GroupVersion().String()
			typeMetaAdder["kind"] = mapping.GroupVersionKind.Kind

			rsc := dClient.Resource(mapping.Resource).Namespace(testNamespace)
			if mapping.Scope == meta.RESTScopeRoot {
				rsc = dClient.Resource(mapping.Resource)
			}
			obj := &unstructured.Unstructured{Object: typeMetaAdder}
			name := obj.GetName()

			DryRunCreateTest(t, rsc, obj, gvResource)

			if _, err := rsc.Create(obj, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create stub for %s: %#v", gvResource, err)
			}

			DryRunUpdateTest(t, rsc, name)
			DryRunPatchTest(t, rsc, name)
			DryRunScalePatchTest(t, rsc, name)
			DryRunScaleUpdateTest(t, rsc, name)
			if resourceToTest.hasDeleteCollection {
				DryRunDeleteCollectionTest(t, rsc, name)
			}
			DryRunDeleteTest(t, rsc, name)

			if err = rsc.Delete(obj.GetName(), metav1.NewDeleteOptions(0)); err != nil {
				t.Fatalf("deleting final object failed: %v", err)
			}
		})
	}
}

func startRealMasterOrDie(t *testing.T, certDir string) *restclient.Config {
	_, defaultServiceClusterIPRange, err := net.ParseCIDR("10.0.0.0/24")
	if err != nil {
		t.Fatal(err)
	}

	listener, _, err := genericapiserveroptions.CreateListener("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	kubeAPIServerOptions := options.NewServerRunOptions()
	kubeAPIServerOptions.InsecureServing.BindPort = 0
	kubeAPIServerOptions.SecureServing.Listener = listener
	kubeAPIServerOptions.SecureServing.ServerCert.CertDirectory = certDir
	kubeAPIServerOptions.Etcd.StorageConfig.ServerList = []string{framework.GetEtcdURL()}
	kubeAPIServerOptions.Etcd.DefaultStorageMediaType = runtime.ContentTypeJSON // force json we can easily interpret the result in etcd
	kubeAPIServerOptions.ServiceClusterIPRange = *defaultServiceClusterIPRange
	kubeAPIServerOptions.Authorization.Modes = []string{"RBAC"}
	kubeAPIServerOptions.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount"}
	completedOptions, err := app.Complete(kubeAPIServerOptions)
	if err != nil {
		t.Fatal(err)
	}
	kubeAPIServerOptions.APIEnablement.RuntimeConfig.Set("api/all=true")

	kubeAPIServer, err := app.CreateServerChain(completedOptions, wait.NeverStop)
	if err != nil {
		t.Fatal(err)
	}
	kubeClientConfig := restclient.CopyConfig(kubeAPIServer.LoopbackClientConfig)

	go func() {
		// Catch panics that occur in this go routine so we get a comprehensible failure
		defer func() {
			if err := recover(); err != nil {
				t.Errorf("Unexpected panic trying to start API master: %#v", err)
			}
		}()

		if err := kubeAPIServer.PrepareRun().Run(wait.NeverStop); err != nil {
			t.Fatal(err)
		}
	}()

	lastHealth := ""
	if err := wait.PollImmediate(time.Second, time.Minute, func() (done bool, err error) {
		// wait for the server to be healthy
		result := clientset.NewForConfigOrDie(kubeClientConfig).RESTClient().Get().AbsPath("/healthz").Do()
		content, _ := result.Raw()
		lastHealth = string(content)
		if errResult := result.Error(); errResult != nil {
			t.Log(errResult)
			return false, nil
		}
		var status int
		result.StatusCode(&status)
		return status == http.StatusOK, nil
	}); err != nil {
		t.Log(lastHealth)
		t.Fatal(err)
	}

	// this test makes lots of requests, don't be slow
	kubeClientConfig.QPS = 99999
	kubeClientConfig.Burst = 9999

	return kubeClientConfig
}

func gvr(g, v, r string) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: g, Version: v, Resource: r}
}

type resourceToTest struct {
	gvk                 schema.GroupVersionKind
	gvr                 schema.GroupVersionResource
	namespaced          bool
	hasDeleteCollection bool
}

func getResourcesToTest(serverResources []*metav1.APIResourceList, isOAPI bool, t *testing.T) []resourceToTest {
	resourcesToTest := []resourceToTest{}

	for _, discoveryGroup := range serverResources {
		for _, discoveryResource := range discoveryGroup.APIResources {
			// this is a subresource, skip it
			if strings.Contains(discoveryResource.Name, "/") {
				continue
			}
			hasCreate := false
			hasGet := false
			hasDeleteCollection := false
			for _, verb := range discoveryResource.Verbs {
				if string(verb) == "get" {
					hasGet = true
				}
				if string(verb) == "create" {
					hasCreate = true
				}
				if string(verb) == "deletecollection" {
					hasDeleteCollection = true
				}
			}
			if !(hasCreate && hasGet) {
				continue
			}

			resourceGV, err := schema.ParseGroupVersion(discoveryGroup.GroupVersion)
			if err != nil {
				t.Fatal(err)
			}
			gvk := resourceGV.WithKind(discoveryResource.Kind)
			if len(discoveryResource.Group) > 0 || len(discoveryResource.Version) > 0 {
				gvk = schema.GroupVersionKind{
					Group:   discoveryResource.Group,
					Version: discoveryResource.Version,
					Kind:    discoveryResource.Kind,
				}
			}
			gvr := resourceGV.WithResource(discoveryResource.Name)

			resourcesToTest = append(resourcesToTest, resourceToTest{
				gvk:                 gvk,
				gvr:                 gvr,
				namespaced:          discoveryResource.Namespaced,
				hasDeleteCollection: hasDeleteCollection,
			})
		}
	}

	return resourcesToTest
}
