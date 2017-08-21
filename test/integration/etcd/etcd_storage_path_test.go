/*
Copyright 2017 The Kubernetes Authors.

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
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"mime"
	"net"
	"net/http"
	"os"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	clientset "k8s.io/client-go/kubernetes"
	kclient "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	kapi "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/test/integration/framework"

	// install all APIs
	_ "k8s.io/kubernetes/pkg/master" // TODO what else is needed

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
)

// Etcd data for all persisted objects.
var etcdStorageData = map[schema.GroupVersionResource]struct {
	stub             string                   // Valid JSON stub to use during create
	prerequisites    []prerequisite           // Optional, ordered list of JSON objects to create before stub
	expectedEtcdPath string                   // Expected location of object in etcd, do not use any variables, constants, etc to derive this value - always supply the full raw string
	expectedGVK      *schema.GroupVersionKind // The GVK that we expect this object to be stored as - leave this nil to use the default
}{
	// k8s.io/kubernetes/pkg/api/v1
	gvr("", "v1", "configmaps"): {
		stub:             `{"data": {"foo": "bar"}, "metadata": {"name": "cm1"}}`,
		expectedEtcdPath: "/registry/configmaps/etcdstoragepathtestnamespace/cm1",
	},
	gvr("", "v1", "services"): {
		stub:             `{"metadata": {"name": "service1"}, "spec": {"externalName": "service1name", "ports": [{"port": 10000, "targetPort": 11000}], "selector": {"test": "data"}}}`,
		expectedEtcdPath: "/registry/services/specs/etcdstoragepathtestnamespace/service1",
	},
	gvr("", "v1", "podtemplates"): {
		stub:             `{"metadata": {"name": "pt1name"}, "template": {"metadata": {"labels": {"pt": "01"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container9"}]}}}`,
		expectedEtcdPath: "/registry/podtemplates/etcdstoragepathtestnamespace/pt1name",
	},
	gvr("", "v1", "pods"): {
		stub:             `{"metadata": {"name": "pod1"}, "spec": {"containers": [{"image": "fedora:latest", "name": "container7", "resources": {"limits": {"cpu": "1M"}, "requests": {"cpu": "1M"}}}]}}`,
		expectedEtcdPath: "/registry/pods/etcdstoragepathtestnamespace/pod1",
	},
	gvr("", "v1", "endpoints"): {
		stub:             `{"metadata": {"name": "ep1name"}, "subsets": [{"addresses": [{"hostname": "bar-001", "ip": "192.168.3.1"}], "ports": [{"port": 8000}]}]}`,
		expectedEtcdPath: "/registry/services/endpoints/etcdstoragepathtestnamespace/ep1name",
	},
	gvr("", "v1", "resourcequotas"): {
		stub:             `{"metadata": {"name": "rq1name"}, "spec": {"hard": {"cpu": "5M"}}}`,
		expectedEtcdPath: "/registry/resourcequotas/etcdstoragepathtestnamespace/rq1name",
	},
	gvr("", "v1", "limitranges"): {
		stub:             `{"metadata": {"name": "lr1name"}, "spec": {"limits": [{"type": "Pod"}]}}`,
		expectedEtcdPath: "/registry/limitranges/etcdstoragepathtestnamespace/lr1name",
	},
	gvr("", "v1", "namespaces"): {
		stub:             `{"metadata": {"name": "namespace1"}, "spec": {"finalizers": ["kubernetes"]}}`,
		expectedEtcdPath: "/registry/namespaces/namespace1",
	},
	gvr("", "v1", "nodes"): {
		stub:             `{"metadata": {"name": "node1"}, "spec": {"unschedulable": true}}`,
		expectedEtcdPath: "/registry/minions/node1",
	},
	gvr("", "v1", "persistentvolumes"): {
		stub:             `{"metadata": {"name": "pv1name"}, "spec": {"accessModes": ["ReadWriteOnce"], "capacity": {"storage": "3M"}, "hostPath": {"path": "/tmp/test/"}}}`,
		expectedEtcdPath: "/registry/persistentvolumes/pv1name",
	},
	gvr("", "v1", "events"): {
		stub:             `{"involvedObject": {"namespace": "etcdstoragepathtestnamespace"}, "message": "some data here", "metadata": {"name": "event1"}}`,
		expectedEtcdPath: "/registry/events/etcdstoragepathtestnamespace/event1",
	},
	gvr("", "v1", "persistentvolumeclaims"): {
		stub:             `{"metadata": {"name": "pvc1"}, "spec": {"accessModes": ["ReadWriteOnce"], "resources": {"limits": {"storage": "1M"}, "requests": {"storage": "2M"}}, "selector": {"matchLabels": {"pvc": "stuff"}}}}`,
		expectedEtcdPath: "/registry/persistentvolumeclaims/etcdstoragepathtestnamespace/pvc1",
	},
	gvr("", "v1", "serviceaccounts"): {
		stub:             `{"metadata": {"name": "sa1name"}, "secrets": [{"name": "secret00"}]}`,
		expectedEtcdPath: "/registry/serviceaccounts/etcdstoragepathtestnamespace/sa1name",
	},
	gvr("", "v1", "secrets"): {
		stub:             `{"data": {"key": "ZGF0YSBmaWxl"}, "metadata": {"name": "secret1"}}`,
		expectedEtcdPath: "/registry/secrets/etcdstoragepathtestnamespace/secret1",
	},
	gvr("", "v1", "replicationcontrollers"): {
		stub:             `{"metadata": {"name": "rc1"}, "spec": {"selector": {"new": "stuff"}, "template": {"metadata": {"labels": {"new": "stuff"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container8"}]}}}}`,
		expectedEtcdPath: "/registry/controllers/etcdstoragepathtestnamespace/rc1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/apps/v1beta1
	gvr("apps", "v1beta1", "statefulsets"): {
		stub:             `{"metadata": {"name": "ss1"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}}}}`,
		expectedEtcdPath: "/registry/statefulsets/etcdstoragepathtestnamespace/ss1",
	},
	gvr("apps", "v1beta1", "deployments"): {
		stub:             `{"metadata": {"name": "deployment2"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
		expectedEtcdPath: "/registry/deployments/etcdstoragepathtestnamespace/deployment2",
		expectedGVK:      gvkP("extensions", "v1beta1", "Deployment"),
	},
	gvr("apps", "v1beta1", "controllerrevisions"): {
		stub:             `{"metadata":{"name":"crs1"},"data":{"name":"abc","namespace":"default","creationTimestamp":null,"Spec":{"Replicas":0,"Selector":{"matchLabels":{"foo":"bar"}},"Template":{"creationTimestamp":null,"labels":{"foo":"bar"},"Spec":{"Volumes":null,"InitContainers":null,"Containers":null,"RestartPolicy":"Always","TerminationGracePeriodSeconds":null,"ActiveDeadlineSeconds":null,"DNSPolicy":"ClusterFirst","NodeSelector":null,"ServiceAccountName":"","AutomountServiceAccountToken":null,"NodeName":"","SecurityContext":null,"ImagePullSecrets":null,"Hostname":"","Subdomain":"","Affinity":null,"SchedulerName":"","Tolerations":null,"HostAliases":null}},"VolumeClaimTemplates":null,"ServiceName":""},"Status":{"ObservedGeneration":null,"Replicas":0}},"revision":0}`,
		expectedEtcdPath: "/registry/controllerrevisions/etcdstoragepathtestnamespace/crs1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/apps/v1beta2
	gvr("apps", "v1beta2", "statefulsets"): {
		stub:             `{"metadata": {"name": "ss2"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}}}}`,
		expectedEtcdPath: "/registry/statefulsets/etcdstoragepathtestnamespace/ss2",
		expectedGVK:      gvkP("apps", "v1beta1", "StatefulSet"),
	},
	gvr("apps", "v1beta2", "deployments"): {
		stub:             `{"metadata": {"name": "deployment3"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
		expectedEtcdPath: "/registry/deployments/etcdstoragepathtestnamespace/deployment3",
		expectedGVK:      gvkP("extensions", "v1beta1", "Deployment"),
	},
	gvr("apps", "v1beta2", "daemonsets"): {
		stub:             `{"metadata": {"name": "ds5"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
		expectedEtcdPath: "/registry/daemonsets/etcdstoragepathtestnamespace/ds5",
		expectedGVK:      gvkP("extensions", "v1beta1", "DaemonSet"),
	},
	gvr("apps", "v1beta2", "replicasets"): {
		stub:             `{"metadata": {"name": "rs2"}, "spec": {"selector": {"matchLabels": {"g": "h"}}, "template": {"metadata": {"labels": {"g": "h"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container4"}]}}}}`,
		expectedEtcdPath: "/registry/replicasets/etcdstoragepathtestnamespace/rs2",
		expectedGVK:      gvkP("extensions", "v1beta1", "ReplicaSet"),
	},
	gvr("apps", "v1beta2", "controllerrevisions"): {
		stub:             `{"metadata":{"name":"crs2"},"data":{"name":"abc","namespace":"default","creationTimestamp":null,"Spec":{"Replicas":0,"Selector":{"matchLabels":{"foo":"bar"}},"Template":{"creationTimestamp":null,"labels":{"foo":"bar"},"Spec":{"Volumes":null,"InitContainers":null,"Containers":null,"RestartPolicy":"Always","TerminationGracePeriodSeconds":null,"ActiveDeadlineSeconds":null,"DNSPolicy":"ClusterFirst","NodeSelector":null,"ServiceAccountName":"","AutomountServiceAccountToken":null,"NodeName":"","SecurityContext":null,"ImagePullSecrets":null,"Hostname":"","Subdomain":"","Affinity":null,"SchedulerName":"","Tolerations":null,"HostAliases":null}},"VolumeClaimTemplates":null,"ServiceName":""},"Status":{"ObservedGeneration":null,"Replicas":0}},"revision":0}`,
		expectedEtcdPath: "/registry/controllerrevisions/etcdstoragepathtestnamespace/crs2",
		expectedGVK:      gvkP("apps", "v1beta1", "ControllerRevision"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/autoscaling/v1
	gvr("autoscaling", "v1", "horizontalpodautoscalers"): {
		stub:             `{"metadata": {"name": "hpa2"}, "spec": {"maxReplicas": 3, "scaleTargetRef": {"kind": "something", "name": "cross"}}}`,
		expectedEtcdPath: "/registry/horizontalpodautoscalers/etcdstoragepathtestnamespace/hpa2",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/autoscaling/v2alpha1
	gvr("autoscaling", "v2alpha1", "horizontalpodautoscalers"): {
		stub:             `{"metadata": {"name": "hpa1"}, "spec": {"maxReplicas": 3, "scaleTargetRef": {"kind": "something", "name": "cross"}}}`,
		expectedEtcdPath: "/registry/horizontalpodautoscalers/etcdstoragepathtestnamespace/hpa1",
		expectedGVK:      gvkP("autoscaling", "v1", "HorizontalPodAutoscaler"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/batch/v1
	gvr("batch", "v1", "jobs"): {
		stub:             `{"metadata": {"name": "job1"}, "spec": {"manualSelector": true, "selector": {"matchLabels": {"controller-uid": "uid1"}}, "template": {"metadata": {"labels": {"controller-uid": "uid1"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container1"}], "dnsPolicy": "ClusterFirst", "restartPolicy": "Never"}}}}`,
		expectedEtcdPath: "/registry/jobs/etcdstoragepathtestnamespace/job1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/batch/v1beta1
	gvr("batch", "v1beta1", "cronjobs"): {
		stub:             `{"metadata": {"name": "cjv1beta1"}, "spec": {"jobTemplate": {"spec": {"template": {"metadata": {"labels": {"controller-uid": "uid0"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container0"}], "dnsPolicy": "ClusterFirst", "restartPolicy": "Never"}}}}, "schedule": "* * * * *"}}`,
		expectedEtcdPath: "/registry/cronjobs/etcdstoragepathtestnamespace/cjv1beta1",
		// TODO this needs to be updated to batch/v1beta1 when it's enabed by default
		expectedGVK: gvkP("batch", "v2alpha1", "CronJob"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/batch/v2alpha1
	gvr("batch", "v2alpha1", "cronjobs"): {
		stub:             `{"metadata": {"name": "cjv2alpha1"}, "spec": {"jobTemplate": {"spec": {"template": {"metadata": {"labels": {"controller-uid": "uid0"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container0"}], "dnsPolicy": "ClusterFirst", "restartPolicy": "Never"}}}}, "schedule": "* * * * *"}}`,
		expectedEtcdPath: "/registry/cronjobs/etcdstoragepathtestnamespace/cjv2alpha1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/certificates/v1beta1
	gvr("certificates.k8s.io", "v1beta1", "certificatesigningrequests"): {
		stub:             `{"metadata": {"name": "csr1"}, "spec": {"request": "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0KTUlJQnlqQ0NBVE1DQVFBd2dZa3hDekFKQmdOVkJBWVRBbFZUTVJNd0VRWURWUVFJRXdwRFlXeHBabTl5Ym1saApNUll3RkFZRFZRUUhFdzFOYjNWdWRHRnBiaUJXYVdWM01STXdFUVlEVlFRS0V3cEhiMjluYkdVZ1NXNWpNUjh3CkhRWURWUVFMRXhaSmJtWnZjbTFoZEdsdmJpQlVaV05vYm05c2IyZDVNUmN3RlFZRFZRUURFdzUzZDNjdVoyOXYKWjJ4bExtTnZiVENCbnpBTkJna3Foa2lHOXcwQkFRRUZBQU9CalFBd2dZa0NnWUVBcFp0WUpDSEo0VnBWWEhmVgpJbHN0UVRsTzRxQzAzaGpYK1prUHl2ZFlkMVE0K3FiQWVUd1htQ1VLWUhUaFZSZDVhWFNxbFB6eUlCd2llTVpyCldGbFJRZGRaMUl6WEFsVlJEV3dBbzYwS2VjcWVBWG5uVUsrNWZYb1RJL1VnV3NocmU4dEoreC9UTUhhUUtSL0oKY0lXUGhxYVFoc0p1elpidkFkR0E4MEJMeGRNQ0F3RUFBYUFBTUEwR0NTcUdTSWIzRFFFQkJRVUFBNEdCQUlobAo0UHZGcStlN2lwQVJnSTVaTStHWng2bXBDejQ0RFRvMEprd2ZSRGYrQnRyc2FDMHE2OGVUZjJYaFlPc3E0ZmtIClEwdUEwYVZvZzNmNWlKeENhM0hwNWd4YkpRNnpWNmtKMFRFc3VhYU9oRWtvOXNkcENvUE9uUkJtMmkvWFJEMkQKNmlOaDhmOHowU2hHc0ZxakRnRkh5RjNvK2xVeWorVUM2SDFRVzdibgotLS0tLUVORCBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0="}}`,
		expectedEtcdPath: "/registry/certificatesigningrequests/csr1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/extensions/v1beta1
	gvr("extensions", "v1beta1", "daemonsets"): {
		stub:             `{"metadata": {"name": "ds1"}, "spec": {"selector": {"matchLabels": {"u": "t"}}, "template": {"metadata": {"labels": {"u": "t"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container5"}]}}}}`,
		expectedEtcdPath: "/registry/daemonsets/etcdstoragepathtestnamespace/ds1",
	},
	gvr("extensions", "v1beta1", "podsecuritypolicies"): {
		stub:             `{"metadata": {"name": "psp1"}, "spec": {"fsGroup": {"rule": "RunAsAny"}, "privileged": true, "runAsUser": {"rule": "RunAsAny"}, "seLinux": {"rule": "MustRunAs"}, "supplementalGroups": {"rule": "RunAsAny"}}}`,
		expectedEtcdPath: "/registry/podsecuritypolicy/psp1",
	},
	gvr("extensions", "v1beta1", "ingresses"): {
		stub:             `{"metadata": {"name": "ingress1"}, "spec": {"backend": {"serviceName": "service", "servicePort": 5000}}}`,
		expectedEtcdPath: "/registry/ingress/etcdstoragepathtestnamespace/ingress1",
	},
	gvr("extensions", "v1beta1", "networkpolicies"): {
		stub:             `{"metadata": {"name": "np1"}, "spec": {"podSelector": {"matchLabels": {"e": "f"}}}}`,
		expectedEtcdPath: "/registry/networkpolicies/etcdstoragepathtestnamespace/np1",
	},
	gvr("extensions", "v1beta1", "deployments"): {
		stub:             `{"metadata": {"name": "deployment1"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
		expectedEtcdPath: "/registry/deployments/etcdstoragepathtestnamespace/deployment1",
	},
	gvr("extensions", "v1beta1", "replicasets"): {
		stub:             `{"metadata": {"name": "rs1"}, "spec": {"selector": {"matchLabels": {"g": "h"}}, "template": {"metadata": {"labels": {"g": "h"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container4"}]}}}}`,
		expectedEtcdPath: "/registry/replicasets/etcdstoragepathtestnamespace/rs1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/networking/v1
	gvr("networking.k8s.io", "v1", "networkpolicies"): {
		stub:             `{"metadata": {"name": "np2"}, "spec": {"podSelector": {"matchLabels": {"e": "f"}}}}`,
		expectedEtcdPath: "/registry/networkpolicies/etcdstoragepathtestnamespace/np2",
		expectedGVK:      gvkP("extensions", "v1beta1", "NetworkPolicy"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/policy/v1beta1
	gvr("policy", "v1beta1", "poddisruptionbudgets"): {
		stub:             `{"metadata": {"name": "pdb1"}, "spec": {"selector": {"matchLabels": {"anokkey": "anokvalue"}}}}`,
		expectedEtcdPath: "/registry/poddisruptionbudgets/etcdstoragepathtestnamespace/pdb1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/storage/v1beta1
	gvr("storage.k8s.io", "v1beta1", "storageclasses"): {
		stub:             `{"metadata": {"name": "sc1"}, "provisioner": "aws"}`,
		expectedEtcdPath: "/registry/storageclasses/sc1",
		expectedGVK:      gvkP("storage.k8s.io", "v1", "StorageClass"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/storage/v1
	gvr("storage.k8s.io", "v1", "storageclasses"): {
		stub:             `{"metadata": {"name": "sc2"}, "provisioner": "aws"}`,
		expectedEtcdPath: "/registry/storageclasses/sc2",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/settings/v1alpha1
	gvr("settings.k8s.io", "v1alpha1", "podpresets"): {
		stub:             `{"metadata": {"name": "podpre1"}, "spec": {"env": [{"name": "FOO"}]}}`,
		expectedEtcdPath: "/registry/podpresets/etcdstoragepathtestnamespace/podpre1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/rbac/v1alpha1
	gvr("rbac.authorization.k8s.io", "v1alpha1", "roles"): {
		stub:             `{"metadata": {"name": "role1"}, "rules": [{"apiGroups": ["v1"], "resources": ["events"], "verbs": ["watch"]}]}`,
		expectedEtcdPath: "/registry/roles/etcdstoragepathtestnamespace/role1",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1beta1", "Role"),
	},
	gvr("rbac.authorization.k8s.io", "v1alpha1", "clusterroles"): {
		stub:             `{"metadata": {"name": "crole1"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
		expectedEtcdPath: "/registry/clusterroles/crole1",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1beta1", "ClusterRole"),
	},
	gvr("rbac.authorization.k8s.io", "v1alpha1", "rolebindings"): {
		stub:             `{"metadata": {"name": "roleb1"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/rolebindings/etcdstoragepathtestnamespace/roleb1",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1beta1", "RoleBinding"),
	},
	gvr("rbac.authorization.k8s.io", "v1alpha1", "clusterrolebindings"): {
		stub:             `{"metadata": {"name": "croleb1"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/clusterrolebindings/croleb1",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1beta1", "ClusterRoleBinding"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/rbac/v1beta1
	gvr("rbac.authorization.k8s.io", "v1beta1", "roles"): {
		stub:             `{"metadata": {"name": "role2"}, "rules": [{"apiGroups": ["v1"], "resources": ["events"], "verbs": ["watch"]}]}`,
		expectedEtcdPath: "/registry/roles/etcdstoragepathtestnamespace/role2",
	},
	gvr("rbac.authorization.k8s.io", "v1beta1", "clusterroles"): {
		stub:             `{"metadata": {"name": "crole2"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
		expectedEtcdPath: "/registry/clusterroles/crole2",
	},
	gvr("rbac.authorization.k8s.io", "v1beta1", "rolebindings"): {
		stub:             `{"metadata": {"name": "roleb2"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/rolebindings/etcdstoragepathtestnamespace/roleb2",
	},
	gvr("rbac.authorization.k8s.io", "v1beta1", "clusterrolebindings"): {
		stub:             `{"metadata": {"name": "croleb2"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/clusterrolebindings/croleb2",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/rbac/v1
	gvr("rbac.authorization.k8s.io", "v1", "roles"): {
		stub:             `{"metadata": {"name": "role3"}, "rules": [{"apiGroups": ["v1"], "resources": ["events"], "verbs": ["watch"]}]}`,
		expectedEtcdPath: "/registry/roles/etcdstoragepathtestnamespace/role3",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1beta1", "Role"),
	},
	gvr("rbac.authorization.k8s.io", "v1", "clusterroles"): {
		stub:             `{"metadata": {"name": "crole3"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
		expectedEtcdPath: "/registry/clusterroles/crole3",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1beta1", "ClusterRole"),
	},
	gvr("rbac.authorization.k8s.io", "v1", "rolebindings"): {
		stub:             `{"metadata": {"name": "roleb3"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/rolebindings/etcdstoragepathtestnamespace/roleb3",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1beta1", "RoleBinding"),
	},
	gvr("rbac.authorization.k8s.io", "v1", "clusterrolebindings"): {
		stub:             `{"metadata": {"name": "croleb3"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/clusterrolebindings/croleb3",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1beta1", "ClusterRoleBinding"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/admissionregistration/v1alpha1
	gvr("admissionregistration.k8s.io", "v1alpha1", "initializerconfigurations"): {
		stub:             `{"metadata":{"name":"ic1"},"initializers":[{"name":"initializer.k8s.io","rules":[{"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore"}]}`,
		expectedEtcdPath: "/registry/initializerconfigurations/ic1",
	},
	gvr("admissionregistration.k8s.io", "v1alpha1", "externaladmissionhookconfigurations"): {
		stub:             `{"metadata":{"name":"hook1","creationTimestamp":null},"externalAdmissionHooks":[{"name":"externaladmissionhook.k8s.io","clientConfig":{"service":{"namespace":"","name":""},"caBundle":null},"rules":[{"operations":["CREATE"],"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore"}]}`,
		expectedEtcdPath: "/registry/externaladmissionhookconfigurations/hook1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/scheduling/v1alpha1
	gvr("scheduling.k8s.io", "v1alpha1", "priorityclasses"): {
		stub:             `{"metadata":{"name":"pc1"},"Value":1000}`,
		expectedEtcdPath: "/registry/priorityclasses/pc1",
	},
	// --
}

// Be very careful when whitelisting an object as ephemeral.
// Doing so removes the safety we gain from this test by skipping that object.
var ephemeralWhiteList = createEphemeralWhiteList(
	// k8s.io/kubernetes/federation/apis/federation/v1beta1
	gvr("federation", "v1beta1", "clusters"), // we cannot create this
	// --

	// k8s.io/kubernetes/pkg/api/v1
	gvr("", "v1", "bindings"),             // annotation on pod, not stored in etcd
	gvr("", "v1", "rangeallocations"),     // stored in various places in etcd but cannot be directly created
	gvr("", "v1", "componentstatuses"),    // status info not stored in etcd
	gvr("", "v1", "serializedreferences"), // used for serilization, not stored in etcd
	gvr("", "v1", "nodeconfigsources"),    // subfield of node.spec, but shouldn't be directly created
	gvr("", "v1", "podstatusresults"),     // wrapper object not stored in etcd
	// --

	// k8s.io/kubernetes/pkg/apis/authentication/v1beta1
	gvr("authentication.k8s.io", "v1beta1", "tokenreviews"), // not stored in etcd
	// --

	// k8s.io/kubernetes/pkg/apis/authentication/v1
	gvr("authentication.k8s.io", "v1", "tokenreviews"), // not stored in etcd
	// --

	// k8s.io/kubernetes/pkg/apis/authorization/v1beta1

	// SAR objects that are not stored in etcd
	gvr("authorization.k8s.io", "v1beta1", "selfsubjectaccessreviews"),
	gvr("authorization.k8s.io", "v1beta1", "localsubjectaccessreviews"),
	gvr("authorization.k8s.io", "v1beta1", "subjectaccessreviews"),
	// --

	// k8s.io/kubernetes/pkg/apis/authorization/v1

	// SAR objects that are not stored in etcd
	gvr("authorization.k8s.io", "v1", "selfsubjectaccessreviews"),
	gvr("authorization.k8s.io", "v1", "localsubjectaccessreviews"),
	gvr("authorization.k8s.io", "v1", "subjectaccessreviews"),
	// --

	// k8s.io/kubernetes/pkg/apis/autoscaling/v1
	gvr("autoscaling", "v1", "scales"), // not stored in etcd, part of kapiv1.ReplicationController
	// --

	// k8s.io/kubernetes/pkg/apis/apps/v1beta1
	gvr("apps", "v1beta1", "scales"),              // not stored in etcd, part of kapiv1.ReplicationController
	gvr("apps", "v1beta1", "deploymentrollbacks"), // used to rollback deployment, not stored in etcd
	// --

	// k8s.io/kubernetes/pkg/apis/apps/v1beta2
	gvr("apps", "v1beta2", "scales"), // not stored in etcd, part of kapiv1.ReplicationController
	// --

	// k8s.io/kubernetes/pkg/apis/batch/v1beta1
	gvr("batch", "v1beta1", "jobtemplates"), // not stored in etcd
	// --

	// k8s.io/kubernetes/pkg/apis/batch/v2alpha1
	gvr("batch", "v2alpha1", "jobtemplates"), // not stored in etcd
	// --

	// k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1
	gvr("componentconfig", "v1alpha1", "kubeschedulerconfigurations"), // not stored in etcd
	gvr("componentconfig", "v1alpha1", "kubeproxyconfigurations"),     // not stored in etcd
	// --

	// k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1alpha1
	gvr("kubeletconfig", "v1alpha1", "kubeletconfigurations"), // not stored in etcd
	// --

	// k8s.io/kubernetes/pkg/apis/extensions/v1beta1
	gvr("extensions", "v1beta1", "deploymentrollbacks"),          // used to rollback deployment, not stored in etcd
	gvr("extensions", "v1beta1", "replicationcontrollerdummies"), // not stored in etcd
	gvr("extensions", "v1beta1", "scales"),                       // not stored in etcd, part of kapiv1.ReplicationController
	gvr("extensions", "v1beta1", "thirdpartyresourcedatas"),      // we cannot create this
	gvr("extensions", "v1beta1", "thirdpartyresources"),          // these have been removed from the API server, but kept for the client
	// --

	// k8s.io/kubernetes/pkg/apis/imagepolicy/v1alpha1
	gvr("imagepolicy.k8s.io", "v1alpha1", "imagereviews"), // not stored in etcd
	// --

	// k8s.io/kubernetes/pkg/apis/policy/v1beta1
	gvr("policy", "v1beta1", "evictions"), // not stored in etcd, deals with evicting kapiv1.Pod
	// --

	// k8s.io/kubernetes/pkg/apis/admission/v1alpha1
	gvr("admission.k8s.io", "v1alpha1", "admissionreviews"), // not stored in etcd, call out to webhooks.
	// --
)

// Only add kinds to this list when there is no way to create the object
var kindWhiteList = sets.NewString(
	// k8s.io/kubernetes/pkg/api/v1
	"DeleteOptions",
	"ExportOptions",
	"ListOptions",
	"NodeProxyOptions",
	"PodAttachOptions",
	"PodExecOptions",
	"PodLogOptions",
	"PodProxyOptions",
	"ServiceProxyOptions",
	"GetOptions",
	"APIGroup",
	"PodPortForwardOptions",
	"APIVersions",
	// --

	// k8s.io/kubernetes/pkg/watch/versioned
	"WatchEvent",
	// --

	// k8s.io/kubernetes/pkg/api/unversioned
	"Status",
	// --
)

// namespace used for all tests, do not change this
const testNamespace = "etcdstoragepathtestnamespace"

// TestEtcdStoragePath tests to make sure that all objects are stored in an expected location in etcd.
// It will start failing when a new type is added to ensure that all future types are added to this test.
// It will also fail when a type gets moved to a different location. Be very careful in this situation because
// it essentially means that you will be break old clusters unless you create some migration path for the old data.
func TestEtcdStoragePath(t *testing.T) {
	certDir, _ := ioutil.TempDir("", "test-integration-etcd")
	defer os.RemoveAll(certDir)

	client, kvClient, mapper := startRealMasterOrDie(t, certDir)
	defer func() {
		dumpEtcdKVOnFailure(t, kvClient)
	}()

	kindSeen := sets.NewString()
	pathSeen := map[string][]schema.GroupVersionResource{}
	etcdSeen := map[schema.GroupVersionResource]empty{}
	ephemeralSeen := map[schema.GroupVersionResource]empty{}
	cohabitatingResources := map[string]map[schema.GroupVersionKind]empty{}

	for gvk, apiType := range kapi.Scheme.AllKnownTypes() {
		// we do not care about internal objects or lists // TODO make sure this is always true
		if gvk.Version == runtime.APIVersionInternal || strings.HasSuffix(apiType.Name(), "List") {
			continue
		}

		kind := gvk.Kind
		pkgPath := apiType.PkgPath()

		if kindWhiteList.Has(kind) {
			kindSeen.Insert(kind)
			continue
		}

		mapping, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
		if err != nil {
			t.Errorf("unexpected error getting mapping for %s from %s with GVK %s: %v", kind, pkgPath, gvk, err)
			continue
		}

		gvResource := gvk.GroupVersion().WithResource(mapping.Resource)
		etcdSeen[gvResource] = empty{}

		testData, hasTest := etcdStorageData[gvResource]
		_, isEphemeral := ephemeralWhiteList[gvResource]

		if !hasTest && !isEphemeral {
			t.Errorf("no test data for %s from %s.  Please add a test for your new type to etcdStorageData.", kind, pkgPath)
			continue
		}

		if hasTest && isEphemeral {
			t.Errorf("duplicate test data for %s from %s.  Object has both test data and is ephemeral.", kind, pkgPath)
			continue
		}

		if isEphemeral { // TODO it would be nice if we could remove this and infer if an object is not stored in etcd
			// t.Logf("Skipping test for %s from %s", kind, pkgPath)
			ephemeralSeen[gvResource] = empty{}
			delete(etcdSeen, gvResource)
			continue
		}

		if len(testData.expectedEtcdPath) == 0 {
			t.Errorf("empty test data for %s from %s", kind, pkgPath)
			continue
		}

		shouldCreate := len(testData.stub) != 0 // try to create only if we have a stub

		var input *metaObject
		if shouldCreate {
			if input, err = jsonToMetaObject([]byte(testData.stub)); err != nil || input.isEmpty() {
				t.Errorf("invalid test data for %s from %s: %v", kind, pkgPath, err)
				continue
			}
		}

		func() { // forces defer to run per iteration of the for loop
			all := &[]cleanupData{}
			defer func() {
				if !t.Failed() { // do not cleanup if test has already failed since we may need things in the etcd dump
					if err := client.cleanup(all); err != nil {
						t.Fatalf("failed to clean up etcd: %#v", err)
					}
				}
			}()

			if err := client.createPrerequisites(mapper, testNamespace, testData.prerequisites, all); err != nil {
				t.Errorf("failed to create prerequisites for %s from %s: %#v", kind, pkgPath, err)
				return
			}

			if shouldCreate { // do not try to create items with no stub
				if err := client.create(testData.stub, testNamespace, mapping, all); err != nil {
					t.Errorf("failed to create stub for %s from %s: %#v", kind, pkgPath, err)
					return
				}
			}

			output, err := getFromEtcd(kvClient, testData.expectedEtcdPath)
			if err != nil {
				t.Errorf("failed to get from etcd for %s from %s: %#v", kind, pkgPath, err)
				return
			}

			expectedGVK := gvk
			if testData.expectedGVK != nil {
				if gvk == *testData.expectedGVK {
					t.Errorf("GVK override %s for %s from %s is unnecessary or something was changed incorrectly", testData.expectedGVK, kind, pkgPath)
				}
				expectedGVK = *testData.expectedGVK
			}

			actualGVK := output.getGVK()
			if actualGVK != expectedGVK {
				t.Errorf("GVK for %s from %s does not match, expected %s got %s", kind, pkgPath, expectedGVK, actualGVK)
			}

			if !apiequality.Semantic.DeepDerivative(input, output) {
				t.Errorf("Test stub for %s from %s does not match: %s", kind, pkgPath, diff.ObjectGoPrintDiff(input, output))
			}

			addGVKToEtcdBucket(cohabitatingResources, actualGVK, getEtcdBucket(testData.expectedEtcdPath))
			pathSeen[testData.expectedEtcdPath] = append(pathSeen[testData.expectedEtcdPath], gvResource)
		}()
	}

	if inEtcdData, inEtcdSeen := diffMaps(etcdStorageData, etcdSeen); len(inEtcdData) != 0 || len(inEtcdSeen) != 0 {
		t.Errorf("etcd data does not match the types we saw:\nin etcd data but not seen:\n%s\nseen but not in etcd data:\n%s", inEtcdData, inEtcdSeen)
	}

	if inEphemeralWhiteList, inEphemeralSeen := diffMaps(ephemeralWhiteList, ephemeralSeen); len(inEphemeralWhiteList) != 0 || len(inEphemeralSeen) != 0 {
		t.Errorf("ephemeral whitelist does not match the types we saw:\nin ephemeral whitelist but not seen:\n%s\nseen but not in ephemeral whitelist:\n%s", inEphemeralWhiteList, inEphemeralSeen)
	}

	if inKindData, inKindSeen := diffMaps(kindWhiteList, kindSeen); len(inKindData) != 0 || len(inKindSeen) != 0 {
		t.Errorf("kind whitelist data does not match the types we saw:\nin kind whitelist but not seen:\n%s\nseen but not in kind whitelist:\n%s", inKindData, inKindSeen)
	}

	for bucket, gvks := range cohabitatingResources {
		if len(gvks) != 1 {
			gvkStrings := []string{}
			for key := range gvks {
				gvkStrings = append(gvkStrings, keyStringer(key))
			}
			t.Errorf("cohabitating resources in etcd bucket %s have inconsistent GVKs\nyou may need to use DefaultStorageFactory.AddCohabitatingResources to sync the GVK of these resources:\n%s", bucket, gvkStrings)
		}
	}

	for path, gvrs := range pathSeen {
		if len(gvrs) != 1 {
			gvrStrings := []string{}
			for _, key := range gvrs {
				gvrStrings = append(gvrStrings, keyStringer(key))
			}
			t.Errorf("invalid test data, please ensure all expectedEtcdPath are unique, path %s has duplicate GVRs:\n%s", path, gvrStrings)
		}
	}
}

func startRealMasterOrDie(t *testing.T, certDir string) (*allClient, clientv3.KV, meta.RESTMapper) {
	_, defaultServiceClusterIPRange, err := net.ParseCIDR("10.0.0.0/24")
	if err != nil {
		t.Fatal(err)
	}

	kubeClientConfigValue := atomic.Value{}
	storageConfigValue := atomic.Value{}

	go func() {
		for {
			kubeAPIServerOptions := options.NewServerRunOptions()
			kubeAPIServerOptions.SecureServing.BindAddress = net.ParseIP("127.0.0.1")
			kubeAPIServerOptions.SecureServing.ServerCert.CertDirectory = certDir
			kubeAPIServerOptions.Etcd.StorageConfig.ServerList = []string{framework.GetEtcdURL()}
			kubeAPIServerOptions.Etcd.DefaultStorageMediaType = runtime.ContentTypeJSON // TODO use protobuf?
			kubeAPIServerOptions.ServiceClusterIPRange = *defaultServiceClusterIPRange
			kubeAPIServerOptions.Authorization.Mode = "RBAC"

			// always get a fresh port in case something claimed the old one
			kubePort, err := framework.FindFreeLocalPort()
			if err != nil {
				t.Fatal(err)
			}

			kubeAPIServerOptions.SecureServing.BindPort = kubePort

			tunneler, proxyTransport, err := app.CreateNodeDialer(kubeAPIServerOptions)
			if err != nil {
				t.Fatal(err)
			}
			kubeAPIServerConfig, sharedInformers, _, _, _, err := app.CreateKubeAPIServerConfig(kubeAPIServerOptions, tunneler, proxyTransport)
			if err != nil {
				t.Fatal(err)
			}

			kubeAPIServerConfig.APIResourceConfigSource = &allResourceSource{} // force enable all resources

			kubeAPIServer, err := app.CreateKubeAPIServer(kubeAPIServerConfig, genericapiserver.EmptyDelegate, sharedInformers)
			if err != nil {
				t.Fatal(err)
			}

			kubeClientConfigValue.Store(kubeAPIServerConfig.GenericConfig.LoopbackClientConfig)
			storageConfigValue.Store(kubeAPIServerOptions.Etcd.StorageConfig)

			if err := kubeAPIServer.GenericAPIServer.PrepareRun().Run(wait.NeverStop); err != nil {
				t.Log(err)
			}

			time.Sleep(100 * time.Millisecond)
		}
	}()

	if err := wait.PollImmediate(100*time.Millisecond, 10*time.Second, func() (done bool, err error) {
		obj := kubeClientConfigValue.Load()
		if obj == nil {
			return false, nil
		}
		kubeClientConfig := kubeClientConfigValue.Load().(*restclient.Config)
		kubeClient, err := kclient.NewForConfig(kubeClientConfig)
		if err != nil {
			// this happens because we race the API server start
			t.Log(err)
			return false, nil
		}
		if _, err := kubeClient.Discovery().ServerVersion(); err != nil {
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Fatal(err)
	}

	kubeClientConfig := kubeClientConfigValue.Load().(*restclient.Config)
	storageConfig := storageConfigValue.Load().(storagebackend.Config)

	kubeClient := clientset.NewForConfigOrDie(kubeClientConfig)
	if _, err := kubeClient.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}); err != nil {
		t.Fatal(err)
	}

	client, err := newClient(*kubeClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	kvClient, err := getEtcdKVClient(storageConfig)
	if err != nil {
		t.Fatal(err)
	}

	mapper, _ := util.NewFactory(clientcmd.NewDefaultClientConfig(*clientcmdapi.NewConfig(), &clientcmd.ConfigOverrides{})).Object()

	return client, kvClient, mapper
}

func dumpEtcdKVOnFailure(t *testing.T, kvClient clientv3.KV) {
	if t.Failed() {
		response, err := kvClient.Get(context.Background(), "/", clientv3.WithPrefix())
		if err != nil {
			t.Fatal(err)
		}

		for _, kv := range response.Kvs {
			t.Error(string(kv.Key), "->", string(kv.Value))
		}
	}
}

func addGVKToEtcdBucket(cohabitatingResources map[string]map[schema.GroupVersionKind]empty, gvk schema.GroupVersionKind, bucket string) {
	if cohabitatingResources[bucket] == nil {
		cohabitatingResources[bucket] = map[schema.GroupVersionKind]empty{}
	}
	cohabitatingResources[bucket][gvk] = empty{}
}

// getEtcdBucket assumes the last segment of the given etcd path is the name of the object.
// Thus it strips that segment to extract the object's storage "bucket" in etcd. We expect
// all objects that share the a bucket (cohabitating resources) to be stored as the same GVK.
func getEtcdBucket(path string) string {
	idx := strings.LastIndex(path, "/")
	if idx == -1 {
		panic("path with no slashes " + path)
	}
	bucket := path[:idx]
	if len(bucket) == 0 {
		panic("invalid bucket for path " + path)
	}
	return bucket
}

// stable fields to compare as a sanity check
type metaObject struct {
	// all of type meta
	Kind       string `json:"kind,omitempty" protobuf:"bytes,1,opt,name=kind"`
	APIVersion string `json:"apiVersion,omitempty" protobuf:"bytes,2,opt,name=apiVersion"`

	// parts of object meta
	Metadata struct {
		Name      string `json:"name,omitempty" protobuf:"bytes,1,opt,name=name"`
		Namespace string `json:"namespace,omitempty" protobuf:"bytes,2,opt,name=namespace"`
	} `json:"metadata,omitempty" protobuf:"bytes,3,opt,name=metadata"`
}

func (obj *metaObject) getGVK() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

func (obj *metaObject) isEmpty() bool {
	return obj == nil || *obj == metaObject{} // compare to zero value since all fields are strings
}

type prerequisite struct {
	gvrData schema.GroupVersionResource
	stub    string
}

type empty struct{}

type cleanupData struct {
	obj     runtime.Object
	mapping *meta.RESTMapping
}

func gvr(g, v, r string) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: g, Version: v, Resource: r}
}

func gvkP(g, v, k string) *schema.GroupVersionKind {
	return &schema.GroupVersionKind{Group: g, Version: v, Kind: k}
}

func createEphemeralWhiteList(gvrs ...schema.GroupVersionResource) map[schema.GroupVersionResource]empty {
	ephemeral := map[schema.GroupVersionResource]empty{}
	for _, gvResource := range gvrs {
		if _, ok := ephemeral[gvResource]; ok {
			panic("invalid ephemeral whitelist contains duplicate keys")
		}
		ephemeral[gvResource] = empty{}
	}
	return ephemeral
}

func jsonToMetaObject(stub []byte) (*metaObject, error) {
	obj := &metaObject{}
	if err := json.Unmarshal(stub, obj); err != nil {
		return nil, err
	}
	return obj, nil
}

func keyStringer(i interface{}) string {
	base := "\n\t"
	switch key := i.(type) {
	case string:
		return base + key
	case schema.GroupVersionResource:
		return base + key.String()
	case schema.GroupVersionKind:
		return base + key.String()
	default:
		panic("unexpected type")
	}
}

type allClient struct {
	client  *http.Client
	config  *restclient.Config
	backoff restclient.BackoffManager
}

func (c *allClient) verb(verb string, gvk schema.GroupVersionKind) (*restclient.Request, error) {
	apiPath := "/apis"
	if gvk.Group == kapi.GroupName {
		apiPath = "/api"
	}
	baseURL, versionedAPIPath, err := restclient.DefaultServerURL(c.config.Host, apiPath, gvk.GroupVersion(), true)
	if err != nil {
		return nil, err
	}
	contentConfig := c.config.ContentConfig
	gv := gvk.GroupVersion()
	contentConfig.GroupVersion = &gv
	serializers, err := createSerializers(contentConfig)
	if err != nil {
		return nil, err
	}
	return restclient.NewRequest(c.client, verb, baseURL, versionedAPIPath, contentConfig, *serializers, c.backoff, c.config.RateLimiter), nil
}

func (c *allClient) create(stub, ns string, mapping *meta.RESTMapping, all *[]cleanupData) error {
	req, err := c.verb("POST", mapping.GroupVersionKind)
	if err != nil {
		return err
	}
	namespaced := mapping.Scope.Name() == meta.RESTScopeNameNamespace
	output, err := req.NamespaceIfScoped(ns, namespaced).Resource(mapping.Resource).Body(strings.NewReader(stub)).Do().Get()
	if err != nil {
		return err
	}
	*all = append(*all, cleanupData{output, mapping})
	return nil
}

func (c *allClient) destroy(obj runtime.Object, mapping *meta.RESTMapping) error {
	req, err := c.verb("DELETE", mapping.GroupVersionKind)
	if err != nil {
		return err
	}
	namespaced := mapping.Scope.Name() == meta.RESTScopeNameNamespace
	name, err := mapping.MetadataAccessor.Name(obj)
	if err != nil {
		return err
	}
	ns, err := mapping.MetadataAccessor.Namespace(obj)
	if err != nil {
		return err
	}
	return req.NamespaceIfScoped(ns, namespaced).Resource(mapping.Resource).Name(name).Do().Error()
}

func (c *allClient) cleanup(all *[]cleanupData) error {
	for i := len(*all) - 1; i >= 0; i-- { // delete in reverse order in case creation order mattered
		obj := (*all)[i].obj
		mapping := (*all)[i].mapping

		if err := c.destroy(obj, mapping); err != nil {
			return err
		}
	}
	return nil
}

func (c *allClient) createPrerequisites(mapper meta.RESTMapper, ns string, prerequisites []prerequisite, all *[]cleanupData) error {
	for _, prerequisite := range prerequisites {
		gvk, err := mapper.KindFor(prerequisite.gvrData)
		if err != nil {
			return err
		}
		mapping, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)
		if err != nil {
			return err
		}
		if err := c.create(prerequisite.stub, ns, mapping, all); err != nil {
			return err
		}
	}
	return nil
}

func newClient(config restclient.Config) (*allClient, error) {
	config.ContentConfig.NegotiatedSerializer = kapi.Codecs
	config.ContentConfig.ContentType = "application/json"
	config.Timeout = 30 * time.Second
	config.RateLimiter = flowcontrol.NewTokenBucketRateLimiter(3, 10)

	transport, err := restclient.TransportFor(&config)
	if err != nil {
		return nil, err
	}

	client := &http.Client{
		Transport: transport,
		Timeout:   config.Timeout,
	}

	backoff := &restclient.URLBackoff{
		Backoff: flowcontrol.NewBackOff(1*time.Second, 10*time.Second),
	}

	return &allClient{
		client:  client,
		config:  &config,
		backoff: backoff,
	}, nil
}

// copied from restclient
func createSerializers(config restclient.ContentConfig) (*restclient.Serializers, error) {
	mediaTypes := config.NegotiatedSerializer.SupportedMediaTypes()
	contentType := config.ContentType
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return nil, fmt.Errorf("the content type specified in the client configuration is not recognized: %v", err)
	}
	info, ok := runtime.SerializerInfoForMediaType(mediaTypes, mediaType)
	if !ok {
		if len(contentType) != 0 || len(mediaTypes) == 0 {
			return nil, fmt.Errorf("no serializers registered for %s", contentType)
		}
		info = mediaTypes[0]
	}

	internalGV := schema.GroupVersions{
		{
			Group:   config.GroupVersion.Group,
			Version: runtime.APIVersionInternal,
		},
		// always include the legacy group as a decoding target to handle non-error `Status` return types
		{
			Group:   "",
			Version: runtime.APIVersionInternal,
		},
	}

	s := &restclient.Serializers{
		Encoder: config.NegotiatedSerializer.EncoderForVersion(info.Serializer, *config.GroupVersion),
		Decoder: config.NegotiatedSerializer.DecoderToVersion(info.Serializer, internalGV),

		RenegotiatedDecoder: func(contentType string, params map[string]string) (runtime.Decoder, error) {
			info, ok := runtime.SerializerInfoForMediaType(mediaTypes, contentType)
			if !ok {
				return nil, fmt.Errorf("serializer for %s not registered", contentType)
			}
			return config.NegotiatedSerializer.DecoderToVersion(info.Serializer, internalGV), nil
		},
	}
	if info.StreamSerializer != nil {
		s.StreamingSerializer = info.StreamSerializer.Serializer
		s.Framer = info.StreamSerializer.Framer
	}

	return s, nil
}

func getFromEtcd(keys clientv3.KV, path string) (*metaObject, error) {
	response, err := keys.Get(context.Background(), path)
	if err != nil {
		return nil, err
	}
	if response.More || response.Count != 1 || len(response.Kvs) != 1 {
		return nil, fmt.Errorf("Invalid etcd response (not found == %v): %#v", response.Count == 0, response)
	}
	return jsonToMetaObject(response.Kvs[0].Value)
}

func diffMaps(a, b interface{}) ([]string, []string) {
	inA := diffMapKeys(a, b, keyStringer)
	inB := diffMapKeys(b, a, keyStringer)
	return inA, inB
}

func diffMapKeys(a, b interface{}, stringer func(interface{}) string) []string {
	av := reflect.ValueOf(a)
	bv := reflect.ValueOf(b)
	ret := []string{}

	for _, ka := range av.MapKeys() {
		kat := ka.Interface()
		found := false
		for _, kb := range bv.MapKeys() {
			kbt := kb.Interface()
			if kat == kbt {
				found = true
				break
			}
		}
		if !found {
			ret = append(ret, stringer(kat))
		}
	}

	return ret
}

func getEtcdKVClient(config storagebackend.Config) (clientv3.KV, error) {
	tlsInfo := transport.TLSInfo{
		CertFile: config.CertFile,
		KeyFile:  config.KeyFile,
		CAFile:   config.CAFile,
	}

	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		return nil, err
	}

	cfg := clientv3.Config{
		Endpoints: config.ServerList,
		TLS:       tlsConfig,
	}

	c, err := clientv3.New(cfg)
	if err != nil {
		return nil, err
	}

	return clientv3.NewKV(c), nil
}

type allResourceSource struct{}

func (*allResourceSource) AnyVersionOfResourceEnabled(resource schema.GroupResource) bool { return true }
func (*allResourceSource) AllResourcesForVersionEnabled(version schema.GroupVersion) bool { return true }
func (*allResourceSource) AnyResourcesForGroupEnabled(group string) bool                  { return true }
func (*allResourceSource) ResourceEnabled(resource schema.GroupVersionResource) bool      { return true }
func (*allResourceSource) AnyResourcesForVersionEnabled(version schema.GroupVersion) bool { return true }
