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
	"net"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapiserveroptions "k8s.io/apiserver/pkg/server/options"
	cacheddiscovery "k8s.io/client-go/discovery/cached"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	"k8s.io/kubernetes/cmd/kube-apiserver/app"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"

	// install all APIs
	_ "k8s.io/kubernetes/pkg/master" // TODO what else is needed

	"github.com/coreos/etcd/clientv3"
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
		expectedGVK:      gvkP("apps", "v1", "StatefulSet"),
	},
	gvr("apps", "v1beta1", "deployments"): {
		stub:             `{"metadata": {"name": "deployment2"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
		expectedEtcdPath: "/registry/deployments/etcdstoragepathtestnamespace/deployment2",
		expectedGVK:      gvkP("apps", "v1", "Deployment"),
	},
	gvr("apps", "v1beta1", "controllerrevisions"): {
		stub:             `{"metadata":{"name":"crs1"},"data":{"name":"abc","namespace":"default","creationTimestamp":null,"Spec":{"Replicas":0,"Selector":{"matchLabels":{"foo":"bar"}},"Template":{"creationTimestamp":null,"labels":{"foo":"bar"},"Spec":{"Volumes":null,"InitContainers":null,"Containers":null,"RestartPolicy":"Always","TerminationGracePeriodSeconds":null,"ActiveDeadlineSeconds":null,"DNSPolicy":"ClusterFirst","NodeSelector":null,"ServiceAccountName":"","AutomountServiceAccountToken":null,"NodeName":"","SecurityContext":null,"ImagePullSecrets":null,"Hostname":"","Subdomain":"","Affinity":null,"SchedulerName":"","Tolerations":null,"HostAliases":null}},"VolumeClaimTemplates":null,"ServiceName":""},"Status":{"ObservedGeneration":null,"Replicas":0}},"revision":0}`,
		expectedEtcdPath: "/registry/controllerrevisions/etcdstoragepathtestnamespace/crs1",
		expectedGVK:      gvkP("apps", "v1", "ControllerRevision"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/apps/v1beta2
	gvr("apps", "v1beta2", "statefulsets"): {
		stub:             `{"metadata": {"name": "ss2"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}}}}`,
		expectedEtcdPath: "/registry/statefulsets/etcdstoragepathtestnamespace/ss2",
		expectedGVK:      gvkP("apps", "v1", "StatefulSet"),
	},
	gvr("apps", "v1beta2", "deployments"): {
		stub:             `{"metadata": {"name": "deployment3"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
		expectedEtcdPath: "/registry/deployments/etcdstoragepathtestnamespace/deployment3",
		expectedGVK:      gvkP("apps", "v1", "Deployment"),
	},
	gvr("apps", "v1beta2", "daemonsets"): {
		stub:             `{"metadata": {"name": "ds5"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
		expectedEtcdPath: "/registry/daemonsets/etcdstoragepathtestnamespace/ds5",
		expectedGVK:      gvkP("apps", "v1", "DaemonSet"),
	},
	gvr("apps", "v1beta2", "replicasets"): {
		stub:             `{"metadata": {"name": "rs2"}, "spec": {"selector": {"matchLabels": {"g": "h"}}, "template": {"metadata": {"labels": {"g": "h"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container4"}]}}}}`,
		expectedEtcdPath: "/registry/replicasets/etcdstoragepathtestnamespace/rs2",
		expectedGVK:      gvkP("apps", "v1", "ReplicaSet"),
	},
	gvr("apps", "v1beta2", "controllerrevisions"): {
		stub:             `{"metadata":{"name":"crs2"},"data":{"name":"abc","namespace":"default","creationTimestamp":null,"Spec":{"Replicas":0,"Selector":{"matchLabels":{"foo":"bar"}},"Template":{"creationTimestamp":null,"labels":{"foo":"bar"},"Spec":{"Volumes":null,"InitContainers":null,"Containers":null,"RestartPolicy":"Always","TerminationGracePeriodSeconds":null,"ActiveDeadlineSeconds":null,"DNSPolicy":"ClusterFirst","NodeSelector":null,"ServiceAccountName":"","AutomountServiceAccountToken":null,"NodeName":"","SecurityContext":null,"ImagePullSecrets":null,"Hostname":"","Subdomain":"","Affinity":null,"SchedulerName":"","Tolerations":null,"HostAliases":null}},"VolumeClaimTemplates":null,"ServiceName":""},"Status":{"ObservedGeneration":null,"Replicas":0}},"revision":0}`,
		expectedEtcdPath: "/registry/controllerrevisions/etcdstoragepathtestnamespace/crs2",
		expectedGVK:      gvkP("apps", "v1", "ControllerRevision"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/apps/v1
	gvr("apps", "v1", "daemonsets"): {
		stub:             `{"metadata": {"name": "ds6"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
		expectedEtcdPath: "/registry/daemonsets/etcdstoragepathtestnamespace/ds6",
	},
	gvr("apps", "v1", "deployments"): {
		stub:             `{"metadata": {"name": "deployment4"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
		expectedEtcdPath: "/registry/deployments/etcdstoragepathtestnamespace/deployment4",
	},
	gvr("apps", "v1", "statefulsets"): {
		stub:             `{"metadata": {"name": "ss3"}, "spec": {"selector": {"matchLabels": {"a": "b"}}, "template": {"metadata": {"labels": {"a": "b"}}}}}`,
		expectedEtcdPath: "/registry/statefulsets/etcdstoragepathtestnamespace/ss3",
	},
	gvr("apps", "v1", "replicasets"): {
		stub:             `{"metadata": {"name": "rs3"}, "spec": {"selector": {"matchLabels": {"g": "h"}}, "template": {"metadata": {"labels": {"g": "h"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container4"}]}}}}`,
		expectedEtcdPath: "/registry/replicasets/etcdstoragepathtestnamespace/rs3",
	},
	gvr("apps", "v1", "controllerrevisions"): {
		stub:             `{"metadata":{"name":"crs3"},"data":{"name":"abc","namespace":"default","creationTimestamp":null,"Spec":{"Replicas":0,"Selector":{"matchLabels":{"foo":"bar"}},"Template":{"creationTimestamp":null,"labels":{"foo":"bar"},"Spec":{"Volumes":null,"InitContainers":null,"Containers":null,"RestartPolicy":"Always","TerminationGracePeriodSeconds":null,"ActiveDeadlineSeconds":null,"DNSPolicy":"ClusterFirst","NodeSelector":null,"ServiceAccountName":"","AutomountServiceAccountToken":null,"NodeName":"","SecurityContext":null,"ImagePullSecrets":null,"Hostname":"","Subdomain":"","Affinity":null,"SchedulerName":"","Tolerations":null,"HostAliases":null}},"VolumeClaimTemplates":null,"ServiceName":""},"Status":{"ObservedGeneration":null,"Replicas":0}},"revision":0}`,
		expectedEtcdPath: "/registry/controllerrevisions/etcdstoragepathtestnamespace/crs3",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/autoscaling/v1
	gvr("autoscaling", "v1", "horizontalpodautoscalers"): {
		stub:             `{"metadata": {"name": "hpa2"}, "spec": {"maxReplicas": 3, "scaleTargetRef": {"kind": "something", "name": "cross"}}}`,
		expectedEtcdPath: "/registry/horizontalpodautoscalers/etcdstoragepathtestnamespace/hpa2",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/autoscaling/v2beta1
	gvr("autoscaling", "v2beta1", "horizontalpodautoscalers"): {
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
	},
	// --

	// k8s.io/kubernetes/pkg/apis/batch/v2alpha1
	gvr("batch", "v2alpha1", "cronjobs"): {
		stub:             `{"metadata": {"name": "cjv2alpha1"}, "spec": {"jobTemplate": {"spec": {"template": {"metadata": {"labels": {"controller-uid": "uid0"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container0"}], "dnsPolicy": "ClusterFirst", "restartPolicy": "Never"}}}}, "schedule": "* * * * *"}}`,
		expectedEtcdPath: "/registry/cronjobs/etcdstoragepathtestnamespace/cjv2alpha1",
		expectedGVK:      gvkP("batch", "v1beta1", "CronJob"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/certificates/v1beta1
	gvr("certificates.k8s.io", "v1beta1", "certificatesigningrequests"): {
		stub:             `{"metadata": {"name": "csr1"}, "spec": {"request": "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0KTUlJQnlqQ0NBVE1DQVFBd2dZa3hDekFKQmdOVkJBWVRBbFZUTVJNd0VRWURWUVFJRXdwRFlXeHBabTl5Ym1saApNUll3RkFZRFZRUUhFdzFOYjNWdWRHRnBiaUJXYVdWM01STXdFUVlEVlFRS0V3cEhiMjluYkdVZ1NXNWpNUjh3CkhRWURWUVFMRXhaSmJtWnZjbTFoZEdsdmJpQlVaV05vYm05c2IyZDVNUmN3RlFZRFZRUURFdzUzZDNjdVoyOXYKWjJ4bExtTnZiVENCbnpBTkJna3Foa2lHOXcwQkFRRUZBQU9CalFBd2dZa0NnWUVBcFp0WUpDSEo0VnBWWEhmVgpJbHN0UVRsTzRxQzAzaGpYK1prUHl2ZFlkMVE0K3FiQWVUd1htQ1VLWUhUaFZSZDVhWFNxbFB6eUlCd2llTVpyCldGbFJRZGRaMUl6WEFsVlJEV3dBbzYwS2VjcWVBWG5uVUsrNWZYb1RJL1VnV3NocmU4dEoreC9UTUhhUUtSL0oKY0lXUGhxYVFoc0p1elpidkFkR0E4MEJMeGRNQ0F3RUFBYUFBTUEwR0NTcUdTSWIzRFFFQkJRVUFBNEdCQUlobAo0UHZGcStlN2lwQVJnSTVaTStHWng2bXBDejQ0RFRvMEprd2ZSRGYrQnRyc2FDMHE2OGVUZjJYaFlPc3E0ZmtIClEwdUEwYVZvZzNmNWlKeENhM0hwNWd4YkpRNnpWNmtKMFRFc3VhYU9oRWtvOXNkcENvUE9uUkJtMmkvWFJEMkQKNmlOaDhmOHowU2hHc0ZxakRnRkh5RjNvK2xVeWorVUM2SDFRVzdibgotLS0tLUVORCBDRVJUSUZJQ0FURSBSRVFVRVNULS0tLS0="}}`,
		expectedEtcdPath: "/registry/certificatesigningrequests/csr1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/coordination/v1beta1
	gvr("coordination.k8s.io", "v1beta1", "leases"): {
		stub:             `{"metadata": {"name": "lease1"}, "spec": {"holderIdentity": "holder", "leaseDurationSeconds": 5}}`,
		expectedEtcdPath: "/registry/leases/etcdstoragepathtestnamespace/lease1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/events/v1beta1
	gvr("events.k8s.io", "v1beta1", "events"): {
		stub:             `{"metadata": {"name": "event2"}, "regarding": {"namespace": "etcdstoragepathtestnamespace"}, "note": "some data here", "eventTime": "2017-08-09T15:04:05.000000Z", "reportingInstance": "node-xyz", "reportingController": "k8s.io/my-controller", "action": "DidNothing", "reason": "Laziness"}`,
		expectedEtcdPath: "/registry/events/etcdstoragepathtestnamespace/event2",
		expectedGVK:      gvkP("", "v1", "Event"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/extensions/v1beta1
	gvr("extensions", "v1beta1", "daemonsets"): {
		stub:             `{"metadata": {"name": "ds1"}, "spec": {"selector": {"matchLabels": {"u": "t"}}, "template": {"metadata": {"labels": {"u": "t"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container5"}]}}}}`,
		expectedEtcdPath: "/registry/daemonsets/etcdstoragepathtestnamespace/ds1",
		expectedGVK:      gvkP("apps", "v1", "DaemonSet"),
	},
	gvr("extensions", "v1beta1", "podsecuritypolicies"): {
		stub:             `{"metadata": {"name": "psp1"}, "spec": {"fsGroup": {"rule": "RunAsAny"}, "privileged": true, "runAsUser": {"rule": "RunAsAny"}, "seLinux": {"rule": "MustRunAs"}, "supplementalGroups": {"rule": "RunAsAny"}}}`,
		expectedEtcdPath: "/registry/podsecuritypolicy/psp1",
		expectedGVK:      gvkP("policy", "v1beta1", "PodSecurityPolicy"),
	},
	gvr("extensions", "v1beta1", "ingresses"): {
		stub:             `{"metadata": {"name": "ingress1"}, "spec": {"backend": {"serviceName": "service", "servicePort": 5000}}}`,
		expectedEtcdPath: "/registry/ingress/etcdstoragepathtestnamespace/ingress1",
	},
	gvr("extensions", "v1beta1", "networkpolicies"): {
		stub:             `{"metadata": {"name": "np1"}, "spec": {"podSelector": {"matchLabels": {"e": "f"}}}}`,
		expectedEtcdPath: "/registry/networkpolicies/etcdstoragepathtestnamespace/np1",
		expectedGVK:      gvkP("networking.k8s.io", "v1", "NetworkPolicy"),
	},
	gvr("extensions", "v1beta1", "deployments"): {
		stub:             `{"metadata": {"name": "deployment1"}, "spec": {"selector": {"matchLabels": {"f": "z"}}, "template": {"metadata": {"labels": {"f": "z"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container6"}]}}}}`,
		expectedEtcdPath: "/registry/deployments/etcdstoragepathtestnamespace/deployment1",
		expectedGVK:      gvkP("apps", "v1", "Deployment"),
	},
	gvr("extensions", "v1beta1", "replicasets"): {
		stub:             `{"metadata": {"name": "rs1"}, "spec": {"selector": {"matchLabels": {"g": "h"}}, "template": {"metadata": {"labels": {"g": "h"}}, "spec": {"containers": [{"image": "fedora:latest", "name": "container4"}]}}}}`,
		expectedEtcdPath: "/registry/replicasets/etcdstoragepathtestnamespace/rs1",
		expectedGVK:      gvkP("apps", "v1", "ReplicaSet"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/networking/v1
	gvr("networking.k8s.io", "v1", "networkpolicies"): {
		stub:             `{"metadata": {"name": "np2"}, "spec": {"podSelector": {"matchLabels": {"e": "f"}}}}`,
		expectedEtcdPath: "/registry/networkpolicies/etcdstoragepathtestnamespace/np2",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/policy/v1beta1
	gvr("policy", "v1beta1", "poddisruptionbudgets"): {
		stub:             `{"metadata": {"name": "pdb1"}, "spec": {"selector": {"matchLabels": {"anokkey": "anokvalue"}}}}`,
		expectedEtcdPath: "/registry/poddisruptionbudgets/etcdstoragepathtestnamespace/pdb1",
	},
	gvr("policy", "v1beta1", "podsecuritypolicies"): {
		stub:             `{"metadata": {"name": "psp2"}, "spec": {"fsGroup": {"rule": "RunAsAny"}, "privileged": true, "runAsUser": {"rule": "RunAsAny"}, "seLinux": {"rule": "MustRunAs"}, "supplementalGroups": {"rule": "RunAsAny"}}}`,
		expectedEtcdPath: "/registry/podsecuritypolicy/psp2",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/storage/v1alpha1
	gvr("storage.k8s.io", "v1alpha1", "volumeattachments"): {
		stub:             `{"metadata": {"name": "va1"}, "spec": {"attacher": "gce", "nodeName": "localhost", "source": {"persistentVolumeName": "pv1"}}}`,
		expectedEtcdPath: "/registry/volumeattachments/va1",
		expectedGVK:      gvkP("storage.k8s.io", "v1beta1", "VolumeAttachment"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/storage/v1beta1
	gvr("storage.k8s.io", "v1beta1", "volumeattachments"): {
		stub:             `{"metadata": {"name": "va2"}, "spec": {"attacher": "gce", "nodeName": "localhost", "source": {"persistentVolumeName": "pv2"}}}`,
		expectedEtcdPath: "/registry/volumeattachments/va2",
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
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "Role"),
	},
	gvr("rbac.authorization.k8s.io", "v1alpha1", "clusterroles"): {
		stub:             `{"metadata": {"name": "crole1"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
		expectedEtcdPath: "/registry/clusterroles/crole1",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "ClusterRole"),
	},
	gvr("rbac.authorization.k8s.io", "v1alpha1", "rolebindings"): {
		stub:             `{"metadata": {"name": "roleb1"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/rolebindings/etcdstoragepathtestnamespace/roleb1",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "RoleBinding"),
	},
	gvr("rbac.authorization.k8s.io", "v1alpha1", "clusterrolebindings"): {
		stub:             `{"metadata": {"name": "croleb1"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/clusterrolebindings/croleb1",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "ClusterRoleBinding"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/rbac/v1beta1
	gvr("rbac.authorization.k8s.io", "v1beta1", "roles"): {
		stub:             `{"metadata": {"name": "role2"}, "rules": [{"apiGroups": ["v1"], "resources": ["events"], "verbs": ["watch"]}]}`,
		expectedEtcdPath: "/registry/roles/etcdstoragepathtestnamespace/role2",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "Role"),
	},
	gvr("rbac.authorization.k8s.io", "v1beta1", "clusterroles"): {
		stub:             `{"metadata": {"name": "crole2"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
		expectedEtcdPath: "/registry/clusterroles/crole2",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "ClusterRole"),
	},
	gvr("rbac.authorization.k8s.io", "v1beta1", "rolebindings"): {
		stub:             `{"metadata": {"name": "roleb2"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/rolebindings/etcdstoragepathtestnamespace/roleb2",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "RoleBinding"),
	},
	gvr("rbac.authorization.k8s.io", "v1beta1", "clusterrolebindings"): {
		stub:             `{"metadata": {"name": "croleb2"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/clusterrolebindings/croleb2",
		expectedGVK:      gvkP("rbac.authorization.k8s.io", "v1", "ClusterRoleBinding"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/rbac/v1
	gvr("rbac.authorization.k8s.io", "v1", "roles"): {
		stub:             `{"metadata": {"name": "role3"}, "rules": [{"apiGroups": ["v1"], "resources": ["events"], "verbs": ["watch"]}]}`,
		expectedEtcdPath: "/registry/roles/etcdstoragepathtestnamespace/role3",
	},
	gvr("rbac.authorization.k8s.io", "v1", "clusterroles"): {
		stub:             `{"metadata": {"name": "crole3"}, "rules": [{"nonResourceURLs": ["/version"], "verbs": ["get"]}]}`,
		expectedEtcdPath: "/registry/clusterroles/crole3",
	},
	gvr("rbac.authorization.k8s.io", "v1", "rolebindings"): {
		stub:             `{"metadata": {"name": "roleb3"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/rolebindings/etcdstoragepathtestnamespace/roleb3",
	},
	gvr("rbac.authorization.k8s.io", "v1", "clusterrolebindings"): {
		stub:             `{"metadata": {"name": "croleb3"}, "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "ClusterRole", "name": "somecr"}, "subjects": [{"apiVersion": "rbac.authorization.k8s.io/v1alpha1", "kind": "Group", "name": "system:authenticated"}]}`,
		expectedEtcdPath: "/registry/clusterrolebindings/croleb3",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/admissionregistration/v1alpha1
	gvr("admissionregistration.k8s.io", "v1alpha1", "initializerconfigurations"): {
		stub:             `{"metadata":{"name":"ic1"},"initializers":[{"name":"initializer.k8s.io","rules":[{"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore"}]}`,
		expectedEtcdPath: "/registry/initializerconfigurations/ic1",
	},
	// k8s.io/kubernetes/pkg/apis/admissionregistration/v1beta1
	gvr("admissionregistration.k8s.io", "v1beta1", "validatingwebhookconfigurations"): {
		stub:             `{"metadata":{"name":"hook1","creationTimestamp":null},"webhooks":[{"name":"externaladmissionhook.k8s.io","clientConfig":{"service":{"namespace":"ns","name":"n"},"caBundle":null},"rules":[{"operations":["CREATE"],"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore"}]}`,
		expectedEtcdPath: "/registry/validatingwebhookconfigurations/hook1",
	},
	gvr("admissionregistration.k8s.io", "v1beta1", "mutatingwebhookconfigurations"): {
		stub:             `{"metadata":{"name":"hook1","creationTimestamp":null},"webhooks":[{"name":"externaladmissionhook.k8s.io","clientConfig":{"service":{"namespace":"ns","name":"n"},"caBundle":null},"rules":[{"operations":["CREATE"],"apiGroups":["group"],"apiVersions":["version"],"resources":["resource"]}],"failurePolicy":"Ignore"}]}`,
		expectedEtcdPath: "/registry/mutatingwebhookconfigurations/hook1",
	},
	// --

	// k8s.io/kubernetes/pkg/apis/scheduling/v1alpha1
	gvr("scheduling.k8s.io", "v1alpha1", "priorityclasses"): {
		stub:             `{"metadata":{"name":"pc1"},"Value":1000}`,
		expectedEtcdPath: "/registry/priorityclasses/pc1",
		expectedGVK:      gvkP("scheduling.k8s.io", "v1beta1", "PriorityClass"),
	},
	// --

	// k8s.io/kubernetes/pkg/apis/scheduling/v1beta1
	gvr("scheduling.k8s.io", "v1beta1", "priorityclasses"): {
		stub:             `{"metadata":{"name":"pc2"},"Value":1000}`,
		expectedEtcdPath: "/registry/priorityclasses/pc2",
	},
	// --

	// k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1
	// depends on aggregator using the same ungrouped RESTOptionsGetter as the kube apiserver, not SimpleRestOptionsFactory in aggregator.go
	gvr("apiregistration.k8s.io", "v1beta1", "apiservices"): {
		stub:             `{"metadata": {"name": "as1.foo.com"}, "spec": {"group": "foo.com", "version": "as1", "groupPriorityMinimum":100, "versionPriority":10}}`,
		expectedEtcdPath: "/registry/apiregistration.k8s.io/apiservices/as1.foo.com",
	},
	// --

	// k8s.io/kube-aggregator/pkg/apis/apiregistration/v1
	// depends on aggregator using the same ungrouped RESTOptionsGetter as the kube apiserver, not SimpleRestOptionsFactory in aggregator.go
	gvr("apiregistration.k8s.io", "v1", "apiservices"): {
		stub:             `{"metadata": {"name": "as2.foo.com"}, "spec": {"group": "foo.com", "version": "as2", "groupPriorityMinimum":100, "versionPriority":10}}`,
		expectedEtcdPath: "/registry/apiregistration.k8s.io/apiservices/as2.foo.com",
		expectedGVK:      gvkP("apiregistration.k8s.io", "v1beta1", "APIService"),
	},
	// --

	// k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1
	gvr("apiextensions.k8s.io", "v1beta1", "customresourcedefinitions"): {
		stub:             `{"metadata": {"name": "openshiftwebconsoleconfigs.webconsole.operator.openshift.io"},"spec": {"scope": "Cluster","group": "webconsole.operator.openshift.io","version": "v1alpha1","names": {"kind": "OpenShiftWebConsoleConfig","plural": "openshiftwebconsoleconfigs","singular": "openshiftwebconsoleconfig"}}}`,
		expectedEtcdPath: "/registry/apiextensions.k8s.io/customresourcedefinitions/openshiftwebconsoleconfigs.webconsole.operator.openshift.io",
	},
	// --

}

// Only add kinds to this list when this a virtual resource with get and create verbs that doesn't actually
// store into it's kind.  We've used this downstream for mappings before.
var kindWhiteList = sets.NewString()

// namespace used for all tests, do not change this
const testNamespace = "etcdstoragepathtestnamespace"

// TestEtcdStoragePath tests to make sure that all objects are stored in an expected location in etcd.
// It will start failing when a new type is added to ensure that all future types are added to this test.
// It will also fail when a type gets moved to a different location. Be very careful in this situation because
// it essentially means that you will be break old clusters unless you create some migration path for the old data.
func TestEtcdStoragePath(t *testing.T) {
	certDir, _ := ioutil.TempDir("", "test-integration-etcd")
	defer os.RemoveAll(certDir)

	clientConfig, kvClient := startRealMasterOrDie(t, certDir)
	defer func() {
		dumpEtcdKVOnFailure(t, kvClient)
	}()

	client := &allClient{dynamicClient: dynamic.NewForConfigOrDie(clientConfig)}
	kubeClient := clientset.NewForConfigOrDie(clientConfig)
	if _, err := kubeClient.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespace}}); err != nil {
		t.Fatal(err)
	}

	discoveryClient := cacheddiscovery.NewMemCacheClient(kubeClient.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(discoveryClient)
	restMapper.Reset()

	resourcesToPersist := []resourceToPersist{}
	serverResources, err := kubeClient.Discovery().ServerResources()
	if err != nil {
		t.Fatal(err)
	}
	resourcesToPersist = append(resourcesToPersist, getResourcesToPersist(serverResources, false, t)...)

	kindSeen := sets.NewString()
	pathSeen := map[string][]schema.GroupVersionResource{}
	etcdSeen := map[schema.GroupVersionResource]empty{}
	cohabitatingResources := map[string]map[schema.GroupVersionKind]empty{}

	for _, resourceToPersist := range resourcesToPersist {
		t.Run(resourceToPersist.gvr.String(), func(t *testing.T) {
			gvk := resourceToPersist.gvk
			gvResource := resourceToPersist.gvr
			kind := gvk.Kind

			mapping := &meta.RESTMapping{
				Resource:         resourceToPersist.gvr,
				GroupVersionKind: resourceToPersist.gvk,
				Scope:            meta.RESTScopeRoot,
			}
			if resourceToPersist.namespaced {
				mapping.Scope = meta.RESTScopeNamespace
			}

			if kindWhiteList.Has(kind) {
				kindSeen.Insert(kind)
				t.Skip("whitelisted")
			}

			etcdSeen[gvResource] = empty{}
			testData, hasTest := etcdStorageData[gvResource]

			if !hasTest {
				t.Fatalf("no test data for %s.  Please add a test for your new type to etcdStorageData.", gvResource)
			}

			if len(testData.expectedEtcdPath) == 0 {
				t.Fatalf("empty test data for %s", gvResource)
			}

			shouldCreate := len(testData.stub) != 0 // try to create only if we have a stub

			var input *metaObject
			if shouldCreate {
				if input, err = jsonToMetaObject([]byte(testData.stub)); err != nil || input.isEmpty() {
					t.Fatalf("invalid test data for %s: %v", gvResource, err)
				}
			}

			all := &[]cleanupData{}
			defer func() {
				if !t.Failed() { // do not cleanup if test has already failed since we may need things in the etcd dump
					if err := client.cleanup(all); err != nil {
						t.Fatalf("failed to clean up etcd: %#v", err)
					}
				}
			}()

			if err := client.createPrerequisites(restMapper, testNamespace, testData.prerequisites, all); err != nil {
				t.Fatalf("failed to create prerequisites for %s: %#v", gvResource, err)
			}

			if shouldCreate { // do not try to create items with no stub
				if err := client.create(testData.stub, testNamespace, mapping, all); err != nil {
					t.Fatalf("failed to create stub for %s: %#v", gvResource, err)
				}
			}

			output, err := getFromEtcd(kvClient, testData.expectedEtcdPath)
			if err != nil {
				t.Fatalf("failed to get from etcd for %s: %#v", gvResource, err)
			}

			expectedGVK := gvk
			if testData.expectedGVK != nil {
				if gvk == *testData.expectedGVK {
					t.Errorf("GVK override %s for %s is unnecessary or something was changed incorrectly", testData.expectedGVK, gvk)
				}
				expectedGVK = *testData.expectedGVK
			}

			actualGVK := output.getGVK()
			if actualGVK != expectedGVK {
				t.Errorf("GVK for %s does not match, expected %s got %s", kind, expectedGVK, actualGVK)
			}

			if !apiequality.Semantic.DeepDerivative(input, output) {
				t.Errorf("Test stub for %s does not match: %s", kind, diff.ObjectGoPrintDiff(input, output))
			}

			addGVKToEtcdBucket(cohabitatingResources, actualGVK, getEtcdBucket(testData.expectedEtcdPath))
			pathSeen[testData.expectedEtcdPath] = append(pathSeen[testData.expectedEtcdPath], mapping.Resource)
		})
	}

	if inEtcdData, inEtcdSeen := diffMaps(etcdStorageData, etcdSeen); len(inEtcdData) != 0 || len(inEtcdSeen) != 0 {
		t.Errorf("etcd data does not match the types we saw:\nin etcd data but not seen:\n%s\nseen but not in etcd data:\n%s", inEtcdData, inEtcdSeen)
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

func startRealMasterOrDie(t *testing.T, certDir string) (*restclient.Config, clientv3.KV) {
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

	kvClient, err := integration.GetEtcdKVClient(kubeAPIServerOptions.Etcd.StorageConfig)
	if err != nil {
		t.Fatal(err)
	}

	return kubeClientConfig, kvClient
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
	obj      *unstructured.Unstructured
	resource schema.GroupVersionResource
}

func gvr(g, v, r string) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: g, Version: v, Resource: r}
}

func gvkP(g, v, k string) *schema.GroupVersionKind {
	return &schema.GroupVersionKind{Group: g, Version: v, Kind: k}
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
	dynamicClient dynamic.Interface
}

func (c *allClient) create(stub, ns string, mapping *meta.RESTMapping, all *[]cleanupData) error {
	// we don't require GVK on the data we provide, so we fill it in here.  We could, but that seems extraneous.
	typeMetaAdder := map[string]interface{}{}
	err := json.Unmarshal([]byte(stub), &typeMetaAdder)
	if err != nil {
		return err
	}
	typeMetaAdder["apiVersion"] = mapping.GroupVersionKind.GroupVersion().String()
	typeMetaAdder["kind"] = mapping.GroupVersionKind.Kind

	if mapping.Scope == meta.RESTScopeRoot {
		ns = ""
	}
	obj := &unstructured.Unstructured{Object: typeMetaAdder}
	actual, err := c.dynamicClient.Resource(mapping.Resource).Namespace(ns).Create(obj, metav1.CreateOptions{})
	if err != nil {
		return err
	}

	*all = append(*all, cleanupData{actual, mapping.Resource})
	return nil
}

func (c *allClient) cleanup(all *[]cleanupData) error {
	for i := len(*all) - 1; i >= 0; i-- { // delete in reverse order in case creation order mattered
		obj := (*all)[i].obj
		gvr := (*all)[i].resource

		if err := c.dynamicClient.Resource(gvr).Namespace(obj.GetNamespace()).Delete(obj.GetName(), nil); err != nil {
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

type resourceToPersist struct {
	gvk        schema.GroupVersionKind
	gvr        schema.GroupVersionResource
	golangType reflect.Type
	namespaced bool
}

func getResourcesToPersist(serverResources []*metav1.APIResourceList, isOAPI bool, t *testing.T) []resourceToPersist {
	resourcesToPersist := []resourceToPersist{}

	for _, discoveryGroup := range serverResources {
		for _, discoveryResource := range discoveryGroup.APIResources {
			// this is a subresource, skip it
			if strings.Contains(discoveryResource.Name, "/") {
				continue
			}
			hasCreate := false
			hasGet := false
			for _, verb := range discoveryResource.Verbs {
				if string(verb) == "get" {
					hasGet = true
				}
				if string(verb) == "create" {
					hasCreate = true
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

			resourcesToPersist = append(resourcesToPersist, resourceToPersist{
				gvk:        gvk,
				gvr:        gvr,
				namespaced: discoveryResource.Namespaced,
			})
		}
	}

	return resourcesToPersist
}
