package util

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/kubelet/log/api/v1beta1"
)

var testcases = []struct {
	pod             *v1.Pod
	logPolicyExists bool
	logPolicy       *v1beta1.PodLogPolicy
	configMapNames  sets.String
}{
	{
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "test",
				Name:      "test-pod-1",
				Annotations: map[string]string{
					v1beta1.PodLogPolicyLabelKey: `{
  "plugin_name": "logexporter",
  "safe_deletion_enabled": false,
  "container_log_policies": [
    {
      "container_name": "container1",
      "name": "std",
      "path": "-",
      "plugin_configmap": "container1-stdlog"
    }, {
      "container_name": "container1",
      "name": "app",
      "path": "/var/log/app",
      "volume_name": "container1-applog",
      "plugin_configmap": "container1-applog"
    }, {
      "container_name": "container1",
      "name": "audit",
      "path": "/var/log/audit",
      "volume_name": "container1-auditlog",
      "plugin_configmap": "container1-auditlog"
    }, {
      "container_name": "container2",
      "name": "app",
      "path": "/var/log/app",
      "volume_name": "container2-applog",
      "plugin_configmap": "container2-applog"
    }, {
      "container_name": "container2",
      "name": "audit",
      "path": "/var/log/audit",
      "volume_name": "container2-auditlog",
      "plugin_configmap": "container2-auditlog"
    }
  ]
}
`,
				},
			},
		},
		true,
		&v1beta1.PodLogPolicy{
			PluginName:          "logexporter",
			SafeDeletionEnabled: false,
			ContainerLogPolicies: []v1beta1.ContainerLogPolicy{
				{
					ContainerName:   "container1",
					Name:            "std",
					Path:            "-",
					PluginConfigMap: "container1-stdlog",
				},
				{
					ContainerName:   "container1",
					Name:            "app",
					Path:            "/var/log/app",
					VolumeName:      "container1-applog",
					PluginConfigMap: "container1-applog",
				},
				{
					ContainerName:   "container1",
					Name:            "audit",
					Path:            "/var/log/audit",
					VolumeName:      "container1-auditlog",
					PluginConfigMap: "container1-auditlog",
				},
				{
					ContainerName:   "container2",
					Name:            "app",
					Path:            "/var/log/app",
					VolumeName:      "container2-applog",
					PluginConfigMap: "container2-applog",
				},
				{
					ContainerName:   "container2",
					Name:            "audit",
					Path:            "/var/log/audit",
					VolumeName:      "container2-auditlog",
					PluginConfigMap: "container2-auditlog",
				},
			},
		},
		sets.NewString(
			"container1-stdlog",
			"container1-applog",
			"container1-auditlog",
			"container2-applog",
			"container2-auditlog",
		),
	},
	{
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "test",
				Name:      "test-pod-2",
				Annotations: map[string]string{
					v1beta1.PodLogPolicyLabelKey: "invalid json",
				},
			},
		},
		true,
		nil,
		sets.NewString(),
	},
	{
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "test",
				Name:      "test-pod-3",
			},
		},
		false,
		nil,
		sets.NewString(),
	},
}

func TestIsPodLogPolicyExists(t *testing.T) {
	for _, tc := range testcases {
		actual := IsPodLogPolicyExists(tc.pod)
		if actual != tc.logPolicyExists {
			t.Errorf("test IsPodLogPolicyExists error, expected: %v, actual: %v", tc.logPolicyExists, actual)
		}
	}
}

func TestGetPodLogPolicy(t *testing.T) {
	for _, tc := range testcases {
		actual, _ := GetPodLogPolicy(tc.pod)
		if !reflect.DeepEqual(actual, tc.logPolicy) {
			t.Errorf("test GetPodLogPolicy error, expected: %v, actual: %v", tc.logPolicy, actual)
		}
	}
}

func TestGetPodLogConfigMapNames(t *testing.T) {
	for _, tc := range testcases {
		actual := GetPodLogConfigMapNames(tc.pod)
		if !actual.Equal(tc.configMapNames) {

			t.Errorf("test GetPodLogConfigMapNames error, expected: %v, actual: %v", tc.configMapNames, actual)
		}
	}
}
