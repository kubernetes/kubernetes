package manager

import (
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/logplugin/v1beta1"
	"k8s.io/kubernetes/pkg/kubelet/log/api"
	"k8s.io/kubernetes/pkg/kubelet/log/api/util"
	"k8s.io/kubernetes/pkg/kubelet/log/policy"
)

var testcases = []struct {
	pod                 *v1.Pod
	configs             []*pluginapi.Config
	state               pluginapi.State
	finished            bool
	safeDeletionEnabled bool
}{
	{
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "test",
				Name:      "test-pod-1",
				UID:       "test-pod-1-uid",
				Annotations: map[string]string{
					api.PodLogPolicyLabelKey: `{
  "plugin_name": "logexporter",
  "safe_deletion_enabled": false,
  "container_log_policies": [
    {
      "container_name": "container1",
      "name": "std",
      "path": "-",
      "plugin_configmap": "container1-stdlog"
    }
  ]
}
`,
				},
			},
		},
		[]*pluginapi.Config{
			{
				Metadata: &pluginapi.ConfigMeta{
					Name:   "config-1",
					PodUID: "test-pod-1-uid",
				},
				Spec: &pluginapi.ConfigSpec{},
			},
		},
		pluginapi.State_Running,
		false,
		false,
	},
	{
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "test",
				Name:      "test-pod-2",
				UID:       "test-pod-2-uid",
				Annotations: map[string]string{
					api.PodLogPolicyLabelKey: `{
  "plugin_name": "logexporter",
  "safe_deletion_enabled": true,
  "container_log_policies": [
    {
      "container_name": "container1",
      "name": "std",
      "path": "-",
      "plugin_configmap": "container1-stdlog"
    }
  ]
}
`,
				},
			},
		},
		nil,
		pluginapi.State_NotFound,
		true,
		true,
	},
	{
		&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: "test",
				Name:      "test-pod-3",
				UID:       "test-pod-3-uid",
				Annotations: map[string]string{
					api.PodLogPolicyLabelKey: `{
  "plugin_name": "logexporter",
  "safe_deletion_enabled": true,
  "container_log_policies": [
    {
      "container_name": "container1",
      "name": "std",
      "path": "-",
      "plugin_configmap": "container1-stdlog"
    }
  ]
}
`,
				},
			},
		},
		[]*pluginapi.Config{
			{
				Metadata: &pluginapi.ConfigMeta{
					Name:   "config-3",
					PodUID: "test-pod-3-uid",
				},
				Spec: &pluginapi.ConfigSpec{},
			},
		},
		pluginapi.State_Idle,
		true,
		true,
	},
}

func TestCollectFinished(t *testing.T) {
	manager := &ManagerImpl{
		logPlugins:          make(map[string]pluginEndpoint),
		policyStatusManager: policy.NewPolicyStatusManager(),
		pluginStatusManager: newPluginStatusManager(),
	}

	socketPath := "/tmp/mock.sock"
	logPluginName := "logexporter"

	p, ep := setUpEndpoint(t, socketPath, logPluginName)
	defer cleanUpEndpoint(p, ep)
	manager.logPlugins[logPluginName] = ep

	configs := make([]*pluginapi.Config, 0)
	for _, tc := range testcases {
		logPolicy, err := util.GetPodLogPolicy(tc.pod)
		if err != nil {
			t.Fatalf("unexpected error, %v", err)
		}
		manager.policyStatusManager.UpdateLogPolicy(tc.pod.UID, logPolicy)
		for _, config := range tc.configs {
			_, err := ep.addConfig(config)
			if err != nil {
				t.Fatalf("unexpected error, %v", err)
			}
			p.setState(config.Metadata.Name, tc.state)
		}
		if tc.configs == nil {
			continue
		}
		configs = append(configs, tc.configs...)
		manager.pluginStatusManager.updateAllLogConfigs(configs, ep.name())

		isFinished, _ := manager.isCollectFinished(tc.pod, logPolicy)
		manager.policyStatusManager.UpdateCollectFinishedStatus(tc.pod.UID, isFinished)
	}

	for _, tc := range testcases {
		actual := manager.IsCollectFinished(tc.pod.UID)
		if actual != tc.finished {
			t.Errorf("test CollectFinished failed, expected: %t, actual: %t", tc.finished, actual)
		}
	}
}

func TestSafeDeletionEnabled(t *testing.T) {
	for _, tc := range testcases {
		logPolicy, err := util.GetPodLogPolicy(tc.pod)
		if err != nil {
			t.Fatalf("unexpected error, %v", err)
		}
		if logPolicy.SafeDeletionEnabled != tc.safeDeletionEnabled {
			t.Errorf("test SafeDeletionEnabled failed, expected: %t, actual: %t", tc.safeDeletionEnabled, logPolicy.SafeDeletionEnabled)
		}
	}
}
