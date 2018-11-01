package manager

import (
	"path"
	"reflect"
	"testing"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/logplugin/v1alpha1"
)

var (
	socketName = "mock.sock"
)

func setUpEndpoint(t *testing.T, socketPath, name string) (*LogPluginStub, pluginEndpoint) {
	p := NewLogPluginStub(socketPath, name)
	err := p.Start()
	if err != nil {
		t.Fatalf("unexpected error, %v", err)
	}

	ep, err := newEndpointImpl(socketPath, name)
	if err != nil {
		t.Fatalf("unexpected error, %v", err)
	}

	return p, ep
}

func cleanUpEndpoint(p *LogPluginStub, ep pluginEndpoint) {
	p.Stop()
	ep.stop()
}

func TestNewEndpoint(t *testing.T) {
	socketPath := path.Join("/tmp", socketName)
	p, ep := setUpEndpoint(t, socketPath, "mock")
	defer cleanUpEndpoint(p, ep)
}

func TestEndpoint(t *testing.T) {
	socketPath := path.Join("/tmp", socketName)
	p, ep := setUpEndpoint(t, socketPath, "mock")
	defer cleanUpEndpoint(p, ep)

	// add config
	config := &pluginapi.Config{
		Metadata: &pluginapi.ConfigMeta{
			Name:          "config-1",
			PodNamespace:  "test",
			PodName:       "test-pod",
			PodUID:        "test-uid",
			ContainerName: "container-1",
		},
		Spec: &pluginapi.ConfigSpec{
			Content:  "",
			Category: "std",
			Path:     "test-path",
		},
	}
	_, err := ep.addConfig(config)
	if err != nil {
		t.Fatalf("unexpected error, %v", err)
	}
	p.setState(config.Metadata.Name, pluginapi.State_Running)

	// list config
	listConfigRsp, err := ep.listConfig()
	if err != nil {
		t.Fatalf("unexpected error, %v", err)
	}
	for _, c := range listConfigRsp.Configs {
		if !reflect.DeepEqual(c.Metadata, config.Metadata) {
			t.Errorf("unexpected metadata, expected: %s, actual: %s", config.Metadata, c.Metadata)
		}
		if c.Spec.Category != config.Spec.Category {
			t.Errorf("unexpected category, expected: %s, actual: %s", config.Spec.Category, c.Spec.Category)
		}
		if c.Spec.Path != config.Spec.Path {
			t.Errorf("unexpected path, expected: %s, actual: %s", config.Spec.Path, c.Spec.Path)
		}
	}

	// get state
	getStateRsp, err := ep.getState("config-1")
	if err != nil {
		t.Fatalf("unexpected error, %v", err)
	}
	if getStateRsp.State != pluginapi.State_Running {
		t.Errorf("unexpected state, expected: %s, actual: %s", pluginapi.State_Running, getStateRsp.State)
	}

	getStateRsp, err = ep.getState("config-2")
	if err != nil {
		t.Fatalf("unexpected error, %v", err)
	}
	if getStateRsp.State != pluginapi.State_NotFound {
		t.Errorf("unexpected state, expected: %s, actual: %s", pluginapi.State_NotFound, getStateRsp.State)
	}

	// delete config
	_, err = ep.delConfig("config-1")
	if err != nil {
		t.Fatalf("unexpected error, %v", err)
	}

	// list config
	listConfigRsp, err = ep.listConfig()
	if err != nil {
		t.Fatalf("unexpected error, %v", err)
	}
	for _, c := range listConfigRsp.Configs {
		if reflect.DeepEqual(c.Metadata, config.Metadata) {
			t.Errorf("unexpected metadata, expected: %s, actual: %s", config.Metadata, c.Metadata)
		}
		if c.Spec.Category != config.Spec.Category {
			t.Errorf("unexpected category, expected: %s, actual: %s", config.Spec.Category, c.Spec.Category)
		}
		if c.Spec.Path != config.Spec.Path {
			t.Errorf("unexpected path, expected: %s, actual: %s", config.Spec.Path, c.Spec.Path)
		}
	}
}
