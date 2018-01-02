package convert

import (
	"testing"

	swarmtypes "github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/api/types/swarm/runtime"
	swarmapi "github.com/docker/swarmkit/api"
	google_protobuf3 "github.com/gogo/protobuf/types"
)

func TestServiceConvertFromGRPCRuntimeContainer(t *testing.T) {
	gs := swarmapi.Service{
		Meta: swarmapi.Meta{
			Version: swarmapi.Version{
				Index: 1,
			},
			CreatedAt: nil,
			UpdatedAt: nil,
		},
		SpecVersion: &swarmapi.Version{
			Index: 1,
		},
		Spec: swarmapi.ServiceSpec{
			Task: swarmapi.TaskSpec{
				Runtime: &swarmapi.TaskSpec_Container{
					Container: &swarmapi.ContainerSpec{
						Image: "alpine:latest",
					},
				},
			},
		},
	}

	svc, err := ServiceFromGRPC(gs)
	if err != nil {
		t.Fatal(err)
	}

	if svc.Spec.TaskTemplate.Runtime != swarmtypes.RuntimeContainer {
		t.Fatalf("expected type %s; received %T", swarmtypes.RuntimeContainer, svc.Spec.TaskTemplate.Runtime)
	}
}

func TestServiceConvertFromGRPCGenericRuntimePlugin(t *testing.T) {
	kind := string(swarmtypes.RuntimePlugin)
	url := swarmtypes.RuntimeURLPlugin
	gs := swarmapi.Service{
		Meta: swarmapi.Meta{
			Version: swarmapi.Version{
				Index: 1,
			},
			CreatedAt: nil,
			UpdatedAt: nil,
		},
		SpecVersion: &swarmapi.Version{
			Index: 1,
		},
		Spec: swarmapi.ServiceSpec{
			Task: swarmapi.TaskSpec{
				Runtime: &swarmapi.TaskSpec_Generic{
					Generic: &swarmapi.GenericRuntimeSpec{
						Kind: kind,
						Payload: &google_protobuf3.Any{
							TypeUrl: string(url),
						},
					},
				},
			},
		},
	}

	svc, err := ServiceFromGRPC(gs)
	if err != nil {
		t.Fatal(err)
	}

	if svc.Spec.TaskTemplate.Runtime != swarmtypes.RuntimePlugin {
		t.Fatalf("expected type %s; received %T", swarmtypes.RuntimePlugin, svc.Spec.TaskTemplate.Runtime)
	}
}

func TestServiceConvertToGRPCGenericRuntimePlugin(t *testing.T) {
	s := swarmtypes.ServiceSpec{
		TaskTemplate: swarmtypes.TaskSpec{
			Runtime:    swarmtypes.RuntimePlugin,
			PluginSpec: &runtime.PluginSpec{},
		},
		Mode: swarmtypes.ServiceMode{
			Global: &swarmtypes.GlobalService{},
		},
	}

	svc, err := ServiceSpecToGRPC(s)
	if err != nil {
		t.Fatal(err)
	}

	v, ok := svc.Task.Runtime.(*swarmapi.TaskSpec_Generic)
	if !ok {
		t.Fatal("expected type swarmapi.TaskSpec_Generic")
	}

	if v.Generic.Payload.TypeUrl != string(swarmtypes.RuntimeURLPlugin) {
		t.Fatalf("expected url %s; received %s", swarmtypes.RuntimeURLPlugin, v.Generic.Payload.TypeUrl)
	}
}

func TestServiceConvertToGRPCContainerRuntime(t *testing.T) {
	image := "alpine:latest"
	s := swarmtypes.ServiceSpec{
		TaskTemplate: swarmtypes.TaskSpec{
			ContainerSpec: &swarmtypes.ContainerSpec{
				Image: image,
			},
		},
		Mode: swarmtypes.ServiceMode{
			Global: &swarmtypes.GlobalService{},
		},
	}

	svc, err := ServiceSpecToGRPC(s)
	if err != nil {
		t.Fatal(err)
	}

	v, ok := svc.Task.Runtime.(*swarmapi.TaskSpec_Container)
	if !ok {
		t.Fatal("expected type swarmapi.TaskSpec_Container")
	}

	if v.Container.Image != image {
		t.Fatalf("expected image %s; received %s", image, v.Container.Image)
	}
}

func TestServiceConvertToGRPCGenericRuntimeCustom(t *testing.T) {
	s := swarmtypes.ServiceSpec{
		TaskTemplate: swarmtypes.TaskSpec{
			Runtime: "customruntime",
		},
		Mode: swarmtypes.ServiceMode{
			Global: &swarmtypes.GlobalService{},
		},
	}

	if _, err := ServiceSpecToGRPC(s); err != ErrUnsupportedRuntime {
		t.Fatal(err)
	}
}
