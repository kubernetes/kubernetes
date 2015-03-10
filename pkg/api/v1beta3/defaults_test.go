/*
Copyright 2015 Google Inc. All rights reserved.

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

package v1beta3_test

import (
	"reflect"
	"testing"

	current "github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	data, err := current.Codec.Encode(obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = current.Codec.DecodeInto(data, obj2)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	return obj2
}

func TestSetDefaultService(t *testing.T) {
	svc := &current.Service{}
	obj2 := roundTrip(t, runtime.Object(svc))
	svc2 := obj2.(*current.Service)
	if svc2.Spec.Protocol != current.ProtocolTCP {
		t.Errorf("Expected default protocol :%s, got: %s", current.ProtocolTCP, svc2.Spec.Protocol)
	}
	if svc2.Spec.SessionAffinity != current.AffinityTypeNone {
		t.Errorf("Expected default sesseion affinity type:%s, got: %s", current.AffinityTypeNone, svc2.Spec.SessionAffinity)
	}
}

func TestSetDefaulPodSpec(t *testing.T) {
	bp := &current.BoundPod{}
	bp.Spec.Volumes = []current.Volume{{}}

	obj2 := roundTrip(t, runtime.Object(bp))
	bp2 := obj2.(*current.BoundPod)
	if bp2.Spec.DNSPolicy != current.DNSClusterFirst {
		t.Errorf("Expected default dns policy :%s, got: %s", current.DNSClusterFirst, bp2.Spec.DNSPolicy)
	}
	policy := bp2.Spec.RestartPolicy
	if policy.Never != nil || policy.OnFailure != nil || policy.Always == nil {
		t.Errorf("Expected only policy.Always is set, got: %s", policy)
	}
	vsource := bp2.Spec.Volumes[0].VolumeSource
	if vsource.EmptyDir == nil {
		t.Errorf("Expected non-empty volume is set, got: %s", vsource.EmptyDir)
	}
}

func TestSetDefaultContainer(t *testing.T) {
	bp := &current.BoundPod{}
	bp.Spec.Containers = []current.Container{{}}
	bp.Spec.Containers[0].Ports = []current.ContainerPort{{}}

	obj2 := roundTrip(t, runtime.Object(bp))
	bp2 := obj2.(*current.BoundPod)

	container := bp2.Spec.Containers[0]
	if container.TerminationMessagePath != current.TerminationMessagePathDefault {
		t.Errorf("Expected termination message path: %s, got: %s",
			current.TerminationMessagePathDefault, container.TerminationMessagePath)
	}
	if container.ImagePullPolicy != current.PullIfNotPresent {
		t.Errorf("Expected image pull policy: %s, got: %s",
			current.PullIfNotPresent, container.ImagePullPolicy)
	}
	if container.Ports[0].Protocol != current.ProtocolTCP {
		t.Errorf("Expected protocol: %s, got: %s",
			current.ProtocolTCP, container.Ports[0].Protocol)
	}
}

func TestSetDefaultSecret(t *testing.T) {
	s := &current.Secret{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*current.Secret)

	if s2.Type != current.SecretTypeOpaque {
		t.Errorf("Expected secret type %v, got %v", current.SecretTypeOpaque, s2.Type)
	}
}

func TestSetDefaulEndpointsProtocol(t *testing.T) {
	in := &current.Endpoints{}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*current.Endpoints)

	if out.Protocol != current.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", current.ProtocolTCP, out.Protocol)
	}
}

func TestSetDefaulServiceDestinationPort(t *testing.T) {
	in := &current.Service{Spec: current.ServiceSpec{Port: 1234}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*current.Service)
	if out.Spec.ContainerPort.Kind != util.IntstrInt || out.Spec.ContainerPort.IntVal != 1234 {
		t.Errorf("Expected ContainerPort to be defaulted, got %s", out.Spec.ContainerPort)
	}

	in = &current.Service{Spec: current.ServiceSpec{Port: 1234, ContainerPort: util.NewIntOrStringFromInt(5678)}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*current.Service)
	if out.Spec.ContainerPort.Kind != util.IntstrInt || out.Spec.ContainerPort.IntVal != 5678 {
		t.Errorf("Expected ContainerPort to be unchanged, got %s", out.Spec.ContainerPort)
	}
}

func TestSetDefaultNamespace(t *testing.T) {
	s := &current.Namespace{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*current.Namespace)

	if s2.Status.Phase != current.NamespaceActive {
		t.Errorf("Expected phase %v, got %v", current.NamespaceActive, s2.Status.Phase)
	}
}
