/*
Copyright The Kubernetes Authors.

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

package kuberuntime

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/runtimeclass"
	rctest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
)

type handlerImageService struct {
	internalapi.ImageManagerService
	images   map[string]*runtimeapi.Image
	requests []*runtimeapi.ImageSpec
	failAt   int
	err      error
}

func (s *handlerImageService) ImageStatus(_ context.Context, spec *runtimeapi.ImageSpec, _ bool) (*runtimeapi.ImageStatusResponse, error) {
	s.requests = append(s.requests, proto.Clone(spec).(*runtimeapi.ImageSpec))
	if s.err != nil && (s.failAt == 0 || len(s.requests) == s.failAt) {
		return nil, s.err
	}
	return &runtimeapi.ImageStatusResponse{Image: s.images[spec.RuntimeHandler]}, nil
}

type imageUserRuntimeRecorder struct {
	internalapi.RuntimeService
	configs []*runtimeapi.ContainerConfig
}

func (s *imageUserRuntimeRecorder) CreateContainer(ctx context.Context, sandboxID string, config *runtimeapi.ContainerConfig, sandbox *runtimeapi.PodSandboxConfig) (string, error) {
	s.configs = append(s.configs, proto.Clone(config).(*runtimeapi.ContainerConfig))
	return s.RuntimeService.CreateContainer(ctx, sandboxID, config, sandbox)
}

func TestOrdinaryContainerUsesHandlerImageUser(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("image-derived Linux credentials")
	}
	for _, enabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("image-handler=%t", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClassInImageCriAPI, enabled)
			ctx := ktesting.Init(t)
			runtimeService, images, m, err := createTestRuntimeManager(ctx)
			require.NoError(t, err)
			classes := runtimeclass.NewManager(fake.NewClientset(rctest.NewRuntimeClass("class-a", "handler-a"), rctest.NewRuntimeClass("class-b", "handler-b")))
			defer rctest.StartManagerSync(ktesting.Init(t), classes)()
			m.runtimeClassManager = classes
			imageService := &handlerImageService{ImageManagerService: images, images: map[string]*runtimeapi.Image{
				"":          {Id: "image-default", Uid: &runtimeapi.Int64Value{Value: 1000}},
				"handler-a": {Id: "image-a", Uid: &runtimeapi.Int64Value{Value: 1111}},
				"handler-b": {Id: "image-b", Username: "handler-user"},
			}}
			m.imageService = imageService
			recorder := &imageUserRuntimeRecorder{RuntimeService: runtimeService}
			m.runtimeService = recorder
			for _, class := range []string{"class-a", "class-b"} {
				pod := newTestPod()
				pod.Name = class
				pod.UID = types.UID(class)
				pod.Spec.RuntimeClassName = &class
				pod.Spec.SecurityContext = nil
				pod.Spec.Containers[0].SecurityContext = nil
				pod.Spec.Containers[0].Image = "same-image:latest"
				pod.Spec.Containers[0].ImagePullPolicy = v1.PullIfNotPresent
				require.Nil(t, pod.Spec.RestoreFrom)
				sandbox, err := m.generatePodSandboxConfig(ctx, pod, 0)
				require.NoError(t, err)
				id, err := m.runtimeService.RunPodSandbox(ctx, sandbox, "")
				require.NoError(t, err)
				_, err = m.startContainer(ctx, id, sandbox, containerStartSpec(&pod.Spec.Containers[0]), pod, &kubecontainer.PodStatus{}, nil, "", nil, nil)
				require.NoError(t, err)
			}
			require.Len(t, recorder.configs, 2)
			for i, handler := range []string{"handler-a", "handler-b"} {
				want := handler
				if !enabled {
					want = ""
				}
				credentials := recorder.configs[i].GetLinux().GetSecurityContext()
				image := imageService.images[want]
				require.Equal(t, image.Uid, credentials.RunAsUser)
				require.Equal(t, image.Username, credentials.RunAsUsername)
				require.Equal(t, image.Id, recorder.configs[i].Image.Image)
			}
			for _, spec := range imageService.requests {
				if enabled {
					require.Contains(t, []string{"handler-a", "handler-b"}, spec.RuntimeHandler)
				} else {
					require.Empty(t, spec.RuntimeHandler)
				}
			}
			require.Len(t, runtimeService.Containers, 2)
			for _, container := range runtimeService.Containers {
				require.Equal(t, runtimeapi.ContainerState_CONTAINER_RUNNING, container.State)
			}
		})
	}
}

func TestImageUserLookupErrorsPreserveBoundary(t *testing.T) {
	ctx := ktesting.Init(t)
	_, images, m, err := createTestRuntimeManager(ctx)
	require.NoError(t, err)
	sentinel := errors.New("image metadata unavailable")
	m.imageService = &handlerImageService{ImageManagerService: images, err: sentinel}
	_, _, err = m.getImageUser(ctx, "image", "handler")
	require.ErrorIs(t, err, sentinel)
}

func TestOrdinaryContainerMissingRuntimeClass(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("image-handler=%t", enabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClassInImageCriAPI, enabled)
			ctx := ktesting.Init(t)
			runtimeService, _, m, err := createTestRuntimeManager(ctx)
			require.NoError(t, err)
			classes := runtimeclass.NewManager(rctest.NewPopulatedClient())
			defer rctest.StartManagerSync(ktesting.Init(t), classes)()
			m.runtimeClassManager = classes
			pod := newTestPod()
			sandbox, err := m.generatePodSandboxConfig(ctx, pod, 0)
			require.NoError(t, err)
			id, err := m.runtimeService.RunPodSandbox(ctx, sandbox, "")
			require.NoError(t, err)
			pod.Spec.RuntimeClassName = new("missing")
			_, err = m.startContainer(ctx, id, sandbox, containerStartSpec(&pod.Spec.Containers[0]), pod, &kubecontainer.PodStatus{}, nil, "", nil, nil)
			if enabled {
				require.True(t, apierrors.IsNotFound(err), "%v", err)
			} else {
				require.ErrorIs(t, err, ErrCreateContainerConfig)
			}
			require.Empty(t, runtimeService.Containers)
			require.NotContains(t, runtimeService.Called, "CreateContainer")
			require.NotContains(t, runtimeService.Called, "StartContainer")
		})
	}
}

func TestOrdinaryContainerImageUserFailureDoesNotCreate(t *testing.T) {
	ctx := ktesting.Init(t)
	runtimeService, images, m, err := createTestRuntimeManager(ctx)
	require.NoError(t, err)
	sentinel := errors.New("image user metadata unavailable")
	m.imageService = &handlerImageService{ImageManagerService: images, images: map[string]*runtimeapi.Image{"": {Id: "image"}}, failAt: 2, err: sentinel}
	pod := newTestPod()
	pod.Spec.Containers[0].ImagePullPolicy = v1.PullIfNotPresent
	sandbox, err := m.generatePodSandboxConfig(ctx, pod, 0)
	require.NoError(t, err)
	id, err := m.runtimeService.RunPodSandbox(ctx, sandbox, "")
	require.NoError(t, err)
	message, err := m.startContainer(ctx, id, sandbox, containerStartSpec(&pod.Spec.Containers[0]), pod, &kubecontainer.PodStatus{}, nil, "", nil, nil)
	require.ErrorIs(t, err, ErrCreateContainerConfig)
	require.Contains(t, message, sentinel.Error())
	require.Empty(t, runtimeService.Containers)
	require.NotContains(t, runtimeService.Called, "CreateContainer")
	require.NotContains(t, runtimeService.Called, "StartContainer")
}
