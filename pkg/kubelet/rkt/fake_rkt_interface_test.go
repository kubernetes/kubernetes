/*
Copyright 2015 The Kubernetes Authors.

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

package rkt

import (
	"fmt"
	"strconv"
	"strings"
	"sync"

	"github.com/coreos/go-systemd/dbus"
	rktapi "github.com/coreos/rkt/api/v1alpha"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"k8s.io/apimachinery/pkg/types"
	kubetypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// fakeRktInterface mocks the rktapi.PublicAPIClient interface for testing purpose.
type fakeRktInterface struct {
	sync.Mutex
	info       rktapi.Info
	images     []*rktapi.Image
	podFilters []*rktapi.PodFilter
	pods       []*rktapi.Pod
	called     []string
	err        error
}

func newFakeRktInterface() *fakeRktInterface {
	return &fakeRktInterface{}
}

func (f *fakeRktInterface) CleanCalls() {
	f.Lock()
	defer f.Unlock()
	f.called = nil
}

func (f *fakeRktInterface) GetInfo(ctx context.Context, in *rktapi.GetInfoRequest, opts ...grpc.CallOption) (*rktapi.GetInfoResponse, error) {
	f.Lock()
	defer f.Unlock()

	f.called = append(f.called, "GetInfo")
	return &rktapi.GetInfoResponse{Info: &f.info}, f.err
}

func (f *fakeRktInterface) ListPods(ctx context.Context, in *rktapi.ListPodsRequest, opts ...grpc.CallOption) (*rktapi.ListPodsResponse, error) {
	f.Lock()
	defer f.Unlock()

	f.called = append(f.called, "ListPods")
	f.podFilters = in.Filters
	return &rktapi.ListPodsResponse{Pods: f.pods}, f.err
}

func (f *fakeRktInterface) InspectPod(ctx context.Context, in *rktapi.InspectPodRequest, opts ...grpc.CallOption) (*rktapi.InspectPodResponse, error) {
	f.Lock()
	defer f.Unlock()

	f.called = append(f.called, "InspectPod")
	for _, pod := range f.pods {
		if pod.Id == in.Id {
			return &rktapi.InspectPodResponse{Pod: pod}, f.err
		}
	}
	return &rktapi.InspectPodResponse{}, fmt.Errorf("pod %q not found", in.Id)
}

func (f *fakeRktInterface) ListImages(ctx context.Context, in *rktapi.ListImagesRequest, opts ...grpc.CallOption) (*rktapi.ListImagesResponse, error) {
	f.Lock()
	defer f.Unlock()

	f.called = append(f.called, "ListImages")
	return &rktapi.ListImagesResponse{Images: f.images}, f.err
}

func (f *fakeRktInterface) InspectImage(ctx context.Context, in *rktapi.InspectImageRequest, opts ...grpc.CallOption) (*rktapi.InspectImageResponse, error) {
	return nil, fmt.Errorf("Not implemented")
}

func (f *fakeRktInterface) ListenEvents(ctx context.Context, in *rktapi.ListenEventsRequest, opts ...grpc.CallOption) (rktapi.PublicAPI_ListenEventsClient, error) {
	return nil, fmt.Errorf("Not implemented")
}

func (f *fakeRktInterface) GetLogs(ctx context.Context, in *rktapi.GetLogsRequest, opts ...grpc.CallOption) (rktapi.PublicAPI_GetLogsClient, error) {
	return nil, fmt.Errorf("Not implemented")
}

// fakeSystemd mocks the systemdInterface for testing purpose.
// TODO(yifan): Remove this once we have a package for launching rkt pods.
// See https://github.com/coreos/rkt/issues/1769.
type fakeSystemd struct {
	sync.Mutex
	called           []string
	resetFailedUnits []string
	version          string
	err              error
}

func newFakeSystemd() *fakeSystemd {
	return &fakeSystemd{}
}

func (f *fakeSystemd) CleanCalls() {
	f.Lock()
	defer f.Unlock()
	f.called = nil
}

func (f *fakeSystemd) Version() (systemdVersion, error) {
	f.Lock()
	defer f.Unlock()

	f.called = append(f.called, "Version")
	v, _ := strconv.Atoi(f.version)
	return systemdVersion(v), f.err
}

func (f *fakeSystemd) ListUnits() ([]dbus.UnitStatus, error) {
	return nil, fmt.Errorf("Not implemented")
}

func (f *fakeSystemd) StopUnit(name string, mode string, ch chan<- string) (int, error) {
	return 0, fmt.Errorf("Not implemented")
}

func (f *fakeSystemd) RestartUnit(name string, mode string, ch chan<- string) (int, error) {
	return 0, fmt.Errorf("Not implemented")
}

func (f *fakeSystemd) ResetFailedUnit(name string) error {
	f.called = append(f.called, "ResetFailedUnit")
	f.resetFailedUnits = append(f.resetFailedUnits, name)
	return f.err
}

type fakeRktCli struct {
	sync.Mutex
	cmds   []string
	result []string
	err    error
}

func newFakeRktCli() *fakeRktCli {
	return &fakeRktCli{
		cmds:   []string{},
		result: []string{},
	}
}

func (f *fakeRktCli) RunCommand(config *Config, args ...string) (result []string, err error) {
	f.Lock()
	defer f.Unlock()
	cmd := append([]string{"rkt"}, args...)
	f.cmds = append(f.cmds, strings.Join(cmd, " "))
	return f.result, f.err
}

func (f *fakeRktCli) Reset() {
	f.cmds = []string{}
	f.result = []string{}
	f.err = nil
}

type fakePodGetter struct {
	pods map[types.UID]*v1.Pod
}

func newFakePodGetter() *fakePodGetter {
	return &fakePodGetter{pods: make(map[types.UID]*v1.Pod)}
}

func (f fakePodGetter) GetPodByUID(uid types.UID) (*v1.Pod, bool) {
	p, found := f.pods[uid]
	return p, found
}

type fakeUnitGetter struct {
	networkNamespace kubecontainer.ContainerID
	callServices     []string
}

func newfakeUnitGetter() *fakeUnitGetter {
	return &fakeUnitGetter{
		networkNamespace: kubecontainer.ContainerID{},
	}
}

func (f *fakeUnitGetter) getNetworkNamespace(uid kubetypes.UID, latestPod *rktapi.Pod) (kubecontainer.ContainerID, error) {
	return kubecontainer.ContainerID{ID: "42"}, nil
}

func (f *fakeUnitGetter) getKubernetesDirective(serviceFilePath string) (podServiceDirective, error) {
	podService := podServiceDirective{
		id:               "fake",
		name:             "fake",
		namespace:        "fake",
		hostNetwork:      true,
		networkNamespace: kubecontainer.ContainerID{ID: "42"},
	}
	return podService, nil
}
