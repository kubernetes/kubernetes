/*
Copyright 2023 The Kubernetes Authors.

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

package dra

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	v1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	drapbv1 "k8s.io/kubelet/pkg/apis/dra/v1alpha3"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/plugin"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

const (
	driverName      = "test-cdi-device"
	driverClassName = "test"
)

type fakeDRADriverGRPCServer struct {
	drapbv1.UnimplementedNodeServer
	driverName             string
	timeout                *time.Duration
	prepareResourceCalls   atomic.Uint32
	unprepareResourceCalls atomic.Uint32
}

func (s *fakeDRADriverGRPCServer) NodePrepareResources(ctx context.Context, req *drapbv1.NodePrepareResourcesRequest) (*drapbv1.NodePrepareResourcesResponse, error) {
	s.prepareResourceCalls.Add(1)

	if s.timeout != nil {
		time.Sleep(*s.timeout)
	}
	deviceName := "claim-" + req.Claims[0].Uid
	result := s.driverName + "/" + driverClassName + "=" + deviceName
	return &drapbv1.NodePrepareResourcesResponse{Claims: map[string]*drapbv1.NodePrepareResourceResponse{req.Claims[0].Uid: {CDIDevices: []string{result}}}}, nil
}

func (s *fakeDRADriverGRPCServer) NodeUnprepareResources(ctx context.Context, req *drapbv1.NodeUnprepareResourcesRequest) (*drapbv1.NodeUnprepareResourcesResponse, error) {
	s.unprepareResourceCalls.Add(1)

	if s.timeout != nil {
		time.Sleep(*s.timeout)
	}
	return &drapbv1.NodeUnprepareResourcesResponse{Claims: map[string]*drapbv1.NodeUnprepareResourceResponse{req.Claims[0].Uid: {}}}, nil
}

type tearDown func()

type fakeDRAServerInfo struct {
	// fake DRA server
	server *fakeDRADriverGRPCServer
	// fake DRA plugin socket name
	socketName string
	// teardownFn stops fake gRPC server
	teardownFn tearDown
}

func setupFakeDRADriverGRPCServer(shouldTimeout bool) (fakeDRAServerInfo, error) {
	socketDir, err := os.MkdirTemp("", "dra")
	if err != nil {
		return fakeDRAServerInfo{
			server:     nil,
			socketName: "",
			teardownFn: nil,
		}, err
	}

	socketName := filepath.Join(socketDir, "server.sock")
	stopCh := make(chan struct{})

	teardown := func() {
		close(stopCh)
		os.RemoveAll(socketName)
	}

	l, err := net.Listen("unix", socketName)
	if err != nil {
		teardown()
		return fakeDRAServerInfo{
			server:     nil,
			socketName: "",
			teardownFn: nil,
		}, err
	}

	s := grpc.NewServer()
	fakeDRADriverGRPCServer := &fakeDRADriverGRPCServer{
		driverName: driverName,
	}
	if shouldTimeout {
		timeout := plugin.PluginClientTimeout + time.Second
		fakeDRADriverGRPCServer.timeout = &timeout
	}

	drapbv1.RegisterNodeServer(s, fakeDRADriverGRPCServer)

	go func() {
		go s.Serve(l)
		<-stopCh
		s.GracefulStop()
	}()

	return fakeDRAServerInfo{
		server:     fakeDRADriverGRPCServer,
		socketName: socketName,
		teardownFn: teardown,
	}, nil
}

func TestNewManagerImpl(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	for _, test := range []struct {
		description        string
		stateFileDirectory string
		wantErr            bool
	}{
		{
			description:        "invalid directory path",
			stateFileDirectory: "",
			wantErr:            true,
		},
		{
			description:        "valid directory path",
			stateFileDirectory: t.TempDir(),
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			manager, err := NewManagerImpl(kubeClient, test.stateFileDirectory)
			if test.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, manager.cache)
			assert.NotNil(t, manager.kubeClient)
		})
	}
}

func TestGetResources(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()
	resourceClaimName := "test-pod-claim-1"

	for _, test := range []struct {
		description string
		container   *v1.Container
		pod         *v1.Pod
		claimInfo   *ClaimInfo
		wantErr     bool
	}{
		{
			description: "claim info with annotations",
			container: &v1.Container{
				Name: "test-container",
				Resources: v1.ResourceRequirements{
					Claims: []v1.ResourceClaim{
						{
							Name: "test-pod-claim-1",
						},
					},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name:   "test-pod-claim-1",
							Source: v1.ClaimSource{ResourceClaimName: &resourceClaimName},
						},
					},
				},
			},
			claimInfo: &ClaimInfo{
				annotations: map[string][]kubecontainer.Annotation{
					"test-plugin": {
						{
							Name:  "test-annotation",
							Value: "123",
						},
					},
				},
				ClaimInfoState: state.ClaimInfoState{
					ClaimName: "test-pod-claim-1",
					CDIDevices: map[string][]string{
						driverName: {"123"},
					},
					Namespace: "test-namespace",
				},
			},
		},
		{
			description: "claim info without annotations",
			container: &v1.Container{
				Name: "test-container",
				Resources: v1.ResourceRequirements{
					Claims: []v1.ResourceClaim{
						{
							Name: "test-pod-claim-1",
						},
					},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name:   "test-pod-claim-1",
							Source: v1.ClaimSource{ResourceClaimName: &resourceClaimName},
						},
					},
				},
			},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimName: "test-pod-claim-1",
					CDIDevices: map[string][]string{
						driverName: {"123"},
					},
					Namespace: "test-namespace",
				},
			},
		},
		{
			description: "no claim info",
			container: &v1.Container{
				Name: "test-container",
				Resources: v1.ResourceRequirements{
					Claims: []v1.ResourceClaim{
						{
							Name: "test-pod-claim-1",
						},
					},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim-1",
						},
					},
				},
			},
			wantErr: true,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			manager, err := NewManagerImpl(kubeClient, t.TempDir())
			assert.NoError(t, err)

			if test.claimInfo != nil {
				manager.cache.add(test.claimInfo)
			}

			containerInfo, err := manager.GetResources(test.pod, test.container)
			if test.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, test.claimInfo.CDIDevices[driverName][0], containerInfo.CDIDevices[0].Name)
		})
	}
}

func TestPrepareResources(t *testing.T) {
	fakeKubeClient := fake.NewSimpleClientset()

	for _, test := range []struct {
		description          string
		driverName           string
		pod                  *v1.Pod
		claimInfo            *ClaimInfo
		resourceClaim        *resourcev1alpha2.ResourceClaim
		wantErr              bool
		wantTimeout          bool
		wantResourceSkipped  bool
		ExpectedPrepareCalls uint32
	}{
		{
			description: "failed to fetch ResourceClaim",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim-0",
							Source: v1.ClaimSource{
								ResourceClaimName: func() *string {
									s := "test-pod-claim-0"
									return &s
								}(),
							},
						},
					},
				},
			},
			wantErr: true,
		},
		{
			description: "plugin does not exist",
			driverName:  "this-plugin-does-not-exist",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim-1",
							Source: v1.ClaimSource{
								ResourceClaimName: func() *string {
									s := "test-pod-claim-1"
									return &s
								}(),
							},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{
									{
										Name: "test-pod-claim-1",
									},
								},
							},
						},
					},
				},
			},
			resourceClaim: &resourcev1alpha2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-claim-1",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: resourcev1alpha2.ResourceClaimSpec{
					ResourceClassName: "test-class",
				},
				Status: resourcev1alpha2.ResourceClaimStatus{
					DriverName: driverName,
					Allocation: &resourcev1alpha2.AllocationResult{
						ResourceHandles: []resourcev1alpha2.ResourceHandle{
							{Data: "test-data", DriverName: driverName},
						},
					},
					ReservedFor: []resourcev1alpha2.ResourceClaimConsumerReference{
						{UID: "test-reserved"},
					},
				},
			},
			wantErr: true,
		},
		{
			description: "pod is not allowed to use resource claim",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim-2",
							Source: v1.ClaimSource{
								ResourceClaimName: func() *string {
									s := "test-pod-claim-2"
									return &s
								}(),
							},
						},
					},
				},
			},
			resourceClaim: &resourcev1alpha2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-claim-2",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: resourcev1alpha2.ResourceClaimSpec{
					ResourceClassName: "test-class",
				},
				Status: resourcev1alpha2.ResourceClaimStatus{
					DriverName: driverName,
					Allocation: &resourcev1alpha2.AllocationResult{
						ResourceHandles: []resourcev1alpha2.ResourceHandle{
							{Data: "test-data", DriverName: driverName},
						},
					},
				},
			},
			wantErr: true,
		},
		{
			description: "no container actually uses the claim",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim-3",
							Source: v1.ClaimSource{ResourceClaimName: func() *string {
								s := "test-pod-claim-3"
								return &s
							}()},
						},
					},
				},
			},
			resourceClaim: &resourcev1alpha2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-claim-3",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: resourcev1alpha2.ResourceClaimSpec{
					ResourceClassName: "test-class",
				},
				Status: resourcev1alpha2.ResourceClaimStatus{
					DriverName: driverName,
					Allocation: &resourcev1alpha2.AllocationResult{
						ResourceHandles: []resourcev1alpha2.ResourceHandle{
							{Data: "test-data", DriverName: driverName},
						},
					},
					ReservedFor: []resourcev1alpha2.ResourceClaimConsumerReference{
						{UID: "test-reserved"},
					},
				},
			},
			wantResourceSkipped: true,
		},
		{
			description: "resource already prepared",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim-4",
							Source: v1.ClaimSource{ResourceClaimName: func() *string {
								s := "test-pod-claim-4"
								return &s
							}()},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{
									{
										Name: "test-pod-claim-4",
									},
								},
							},
						},
					},
				},
			},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: driverName,
					ClaimName:  "test-pod-claim-4",
					Namespace:  "test-namespace",
					PodUIDs:    sets.Set[string]{"test-another-pod-reserved": sets.Empty{}},
				},
				prepared: true,
			},
			resourceClaim: &resourcev1alpha2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-claim-4",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: resourcev1alpha2.ResourceClaimSpec{
					ResourceClassName: "test-class",
				},
				Status: resourcev1alpha2.ResourceClaimStatus{
					DriverName: driverName,
					Allocation: &resourcev1alpha2.AllocationResult{
						ResourceHandles: []resourcev1alpha2.ResourceHandle{
							{Data: "test-data", DriverName: driverName},
						},
					},
					ReservedFor: []resourcev1alpha2.ResourceClaimConsumerReference{
						{UID: "test-reserved"},
					},
				},
			},
			wantResourceSkipped: true,
		},
		{
			description: "should timeout",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim-5",
							Source: v1.ClaimSource{ResourceClaimName: func() *string {
								s := "test-pod-claim-5"
								return &s
							}()},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{
									{
										Name: "test-pod-claim-5",
									},
								},
							},
						},
					},
				},
			},
			resourceClaim: &resourcev1alpha2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-claim-5",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: resourcev1alpha2.ResourceClaimSpec{
					ResourceClassName: "test-class",
				},
				Status: resourcev1alpha2.ResourceClaimStatus{
					DriverName: driverName,
					Allocation: &resourcev1alpha2.AllocationResult{
						ResourceHandles: []resourcev1alpha2.ResourceHandle{
							{Data: "test-data", DriverName: driverName},
						},
					},
					ReservedFor: []resourcev1alpha2.ResourceClaimConsumerReference{
						{UID: "test-reserved"},
					},
				},
			},
			wantErr:              true,
			wantTimeout:          true,
			ExpectedPrepareCalls: 1,
		},
		{
			description: "should prepare resource, claim not in cache",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim-6",
							Source: v1.ClaimSource{ResourceClaimName: func() *string {
								s := "test-pod-claim-6"
								return &s
							}()},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{
									{
										Name: "test-pod-claim-6",
									},
								},
							},
						},
					},
				},
			},
			resourceClaim: &resourcev1alpha2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-claim-6",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: resourcev1alpha2.ResourceClaimSpec{
					ResourceClassName: "test-class",
				},
				Status: resourcev1alpha2.ResourceClaimStatus{
					DriverName: driverName,
					Allocation: &resourcev1alpha2.AllocationResult{
						ResourceHandles: []resourcev1alpha2.ResourceHandle{
							{Data: "test-data"},
						},
					},
					ReservedFor: []resourcev1alpha2.ResourceClaimConsumerReference{
						{UID: "test-reserved"},
					},
				},
			},
			ExpectedPrepareCalls: 1,
		},
		{
			description: "should prepare resource. claim in cache, manager did not prepare resource",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim",
							Source: v1.ClaimSource{ResourceClaimName: func() *string {
								s := "test-pod-claim"
								return &s
							}()},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{
									{
										Name: "test-pod-claim",
									},
								},
							},
						},
					},
				},
			},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: driverName,
					ClassName:  "test-class",
					ClaimName:  "test-pod-claim",
					ClaimUID:   "test-reserved",
					Namespace:  "test-namespace",
					PodUIDs:    sets.Set[string]{"test-reserved": sets.Empty{}},
					CDIDevices: map[string][]string{
						driverName: {fmt.Sprintf("%s/%s=some-device", driverName, driverClassName)},
					},
					ResourceHandles: []resourcev1alpha2.ResourceHandle{{Data: "test-data"}},
				},
				annotations: make(map[string][]kubecontainer.Annotation),
				prepared:    false,
			},
			resourceClaim: &resourcev1alpha2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-claim",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: resourcev1alpha2.ResourceClaimSpec{
					ResourceClassName: "test-class",
				},
				Status: resourcev1alpha2.ResourceClaimStatus{
					DriverName: driverName,
					Allocation: &resourcev1alpha2.AllocationResult{
						ResourceHandles: []resourcev1alpha2.ResourceHandle{
							{Data: "test-data"},
						},
					},
					ReservedFor: []resourcev1alpha2.ResourceClaimConsumerReference{
						{UID: "test-reserved"},
					},
				},
			},
			ExpectedPrepareCalls: 1,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			cache, err := newClaimInfoCache(t.TempDir(), draManagerStateFileName)
			if err != nil {
				t.Fatalf("failed to newClaimInfoCache, err:%v", err)
			}

			manager := &ManagerImpl{
				kubeClient: fakeKubeClient,
				cache:      cache,
			}

			if test.resourceClaim != nil {
				if _, err := fakeKubeClient.ResourceV1alpha2().ResourceClaims(test.pod.Namespace).Create(context.Background(), test.resourceClaim, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create ResourceClaim %s: %+v", test.resourceClaim.Name, err)
				}
			}

			draServerInfo, err := setupFakeDRADriverGRPCServer(test.wantTimeout)
			if err != nil {
				t.Fatal(err)
			}
			defer draServerInfo.teardownFn()

			plg := plugin.NewRegistrationHandler()
			if err := plg.RegisterPlugin(test.driverName, draServerInfo.socketName, []string{"1.27"}); err != nil {
				t.Fatalf("failed to register plugin %s, err: %v", test.driverName, err)
			}
			defer plg.DeRegisterPlugin(test.driverName) // for sake of next tests

			if test.claimInfo != nil {
				manager.cache.add(test.claimInfo)
			}

			err = manager.PrepareResources(test.pod)

			assert.Equal(t, test.ExpectedPrepareCalls, draServerInfo.server.prepareResourceCalls.Load())

			if test.wantErr {
				assert.Error(t, err)
				return // PrepareResources returned an error so stopping the subtest here
			} else if test.wantResourceSkipped {
				assert.NoError(t, err)
				return // resource skipped so no need to continue
			}

			assert.NoError(t, err)
			// check the cache contains the expected claim info
			claimName, _, err := resourceclaim.Name(test.pod, &test.pod.Spec.ResourceClaims[0])
			if err != nil {
				t.Fatal(err)
			}
			claimInfo := manager.cache.get(*claimName, test.pod.Namespace)
			if claimInfo == nil {
				t.Fatalf("claimInfo not found in cache for claim %s", *claimName)
			}
			if claimInfo.DriverName != test.resourceClaim.Status.DriverName {
				t.Fatalf("driverName mismatch: expected %s, got %s", test.resourceClaim.Status.DriverName, claimInfo.DriverName)
			}
			if claimInfo.ClassName != test.resourceClaim.Spec.ResourceClassName {
				t.Fatalf("resourceClassName mismatch: expected %s, got %s", test.resourceClaim.Spec.ResourceClassName, claimInfo.ClassName)
			}
			if len(claimInfo.PodUIDs) != 1 || !claimInfo.PodUIDs.Has(string(test.pod.UID)) {
				t.Fatalf("podUIDs mismatch: expected [%s], got %v", test.pod.UID, claimInfo.PodUIDs)
			}
			expectedResourceClaimDriverName := fmt.Sprintf("%s/%s=claim-%s", driverName, driverClassName, string(test.resourceClaim.Status.ReservedFor[0].UID))
			if len(claimInfo.CDIDevices[test.resourceClaim.Status.DriverName]) != 1 || claimInfo.CDIDevices[test.resourceClaim.Status.DriverName][0] != expectedResourceClaimDriverName {
				t.Fatalf("cdiDevices mismatch: expected [%s], got %v", []string{expectedResourceClaimDriverName}, claimInfo.CDIDevices[test.resourceClaim.Status.DriverName])
			}
		})
	}
}

func TestUnprepareResources(t *testing.T) {
	fakeKubeClient := fake.NewSimpleClientset()

	for _, test := range []struct {
		description            string
		driverName             string
		pod                    *v1.Pod
		claimInfo              *ClaimInfo
		wantErr                bool
		wantTimeout            bool
		wantResourceSkipped    bool
		expectedUnprepareCalls uint32
	}{
		{
			description: "plugin does not exist",
			driverName:  "this-plugin-does-not-exist",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "another-claim-test",
							Source: v1.ClaimSource{
								ResourceClaimName: func() *string {
									s := "another-claim-test"
									return &s
								}(),
							},
						},
					},
				},
			},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: driverName,
					ClaimName:  "another-claim-test",
					Namespace:  "test-namespace",
					ResourceHandles: []resourcev1alpha2.ResourceHandle{
						{
							DriverName: driverName,
							Data:       "test data",
						},
					},
				},
			},
			wantErr: true,
		},
		{
			description: "resource claim referenced by other pod(s)",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim-1",
							Source: v1.ClaimSource{ResourceClaimName: func() *string {
								s := "test-pod-claim-1"
								return &s
							}()},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{
									{
										Name: "test-pod-claim-1",
									},
								},
							},
						},
					},
				},
			},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: driverName,
					ClaimName:  "test-pod-claim-1",
					Namespace:  "test-namespace",
					PodUIDs:    sets.Set[string]{"test-reserved": sets.Empty{}, "test-reserved-2": sets.Empty{}},
				},
			},
			wantResourceSkipped: true,
		},
		{
			description: "should timeout",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim-2",
							Source: v1.ClaimSource{ResourceClaimName: func() *string {
								s := "test-pod-claim-2"
								return &s
							}()},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{
									{
										Name: "test-pod-claim-2",
									},
								},
							},
						},
					},
				},
			},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: driverName,
					ClaimName:  "test-pod-claim-2",
					Namespace:  "test-namespace",
					ResourceHandles: []resourcev1alpha2.ResourceHandle{
						{
							DriverName: driverName,
							Data:       "test data",
						},
					},
				},
			},
			wantErr:                true,
			wantTimeout:            true,
			expectedUnprepareCalls: 1,
		},
		{
			description: "should unprepare resource, claim previously prepared by currently running manager",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim-3",
							Source: v1.ClaimSource{ResourceClaimName: func() *string {
								s := "test-pod-claim-3"
								return &s
							}()},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{
									{
										Name: "test-pod-claim-3",
									},
								},
							},
						},
					},
				},
			},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: driverName,
					ClaimName:  "test-pod-claim-3",
					Namespace:  "test-namespace",
					ResourceHandles: []resourcev1alpha2.ResourceHandle{
						{
							DriverName: driverName,
							Data:       "test data",
						},
					},
				},
				prepared: true,
			},
			expectedUnprepareCalls: 1,
		},
		{
			description: "should unprepare resource, claim previously was not prepared by currently running manager",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
					UID:       "test-reserved",
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: "test-pod-claim",
							Source: v1.ClaimSource{ResourceClaimName: func() *string {
								s := "test-pod-claim"
								return &s
							}()},
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{
									{
										Name: "test-pod-claim",
									},
								},
							},
						},
					},
				},
			},
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					DriverName: driverName,
					ClaimName:  "test-pod-claim",
					Namespace:  "test-namespace",
					ResourceHandles: []resourcev1alpha2.ResourceHandle{
						{
							DriverName: driverName,
							Data:       "test data",
						},
					},
				},
				prepared: false,
			},
			expectedUnprepareCalls: 1,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			cache, err := newClaimInfoCache(t.TempDir(), draManagerStateFileName)
			if err != nil {
				t.Fatalf("failed to create a new instance of the claimInfoCache, err: %v", err)
			}

			draServerInfo, err := setupFakeDRADriverGRPCServer(test.wantTimeout)
			if err != nil {
				t.Fatal(err)
			}
			defer draServerInfo.teardownFn()

			plg := plugin.NewRegistrationHandler()
			if err := plg.RegisterPlugin(test.driverName, draServerInfo.socketName, []string{"1.27"}); err != nil {
				t.Fatalf("failed to register plugin %s, err: %v", test.driverName, err)
			}
			defer plg.DeRegisterPlugin(test.driverName) // for sake of next tests

			manager := &ManagerImpl{
				kubeClient: fakeKubeClient,
				cache:      cache,
			}

			if test.claimInfo != nil {
				manager.cache.add(test.claimInfo)
			}

			err = manager.UnprepareResources(test.pod)

			assert.Equal(t, test.expectedUnprepareCalls, draServerInfo.server.unprepareResourceCalls.Load())

			if test.wantErr {
				assert.Error(t, err)
				return // UnprepareResources returned an error so stopping the subtest here
			} else if test.wantResourceSkipped {
				assert.NoError(t, err)
				return // resource skipped so no need to continue
			}

			assert.NoError(t, err)
			// Check that the cache has been updated correctly
			claimName, _, err := resourceclaim.Name(test.pod, &test.pod.Spec.ResourceClaims[0])
			if err != nil {
				t.Fatal(err)
			}
			claimInfo := manager.cache.get(*claimName, test.pod.Namespace)
			if claimInfo != nil {
				t.Fatalf("claimInfo still found in cache after calling UnprepareResources")
			}
		})
	}
}

func TestPodMightNeedToUnprepareResources(t *testing.T) {
	fakeKubeClient := fake.NewSimpleClientset()

	cache, err := newClaimInfoCache(t.TempDir(), draManagerStateFileName)
	if err != nil {
		t.Fatalf("failed to newClaimInfoCache, err:%v", err)
	}

	manager := &ManagerImpl{
		kubeClient: fakeKubeClient,
		cache:      cache,
	}

	podUID := sets.Set[string]{}
	podUID.Insert("test-pod-uid")
	manager.cache.add(&ClaimInfo{
		ClaimInfoState: state.ClaimInfoState{PodUIDs: podUID, ClaimName: "test-claim", Namespace: "test-namespace"},
	})

	testClaimInfo := manager.cache.get("test-claim", "test-namespace")
	testClaimInfo.addPodReference("test-pod-uid")

	manager.PodMightNeedToUnprepareResources("test-pod-uid")
}

func TestGetContainerClaimInfos(t *testing.T) {
	cache, err := newClaimInfoCache(t.TempDir(), draManagerStateFileName)
	if err != nil {
		t.Fatalf("error occur:%v", err)
	}
	manager := &ManagerImpl{
		cache: cache,
	}

	resourceClaimName := "test-resource-claim-1"
	resourceClaimName2 := "test-resource-claim-2"

	for i, test := range []struct {
		expectedClaimName string
		pod               *v1.Pod
		container         *v1.Container
		claimInfo         *ClaimInfo
	}{
		{
			expectedClaimName: resourceClaimName,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name:   "claim1",
							Source: v1.ClaimSource{ResourceClaimName: &resourceClaimName},
						},
					},
				},
			},
			container: &v1.Container{
				Resources: v1.ResourceRequirements{
					Claims: []v1.ResourceClaim{
						{
							Name: "claim1",
						},
					},
				},
			},
			claimInfo: &ClaimInfo{ClaimInfoState: state.ClaimInfoState{ClaimName: resourceClaimName}},
		},
		{
			expectedClaimName: resourceClaimName2,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name:   "claim2",
							Source: v1.ClaimSource{ResourceClaimName: &resourceClaimName2},
						},
					},
				},
			},
			container: &v1.Container{
				Resources: v1.ResourceRequirements{
					Claims: []v1.ResourceClaim{
						{
							Name: "claim2",
						},
					},
				},
			},
			claimInfo: &ClaimInfo{ClaimInfoState: state.ClaimInfoState{ClaimName: resourceClaimName2}},
		},
	} {
		t.Run(fmt.Sprintf("test-%d", i), func(t *testing.T) {
			manager.cache.add(test.claimInfo)

			fakeClaimInfos, err := manager.GetContainerClaimInfos(test.pod, test.container)
			assert.NoError(t, err)
			assert.Equal(t, 1, len(fakeClaimInfos))
			assert.Equal(t, test.expectedClaimName, fakeClaimInfos[0].ClaimInfoState.ClaimName)

			manager.cache.delete(test.pod.Spec.ResourceClaims[0].Name, "default")
			_, err = manager.GetContainerClaimInfos(test.pod, test.container)
			assert.NoError(t, err)
		})
	}
}
