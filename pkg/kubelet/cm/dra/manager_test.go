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
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	metricstestutil "k8s.io/component-base/metrics/testutil"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"

	drahealthv1alpha1 "k8s.io/kubelet/pkg/apis/dra-health/v1alpha1"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/dra/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/resourceupdates"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	driverClassName = "test"
	podName         = "test-pod"
	containerName   = "test-container"
)

var (
	shareID        = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
	shareUID       = types.UID(shareID)
	testPodCounter atomic.Uint32
)

type fakeDRADriverGRPCServer struct {
	drapb.UnimplementedDRAPluginServer
	drahealthv1alpha1.UnimplementedDRAResourceHealthServer
	driverName                 string
	timeout                    *time.Duration
	prepareResourceCalls       atomic.Uint32
	unprepareResourceCalls     atomic.Uint32
	watchResourcesCalls        atomic.Uint32
	prepareResourcesResponse   *drapb.NodePrepareResourcesResponse
	unprepareResourcesResponse *drapb.NodeUnprepareResourcesResponse
	watchResourcesResponses    chan *drahealthv1alpha1.NodeWatchResourcesResponse
	watchResourcesError        error
}

var gatherWithoutDurations = metricstestutil.GathererFunc(func() ([]*metricstestutil.MetricFamily, error) {
	got, err := kubeletmetrics.GetGather().Gather()
	for _, mf := range got {

		for _, m := range mf.Metric {
			if m.Histogram == nil {
				continue
			}
			// Remove everything from a histogram that depends on timing.
			m.Histogram.SampleSum = nil
			m.Histogram.Bucket = nil
		}
	}
	return got, err
})

func (s *fakeDRADriverGRPCServer) NodePrepareResources(ctx context.Context, req *drapb.NodePrepareResourcesRequest) (*drapb.NodePrepareResourcesResponse, error) {
	s.prepareResourceCalls.Add(1)

	if s.timeout != nil {
		time.Sleep(*s.timeout)
	}

	if s.prepareResourcesResponse == nil {
		cdiDeviceName := "claim-" + req.Claims[0].Uid
		cdiID := s.driverName + "/" + driverClassName + "=" + cdiDeviceName
		return &drapb.NodePrepareResourcesResponse{
			Claims: map[string]*drapb.NodePrepareResourceResponse{
				req.Claims[0].Uid: {
					Devices: []*drapb.Device{
						{
							PoolName:     poolName,
							DeviceName:   deviceName,
							RequestNames: []string{req.Claims[0].Name},
							CdiDeviceIds: []string{cdiID},
						},
					},
				},
			},
		}, nil
	}

	return s.prepareResourcesResponse, nil
}

func (s *fakeDRADriverGRPCServer) NodeUnprepareResources(ctx context.Context, req *drapb.NodeUnprepareResourcesRequest) (*drapb.NodeUnprepareResourcesResponse, error) {
	s.unprepareResourceCalls.Add(1)

	if s.timeout != nil {
		time.Sleep(*s.timeout)
	}

	if s.unprepareResourcesResponse == nil {
		return &drapb.NodeUnprepareResourcesResponse{
			Claims: map[string]*drapb.NodeUnprepareResourceResponse{
				req.Claims[0].Uid: {},
			},
		}, nil
	}

	return s.unprepareResourcesResponse, nil
}

func (s *fakeDRADriverGRPCServer) NodeWatchResources(req *drahealthv1alpha1.NodeWatchResourcesRequest, stream drahealthv1alpha1.DRAResourceHealth_NodeWatchResourcesServer) error {
	s.watchResourcesCalls.Add(1)
	logger := klog.FromContext(stream.Context())
	logger.V(4).Info("Fake Server: WatchResources stream started")

	if s.watchResourcesError != nil {
		logger.Error(s.watchResourcesError, "Fake Server: Returning predefined stream error")
		return s.watchResourcesError
	}

	go func() {
		for {
			select {
			case <-stream.Context().Done():
				logger.Info("Fake Server: WatchResources stream context canceled")
				return
			case resp, ok := <-s.watchResourcesResponses:
				if !ok {
					logger.Info("Fake Server: WatchResources response channel closed")
					return
				}
				logger.V(5).Info("Fake Server: Sending health response", "response", resp)
				// Use the stream argument to send
				if err := stream.Send(resp); err != nil {
					logger.Error(err, "Fake Server: Error sending response on stream")
					return
				}
			}
		}
	}()

	logger.V(4).Info("Fake Server: WatchResources RPC call returning control to client.")
	return nil
}

type mockWatchResourcesClient struct {
	mock.Mock
	RecvChan chan struct {
		Resp *drahealthv1alpha1.NodeWatchResourcesResponse
		Err  error
	}
	Ctx context.Context
}

func (m *mockWatchResourcesClient) Recv() (*drahealthv1alpha1.NodeWatchResourcesResponse, error) {
	logger := klog.FromContext(m.Ctx)
	select {
	case <-m.Ctx.Done():
		logger.V(6).Info("mockWatchClient.Recv: Context done", "err", m.Ctx.Err())
		return nil, m.Ctx.Err()
	case item, ok := <-m.RecvChan:
		if !ok {
			logger.V(6).Info("mockWatchClient.Recv: RecvChan closed, returning io.EOF")
			return nil, io.EOF
		}
		return item.Resp, item.Err
	}
}

func (m *mockWatchResourcesClient) Context() context.Context {
	return m.Ctx
}

func (m *mockWatchResourcesClient) Header() (metadata.MD, error) { return nil, nil }
func (m *mockWatchResourcesClient) Trailer() metadata.MD         { return nil }
func (m *mockWatchResourcesClient) CloseSend() error             { return nil }
func (m *mockWatchResourcesClient) RecvMsg(v interface{}) error {
	return fmt.Errorf("RecvMsg not implemented")
}
func (m *mockWatchResourcesClient) SendMsg(v interface{}) error {
	return fmt.Errorf("SendMsg not implemented")
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

func setupFakeDRADriverGRPCServer(ctx context.Context, shouldTimeout bool, pluginClientTimeout *time.Duration, prepareResourcesResponse *drapb.NodePrepareResourcesResponse, unprepareResourcesResponse *drapb.NodeUnprepareResourcesResponse, watchResourcesError error) (fakeDRAServerInfo, error) {
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
		if err := os.Remove(socketName); err != nil {
			logger := klog.FromContext(ctx)
			logger.Error(err, "failed to remove socket file", "path", socketName)
		}
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
		driverName:                 driverName,
		prepareResourcesResponse:   prepareResourcesResponse,
		unprepareResourcesResponse: unprepareResourcesResponse,
		watchResourcesResponses:    make(chan *drahealthv1alpha1.NodeWatchResourcesResponse, 10),
		watchResourcesError:        watchResourcesError,
	}
	if shouldTimeout {
		timeout := *pluginClientTimeout * 2
		fakeDRADriverGRPCServer.timeout = &timeout
	}

	drahealthv1alpha1.RegisterDRAResourceHealthServer(s, fakeDRADriverGRPCServer)
	drapb.RegisterDRAPluginServer(s, fakeDRADriverGRPCServer)

	go func() {
		go func() {
			logger := klog.FromContext(ctx)
			logger.V(4).Info("Starting fake gRPC server", "address", socketName)
			if err := s.Serve(l); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
				logger.Error(err, "failed to serve gRPC")
			}
			logger.V(4).Info("Fake gRPC server stopped serving", "address", socketName)
		}()
		<-stopCh
		logger := klog.FromContext(ctx)
		logger.V(4).Info("Stopping fake gRPC server", "address", socketName)
		s.GracefulStop()
		logger.V(4).Info("Fake gRPC server stopped", "address", socketName)
	}()

	return fakeDRAServerInfo{
		server:     fakeDRADriverGRPCServer,
		socketName: socketName,
		teardownFn: teardown,
	}, nil
}

func genPrepareResourcesResponse(claimUID types.UID) *drapb.NodePrepareResourcesResponse {
	return &drapb.NodePrepareResourcesResponse{
		Claims: map[string]*drapb.NodePrepareResourceResponse{
			string(claimUID): {
				Devices: []*drapb.Device{
					{
						PoolName:     poolName,
						DeviceName:   deviceName,
						RequestNames: []string{requestName},
						CdiDeviceIds: []string{cdiID},
					},
				},
			},
		},
	}
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
			tCtx := ktesting.Init(t)
			manager, err := NewManager(tCtx.Logger(), kubeClient, test.stateFileDirectory)
			if test.wantErr {
				assert.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.NotNil(t, manager.cache)
			assert.NotNil(t, manager.kubeClient)
		})
	}
}

// genTestPod generates pod object
func genTestPod() *v1.Pod {
	claimName := claimName
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			UID:       podUID,
		},
		Spec: v1.PodSpec{
			ResourceClaims: []v1.PodResourceClaim{
				{
					Name:              claimName,
					ResourceClaimName: &claimName,
				},
			},
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Claims: []v1.ResourceClaim{
							{
								Name: claimName,
							},
						},
					},
				},
			},
		},
	}
}

func genTestPodWithClaims(claimNames ...string) *v1.Pod {
	podCounter := testPodCounter.Add(1)

	podName := fmt.Sprintf("test-pod-%d", podCounter)
	podUID := types.UID(fmt.Sprintf("test-pod-uid-%d", podCounter))

	resourceClaims := make([]v1.PodResourceClaim, 0, len(claimNames))
	containerClaims := make([]v1.ResourceClaim, 0, len(claimNames))

	for _, claimName := range claimNames {
		cn := claimName
		resourceClaims = append(resourceClaims, v1.PodResourceClaim{
			Name:              cn,
			ResourceClaimName: &cn,
		})
		containerClaims = append(containerClaims, v1.ResourceClaim{
			Name: cn,
		})
	}

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			UID:       podUID,
		},
		Spec: v1.PodSpec{
			ResourceClaims: resourceClaims,
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Claims: containerClaims,
					},
				},
			},
		},
	}
}

// genTestPodWithExtendedResource generates pod object
func genTestPodWithExtendedResource() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			UID:       podUID,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: containerName,
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							"example.com/gpu": resource.MustParse("1"),
						},
					},
				},
			},
		},
		Status: v1.PodStatus{
			ExtendedResourceClaimStatus: &v1.PodExtendedResourceClaimStatus{
				ResourceClaimName: claimName,
				RequestMappings: []v1.ContainerExtendedResourceRequest{
					{
						ContainerName: containerName,
						ResourceName:  "example.com/gpu",
						RequestName:   "container-0-request-0",
					},
				},
			},
		},
	}
}

// genTestClaim generates resource claim object
func genTestClaim(name, driver, device, podUID string) *resourceapi.ResourceClaim {
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			UID:       types.UID(fmt.Sprintf("%s-uid", name)),
		},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests: []resourceapi.DeviceRequest{
					{
						Name: requestName,
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: className,
						},
					},
				},
			},
		},
		Status: resourceapi.ResourceClaimStatus{
			Allocation: &resourceapi.AllocationResult{
				Devices: resourceapi.DeviceAllocationResult{
					Results: []resourceapi.DeviceRequestAllocationResult{
						{
							Request: requestName,
							Pool:    poolName,
							Device:  device,
							Driver:  driver,
						},
					},
				},
			},
			ReservedFor: []resourceapi.ResourceClaimConsumerReference{
				{UID: types.UID(podUID)},
			},
		},
	}
}

// genTestClaimWithExtendedResource generates resource claim object
func genTestClaimWithExtendedResource(name, driver, device, podUID string) *resourceapi.ResourceClaim {
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			UID:       types.UID(fmt.Sprintf("%s-uid", name)),
		},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests: []resourceapi.DeviceRequest{
					{
						Name: "container-0-request-0",
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: className,
						},
					},
				},
			},
		},
		Status: resourceapi.ResourceClaimStatus{
			Allocation: &resourceapi.AllocationResult{
				Devices: resourceapi.DeviceAllocationResult{
					Results: []resourceapi.DeviceRequestAllocationResult{
						{
							Request: "container-0-request-0",
							Pool:    poolName,
							Device:  device,
							Driver:  driver,
						},
					},
				},
			},
			ReservedFor: []resourceapi.ResourceClaimConsumerReference{
				{UID: types.UID(podUID)},
			},
		},
	}
}

// genTestClaimInfo generates claim info object
func genTestClaimInfo(claimUID types.UID, podUIDs []string, prepared bool) *ClaimInfo {
	return &ClaimInfo{
		ClaimInfoState: state.ClaimInfoState{
			ClaimUID:  claimUID,
			ClaimName: claimName,
			Namespace: namespace,
			PodUIDs:   sets.New[string](podUIDs...),
			DriverState: map[string]state.DriverState{
				driverName: {
					Devices: []state.Device{{
						PoolName:     poolName,
						DeviceName:   deviceName,
						RequestNames: []string{requestName},
						CDIDeviceIDs: []string{cdiID},
					}},
				},
			},
		},
		prepared: prepared,
	}
}

// genTestClaimInfoWithExtendedResource generates claim info object
func genTestClaimInfoWithExtendedResource(podUIDs []string, prepared bool) *ClaimInfo {
	return &ClaimInfo{
		ClaimInfoState: state.ClaimInfoState{
			ClaimUID:  claimUID,
			ClaimName: claimName,
			Namespace: namespace,
			PodUIDs:   sets.New[string](podUIDs...),
			DriverState: map[string]state.DriverState{
				driverName: {
					Devices: []state.Device{{
						PoolName:     poolName,
						DeviceName:   deviceName,
						RequestNames: []string{"container-0-request-0"},
						CDIDeviceIDs: []string{cdiID},
					}},
				},
			},
		},
		prepared: prepared,
	}
}

// genClaimInfoState generates claim info state object
func genClaimInfoState(cdiDeviceID string) state.ClaimInfoState {
	s := state.ClaimInfoState{
		ClaimUID:  claimUID,
		ClaimName: claimName,
		Namespace: namespace,
		PodUIDs:   sets.New[string](podUID),
		DriverState: map[string]state.DriverState{
			driverName: {},
		},
	}
	if cdiDeviceID != "" {
		s.DriverState[driverName] = state.DriverState{Devices: []state.Device{{PoolName: poolName, DeviceName: deviceName, RequestNames: []string{requestName}, CDIDeviceIDs: []string{cdiDeviceID}}}}
	}
	return s
}

func genClaimInfoStateWithExtendedResource(cdiDeviceID string) state.ClaimInfoState {
	s := state.ClaimInfoState{
		ClaimUID:  claimUID,
		ClaimName: claimName,
		Namespace: namespace,
		PodUIDs:   sets.New[string](podUID),
		DriverState: map[string]state.DriverState{
			driverName: {},
		},
	}
	if cdiDeviceID != "" {
		s.DriverState[driverName] = state.DriverState{Devices: []state.Device{{PoolName: poolName, DeviceName: deviceName, RequestNames: []string{"container-0-request-0"}, CDIDeviceIDs: []string{cdiDeviceID}}}}
	}
	return s
}

func genClaimInfoStateWithShareID(cdiDeviceID string) state.ClaimInfoState {
	s := state.ClaimInfoState{
		ClaimUID:  claimUID,
		ClaimName: claimName,
		Namespace: namespace,
		PodUIDs:   sets.New[string](podUID),
		DriverState: map[string]state.DriverState{
			driverName: {},
		},
	}
	if cdiDeviceID != "" {
		s.DriverState[driverName] = state.DriverState{Devices: []state.Device{
			{PoolName: poolName, DeviceName: deviceName, ShareID: &shareUID, RequestNames: []string{requestName}, CDIDeviceIDs: []string{cdiDeviceID}},
		}}
	}
	return s
}

func TestGetResources(t *testing.T) {
	kubeClient := fake.NewSimpleClientset()

	for _, test := range []struct {
		description string
		container   *v1.Container
		pod         *v1.Pod
		claimInfo   *ClaimInfo
		wantErr     bool
	}{
		{
			description: "claim info with devices",
			container: &v1.Container{
				Name: containerName,
				Resources: v1.ResourceRequirements{
					Claims: []v1.ResourceClaim{
						{
							Name: claimName,
						},
					},
				},
			},
			pod:       genTestPod(),
			claimInfo: genTestClaimInfo(claimUID, nil, false),
		},
		{
			description: "nil claiminfo",
			container: &v1.Container{
				Name: containerName,
				Resources: v1.ResourceRequirements{
					Claims: []v1.ResourceClaim{
						{
							Name: claimName,
						},
					},
				},
			},
			pod:     genTestPod(),
			wantErr: true,
		},
		{
			description: "extended resource claim info with devices",
			container: &v1.Container{
				Name: containerName,
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						"example.com/gpu": resource.MustParse("1"),
					},
				},
			},
			pod:       genTestPodWithExtendedResource(),
			claimInfo: genTestClaimInfoWithExtendedResource(nil, false),
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			manager, err := NewManager(tCtx.Logger(), kubeClient, t.TempDir())
			require.NoError(t, err)

			if test.claimInfo != nil {
				manager.cache.add(test.claimInfo)
			}

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, true)
			containerInfo, err := manager.GetResources(test.pod, test.container)
			if test.wantErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, test.claimInfo.DriverState[driverName].Devices[0].CDIDeviceIDs[0], containerInfo.CDIDevices[0].Name)
			}
		})
	}
}

func getFakeNode() (*v1.Node, error) {
	return &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "worker"}}, nil
}

func TestPrepareResources(t *testing.T) {
	claimName := claimName
	fakeKubeClient := fake.NewSimpleClientset()
	anotherClaimUID := types.UID("another-claim-uid")
	shareUID := types.UID(shareID)

	for _, test := range []struct {
		description         string
		driverName          string
		pod                 *v1.Pod
		claimInfo           *ClaimInfo
		claim               *resourceapi.ResourceClaim
		resp                *drapb.NodePrepareResourcesResponse
		wantTimeout         bool
		wantResourceSkipped bool

		expectedErrMsg         string
		expectedClaimInfoState state.ClaimInfoState
		expectedPrepareCalls   uint32
		expectedMetric         string
	}{
		{
			description:    "claim doesn't exist",
			driverName:     driverName,
			pod:            genTestPod(),
			expectedErrMsg: "fetch ResourceClaim ",
			expectedMetric: `# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="true",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="true",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="true",operation_name="PrepareResources"} 1
`,
		},
		{
			description:    "unknown driver",
			pod:            genTestPod(),
			claim:          genTestClaim(claimName, "unknown driver", deviceName, podUID),
			expectedErrMsg: "prepare dynamic resources: DRA driver unknown driver is not registered",
			expectedMetric: `# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="true",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="true",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="true",operation_name="PrepareResources"} 1
`,
		},
		{
			description:            "should prepare resources, driver returns nil value",
			driverName:             driverName,
			pod:                    genTestPod(),
			claim:                  genTestClaim(claimName, driverName, deviceName, podUID),
			resp:                   &drapb.NodePrepareResourcesResponse{Claims: map[string]*drapb.NodePrepareResourceResponse{string(claimUID): nil}},
			expectedClaimInfoState: genClaimInfoState(""),
			expectedPrepareCalls:   1,
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="false",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="false",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="false",operation_name="PrepareResources"} 1
`,
		},
		{
			description:          "driver returns empty result",
			driverName:           driverName,
			pod:                  genTestPod(),
			claim:                genTestClaim(claimName, driverName, deviceName, podUID),
			resp:                 &drapb.NodePrepareResourcesResponse{Claims: map[string]*drapb.NodePrepareResourceResponse{}},
			expectedPrepareCalls: 1,
			expectedErrMsg:       "NodePrepareResources skipped 1 ResourceClaims",
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="true",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="true",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="true",operation_name="PrepareResources"} 1
`,
		},
		{
			description:    "pod is not allowed to use resource claim",
			driverName:     driverName,
			pod:            genTestPod(),
			claim:          genTestClaim(claimName, driverName, deviceName, ""),
			expectedErrMsg: "is not allowed to use ResourceClaim ",
			expectedMetric: `# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="true",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="true",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="true",operation_name="PrepareResources"} 1
`,
		},
		{
			description: "no container uses the claim",
			driverName:  driverName,
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: namespace,
					UID:       podUID,
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name:              claimName,
							ResourceClaimName: &claimName,
						},
					},
					Containers: []v1.Container{
						{},
					},
				},
			},
			claim:                genTestClaim(claimName, driverName, deviceName, podUID),
			expectedPrepareCalls: 1,
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="false",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="false",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="false",operation_name="PrepareResources"} 1
`,
			expectedClaimInfoState: genClaimInfoState(cdiID),
			resp:                   genPrepareResourcesResponse(claimUID),
		},
		{
			description:            "resource already prepared",
			driverName:             driverName,
			pod:                    genTestPod(),
			claim:                  genTestClaim(claimName, driverName, deviceName, podUID),
			claimInfo:              genTestClaimInfo(claimUID, []string{podUID}, true),
			expectedClaimInfoState: genClaimInfoState(cdiID),
			resp:                   genPrepareResourcesResponse(claimUID),
			expectedMetric: `# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="false",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="false",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="false",operation_name="PrepareResources"} 1
`,
		},
		{
			description:          "should timeout",
			driverName:           driverName,
			pod:                  genTestPod(),
			claim:                genTestClaim(claimName, driverName, deviceName, podUID),
			wantTimeout:          true,
			expectedPrepareCalls: 1,
			expectedErrMsg:       "NodePrepareResources: rpc error: code = DeadlineExceeded",
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="DeadlineExceeded",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="DeadlineExceeded",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="DeadlineExceeded",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="true",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="true",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="true",operation_name="PrepareResources"} 1
`,
		},
		{
			description:            "should prepare resource, claim not in cache",
			driverName:             driverName,
			pod:                    genTestPod(),
			claim:                  genTestClaim(claimName, driverName, deviceName, podUID),
			expectedClaimInfoState: genClaimInfoState(cdiID),
			resp:                   genPrepareResourcesResponse(claimUID),
			expectedPrepareCalls:   1,
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="false",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="false",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="false",operation_name="PrepareResources"} 1
`,
		},
		{
			description:            "should prepare extended resource claim backed by DRA",
			driverName:             driverName,
			pod:                    genTestPodWithExtendedResource(),
			claim:                  genTestClaimWithExtendedResource(claimName, driverName, deviceName, podUID),
			expectedClaimInfoState: genClaimInfoStateWithExtendedResource(cdiID),
			resp: &drapb.NodePrepareResourcesResponse{Claims: map[string]*drapb.NodePrepareResourceResponse{
				string(claimUID): {
					Devices: []*drapb.Device{
						{
							PoolName:     poolName,
							DeviceName:   deviceName,
							RequestNames: []string{"container-0-request-0"},
							CdiDeviceIds: []string{cdiID},
						},
					},
				},
			}},
			expectedPrepareCalls: 1,
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="false",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="false",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="false",operation_name="PrepareResources"} 1
`,
		},
		{
			description:    "claim UIDs mismatch",
			driverName:     driverName,
			pod:            genTestPod(),
			claim:          genTestClaim(claimName, driverName, deviceName, podUID),
			claimInfo:      genTestClaimInfo(anotherClaimUID, []string{podUID}, false),
			expectedErrMsg: fmt.Sprintf("old ResourceClaim with same name %s and different UID %s still exists", claimName, anotherClaimUID),
			expectedMetric: `# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="true",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="true",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="true",operation_name="PrepareResources"} 1
`,
		},
		{
			description: "should prepare resources with share id",
			driverName:  driverName,
			pod:         genTestPod(),
			claim: func() *resourceapi.ResourceClaim {
				claim := genTestClaim(claimName, driverName, deviceName, podUID)
				claim.Status.Allocation.Devices.Results[0].ShareID = &shareUID
				return claim
			}(),
			expectedClaimInfoState: genClaimInfoStateWithShareID(cdiID),
			resp: &drapb.NodePrepareResourcesResponse{Claims: map[string]*drapb.NodePrepareResourceResponse{
				string(claimUID): {
					Devices: []*drapb.Device{
						{
							PoolName:     poolName,
							DeviceName:   deviceName,
							RequestNames: []string{requestName},
							ShareId:      &shareID,
							CdiDeviceIds: []string{cdiID},
						},
					},
				},
			}},
			expectedPrepareCalls: 1,
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodePrepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="false",operation_name="PrepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="false",operation_name="PrepareResources"} 0
dra_operations_duration_seconds_count{is_error="false",operation_name="PrepareResources"} 1
`,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			backgroundCtx, cancel := context.WithCancel(context.Background())
			defer cancel()

			kubeletmetrics.Register()
			kubeletmetrics.DRAOperationsDuration.Reset()
			kubeletmetrics.DRAGRPCOperationsDuration.Reset()
			defer func() {
				require.NoError(t,
					metricstestutil.GatherAndCompare(gatherWithoutDurations,
						strings.NewReader(test.expectedMetric),
						kubeletmetrics.DRAOperationsDuration.FQName(),
						kubeletmetrics.DRAGRPCOperationsDuration.FQName(),
					),
				)
			}()

			tCtx := ktesting.Init(t)
			backgroundCtx = klog.NewContext(backgroundCtx, tCtx.Logger())

			manager, err := NewManager(tCtx.Logger(), fakeKubeClient, t.TempDir())
			require.NoError(t, err, "create DRA manager")
			manager.initDRAPluginManager(backgroundCtx, getFakeNode, time.Second /* very short wiping delay for testing */)

			if test.claim != nil {
				if _, err := fakeKubeClient.ResourceV1().ResourceClaims(test.pod.Namespace).Create(tCtx, test.claim, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create ResourceClaim %s: %+v", test.claim.Name, err)
				}
				defer func() {
					require.NoError(t, fakeKubeClient.ResourceV1().ResourceClaims(test.pod.Namespace).Delete(tCtx, test.claim.Name, metav1.DeleteOptions{}))
				}()
			}

			var pluginClientTimeout *time.Duration
			if test.wantTimeout {
				timeout := time.Millisecond * 20
				pluginClientTimeout = &timeout
			}

			draServerInfo, err := setupFakeDRADriverGRPCServer(backgroundCtx, test.wantTimeout, pluginClientTimeout, test.resp, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer draServerInfo.teardownFn()
			plg := manager.GetWatcherHandler()
			if err := plg.RegisterPlugin(test.driverName, draServerInfo.socketName, []string{drapb.DRAPluginService}, pluginClientTimeout); err != nil {
				t.Fatalf("failed to register plugin %s, err: %v", test.driverName, err)
			}

			if test.claimInfo != nil {
				manager.cache.add(test.claimInfo)
			}

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAConsumableCapacity, true)
			err = manager.PrepareResources(backgroundCtx, test.pod)

			assert.Equal(t, test.expectedPrepareCalls, draServerInfo.server.prepareResourceCalls.Load())

			if test.expectedErrMsg != "" {
				assert.Error(t, err)
				if err != nil {
					assert.ErrorContains(t, err, test.expectedErrMsg)
				}
				return // PrepareResources returned an error so stopping the test case here
			}

			require.NoError(t, err)

			if test.wantResourceSkipped {
				return // resource skipped so no need to continue
			}

			// check the cache contains the expected claim info
			var podClaimName *string
			if len(test.pod.Spec.ResourceClaims) > 0 {
				podClaimName, _, err = resourceclaim.Name(test.pod, &test.pod.Spec.ResourceClaims[0])
				require.NoError(t, err)
			}
			if podClaimName == nil {
				podClaimName = &claimName
			}
			claimInfoResult, ok := manager.cache.get(*podClaimName, test.pod.Namespace)
			require.True(t, ok, "claimInfo not found in cache")
			require.True(t, claimInfoResult.PodUIDs.Has(string(test.pod.UID)), "podUIDs mismatch")
			assert.Equal(t, test.expectedClaimInfoState.ClaimUID, claimInfoResult.ClaimUID)
			assert.Equal(t, test.expectedClaimInfoState.ClaimName, claimInfoResult.ClaimName)
			assert.Equal(t, test.expectedClaimInfoState.Namespace, claimInfoResult.Namespace)
			assert.Equal(t, test.expectedClaimInfoState.DriverState, claimInfoResult.DriverState)
			assert.True(t, claimInfoResult.prepared, "ClaimInfo should be marked as prepared")
		})
	}
}

// TestPrepareResourcesWithPreparedAndNewClaim verifies that PrepareResources
// correctly handles a pod that references a mix of ResourceClaims:
// - first claim already prepared by a previous pod
// - second claim is new and needs to be prepared
func TestPrepareResourcesWithPreparedAndNewClaim(t *testing.T) {
	tCtx := ktesting.Init(t)

	fakeKubeClient := fake.NewClientset()

	manager, err := NewManager(tCtx.Logger(), fakeKubeClient, t.TempDir())
	require.NoError(t, err)
	manager.initDRAPluginManager(tCtx, getFakeNode, time.Second)

	secondClaimName := fmt.Sprintf("%s-second", claimName)

	// Generate two pods where the second pod reuses an existing claim and adds a new one
	firstPod := genTestPodWithClaims(claimName)
	secondPod := genTestPodWithClaims(claimName, secondClaimName)

	firstClaim := genTestClaim(claimName, driverName, deviceName, string(firstPod.ObjectMeta.UID))
	secondClaim := genTestClaim(secondClaimName, driverName, deviceName, string(secondPod.ObjectMeta.UID))

	// Make firstClaim reserved for first and second pod
	firstClaim.Status.ReservedFor = append(
		firstClaim.Status.ReservedFor,
		resourceapi.ResourceClaimConsumerReference{UID: secondPod.ObjectMeta.UID},
	)

	_, err = fakeKubeClient.ResourceV1().ResourceClaims(namespace).Create(tCtx, firstClaim, metav1.CreateOptions{})
	require.NoError(t, err)

	_, err = fakeKubeClient.ResourceV1().ResourceClaims(namespace).Create(tCtx, secondClaim, metav1.CreateOptions{})
	require.NoError(t, err)

	respFirst := genPrepareResourcesResponse(firstClaim.UID)
	draServerInfo, err := setupFakeDRADriverGRPCServer(tCtx, false, nil, respFirst, nil, nil)
	require.NoError(t, err)
	defer draServerInfo.teardownFn()

	plg := manager.GetWatcherHandler()
	require.NoError(t, plg.RegisterPlugin(driverName, draServerInfo.socketName, []string{drapb.DRAPluginService}, nil))

	err = manager.PrepareResources(tCtx, firstPod)
	require.NoError(t, err)

	assert.Equal(t, uint32(1),
		draServerInfo.server.prepareResourceCalls.Load(),
		"first pod should trigger one prepare call",
	)

	respSecond := genPrepareResourcesResponse(secondClaim.UID)
	draServerInfo.server.prepareResourcesResponse = respSecond

	err = manager.PrepareResources(tCtx, secondPod)
	require.NoError(t, err)

	// second pod triggered exactly one prepare call (new claim only) + previous one call
	assert.Equal(t, uint32(2),
		draServerInfo.server.prepareResourceCalls.Load(),
		"second pod should trigger one prepare call for the new claim",
	)

	for _, claimName := range []string{firstClaim.Name, secondClaim.Name} {
		claimInfo, exists := manager.cache.get(claimName, namespace)
		require.True(t, exists, "claim %s should exist in cache", claimName)
		assert.True(t, claimInfo.prepared, "claim %s should be marked as prepared", claimName)
	}
}

func TestUnprepareResources(t *testing.T) {
	fakeKubeClient := fake.NewSimpleClientset()
	for _, test := range []struct {
		description         string
		driverName          string
		pod                 *v1.Pod
		claimInfo           *ClaimInfo
		claim               *resourceapi.ResourceClaim
		resp                *drapb.NodeUnprepareResourcesResponse
		wantTimeout         bool
		wantResourceSkipped bool

		expectedUnprepareCalls uint32
		expectedErrMsg         string
		expectedMetric         string
	}{
		{
			description: "unknown driver",
			driverName:  driverName,
			pod:         genTestPod(),
			claimInfo: &ClaimInfo{
				ClaimInfoState: state.ClaimInfoState{
					ClaimUID:  claimUID,
					ClaimName: claimName,
					Namespace: namespace,
					PodUIDs:   sets.New[string](string(podUID)),
					DriverState: map[string]state.DriverState{
						"unknown-driver": {
							Devices: []state.Device{{
								PoolName:     poolName,
								DeviceName:   deviceName,
								RequestNames: []string{requestName},
								CDIDeviceIDs: []string{"random-cdi-id"},
							}},
						},
					},
				},
				prepared: true,
			},
			expectedErrMsg:         "unprepare dynamic resources: DRA driver unknown-driver is not registered",
			expectedUnprepareCalls: 0,
			expectedMetric: `# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="true",operation_name="UnprepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="true",operation_name="UnprepareResources"} 0
dra_operations_duration_seconds_count{is_error="true",operation_name="UnprepareResources"} 1
`,
		},
		{
			description:         "resource claim referenced by other pod(s)",
			driverName:          driverName,
			pod:                 genTestPod(),
			claimInfo:           genTestClaimInfo(claimUID, []string{podUID, "another-pod-uid"}, true),
			wantResourceSkipped: true,
			expectedMetric: `# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="false",operation_name="UnprepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="false",operation_name="UnprepareResources"} 0
dra_operations_duration_seconds_count{is_error="false",operation_name="UnprepareResources"} 1
`,
		},
		{
			description:            "should timeout",
			driverName:             driverName,
			pod:                    genTestPod(),
			claimInfo:              genTestClaimInfo(claimUID, []string{podUID}, true),
			wantTimeout:            true,
			expectedUnprepareCalls: 1,
			expectedErrMsg:         "NodeUnprepareResources: rpc error: code = DeadlineExceeded",
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="DeadlineExceeded",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="DeadlineExceeded",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="DeadlineExceeded",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="true",operation_name="UnprepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="true",operation_name="UnprepareResources"} 0
dra_operations_duration_seconds_count{is_error="true",operation_name="UnprepareResources"} 1
`,
		},
		{
			description:            "should fail when driver returns empty response",
			driverName:             driverName,
			pod:                    genTestPod(),
			claimInfo:              genTestClaimInfo(claimUID, []string{podUID}, true),
			resp:                   &drapb.NodeUnprepareResourcesResponse{Claims: map[string]*drapb.NodeUnprepareResourceResponse{}},
			expectedUnprepareCalls: 1,
			expectedErrMsg:         "NodeUnprepareResources skipped 1 ResourceClaims",
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="true",operation_name="UnprepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="true",operation_name="UnprepareResources"} 0
dra_operations_duration_seconds_count{is_error="true",operation_name="UnprepareResources"} 1
`,
		},
		{
			description:            "should unprepare already prepared extended resource backed by DRA",
			driverName:             driverName,
			pod:                    genTestPodWithExtendedResource(),
			claim:                  genTestClaimWithExtendedResource(claimName, driverName, deviceName, podUID),
			claimInfo:              genTestClaimInfoWithExtendedResource([]string{podUID}, true),
			expectedUnprepareCalls: 1,
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="false",operation_name="UnprepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="false",operation_name="UnprepareResources"} 0
dra_operations_duration_seconds_count{is_error="false",operation_name="UnprepareResources"} 1
`,
		},
		{
			description:            "should unprepare already prepared resource",
			driverName:             driverName,
			pod:                    genTestPod(),
			claim:                  genTestClaim(claimName, driverName, deviceName, podUID),
			claimInfo:              genTestClaimInfo(claimUID, []string{podUID}, true),
			expectedUnprepareCalls: 1,
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="false",operation_name="UnprepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="false",operation_name="UnprepareResources"} 0
dra_operations_duration_seconds_count{is_error="false",operation_name="UnprepareResources"} 1
`,
		},
		{
			description:            "should unprepare resource when driver returns nil value",
			driverName:             driverName,
			pod:                    genTestPod(),
			claimInfo:              genTestClaimInfo(claimUID, []string{podUID}, true),
			resp:                   &drapb.NodeUnprepareResourcesResponse{Claims: map[string]*drapb.NodeUnprepareResourceResponse{string(claimUID): nil}},
			expectedUnprepareCalls: 1,
			expectedMetric: `# HELP dra_grpc_operations_duration_seconds [ALPHA] Duration in seconds of the DRA gRPC operations
# TYPE dra_grpc_operations_duration_seconds histogram
dra_grpc_operations_duration_seconds_bucket{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources",le="+Inf"} 1
dra_grpc_operations_duration_seconds_sum{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"} 0
dra_grpc_operations_duration_seconds_count{driver_name="test-driver",grpc_status_code="OK",method_name="/k8s.io.kubelet.pkg.apis.dra.v1.DRAPlugin/NodeUnprepareResources"} 1
# HELP dra_operations_duration_seconds [ALPHA] Latency histogram in seconds for the duration of handling all ResourceClaims referenced by a pod when the pod starts or stops. Identified by the name of the operation (PrepareResources or UnprepareResources) and separated by the success of the operation. The number of failed operations is provided through the histogram's overall count.
# TYPE dra_operations_duration_seconds histogram
dra_operations_duration_seconds_bucket{is_error="false",operation_name="UnprepareResources",le="+Inf"} 1
dra_operations_duration_seconds_sum{is_error="false",operation_name="UnprepareResources"} 0
dra_operations_duration_seconds_count{is_error="false",operation_name="UnprepareResources"} 1
`,
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			backgroundCtx, cancel := context.WithCancel(context.Background())
			defer cancel()

			kubeletmetrics.Register()
			kubeletmetrics.DRAOperationsDuration.Reset()
			kubeletmetrics.DRAGRPCOperationsDuration.Reset()
			defer func() {
				require.NoError(t,
					metricstestutil.GatherAndCompare(gatherWithoutDurations,
						strings.NewReader(test.expectedMetric),
						kubeletmetrics.DRAOperationsDuration.FQName(),
						kubeletmetrics.DRAGRPCOperationsDuration.FQName(),
					),
				)
			}()

			tCtx := ktesting.Init(t)
			backgroundCtx = klog.NewContext(backgroundCtx, tCtx.Logger())

			var pluginClientTimeout *time.Duration
			if test.wantTimeout {
				timeout := time.Millisecond * 20
				pluginClientTimeout = &timeout
			}

			draServerInfo, err := setupFakeDRADriverGRPCServer(backgroundCtx, test.wantTimeout, pluginClientTimeout, nil, test.resp, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer draServerInfo.teardownFn()

			manager, err := NewManager(tCtx.Logger(), fakeKubeClient, t.TempDir())
			require.NoError(t, err, "create DRA manager")
			manager.initDRAPluginManager(backgroundCtx, getFakeNode, time.Second /* very short wiping delay for testing */)

			plg := manager.GetWatcherHandler()
			if err := plg.RegisterPlugin(test.driverName, draServerInfo.socketName, []string{drapb.DRAPluginService}, pluginClientTimeout); err != nil {
				t.Fatalf("failed to register plugin %s, err: %v", test.driverName, err)
			}

			if test.claimInfo != nil {
				manager.cache.add(test.claimInfo)
			}

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, true)
			err = manager.UnprepareResources(backgroundCtx, test.pod)

			assert.Equal(t, test.expectedUnprepareCalls, draServerInfo.server.unprepareResourceCalls.Load())

			if test.expectedErrMsg != "" {
				assert.Error(t, err)
				if err != nil {
					assert.ErrorContains(t, err, test.expectedErrMsg)
				}
				return // PrepareResources returned an error so stopping the test case here
			}

			require.NoError(t, err)

			if test.wantResourceSkipped {
				if test.claimInfo != nil && len(test.claimInfo.PodUIDs) > 1 {
					cachedClaim, exists := manager.cache.get(test.claimInfo.ClaimName, test.claimInfo.Namespace)
					require.True(t, exists, "ClaimInfo should still exist if skipped")
					assert.False(t, cachedClaim.PodUIDs.Has(string(test.pod.UID)), "Pod UID should be removed from skipped claim")
				}
				return // resource skipped so no need to continue
			}

			// Check cache was cleared only on successful unprepare
			var podClaimName *string
			if len(test.pod.Spec.ResourceClaims) > 0 {
				podClaimName, _, err = resourceclaim.Name(test.pod, &test.pod.Spec.ResourceClaims[0])
				require.NoError(t, err)
			}
			claimName := claimName
			if podClaimName == nil {
				podClaimName = &claimName
			}
			assert.False(t, manager.cache.contains(*podClaimName, test.pod.Namespace), "claimInfo should not be found after successful unprepare")
		})
	}
}

func TestPodMightNeedToUnprepareResources(t *testing.T) {
	tCtx := ktesting.Init(t)
	fakeKubeClient := fake.NewSimpleClientset()
	manager, err := NewManager(tCtx.Logger(), fakeKubeClient, t.TempDir())
	require.NoError(t, err, "create DRA manager")

	claimInfo := &ClaimInfo{
		ClaimInfoState: state.ClaimInfoState{PodUIDs: sets.New(podUID), ClaimName: claimName, Namespace: namespace},
	}
	manager.cache.add(claimInfo)
	if !manager.cache.contains(claimName, namespace) {
		t.Fatalf("failed to get claimInfo from cache for claim name %s, namespace %s: err:%v", claimName, namespace, err)
	}
	claimInfo.addPodReference(types.UID(podUID))
	needsUnprepare := manager.PodMightNeedToUnprepareResources(types.UID(podUID))
	assert.True(t, needsUnprepare)
}

func TestGetContainerClaimInfos(t *testing.T) {
	for _, test := range []struct {
		description string
		pod         *v1.Pod
		claimInfo   *ClaimInfo

		expectedClaimName string
		expectedErrMsg    string
	}{
		{
			description:       "should get claim info",
			expectedClaimName: claimName,
			pod:               genTestPod(),
			claimInfo:         genTestClaimInfo(claimUID, []string{podUID}, false),
		},
		{
			description:       "should get extended resource claim info",
			expectedClaimName: claimName,
			pod:               genTestPodWithExtendedResource(),
			claimInfo:         genTestClaimInfoWithExtendedResource([]string{podUID}, false),
		},
		{
			description:    "should fail when claim info not found",
			pod:            genTestPod(),
			claimInfo:      &ClaimInfo{},
			expectedErrMsg: "unable to get information for ResourceClaim ",
		},
		{
			description: "should fail when none of the supported fields are set",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: namespace,
					UID:       podUID,
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: claimName,
							// missing ResourceClaimName or ResourceClaimTemplateName
						},
					},
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{
									{
										Name: claimName,
									},
								},
							},
						},
					},
				},
			},
			claimInfo:      genTestClaimInfo(claimUID, []string{podUID}, false),
			expectedErrMsg: "none of the supported fields are set",
		},
		{
			description:    "should fail when claim info is not cached",
			pod:            genTestPod(),
			expectedErrMsg: "unable to get information for ResourceClaim ",
		},
	} {
		t.Run(test.description, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			manager, err := NewManager(tCtx.Logger(), nil, t.TempDir())
			require.NoError(t, err, "create DRA manager")

			if test.claimInfo != nil {
				manager.cache.add(test.claimInfo)
			}

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, true)
			claimInfos, err := manager.GetContainerClaimInfos(test.pod, &test.pod.Spec.Containers[0])

			if test.expectedErrMsg != "" {
				assert.Error(t, err)
				if err != nil {
					assert.ErrorContains(t, err, test.expectedErrMsg)
				}
				return
			}

			require.NoError(t, err)
			assert.Len(t, claimInfos, 1)
			assert.Equal(t, test.expectedClaimName, claimInfos[0].ClaimInfoState.ClaimName)
		})
	}
}

// TestParallelPrepareUnprepareResources calls PrepareResources and UnprepareResources APIs in parallel
// to detect possible data races
func TestParallelPrepareUnprepareResources(t *testing.T) {
	tCtx := ktesting.Init(t)

	// Setup and register fake DRA driver
	draServerInfo, err := setupFakeDRADriverGRPCServer(tCtx, false, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer draServerInfo.teardownFn()

	// Create fake Kube client and DRA manager
	fakeKubeClient := fake.NewSimpleClientset()
	manager, err := NewManager(tCtx.Logger(), fakeKubeClient, t.TempDir())
	require.NoError(t, err, "create DRA manager")
	manager.initDRAPluginManager(tCtx, getFakeNode, time.Second /* very short wiping delay for testing */)

	plg := manager.GetWatcherHandler()
	if err := plg.RegisterPlugin(driverName, draServerInfo.socketName, []string{drapb.DRAPluginService}, nil); err != nil {
		t.Fatalf("failed to register plugin %s, err: %v", driverName, err)
	}

	// Call PrepareResources in parallel
	var wgSync, wgStart sync.WaitGroup // groups to sync goroutines
	numGoroutines := 30
	wgSync.Add(numGoroutines)
	wgStart.Add(1)
	for i := 0; i < numGoroutines; i++ {
		go func(t *testing.T, goRoutineNum int) {
			defer wgSync.Done()
			wgStart.Wait() // Wait to start all goroutines at the same time

			var err error
			claimName := fmt.Sprintf("test-pod-claim-%d", goRoutineNum)
			podUID := types.UID(fmt.Sprintf("test-pod-uid-%d", goRoutineNum))
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("test-pod-%d", goRoutineNum),
					Namespace: namespace,
					UID:       podUID,
				},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{
						{
							Name: claimName,
							ResourceClaimName: func() *string {
								s := claimName
								return &s
							}(),
						},
					},
					Containers: []v1.Container{
						{
							Name: fmt.Sprintf("container-%d", goRoutineNum),
							Resources: v1.ResourceRequirements{
								Claims: []v1.ResourceClaim{
									{
										Name: claimName,
									},
								},
							},
						},
					},
				},
			}
			claim := genTestClaim(claimName, driverName, deviceName, string(podUID))

			if _, err = fakeKubeClient.ResourceV1().ResourceClaims(pod.Namespace).Create(tCtx, claim, metav1.CreateOptions{}); err != nil {
				t.Errorf("failed to create ResourceClaim %s: %+v", claim.Name, err)
				return
			}

			defer func() {
				_ = fakeKubeClient.ResourceV1().ResourceClaims(pod.Namespace).Delete(tCtx, claim.Name, metav1.DeleteOptions{})
			}()

			if err = manager.PrepareResources(tCtx, pod); err != nil {
				t.Errorf("GoRoutine %d: pod: %s: PrepareResources failed: %+v", goRoutineNum, pod.Name, err)
				return
			}

			if err = manager.UnprepareResources(tCtx, pod); err != nil {
				t.Errorf("GoRoutine %d: pod: %s: UnprepareResources failed: %+v", goRoutineNum, pod.Name, err)
				return
			}

		}(t, i)
	}
	wgStart.Done() // Start executing goroutines
	wgSync.Wait()  // Wait for all goroutines to finish
}

// TestHandleWatchResourcesStream verifies the manager's ability to process health updates
// received from a DRA plugin's WatchResources stream. It checks if the internal health cache
// is updated correctly, if affected pods are identified, and if update notifications are sent
// through the manager's update channel. It covers various scenarios including health changes, stream errors, and context cancellation.
func TestHandleWatchResourcesStream(t *testing.T) {
	overallTestCtx, overallTestCancel := context.WithCancel(ktesting.Init(t))
	defer overallTestCancel()

	// Helper to create and setup a new manager for each sub-test
	setupNewManagerAndRunStreamTest := func(
		st *testing.T,
		testSpecificCtx context.Context,
		initialClaimInfos ...*ClaimInfo,
	) (
		managerInstance *Manager,
		runTestStreamFunc func(context.Context, chan struct {
			Resp *drahealthv1alpha1.NodeWatchResourcesResponse
			Err  error
		}) (<-chan resourceupdates.Update, chan struct{}, chan error),
	) {
		tCtx := ktesting.Init(t)
		// Fresh manager for each sub-test
		manager, err := NewManager(tCtx.Logger(), nil, st.TempDir())
		require.NoError(st, err)

		for _, ci := range initialClaimInfos {
			manager.cache.add(ci)
		}

		managerInstance = manager

		runTestStreamFunc = func(
			streamCtx context.Context,
			responses chan struct {
				Resp *drahealthv1alpha1.NodeWatchResourcesResponse
				Err  error
			},
		) (<-chan resourceupdates.Update, chan struct{}, chan error) {
			mockStream := &mockWatchResourcesClient{
				RecvChan: responses,
				Ctx:      streamCtx,
			}
			done := make(chan struct{})
			errChan := make(chan error, 1)
			go func() {
				defer close(done)
				// Use a logger that includes sub-test name for clarity
				logger := klog.FromContext(streamCtx).WithName(st.Name())
				hdlCtx := klog.NewContext(streamCtx, logger)

				err := managerInstance.HandleWatchResourcesStream(hdlCtx, mockStream, driverName)
				if err != nil {
					if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, io.EOF) {
						logger.V(4).Info("HandleWatchResourcesStream (test goroutine) exited as expected", "error", err)
					} else {
						// This is an application/stream error, not a standard exit.
						// The sub-test ("StreamError") will assert this specific error.
						logger.V(2).Info("HandleWatchResourcesStream (test goroutine) exited with application/stream error", "error", err)
					}
				} else {
					logger.V(4).Info("HandleWatchResourcesStream (test goroutine) exited cleanly (nil error, likely from EOF)")
				}
				errChan <- err
				close(errChan)
			}()
			return managerInstance.update, done, errChan
		}
		return managerInstance, runTestStreamFunc
	}

	// Test Case 1: Health change for an allocated device
	t.Run("HealthChangeForAllocatedDevice", func(t *testing.T) {
		stCtx, stCancel := context.WithCancel(overallTestCtx)
		defer stCancel()

		// Setup: Create a manager with a relevant claim already in its cache.
		initialClaim := genTestClaimInfo(claimUID, []string{string(podUID)}, true)
		manager, runStreamTest := setupNewManagerAndRunStreamTest(t, stCtx, initialClaim)

		t.Log("HealthChangeForAllocatedDevice: Test Case Started")

		responses := make(chan struct {
			Resp *drahealthv1alpha1.NodeWatchResourcesResponse
			Err  error
		}, 1)
		updateChan, done, streamErrChan := runStreamTest(stCtx, responses)

		// Send the health update message
		unhealthyDeviceMsg := &drahealthv1alpha1.DeviceHealth{
			Device: &drahealthv1alpha1.DeviceIdentifier{
				PoolName:   poolName,
				DeviceName: deviceName,
			},
			Health:          drahealthv1alpha1.HealthStatus_UNHEALTHY,
			LastUpdatedTime: time.Now().Unix(),
		}
		t.Logf("HealthChangeForAllocatedDevice: Sending health update: %+v", unhealthyDeviceMsg)
		responses <- struct {
			Resp *drahealthv1alpha1.NodeWatchResourcesResponse
			Err  error
		}{
			Resp: &drahealthv1alpha1.NodeWatchResourcesResponse{Devices: []*drahealthv1alpha1.DeviceHealth{unhealthyDeviceMsg}},
		}

		t.Log("HealthChangeForAllocatedDevice: Waiting for update on manager channel")
		select {
		case upd := <-updateChan:
			t.Logf("HealthChangeForAllocatedDevice: Received update: %+v", upd)
			assert.ElementsMatch(t, []string{string(podUID)}, upd.PodUIDs, "Expected pod UID in update")
		case <-time.After(2 * time.Second):
			t.Fatal("HealthChangeForAllocatedDevice: Timeout waiting for pod update on manager.update channel")
		}

		// Check cache state
		cachedHealth := manager.healthInfoCache.getHealthInfo(driverName, poolName, deviceName)
		assert.Equal(t, state.DeviceHealthStatus("Unhealthy"), cachedHealth, "Cache update check failed")

		t.Log("HealthChangeForAllocatedDevice: Closing responses channel to signal EOF")
		close(responses)

		t.Log("HealthChangeForAllocatedDevice: Waiting on done channel")
		var finalErr error
		select {
		case <-done:
			finalErr = <-streamErrChan
			t.Log("HealthChangeForAllocatedDevice: done channel closed, stream goroutine finished.")
		case <-time.After(1 * time.Second):
			t.Fatal("HealthChangeForAllocatedDevice: Timed out waiting for HandleWatchResourcesStream to finish after EOF signal")
		}
		// Expect nil (if HandleWatchResourcesStream returns nil on EOF) or io.EOF
		assert.True(t, finalErr == nil || errors.Is(finalErr, io.EOF), "Expected nil or io.EOF, got %v", finalErr)
	})

	// Test Case 2: Health change for a non-allocated device
	t.Run("NonAllocatedDeviceChange", func(t *testing.T) {
		stCtx, stCancel := context.WithCancel(overallTestCtx)
		defer stCancel()

		// Setup: Manager with no specific claims, or claims that don't use "other-device"
		manager, runStreamTest := setupNewManagerAndRunStreamTest(t, stCtx)

		t.Log("NonAllocatedDeviceChange: Test Case Started")
		responses := make(chan struct {
			Resp *drahealthv1alpha1.NodeWatchResourcesResponse
			Err  error
		}, 1)
		updateChan, done, streamErrChan := runStreamTest(stCtx, responses)

		otherDeviceMsg := &drahealthv1alpha1.DeviceHealth{
			Device: &drahealthv1alpha1.DeviceIdentifier{
				PoolName:   poolName,
				DeviceName: "other-device",
			},
			Health:          drahealthv1alpha1.HealthStatus_UNHEALTHY,
			LastUpdatedTime: time.Now().Unix(),
		}
		responses <- struct {
			Resp *drahealthv1alpha1.NodeWatchResourcesResponse
			Err  error
		}{
			Resp: &drahealthv1alpha1.NodeWatchResourcesResponse{Devices: []*drahealthv1alpha1.DeviceHealth{otherDeviceMsg}},
		}

		select {
		case upd := <-updateChan:
			t.Fatalf("NonAllocatedDeviceChange: Unexpected update on manager.update channel: %+v", upd)
			// OK, no update expected on manager.update for this device
		case <-time.After(200 * time.Millisecond):
			t.Log("NonAllocatedDeviceChange: Correctly received no update on manager channel.")
		}

		// Check health cache for the "other-device"
		cachedHealthOther := manager.healthInfoCache.getHealthInfo(driverName, poolName, "other-device")
		assert.Equal(t, state.DeviceHealthStatus("Unhealthy"), cachedHealthOther, "Cache update for other-device failed")

		close(responses)
		var finalErr error
		select {
		case <-done:
			finalErr = <-streamErrChan
			t.Log("NonAllocatedDeviceChange: Stream handler goroutine finished.")
		case <-time.After(1 * time.Second):
			t.Fatal("NonAllocatedDeviceChange: Timeout waiting for stream handler to finish after EOF")
		}
		assert.True(t, finalErr == nil || errors.Is(finalErr, io.EOF), "Expected nil or io.EOF, got %v", finalErr)
	})

	// Test Case 3: No actual health state change (idempotency)
	t.Run("NoActualStateChange", func(t *testing.T) {
		stCtx, stCancel := context.WithCancel(overallTestCtx)
		defer stCancel()

		// Setup: Manager with a claim and the device already marked Unhealthy in health cache
		initialClaim := genTestClaimInfo(claimUID, []string{string(podUID)}, true)
		manager, runStreamTest := setupNewManagerAndRunStreamTest(t, stCtx, initialClaim)

		// Pre-populate health cache
		initialHealth := state.DeviceHealth{PoolName: poolName, DeviceName: deviceName, Health: "Unhealthy", LastUpdated: time.Now().Add(-5 * time.Millisecond), HealthCheckTimeout: DefaultHealthTimeout} // Ensure LastUpdated is slightly in past
		_, err := manager.healthInfoCache.updateHealthInfo(driverName, []state.DeviceHealth{initialHealth})
		require.NoError(t, err, "Failed to pre-populate health cache")

		t.Log("NoActualStateChange: Test Case Started")
		responses := make(chan struct {
			Resp *drahealthv1alpha1.NodeWatchResourcesResponse
			Err  error
		}, 1)
		updateChan, done, streamErrChan := runStreamTest(stCtx, responses)

		// Send the same "Unhealthy" state again
		unhealthyDeviceMsg := &drahealthv1alpha1.DeviceHealth{
			Device: &drahealthv1alpha1.DeviceIdentifier{
				PoolName:   poolName,
				DeviceName: deviceName,
			},
			Health:          drahealthv1alpha1.HealthStatus_UNHEALTHY,
			LastUpdatedTime: time.Now().Unix(),
		}
		responses <- struct {
			Resp *drahealthv1alpha1.NodeWatchResourcesResponse
			Err  error
		}{
			Resp: &drahealthv1alpha1.NodeWatchResourcesResponse{Devices: []*drahealthv1alpha1.DeviceHealth{unhealthyDeviceMsg}},
		}

		select {
		case upd := <-updateChan:
			t.Fatalf("NoActualStateChange: Unexpected update on manager.update channel: %+v", upd)
		case <-time.After(200 * time.Millisecond):
			t.Log("NoActualStateChange: Correctly received no update on manager channel.")
		}

		close(responses)
		var finalErr error
		select {
		case <-done:
			finalErr = <-streamErrChan
			t.Log("NoActualStateChange: Stream handler goroutine finished.")
		case <-time.After(1 * time.Second):
			t.Fatal("NoActualStateChange: Timeout waiting for stream handler to finish after EOF")
		}
		assert.True(t, finalErr == nil || errors.Is(finalErr, io.EOF), "Expected nil or io.EOF, got %v", finalErr)
	})

	// Test Case 4: Stream error
	t.Run("StreamError", func(t *testing.T) {
		stCtx, stCancel := context.WithCancel(overallTestCtx)
		defer stCancel()

		// Get a new manager and the scoped runStreamTest helper
		_, runStreamTest := setupNewManagerAndRunStreamTest(t, stCtx)
		t.Log("StreamError: Test Case Started")

		responses := make(chan struct {
			Resp *drahealthv1alpha1.NodeWatchResourcesResponse
			Err  error
		}, 1)
		_, done, streamErrChan := runStreamTest(stCtx, responses)

		expectedStreamErr := errors.New("simulated mock stream error")
		responses <- struct {
			Resp *drahealthv1alpha1.NodeWatchResourcesResponse
			Err  error
		}{Err: expectedStreamErr}

		t.Log("StreamError: Waiting on done channel")
		var actualErr error
		select {
		case <-done:
			// Read the error propagated from the HandleWatchResourcesStream goroutine
			actualErr = <-streamErrChan
			t.Logf("StreamError: done channel closed. Stream handler returned: %v", actualErr)
		case <-time.After(2 * time.Second):
			t.Fatal("StreamError: Timeout waiting for stream handler to finish after error signal")
		}

		require.Error(t, actualErr, "HandleWatchResourcesStream should have returned an error")
		assert.ErrorIs(t, actualErr, expectedStreamErr)
	})

	// Test Case 5: Context cancellation
	t.Run("ContextCanceled", func(t *testing.T) {
		stCtx, stCancel := context.WithCancel(overallTestCtx)
		// Deliberately do not `defer stCancel()` for this specific test case

		_, runStreamTest := setupNewManagerAndRunStreamTest(t, stCtx)
		t.Log("ContextCanceled: Test Case Started")

		responses := make(chan struct {
			Resp *drahealthv1alpha1.NodeWatchResourcesResponse
			Err  error
		})
		_, done, streamErrChan := runStreamTest(stCtx, responses)

		t.Log("ContextCanceled: Intentionally canceling context for stream handler after a short delay.")
		time.Sleep(50 * time.Millisecond)
		stCancel()

		t.Log("ContextCanceled: Waiting on done channel")
		var finalErr error
		select {
		case <-done:
			finalErr = <-streamErrChan
			t.Log("ContextCanceled: done channel closed. Stream handler finished after context cancellation.")
		case <-time.After(1 * time.Second):
			t.Fatal("ContextCanceled: Timeout waiting for stream handler to finish after context cancellation")
		}
		require.Error(t, finalErr)
		assert.True(t, errors.Is(finalErr, context.Canceled) || errors.Is(finalErr, context.DeadlineExceeded))
	})
}

// TestUpdateAllocatedResourcesStatus verifies that the manager can correctly
// update the PodStatus with health information for different types of DRA claims.
// It covers the main scenarios that were difficult to test reliably in an e2e
// environment due to timing issues:
//  1. Direct claims: Where the pod's resource claim name directly matches the ResourceClaim object name.
//  2. Renamed claims: Where the pod uses a local name for a ResourceClaim that has a different object name.
//  3. Templated claims: Where the claim is generated from a template, and the pod status contains the
//     dynamically generated claim name.
func TestUpdateAllocatedResourcesStatus(t *testing.T) {
	directClaimName := "direct-claim-name"
	renamedClaimObject := "renamed-claim-object"
	templateName := "template-name"
	templatedClaimName := "pod-templated-templated-claim"

	testCases := []struct {
		name                             string
		pod                              *v1.Pod
		claimInfos                       []*ClaimInfo
		initialStatus                    *v1.PodStatus
		expectedAllocatedResourcesStatus []v1.ResourceStatus
	}{
		{
			name: "Direct claim",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-direct", UID: "pod-direct-uid"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{Name: "container1", Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: "claim1"}}}},
					},
					ResourceClaims: []v1.PodResourceClaim{
						{Name: "claim1", ResourceClaimName: &directClaimName},
					},
				},
				Status: v1.PodStatus{
					ResourceClaimStatuses: []v1.PodResourceClaimStatus{
						{Name: "claim1", ResourceClaimName: &directClaimName},
					},
				},
			},
			claimInfos: []*ClaimInfo{
				{
					ClaimInfoState: state.ClaimInfoState{
						ClaimName: directClaimName,
						PodUIDs:   sets.New("pod-direct-uid"),
						DriverState: map[string]state.DriverState{
							"test-driver": {Devices: []state.Device{{PoolName: "pool", DeviceName: "dev-a"}}},
						},
					},
				},
			},
			initialStatus: &v1.PodStatus{ContainerStatuses: []v1.ContainerStatus{{Name: "container1"}}},
			expectedAllocatedResourcesStatus: []v1.ResourceStatus{
				{
					Name: "claim:claim1",
					Resources: []v1.ResourceHealth{
						{ResourceID: "test-driver/pool/dev-a", Health: v1.ResourceHealthStatusHealthy},
					},
				},
			},
		},
		{
			name: "Renamed claim",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-renamed", UID: "pod-renamed-uid"},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{{Name: "renamed-pod-claim", ResourceClaimName: &renamedClaimObject}},
					Containers:     []v1.Container{{Name: "container1", Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: "renamed-pod-claim"}}}}},
				},
				Status: v1.PodStatus{
					ResourceClaimStatuses: []v1.PodResourceClaimStatus{
						{Name: "renamed-pod-claim", ResourceClaimName: &renamedClaimObject},
					},
				},
			},
			claimInfos: []*ClaimInfo{
				{
					ClaimInfoState: state.ClaimInfoState{
						ClaimName: renamedClaimObject,
						PodUIDs:   sets.New("pod-renamed-uid"),
						DriverState: map[string]state.DriverState{
							"test-driver": {Devices: []state.Device{{PoolName: "pool", DeviceName: "dev-b"}}},
						},
					},
				},
			},
			initialStatus: &v1.PodStatus{ContainerStatuses: []v1.ContainerStatus{{Name: "container1"}}},
			expectedAllocatedResourcesStatus: []v1.ResourceStatus{
				{
					Name: "claim:renamed-pod-claim",
					Resources: []v1.ResourceHealth{
						{ResourceID: "test-driver/pool/dev-b", Health: v1.ResourceHealthStatusHealthy},
					},
				},
			},
		},
		{
			name: "Templated claim",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod-templated", UID: "pod-templated-uid"},
				Spec: v1.PodSpec{
					ResourceClaims: []v1.PodResourceClaim{{Name: "templated-claim", ResourceClaimTemplateName: &templateName}},
					Containers:     []v1.Container{{Name: "container1", Resources: v1.ResourceRequirements{Claims: []v1.ResourceClaim{{Name: "templated-claim"}}}}},
				},
				Status: v1.PodStatus{
					ResourceClaimStatuses: []v1.PodResourceClaimStatus{
						{Name: "templated-claim", ResourceClaimName: &templatedClaimName},
					},
				},
			},
			claimInfos: []*ClaimInfo{
				{
					ClaimInfoState: state.ClaimInfoState{
						ClaimName: templatedClaimName,
						PodUIDs:   sets.New("pod-templated-uid"),
						DriverState: map[string]state.DriverState{
							"test-driver": {Devices: []state.Device{{PoolName: "pool", DeviceName: "dev-c"}}},
						},
					},
				},
			},
			initialStatus: &v1.PodStatus{ContainerStatuses: []v1.ContainerStatus{{Name: "container1"}}},
			expectedAllocatedResourcesStatus: []v1.ResourceStatus{
				{
					Name: "claim:templated-claim",
					Resources: []v1.ResourceHealth{
						{ResourceID: "test-driver/pool/dev-c", Health: v1.ResourceHealthStatusHealthy},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			manager, err := NewManager(tCtx.Logger(), nil, t.TempDir())
			require.NoError(t, err)

			for _, ci := range tc.claimInfos {
				manager.cache.add(ci)
			}

			// Set all devices to be healthy for the test.
			devices := []state.DeviceHealth{}
			for _, ci := range tc.claimInfos {
				for driverName, ds := range ci.DriverState {
					for _, dev := range ds.Devices {
						devices = append(devices, state.DeviceHealth{PoolName: dev.PoolName, DeviceName: dev.DeviceName, Health: state.DeviceHealthStatusHealthy})
					}
					_, err := manager.healthInfoCache.updateHealthInfo(driverName, devices)
					require.NoError(t, err)
				}
			}

			status := tc.initialStatus.DeepCopy()
			manager.UpdateAllocatedResourcesStatus(tc.pod, status)

			require.Len(t, status.ContainerStatuses, 1)
			assert.Equal(t, tc.expectedAllocatedResourcesStatus, status.ContainerStatuses[0].AllocatedResourcesStatus)
		})
	}
}
