/*
Copyright 2022 The Kubernetes Authors.

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

package app

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	resourceapi "k8s.io/api/resource/v1alpha2"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/klog/v2"
	drapbv1alpha3 "k8s.io/kubelet/pkg/apis/dra/v1alpha3"
)

type ExamplePlugin struct {
	stopCh  <-chan struct{}
	logger  klog.Logger
	d       kubeletplugin.DRAPlugin
	fileOps FileOperations

	cdiDir     string
	driverName string
	nodeName   string
	instances  sets.Set[string]

	mutex          sync.Mutex
	instancesInUse sets.Set[string]
	prepared       map[ClaimID]any
	gRPCCalls      []GRPCCall

	block bool

	prepareResourcesFailure   error
	failPrepareResourcesMutex sync.Mutex

	unprepareResourcesFailure   error
	failUnprepareResourcesMutex sync.Mutex
}

type GRPCCall struct {
	// FullMethod is the fully qualified, e.g. /package.service/method.
	FullMethod string

	// Request contains the parameters of the call.
	Request interface{}

	// Response contains the reply of the plugin. It is nil for calls that are in progress.
	Response interface{}

	// Err contains the error return value of the plugin. It is nil for calls that are in progress or succeeded.
	Err error
}

// ClaimID contains both claim name and UID to simplify debugging. The
// namespace is not included because it is random in E2E tests and the UID is
// sufficient to make the ClaimID unique.
type ClaimID struct {
	Name string
	UID  string
}

var _ drapbv1alpha3.NodeServer = &ExamplePlugin{}

// getJSONFilePath returns the absolute path where CDI file is/should be.
func (ex *ExamplePlugin) getJSONFilePath(claimUID string) string {
	return filepath.Join(ex.cdiDir, fmt.Sprintf("%s-%s.json", ex.driverName, claimUID))
}

// FileOperations defines optional callbacks for handling CDI files
// and some other configuration.
type FileOperations struct {
	// Create must overwrite the file.
	Create func(name string, content []byte) error

	// Remove must remove the file. It must not return an error when the
	// file does not exist.
	Remove func(name string) error

	// NumResourceInstances determines whether the plugin reports resources
	// instances and how many. A negative value causes it to report "not implemented"
	// in the NodeListAndWatchResources gRPC call.
	NumResourceInstances int
}

// StartPlugin sets up the servers that are necessary for a DRA kubelet plugin.
func StartPlugin(ctx context.Context, cdiDir, driverName string, nodeName string, fileOps FileOperations, opts ...kubeletplugin.Option) (*ExamplePlugin, error) {
	logger := klog.FromContext(ctx)
	if fileOps.Create == nil {
		fileOps.Create = func(name string, content []byte) error {
			return os.WriteFile(name, content, os.FileMode(0644))
		}
	}
	if fileOps.Remove == nil {
		fileOps.Remove = func(name string) error {
			if err := os.Remove(name); err != nil && !os.IsNotExist(err) {
				return err
			}
			return nil
		}
	}
	ex := &ExamplePlugin{
		stopCh:         ctx.Done(),
		logger:         logger,
		fileOps:        fileOps,
		cdiDir:         cdiDir,
		driverName:     driverName,
		nodeName:       nodeName,
		instances:      sets.New[string](),
		instancesInUse: sets.New[string](),
		prepared:       make(map[ClaimID]any),
	}

	for i := 0; i < ex.fileOps.NumResourceInstances; i++ {
		ex.instances.Insert(fmt.Sprintf("instance-%02d", i))
	}

	opts = append(opts,
		kubeletplugin.Logger(logger),
		kubeletplugin.DriverName(driverName),
		kubeletplugin.GRPCInterceptor(ex.recordGRPCCall),
		kubeletplugin.GRPCStreamInterceptor(ex.recordGRPCStream),
	)
	d, err := kubeletplugin.Start(ex, opts...)
	if err != nil {
		return nil, fmt.Errorf("start kubelet plugin: %w", err)
	}
	ex.d = d

	return ex, nil
}

// stop ensures that all servers are stopped and resources freed.
func (ex *ExamplePlugin) Stop() {
	ex.d.Stop()
}

func (ex *ExamplePlugin) IsRegistered() bool {
	status := ex.d.RegistrationStatus()
	if status == nil {
		return false
	}
	return status.PluginRegistered
}

// Block sets a flag to block Node[Un]PrepareResources
// to emulate time consuming or stuck calls
func (ex *ExamplePlugin) Block() {
	ex.block = true
}

func (ex *ExamplePlugin) withLock(mutex *sync.Mutex, f func()) {
	mutex.Lock()
	f()
	mutex.Unlock()
}

// SetNodePrepareResourcesFailureMode sets the failure mode for NodePrepareResources call
// and returns a function to unset the failure mode
func (ex *ExamplePlugin) SetNodePrepareResourcesFailureMode() func() {
	ex.failPrepareResourcesMutex.Lock()
	ex.prepareResourcesFailure = errors.New("simulated PrepareResources failure")
	ex.failPrepareResourcesMutex.Unlock()

	return func() {
		ex.failPrepareResourcesMutex.Lock()
		ex.prepareResourcesFailure = nil
		ex.failPrepareResourcesMutex.Unlock()
	}
}

func (ex *ExamplePlugin) getPrepareResourcesFailure() error {
	ex.failPrepareResourcesMutex.Lock()
	defer ex.failPrepareResourcesMutex.Unlock()
	return ex.prepareResourcesFailure
}

// SetNodeUnprepareResourcesFailureMode sets the failure mode for NodeUnprepareResources call
// and returns a function to unset the failure mode
func (ex *ExamplePlugin) SetNodeUnprepareResourcesFailureMode() func() {
	ex.failUnprepareResourcesMutex.Lock()
	ex.unprepareResourcesFailure = errors.New("simulated UnprepareResources failure")
	ex.failUnprepareResourcesMutex.Unlock()

	return func() {
		ex.failUnprepareResourcesMutex.Lock()
		ex.unprepareResourcesFailure = nil
		ex.failUnprepareResourcesMutex.Unlock()
	}
}

func (ex *ExamplePlugin) getUnprepareResourcesFailure() error {
	ex.failUnprepareResourcesMutex.Lock()
	defer ex.failUnprepareResourcesMutex.Unlock()
	return ex.unprepareResourcesFailure
}

// NodePrepareResource ensures that the CDI file for the claim exists. It uses
// a deterministic name to simplify NodeUnprepareResource (no need to remember
// or discover the name) and idempotency (when called again, the file simply
// gets written again).
func (ex *ExamplePlugin) nodePrepareResource(ctx context.Context, claimName string, claimUID string, resourceHandle string, structuredResourceHandle []*resourceapi.StructuredResourceHandle) ([]string, error) {
	logger := klog.FromContext(ctx)

	// Block to emulate plugin stuckness or slowness.
	// By default the call will not be blocked as ex.block = false.
	if ex.block {
		<-ctx.Done()
		return nil, ctx.Err()
	}

	ex.mutex.Lock()
	defer ex.mutex.Unlock()

	deviceName := "claim-" + claimUID
	vendor := ex.driverName
	class := "test"
	dev := vendor + "/" + class + "=" + deviceName
	claimID := ClaimID{Name: claimName, UID: claimUID}
	if _, ok := ex.prepared[claimID]; ok {
		// Idempotent call, nothing to do.
		return []string{dev}, nil
	}

	// Determine environment variables.
	var p parameters
	var actualResourceHandle any
	var instanceNames []string
	switch len(structuredResourceHandle) {
	case 0:
		// Control plane controller did the allocation.
		if err := json.Unmarshal([]byte(resourceHandle), &p); err != nil {
			return nil, fmt.Errorf("unmarshal resource handle: %w", err)
		}
		actualResourceHandle = resourceHandle
	case 1:
		// Scheduler did the allocation with structured parameters.
		handle := structuredResourceHandle[0]
		if handle == nil {
			return nil, errors.New("unexpected nil StructuredResourceHandle")
		}
		p.NodeName = handle.NodeName
		if err := extractParameters(handle.VendorClassParameters, &p.EnvVars, "admin"); err != nil {
			return nil, err
		}
		if err := extractParameters(handle.VendorClaimParameters, &p.EnvVars, "user"); err != nil {
			return nil, err
		}
		for _, result := range handle.Results {
			if err := extractParameters(result.VendorRequestParameters, &p.EnvVars, "user"); err != nil {
				return nil, err
			}
			namedResources := result.NamedResources
			if namedResources == nil {
				return nil, errors.New("missing named resources allocation result")
			}
			instanceName := namedResources.Name
			if instanceName == "" {
				return nil, errors.New("empty named resources instance name")
			}
			if !ex.instances.Has(instanceName) {
				return nil, fmt.Errorf("unknown allocated instance %q", instanceName)
			}
			if ex.instancesInUse.Has(instanceName) {
				return nil, fmt.Errorf("resource instance %q used more than once", instanceName)
			}
			instanceNames = append(instanceNames, instanceName)
		}
		actualResourceHandle = handle
	default:
		// Huh?
		return nil, fmt.Errorf("invalid length of NodePrepareResourceRequest.StructuredResourceHandle: %d", len(structuredResourceHandle))
	}

	// Sanity check scheduling.
	if p.NodeName != "" && ex.nodeName != "" && p.NodeName != ex.nodeName {
		return nil, fmt.Errorf("claim was allocated for %q, cannot be prepared on %q", p.NodeName, ex.nodeName)
	}

	// CDI wants env variables as set of strings.
	envs := []string{}
	for key, val := range p.EnvVars {
		envs = append(envs, key+"="+val)
	}

	spec := &spec{
		Version: "0.3.0", // This has to be a version accepted by the runtimes.
		Kind:    vendor + "/" + class,
		// At least one device is required and its entry must have more
		// than just the name.
		Devices: []device{
			{
				Name: deviceName,
				ContainerEdits: containerEdits{
					Env: envs,
				},
			},
		},
	}
	filePath := ex.getJSONFilePath(claimUID)
	buffer, err := json.Marshal(spec)
	if err != nil {
		return nil, fmt.Errorf("marshal spec: %w", err)
	}
	if err := ex.fileOps.Create(filePath, buffer); err != nil {
		return nil, fmt.Errorf("failed to write CDI file %v", err)
	}

	ex.prepared[claimID] = actualResourceHandle
	for _, instanceName := range instanceNames {
		ex.instancesInUse.Insert(instanceName)
	}

	logger.V(3).Info("CDI file created", "path", filePath, "device", dev)
	return []string{dev}, nil
}

func extractParameters(parameters runtime.RawExtension, env *map[string]string, kind string) error {
	if len(parameters.Raw) == 0 {
		return nil
	}
	var data map[string]string
	if err := json.Unmarshal(parameters.Raw, &data); err != nil {
		return fmt.Errorf("decoding %s parameters: %v", kind, err)
	}
	if len(data) > 0 && *env == nil {
		*env = make(map[string]string)
	}
	for key, value := range data {
		(*env)[kind+"_"+key] = value
	}
	return nil
}

func (ex *ExamplePlugin) NodePrepareResources(ctx context.Context, req *drapbv1alpha3.NodePrepareResourcesRequest) (*drapbv1alpha3.NodePrepareResourcesResponse, error) {
	resp := &drapbv1alpha3.NodePrepareResourcesResponse{
		Claims: make(map[string]*drapbv1alpha3.NodePrepareResourceResponse),
	}

	if failure := ex.getPrepareResourcesFailure(); failure != nil {
		return resp, failure
	}

	for _, claimReq := range req.Claims {
		cdiDevices, err := ex.nodePrepareResource(ctx, claimReq.Name, claimReq.Uid, claimReq.ResourceHandle, claimReq.StructuredResourceHandle)
		if err != nil {
			resp.Claims[claimReq.Uid] = &drapbv1alpha3.NodePrepareResourceResponse{
				Error: err.Error(),
			}
		} else {
			resp.Claims[claimReq.Uid] = &drapbv1alpha3.NodePrepareResourceResponse{
				CDIDevices: cdiDevices,
			}
		}
	}
	return resp, nil
}

// NodeUnprepareResource removes the CDI file created by
// NodePrepareResource. It's idempotent, therefore it is not an error when that
// file is already gone.
func (ex *ExamplePlugin) nodeUnprepareResource(ctx context.Context, claimName string, claimUID string, resourceHandle string, structuredResourceHandle []*resourceapi.StructuredResourceHandle) error {
	logger := klog.FromContext(ctx)

	// Block to emulate plugin stuckness or slowness.
	// By default the call will not be blocked as ex.block = false.
	if ex.block {
		<-ctx.Done()
		return ctx.Err()
	}

	filePath := ex.getJSONFilePath(claimUID)
	if err := ex.fileOps.Remove(filePath); err != nil {
		return fmt.Errorf("error removing CDI file: %w", err)
	}
	logger.V(3).Info("CDI file removed", "path", filePath)

	ex.mutex.Lock()
	defer ex.mutex.Unlock()

	claimID := ClaimID{Name: claimName, UID: claimUID}
	expectedResourceHandle, ok := ex.prepared[claimID]
	if !ok {
		// Idempotent call, nothing to do.
		return nil
	}

	var actualResourceHandle any = resourceHandle
	if structuredResourceHandle != nil {
		if len(structuredResourceHandle) != 1 {
			return fmt.Errorf("unexpected number of entries in StructuredResourceHandle: %d", len(structuredResourceHandle))
		}
		actualResourceHandle = structuredResourceHandle[0]
	}
	if diff := cmp.Diff(expectedResourceHandle, actualResourceHandle); diff != "" {
		return fmt.Errorf("difference between expected (-) and actual resource handle (+):\n%s", diff)
	}
	delete(ex.prepared, claimID)
	if structuredResourceHandle := structuredResourceHandle; structuredResourceHandle != nil {
		for _, handle := range structuredResourceHandle {
			for _, result := range handle.Results {
				instanceName := result.NamedResources.Name
				ex.instancesInUse.Delete(instanceName)
			}
		}
	}
	delete(ex.prepared, ClaimID{Name: claimName, UID: claimUID})

	return nil
}

func (ex *ExamplePlugin) NodeUnprepareResources(ctx context.Context, req *drapbv1alpha3.NodeUnprepareResourcesRequest) (*drapbv1alpha3.NodeUnprepareResourcesResponse, error) {
	resp := &drapbv1alpha3.NodeUnprepareResourcesResponse{
		Claims: make(map[string]*drapbv1alpha3.NodeUnprepareResourceResponse),
	}

	if failure := ex.getUnprepareResourcesFailure(); failure != nil {
		return resp, failure
	}

	for _, claimReq := range req.Claims {
		err := ex.nodeUnprepareResource(ctx, claimReq.Name, claimReq.Uid, claimReq.ResourceHandle, claimReq.StructuredResourceHandle)
		if err != nil {
			resp.Claims[claimReq.Uid] = &drapbv1alpha3.NodeUnprepareResourceResponse{
				Error: err.Error(),
			}
		} else {
			resp.Claims[claimReq.Uid] = &drapbv1alpha3.NodeUnprepareResourceResponse{}
		}
	}
	return resp, nil
}

func (ex *ExamplePlugin) NodeListAndWatchResources(req *drapbv1alpha3.NodeListAndWatchResourcesRequest, stream drapbv1alpha3.Node_NodeListAndWatchResourcesServer) error {
	if ex.fileOps.NumResourceInstances < 0 {
		ex.logger.Info("Sending no NodeResourcesResponse")
		return status.New(codes.Unimplemented, "node resource support disabled").Err()
	}

	instances := make([]resourceapi.NamedResourcesInstance, len(ex.instances))
	for i, name := range sets.List(ex.instances) {
		instances[i].Name = name
	}
	resp := &drapbv1alpha3.NodeListAndWatchResourcesResponse{
		Resources: []*resourceapi.ResourceModel{
			{
				NamedResources: &resourceapi.NamedResourcesResources{
					Instances: instances,
				},
			},
		},
	}

	ex.logger.Info("Sending NodeListAndWatchResourcesResponse", "response", resp)
	if err := stream.Send(resp); err != nil {
		return err
	}

	// Keep the stream open until the test is done.
	// TODO: test sending more updates later
	<-ex.stopCh
	ex.logger.Info("Done sending NodeListAndWatchResourcesResponse, closing stream")

	return nil
}

func (ex *ExamplePlugin) GetPreparedResources() []ClaimID {
	ex.mutex.Lock()
	defer ex.mutex.Unlock()
	var prepared []ClaimID
	for claimID := range ex.prepared {
		prepared = append(prepared, claimID)
	}
	return prepared
}

func (ex *ExamplePlugin) recordGRPCCall(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
	call := GRPCCall{
		FullMethod: info.FullMethod,
		Request:    req,
	}
	ex.mutex.Lock()
	ex.gRPCCalls = append(ex.gRPCCalls, call)
	index := len(ex.gRPCCalls) - 1
	ex.mutex.Unlock()

	// We don't hold the mutex here to allow concurrent calls.
	call.Response, call.Err = handler(ctx, req)

	ex.mutex.Lock()
	ex.gRPCCalls[index] = call
	ex.mutex.Unlock()

	return call.Response, call.Err
}

func (ex *ExamplePlugin) recordGRPCStream(srv interface{}, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
	call := GRPCCall{
		FullMethod: info.FullMethod,
	}
	ex.mutex.Lock()
	ex.gRPCCalls = append(ex.gRPCCalls, call)
	index := len(ex.gRPCCalls) - 1
	ex.mutex.Unlock()

	// We don't hold the mutex here to allow concurrent calls.
	call.Err = handler(srv, stream)

	ex.mutex.Lock()
	ex.gRPCCalls[index] = call
	ex.mutex.Unlock()

	return call.Err
}

func (ex *ExamplePlugin) GetGRPCCalls() []GRPCCall {
	ex.mutex.Lock()
	defer ex.mutex.Unlock()

	// We must return a new slice, otherwise adding new calls would become
	// visible to the caller. We also need to copy the entries because
	// they get mutated by recordGRPCCall.
	calls := make([]GRPCCall, 0, len(ex.gRPCCalls))
	calls = append(calls, ex.gRPCCalls...)
	return calls
}
