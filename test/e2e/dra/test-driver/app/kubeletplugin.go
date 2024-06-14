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
	"strings"
	"sync"

	"google.golang.org/grpc"

	resourceapi "k8s.io/api/resource/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/klog/v2"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1alpha4"
)

type ExamplePlugin struct {
	stopCh     <-chan struct{}
	logger     klog.Logger
	kubeClient kubernetes.Interface
	d          kubeletplugin.DRAPlugin
	fileOps    FileOperations

	cdiDir     string
	driverName string
	nodeName   string
	instances  sets.Set[string]

	mutex          sync.Mutex
	instancesInUse sets.Set[string]
	prepared       map[ClaimID][]string // instance names
	gRPCCalls      []GRPCCall

	blockPrepareResourcesMutex   sync.Mutex
	blockUnprepareResourcesMutex sync.Mutex

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

var _ drapb.NodeServer = &ExamplePlugin{}

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
func StartPlugin(ctx context.Context, cdiDir, driverName string, kubeClient kubernetes.Interface, nodeName string, fileOps FileOperations, opts ...kubeletplugin.Option) (*ExamplePlugin, error) {
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
		kubeClient:     kubeClient,
		fileOps:        fileOps,
		cdiDir:         cdiDir,
		driverName:     driverName,
		nodeName:       nodeName,
		instances:      sets.New[string](),
		instancesInUse: sets.New[string](),
		prepared:       make(map[ClaimID][]string),
	}

	for i := 0; i < ex.fileOps.NumResourceInstances; i++ {
		ex.instances.Insert(fmt.Sprintf("instance-%02d", i))
	}

	opts = append(opts,
		kubeletplugin.DriverName(driverName),
		kubeletplugin.NodeName(nodeName),
		kubeletplugin.KubeClient(kubeClient),
		kubeletplugin.GRPCInterceptor(ex.recordGRPCCall),
		kubeletplugin.GRPCStreamInterceptor(ex.recordGRPCStream),
	)
	d, err := kubeletplugin.Start(ctx, ex, opts...)
	if err != nil {
		return nil, fmt.Errorf("start kubelet plugin: %w", err)
	}
	ex.d = d

	if fileOps.NumResourceInstances >= 0 {
		instances := make([]resourceapi.NamedResourcesInstance, ex.fileOps.NumResourceInstances)
		for i := 0; i < ex.fileOps.NumResourceInstances; i++ {
			instances[i].Name = fmt.Sprintf("instance-%02d", i)
		}
		nodeResources := []*resourceapi.ResourceModel{
			{
				NamedResources: &resourceapi.NamedResourcesResources{
					Instances: instances,
				},
			},
		}
		ex.d.PublishResources(ctx, nodeResources)
	}

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

// BlockNodePrepareResources locks blockPrepareResourcesMutex and returns unlocking function for it
func (ex *ExamplePlugin) BlockNodePrepareResources() func() {
	ex.blockPrepareResourcesMutex.Lock()
	return func() {
		ex.blockPrepareResourcesMutex.Unlock()
	}
}

// BlockNodeUnprepareResources locks blockUnprepareResourcesMutex and returns unlocking function for it
func (ex *ExamplePlugin) BlockNodeUnprepareResources() func() {
	ex.blockUnprepareResourcesMutex.Lock()
	return func() {
		ex.blockUnprepareResourcesMutex.Unlock()
	}
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
func (ex *ExamplePlugin) nodePrepareResource(ctx context.Context, claimReq *drapb.Claim) ([]string, error) {
	logger := klog.FromContext(ctx)

	// The plugin must retrieve the claim itself to get it in the version
	// that it understands.
	var resourceHandle string
	var structuredResourceHandle *resourceapi.StructuredResourceHandle
	claim, err := ex.kubeClient.ResourceV1alpha3().ResourceClaims(claimReq.Namespace).Get(ctx, claimReq.Name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("retrieve claim %s/%s: %w", claimReq.Namespace, claimReq.Name, err)
	}
	if claim.Status.Allocation == nil {
		return nil, fmt.Errorf("claim %s/%s not allocated", claimReq.Namespace, claimReq.Name)
	}
	if claim.UID != types.UID(claimReq.Uid) {
		return nil, fmt.Errorf("claim %s/%s got replaced", claimReq.Namespace, claimReq.Name)
	}
	haveResources := false
	for _, handle := range claim.Status.Allocation.ResourceHandles {
		if handle.DriverName == ex.driverName {
			haveResources = true
			resourceHandle = handle.Data
			structuredResourceHandle = handle.StructuredData
			break
		}
	}
	if !haveResources {
		// Nothing to do.
		return nil, nil
	}

	ex.mutex.Lock()
	defer ex.mutex.Unlock()
	ex.blockPrepareResourcesMutex.Lock()
	defer ex.blockPrepareResourcesMutex.Unlock()

	deviceName := "claim-" + claimReq.Uid
	vendor := ex.driverName
	class := "test"
	dev := vendor + "/" + class + "=" + deviceName
	claimID := ClaimID{Name: claimReq.Name, UID: claimReq.Uid}
	if _, ok := ex.prepared[claimID]; ok {
		// Idempotent call, nothing to do.
		return []string{dev}, nil
	}

	// Determine environment variables.
	var p parameters
	var instanceNames []string
	if structuredResourceHandle == nil {
		// Control plane controller did the allocation.
		if err := json.Unmarshal([]byte(resourceHandle), &p); err != nil {
			return nil, fmt.Errorf("unmarshal resource handle: %w", err)
		}
	} else {
		// Scheduler did the allocation with structured parameters.
		p.NodeName = structuredResourceHandle.NodeName
		if err := extractParameters(structuredResourceHandle.VendorClassParameters, &p.EnvVars, "admin"); err != nil {
			return nil, err
		}
		if err := extractParameters(structuredResourceHandle.VendorClaimParameters, &p.EnvVars, "user"); err != nil {
			return nil, err
		}
		for _, result := range structuredResourceHandle.Results {
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
	filePath := ex.getJSONFilePath(claimReq.Uid)
	buffer, err := json.Marshal(spec)
	if err != nil {
		return nil, fmt.Errorf("marshal spec: %w", err)
	}
	if err := ex.fileOps.Create(filePath, buffer); err != nil {
		return nil, fmt.Errorf("failed to write CDI file %v", err)
	}

	ex.prepared[claimID] = instanceNames
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

func (ex *ExamplePlugin) NodePrepareResources(ctx context.Context, req *drapb.NodePrepareResourcesRequest) (*drapb.NodePrepareResourcesResponse, error) {
	resp := &drapb.NodePrepareResourcesResponse{
		Claims: make(map[string]*drapb.NodePrepareResourceResponse),
	}

	if failure := ex.getPrepareResourcesFailure(); failure != nil {
		return resp, failure
	}

	for _, claimReq := range req.Claims {
		cdiDevices, err := ex.nodePrepareResource(ctx, claimReq)
		if err != nil {
			resp.Claims[claimReq.Uid] = &drapb.NodePrepareResourceResponse{
				Error: err.Error(),
			}
		} else {
			resp.Claims[claimReq.Uid] = &drapb.NodePrepareResourceResponse{
				CDIDevices: cdiDevices,
			}
		}
	}
	return resp, nil
}

// NodeUnprepareResource removes the CDI file created by
// NodePrepareResource. It's idempotent, therefore it is not an error when that
// file is already gone.
func (ex *ExamplePlugin) nodeUnprepareResource(ctx context.Context, claimReq *drapb.Claim) error {
	ex.blockUnprepareResourcesMutex.Lock()
	defer ex.blockUnprepareResourcesMutex.Unlock()

	logger := klog.FromContext(ctx)

	filePath := ex.getJSONFilePath(claimReq.Uid)
	if err := ex.fileOps.Remove(filePath); err != nil {
		return fmt.Errorf("error removing CDI file: %w", err)
	}
	logger.V(3).Info("CDI file removed", "path", filePath)

	ex.mutex.Lock()
	defer ex.mutex.Unlock()

	claimID := ClaimID{Name: claimReq.Name, UID: claimReq.Uid}
	instanceNames, ok := ex.prepared[claimID]
	if !ok {
		// Idempotent call, nothing to do.
		return nil
	}

	delete(ex.prepared, claimID)
	for _, instanceName := range instanceNames {
		ex.instancesInUse.Delete(instanceName)
	}

	return nil
}

func (ex *ExamplePlugin) NodeUnprepareResources(ctx context.Context, req *drapb.NodeUnprepareResourcesRequest) (*drapb.NodeUnprepareResourcesResponse, error) {
	resp := &drapb.NodeUnprepareResourcesResponse{
		Claims: make(map[string]*drapb.NodeUnprepareResourceResponse),
	}

	if failure := ex.getUnprepareResourcesFailure(); failure != nil {
		return resp, failure
	}

	for _, claimReq := range req.Claims {
		err := ex.nodeUnprepareResource(ctx, claimReq)
		if err != nil {
			resp.Claims[claimReq.Uid] = &drapb.NodeUnprepareResourceResponse{
				Error: err.Error(),
			}
		} else {
			resp.Claims[claimReq.Uid] = &drapb.NodeUnprepareResourceResponse{}
		}
	}
	return resp, nil
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

// CountCalls counts GRPC calls with the given method suffix.
func (ex *ExamplePlugin) CountCalls(methodSuffix string) int {
	count := 0
	for _, call := range ex.GetGRPCCalls() {
		if strings.HasSuffix(call.FullMethod, methodSuffix) {
			count += 1
		}
	}
	return count
}
