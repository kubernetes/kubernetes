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
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"google.golang.org/grpc"

	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/klog/v2"
	drapbv1alpha2 "k8s.io/kubelet/pkg/apis/dra/v1alpha2"
	drapbv1alpha3 "k8s.io/kubelet/pkg/apis/dra/v1alpha3"
)

type ExamplePlugin struct {
	logger  klog.Logger
	d       kubeletplugin.DRAPlugin
	fileOps FileOperations

	cdiDir     string
	driverName string
	nodeName   string

	mutex     sync.Mutex
	prepared  map[ClaimID]bool
	gRPCCalls []GRPCCall

	block bool
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

var _ drapbv1alpha2.NodeServer = &ExamplePlugin{}

// getJSONFilePath returns the absolute path where CDI file is/should be.
func (ex *ExamplePlugin) getJSONFilePath(claimUID string) string {
	return filepath.Join(ex.cdiDir, fmt.Sprintf("%s-%s.json", ex.driverName, claimUID))
}

// FileOperations defines optional callbacks for handling CDI files.
type FileOperations struct {
	// Create must overwrite the file.
	Create func(name string, content []byte) error

	// Remove must remove the file. It must not return an error when the
	// file does not exist.
	Remove func(name string) error
}

// StartPlugin sets up the servers that are necessary for a DRA kubelet plugin.
func StartPlugin(logger klog.Logger, cdiDir, driverName string, nodeName string, fileOps FileOperations, opts ...kubeletplugin.Option) (*ExamplePlugin, error) {
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
		logger:     logger,
		fileOps:    fileOps,
		cdiDir:     cdiDir,
		driverName: driverName,
		nodeName:   nodeName,
		prepared:   make(map[ClaimID]bool),
	}

	opts = append(opts,
		kubeletplugin.Logger(logger),
		kubeletplugin.DriverName(driverName),
		kubeletplugin.GRPCInterceptor(ex.recordGRPCCall),
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

// NodePrepareResource ensures that the CDI file for the claim exists. It uses
// a deterministic name to simplify NodeUnprepareResource (no need to remember
// or discover the name) and idempotency (when called again, the file simply
// gets written again).
func (ex *ExamplePlugin) NodePrepareResource(ctx context.Context, req *drapbv1alpha2.NodePrepareResourceRequest) (*drapbv1alpha2.NodePrepareResourceResponse, error) {
	logger := klog.FromContext(ctx)

	// Block to emulate plugin stuckness or slowness.
	// By default the call will not be blocked as ex.block = false.
	if ex.block {
		<-ctx.Done()
		return nil, ctx.Err()
	}

	// Determine environment variables.
	var p parameters
	if err := json.Unmarshal([]byte(req.ResourceHandle), &p); err != nil {
		return nil, fmt.Errorf("unmarshal resource handle: %w", err)
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

	deviceName := "claim-" + req.ClaimUid
	vendor := ex.driverName
	class := "test"
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
	filePath := ex.getJSONFilePath(req.ClaimUid)
	buffer, err := json.Marshal(spec)
	if err != nil {
		return nil, fmt.Errorf("marshal spec: %w", err)
	}
	if err := ex.fileOps.Create(filePath, buffer); err != nil {
		return nil, fmt.Errorf("failed to write CDI file %v", err)
	}

	dev := vendor + "/" + class + "=" + deviceName
	resp := &drapbv1alpha2.NodePrepareResourceResponse{CdiDevices: []string{dev}}

	ex.mutex.Lock()
	defer ex.mutex.Unlock()
	ex.prepared[ClaimID{Name: req.ClaimName, UID: req.ClaimUid}] = true

	logger.V(3).Info("CDI file created", "path", filePath, "device", dev)
	return resp, nil
}

func (ex *ExamplePlugin) NodePrepareResources(ctx context.Context, req *drapbv1alpha3.NodePrepareResourcesRequest) (*drapbv1alpha3.NodePrepareResourcesResponse, error) {
	resp := &drapbv1alpha3.NodePrepareResourcesResponse{
		Claims: make(map[string]*drapbv1alpha3.NodePrepareResourceResponse),
	}
	for _, claimReq := range req.Claims {
		claimResp, err := ex.NodePrepareResource(ctx, &drapbv1alpha2.NodePrepareResourceRequest{
			Namespace:      claimReq.Namespace,
			ClaimName:      claimReq.Name,
			ClaimUid:       claimReq.Uid,
			ResourceHandle: claimReq.ResourceHandle,
		})
		if err != nil {
			resp.Claims[claimReq.Uid] = &drapbv1alpha3.NodePrepareResourceResponse{
				Error: err.Error(),
			}
		} else {
			resp.Claims[claimReq.Uid] = &drapbv1alpha3.NodePrepareResourceResponse{
				CDIDevices: claimResp.CdiDevices,
			}
		}
	}
	return resp, nil
}

// NodeUnprepareResource removes the CDI file created by
// NodePrepareResource. It's idempotent, therefore it is not an error when that
// file is already gone.
func (ex *ExamplePlugin) NodeUnprepareResource(ctx context.Context, req *drapbv1alpha2.NodeUnprepareResourceRequest) (*drapbv1alpha2.NodeUnprepareResourceResponse, error) {
	logger := klog.FromContext(ctx)

	// Block to emulate plugin stuckness or slowness.
	// By default the call will not be blocked as ex.block = false.
	if ex.block {
		<-ctx.Done()
		return nil, ctx.Err()
	}

	filePath := ex.getJSONFilePath(req.ClaimUid)
	if err := ex.fileOps.Remove(filePath); err != nil {
		return nil, fmt.Errorf("error removing CDI file: %w", err)
	}
	logger.V(3).Info("CDI file removed", "path", filePath)

	ex.mutex.Lock()
	defer ex.mutex.Unlock()
	delete(ex.prepared, ClaimID{Name: req.ClaimName, UID: req.ClaimUid})

	return &drapbv1alpha2.NodeUnprepareResourceResponse{}, nil
}

func (ex *ExamplePlugin) NodeUnprepareResources(ctx context.Context, req *drapbv1alpha3.NodeUnprepareResourcesRequest) (*drapbv1alpha3.NodeUnprepareResourcesResponse, error) {
	resp := &drapbv1alpha3.NodeUnprepareResourcesResponse{
		Claims: make(map[string]*drapbv1alpha3.NodeUnprepareResourceResponse),
	}
	for _, claimReq := range req.Claims {
		_, err := ex.NodeUnprepareResource(ctx, &drapbv1alpha2.NodeUnprepareResourceRequest{
			Namespace:      claimReq.Namespace,
			ClaimName:      claimReq.Name,
			ClaimUid:       claimReq.Uid,
			ResourceHandle: claimReq.ResourceHandle,
		})
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
