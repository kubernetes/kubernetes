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

	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/klog/v2"
	drapbv1 "k8s.io/kubelet/pkg/apis/dra/v1alpha1"
)

type ExamplePlugin struct {
	logger  klog.Logger
	d       kubeletplugin.DRAPlugin
	fileOps FileOperations

	cdiDir     string
	driverName string
	nodeName   string

	mutex    sync.Mutex
	prepared map[ClaimID]bool
}

// ClaimID contains both claim name and UID to simplify debugging. The
// namespace is not included because it is random in E2E tests and the UID is
// sufficient to make the ClaimID unique.
type ClaimID struct {
	Name string
	UID  string
}

var _ drapbv1.NodeServer = &ExamplePlugin{}

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
	)
	d, err := kubeletplugin.Start(ex, opts...)
	if err != nil {
		return nil, fmt.Errorf("start kubelet plugin: %v", err)
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

// NodePrepareResource ensures that the CDI file for the claim exists. It uses
// a deterministic name to simplify NodeUnprepareResource (no need to remember
// or discover the name) and idempotency (when called again, the file simply
// gets written again).
func (ex *ExamplePlugin) NodePrepareResource(ctx context.Context, req *drapbv1.NodePrepareResourceRequest) (*drapbv1.NodePrepareResourceResponse, error) {
	logger := klog.FromContext(ctx)

	// Determine environment variables.
	var p parameters
	if err := json.Unmarshal([]byte(req.ResourceHandle), &p); err != nil {
		return nil, fmt.Errorf("unmarshal resource handle: %v", err)
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
		Version: "0.2.0", // This has to be a version accepted by the runtimes.
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
		return nil, fmt.Errorf("marshal spec: %v", err)
	}
	if err := ex.fileOps.Create(filePath, buffer); err != nil {
		return nil, fmt.Errorf("failed to write CDI file %v", err)
	}

	dev := vendor + "/" + class + "=" + deviceName
	resp := &drapbv1.NodePrepareResourceResponse{CdiDevices: []string{dev}}

	ex.mutex.Lock()
	defer ex.mutex.Unlock()
	ex.prepared[ClaimID{Name: req.ClaimName, UID: req.ClaimUid}] = true

	logger.V(3).Info("CDI file created", "path", filePath, "device", dev)
	return resp, nil
}

// NodeUnprepareResource removes the CDI file created by
// NodePrepareResource. It's idempotent, therefore it is not an error when that
// file is already gone.
func (ex *ExamplePlugin) NodeUnprepareResource(ctx context.Context, req *drapbv1.NodeUnprepareResourceRequest) (*drapbv1.NodeUnprepareResourceResponse, error) {
	logger := klog.FromContext(ctx)

	filePath := ex.getJSONFilePath(req.ClaimUid)
	if err := ex.fileOps.Remove(filePath); err != nil {
		return nil, fmt.Errorf("error removing CDI file: %v", err)
	}
	logger.V(3).Info("CDI file removed", "path", filePath)

	ex.mutex.Lock()
	defer ex.mutex.Unlock()
	delete(ex.prepared, ClaimID{Name: req.ClaimName, UID: req.ClaimUid})

	return &drapbv1.NodeUnprepareResourceResponse{}, nil
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
