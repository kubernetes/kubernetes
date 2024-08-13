/*
Copyright 2019 The Kubernetes Authors.

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

package benchmark

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"

	resourceapi "k8s.io/api/resource/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	draapp "k8s.io/kubernetes/test/e2e/dra/test-driver/app"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// createResourceClaimsOp defines an op where resource claims are created.
type createResourceClaimsOp struct {
	// Must be createResourceClaimsOpcode.
	Opcode operationCode
	// Number of claims to create. Parameterizable through CountParam.
	Count int
	// Template parameter for Count.
	CountParam string
	// Namespace the claims should be created in.
	Namespace string
	// Path to spec file describing the claims to create.
	TemplatePath string
}

var _ realOp = &createResourceClaimsOp{}
var _ runnableOp = &createResourceClaimsOp{}

func (op *createResourceClaimsOp) isValid(allowParameterization bool) error {
	if op.Opcode != createResourceClaimsOpcode {
		return fmt.Errorf("invalid opcode %q; expected %q", op.Opcode, createResourceClaimsOpcode)
	}
	if !isValidCount(allowParameterization, op.Count, op.CountParam) {
		return fmt.Errorf("invalid Count=%d / CountParam=%q", op.Count, op.CountParam)
	}
	if op.Namespace == "" {
		return fmt.Errorf("Namespace must be set")
	}
	if op.TemplatePath == "" {
		return fmt.Errorf("TemplatePath must be set")
	}
	return nil
}

func (op *createResourceClaimsOp) collectsMetrics() bool {
	return false
}
func (op *createResourceClaimsOp) patchParams(w *workload) (realOp, error) {
	if op.CountParam != "" {
		var err error
		op.Count, err = w.Params.get(op.CountParam[1:])
		if err != nil {
			return nil, err
		}
	}
	return op, op.isValid(false)
}

func (op *createResourceClaimsOp) requiredNamespaces() []string {
	return []string{op.Namespace}
}

func (op *createResourceClaimsOp) run(tCtx ktesting.TContext) {
	tCtx.Logf("creating %d claims in namespace %q", op.Count, op.Namespace)

	var claimTemplate *resourceapi.ResourceClaim
	if err := getSpecFromFile(&op.TemplatePath, &claimTemplate); err != nil {
		tCtx.Fatalf("parsing ResourceClaim %q: %v", op.TemplatePath, err)
	}
	var createErr error
	var mutex sync.Mutex
	create := func(i int) {
		err := func() error {
			if _, err := tCtx.Client().ResourceV1alpha3().ResourceClaims(op.Namespace).Create(tCtx, claimTemplate.DeepCopy(), metav1.CreateOptions{}); err != nil {
				return fmt.Errorf("create claim: %v", err)
			}
			return nil
		}()
		if err != nil {
			mutex.Lock()
			defer mutex.Unlock()
			createErr = err
		}
	}

	workers := op.Count
	if workers > 30 {
		workers = 30
	}
	workqueue.ParallelizeUntil(tCtx, workers, op.Count, create)
	if createErr != nil {
		tCtx.Fatal(createErr.Error())
	}
}

// createResourceDriverOp defines an op where resource claims are created.
type createResourceDriverOp struct {
	// Must be createResourceDriverOpcode.
	Opcode operationCode
	// Name of the driver, used to reference it in a resource class.
	DriverName string
	// Number of claims to allow per node. Parameterizable through MaxClaimsPerNodeParam.
	MaxClaimsPerNode int
	// Template parameter for MaxClaimsPerNode.
	MaxClaimsPerNodeParam string
	// Nodes matching this glob pattern have resources managed by the driver.
	Nodes string
	// StructuredParameters is true if the controller that is built into the scheduler
	// is used and the control-plane controller is not needed.
	// Because we don't run the kubelet plugin, ResourceSlices must
	// get created for all nodes.
	StructuredParameters bool
}

var _ realOp = &createResourceDriverOp{}
var _ runnableOp = &createResourceDriverOp{}

func (op *createResourceDriverOp) isValid(allowParameterization bool) error {
	if op.Opcode != createResourceDriverOpcode {
		return fmt.Errorf("invalid opcode %q; expected %q", op.Opcode, createResourceDriverOpcode)
	}
	if !isValidCount(allowParameterization, op.MaxClaimsPerNode, op.MaxClaimsPerNodeParam) {
		return fmt.Errorf("invalid MaxClaimsPerNode=%d / MaxClaimsPerNodeParam=%q", op.MaxClaimsPerNode, op.MaxClaimsPerNodeParam)
	}
	if op.DriverName == "" {
		return fmt.Errorf("DriverName must be set")
	}
	if op.Nodes == "" {
		return fmt.Errorf("Nodes must be set")
	}
	return nil
}

func (op *createResourceDriverOp) collectsMetrics() bool {
	return false
}
func (op *createResourceDriverOp) patchParams(w *workload) (realOp, error) {
	if op.MaxClaimsPerNodeParam != "" {
		var err error
		op.MaxClaimsPerNode, err = w.Params.get(op.MaxClaimsPerNodeParam[1:])
		if err != nil {
			return nil, err
		}
	}
	return op, op.isValid(false)
}

func (op *createResourceDriverOp) requiredNamespaces() []string { return nil }

func (op *createResourceDriverOp) run(tCtx ktesting.TContext) {
	tCtx.Logf("creating resource driver %q for nodes matching %q", op.DriverName, op.Nodes)

	// Start the controller side of the DRA test driver such that it simulates
	// per-node resources.
	resources := draapp.Resources{
		DriverName:     op.DriverName,
		NodeLocal:      true,
		MaxAllocations: op.MaxClaimsPerNode,
	}

	nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
	if err != nil {
		tCtx.Fatalf("list nodes: %v", err)
	}
	for _, node := range nodes.Items {
		match, err := filepath.Match(op.Nodes, node.Name)
		if err != nil {
			tCtx.Fatalf("matching glob pattern %q against node name %q: %v", op.Nodes, node.Name, err)
		}
		if match {
			resources.Nodes = append(resources.Nodes, node.Name)
		}
	}

	if op.StructuredParameters {
		for _, nodeName := range resources.Nodes {
			slice := resourceSlice(op.DriverName, nodeName, op.MaxClaimsPerNode)
			_, err := tCtx.Client().ResourceV1alpha3().ResourceSlices().Create(tCtx, slice, metav1.CreateOptions{})
			tCtx.ExpectNoError(err, "create node resource slice")
		}
		tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
			err := tCtx.Client().ResourceV1alpha3().ResourceSlices().DeleteCollection(tCtx,
				metav1.DeleteOptions{},
				metav1.ListOptions{FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + op.DriverName},
			)
			tCtx.ExpectNoError(err, "delete node resource slices")
		})
		// No need for the controller.
		return
	}

	controller := draapp.NewController(tCtx.Client(), resources)
	ctx, cancel := context.WithCancel(tCtx)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		ctx := klog.NewContext(ctx, klog.LoggerWithName(klog.FromContext(ctx), op.DriverName))
		controller.Run(ctx, 5 /* workers */)
	}()
	tCtx.Cleanup(func() {
		tCtx.Logf("stopping resource driver %q", op.DriverName)
		// We must cancel before waiting.
		cancel()
		wg.Wait()
		tCtx.Logf("stopped resource driver %q", op.DriverName)
	})
}

func resourceSlice(driverName, nodeName string, capacity int) *resourceapi.ResourceSlice {
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
		},

		Spec: resourceapi.ResourceSliceSpec{
			Driver:   driverName,
			NodeName: nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               nodeName,
				ResourceSliceCount: 1,
			},
		},
	}

	for i := 0; i < capacity; i++ {
		slice.Spec.Devices = append(slice.Spec.Devices,
			resourceapi.Device{
				Name:  fmt.Sprintf("instance-%d", i),
				Basic: &resourceapi.BasicDevice{},
			},
		)
	}

	return slice
}
