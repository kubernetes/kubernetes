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

package operationexecutor

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	numPluginsToRegister   = 2
	numPluginsToUnregister = 2
)

var _ OperationGenerator = &fakeOperationGenerator{}
var socketDir string

func init() {
	d, err := os.MkdirTemp("", "operation_executor_test")
	if err != nil {
		panic(fmt.Sprintf("Could not create a temp directory: %s", d))
	}
	socketDir = d
}

func TestOperationExecutor_RegisterPlugin_ConcurrentRegisterPlugin(t *testing.T) {
	tCtx, ch, quit, oe := setup(t)
	for i := 0; i < numPluginsToRegister; i++ {
		socketPath := fmt.Sprintf("%s/plugin-%d.sock", socketDir, i)
		err := oe.RegisterPlugin(tCtx, socketPath, uuid.NewUUID(), nil /* plugin handlers */, nil /* actual state of the world updator */)
		assert.NoError(t, err)
	}
	if !isOperationRunConcurrently(ch, quit, numPluginsToRegister) {
		t.Fatalf("Unable to start register operations in Concurrent for plugins")
	}
}

func TestOperationExecutor_RegisterPlugin_SerialRegisterPlugin(t *testing.T) {
	tCtx, ch, quit, oe := setup(t)
	socketPath := fmt.Sprintf("%s/plugin-serial.sock", socketDir)

	// First registration should not fail.
	err := oe.RegisterPlugin(tCtx, socketPath, uuid.NewUUID(), nil /* plugin handlers */, nil /* actual state of the world updator */)
	assert.NoError(t, err)

	for i := 1; i < numPluginsToRegister; i++ {
		err := oe.RegisterPlugin(tCtx, socketPath, uuid.NewUUID(), nil /* plugin handlers */, nil /* actual state of the world updator */)
		if err == nil {
			t.Fatalf("RegisterPlugin did not fail. Expected: <Failed to create operation with name \"%s\". An operation with that name is already executing.> Actual: <no error>", socketPath)
		}

	}
	if !isOperationRunSerially(ch, quit) {
		t.Fatalf("Unable to start register operations serially for plugins")
	}
}

func TestOperationExecutor_UnregisterPlugin_ConcurrentUnregisterPlugin(t *testing.T) {
	tCtx, ch, quit, oe := setup(t)
	for i := 0; i < numPluginsToUnregister; i++ {
		socketPath := "socket-path" + strconv.Itoa(i)
		pluginInfo := cache.PluginInfo{SocketPath: socketPath}
		_ = oe.UnregisterPlugin(tCtx, pluginInfo, nil /* actual state of the world updator */)
	}
	if !isOperationRunConcurrently(ch, quit, numPluginsToUnregister) {
		t.Fatalf("Unable to start unregister operations in Concurrent for plugins")
	}
}

func TestOperationExecutor_UnregisterPlugin_SerialUnregisterPlugin(t *testing.T) {
	tCtx, ch, quit, oe := setup(t)
	socketPath := fmt.Sprintf("%s/plugin-serial.sock", socketDir)
	for i := 0; i < numPluginsToUnregister; i++ {
		pluginInfo := cache.PluginInfo{SocketPath: socketPath}
		_ = oe.UnregisterPlugin(tCtx, pluginInfo, nil /* actual state of the world updator */)
	}
	if !isOperationRunSerially(ch, quit) {
		t.Fatalf("Unable to start unregister operations serially for plugins")
	}
}

type fakeOperationGenerator struct {
	ch   chan interface{}
	quit chan interface{}
}

func newFakeOperationGenerator(ch chan interface{}, quit chan interface{}) OperationGenerator {
	return &fakeOperationGenerator{
		ch:   ch,
		quit: quit,
	}
}

func (fopg *fakeOperationGenerator) GenerateRegisterPluginFunc(
	ctx context.Context,
	socketPath string,
	pluginUUID types.UID,
	pluginHandlers map[string]cache.PluginHandler,
	actualStateOfWorldUpdater ActualStateOfWorldUpdater) func() error {

	opFunc := func() error {
		startOperationAndBlock(fopg.ch, fopg.quit)
		return nil
	}
	return opFunc
}

func (fopg *fakeOperationGenerator) GenerateUnregisterPluginFunc(
	ctx context.Context,
	pluginInfo cache.PluginInfo,
	actualStateOfWorldUpdater ActualStateOfWorldUpdater) func() error {
	opFunc := func() error {
		startOperationAndBlock(fopg.ch, fopg.quit)
		return nil
	}
	return opFunc
}

func isOperationRunSerially(ch <-chan interface{}, quit chan<- interface{}) bool {
	defer close(quit)
	numOperationsStarted := 0
loop:
	for {
		select {
		case <-ch:
			numOperationsStarted++
			if numOperationsStarted > 1 {
				return false
			}
		case <-time.After(5 * time.Second):
			break loop
		}
	}
	return true
}

func isOperationRunConcurrently(ch <-chan interface{}, quit chan<- interface{}, numOperationsToRun int) bool {
	defer close(quit)
	numOperationsStarted := 0
loop:
	for {
		select {
		case <-ch:
			numOperationsStarted++
			if numOperationsStarted == numOperationsToRun {
				return true
			}
		case <-time.After(5 * time.Second):
			break loop
		}
	}
	return false
}

func setup(t *testing.T) (context.Context, chan interface{}, chan interface{}, OperationExecutor) {
	tCtx := ktesting.Init(t)
	// Use buffered channel to prevent goroutine leak.
	// If time.After fires before ch receive, the goroutine sending to ch would block forever.
	// Buffer size accommodates maximum concurrent operations.
	ch, quit := make(chan interface{}, numPluginsToRegister*2), make(chan interface{})
	return tCtx, ch, quit, NewOperationExecutor(newFakeOperationGenerator(ch, quit))
}

// This function starts by writing to ch and blocks on the quit channel
// until it is closed by the currently running test
func startOperationAndBlock(ch chan<- interface{}, quit <-chan interface{}) {
	ch <- nil
	<-quit
}
