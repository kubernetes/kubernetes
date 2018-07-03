/*
Copyright 2018 The Kubernetes Authors.

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

package pluginwatcher

import (
	"errors"
	"fmt"
	"reflect"
	"sync"
	"time"

	"golang.org/x/net/context"

	v1beta1 "k8s.io/kubernetes/pkg/kubelet/util/pluginwatcher/example_plugin_apis/v1beta1"
	v1beta2 "k8s.io/kubernetes/pkg/kubelet/util/pluginwatcher/example_plugin_apis/v1beta2"
)

type exampleHandler struct {
	registeredPlugins       map[string]struct{}
	mutex                   sync.Mutex
	chanForHandlerAckErrors chan error // for testing
}

// NewExampleHandler provide a example handler
func NewExampleHandler() *exampleHandler {
	return &exampleHandler{
		chanForHandlerAckErrors: make(chan error),
		registeredPlugins:       make(map[string]struct{}),
	}
}

func (h *exampleHandler) Cleanup() error {
	h.mutex.Lock()
	defer h.mutex.Unlock()
	h.registeredPlugins = make(map[string]struct{})
	return nil
}

func (h *exampleHandler) Handler(pluginName string, endpoint string, versions []string, sockPath string) (chan bool, error) {

	// check for supported versions
	if !reflect.DeepEqual([]string{"v1beta1", "v1beta2"}, versions) {
		return nil, fmt.Errorf("not the supported versions: %s", versions)
	}

	// this handler expects non-empty endpoint as an example
	if len(endpoint) == 0 {
		return nil, errors.New("expecting non empty endpoint")
	}

	_, conn, err := dial(sockPath)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	// The plugin handler should be able to use any listed service API version.
	v1beta1Client := v1beta1.NewExampleClient(conn)
	v1beta2Client := v1beta2.NewExampleClient(conn)

	// Tests v1beta1 GetExampleInfo
	if _, err = v1beta1Client.GetExampleInfo(context.Background(), &v1beta1.ExampleRequest{}); err != nil {
		return nil, err
	}

	// Tests v1beta2 GetExampleInfo
	if _, err = v1beta2Client.GetExampleInfo(context.Background(), &v1beta2.ExampleRequest{}); err != nil {
		return nil, err
	}

	// handle registered plugin
	h.mutex.Lock()
	if _, exist := h.registeredPlugins[pluginName]; exist {
		h.mutex.Unlock()
		return nil, fmt.Errorf("plugin %s already registered", pluginName)
	}
	h.registeredPlugins[pluginName] = struct{}{}
	h.mutex.Unlock()

	chanForAckOfNotification := make(chan bool)
	go func() {
		select {
		case <-chanForAckOfNotification:
			// TODO: handle the negative scenario
			close(chanForAckOfNotification)
		case <-time.After(time.Second):
			h.chanForHandlerAckErrors <- errors.New("Timed out while waiting for notification ack")
		}
	}()
	return chanForAckOfNotification, nil
}
