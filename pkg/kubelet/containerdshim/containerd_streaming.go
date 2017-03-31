/*
Copyright 2016 The Kubernetes Authors.

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

package containerdshim

import (
	"fmt"
	"time"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// P2
func (cs *containerdService) ExecSync(containerID string, cmd []string, timeout time.Duration) (stdout []byte, stderr []byte, err error) {
	return nil, nil, fmt.Errorf("not implemented")
}

// P3
func (cs *containerdService) Exec(req *runtimeapi.ExecRequest) (*runtimeapi.ExecResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

// P3
func (cs *containerdService) Attach(req *runtimeapi.AttachRequest) (*runtimeapi.AttachResponse, error) {
	return nil, fmt.Errorf("not implemented")
}

// P3
func (cs *containerdService) PortForward(req *runtimeapi.PortForwardRequest) (*runtimeapi.PortForwardResponse, error) {
	return nil, fmt.Errorf("not implemented")
}
