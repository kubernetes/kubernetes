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

package node

import (
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2etodokubectl "k8s.io/kubernetes/test/e2e/framework/todo/kubectl"
)

// WaitForSSHTunnels waits for establishing SSH tunnel to busybox pod.
func WaitForSSHTunnels(namespace string) {
	e2elog.Logf("Waiting for SSH tunnels to establish")
	e2etodokubectl.RunKubectl(namespace, "run", "ssh-tunnel-test",
		"--image=busybox",
		"--restart=Never",
		"--command", "--",
		"echo", "Hello")
	defer e2etodokubectl.RunKubectl(namespace, "delete", "pod", "ssh-tunnel-test")

	// allow up to a minute for new ssh tunnels to establish
	wait.PollImmediate(5*time.Second, time.Minute, func() (bool, error) {
		_, err := e2etodokubectl.RunKubectl(namespace, "logs", "ssh-tunnel-test")
		return err == nil, nil
	})
}
