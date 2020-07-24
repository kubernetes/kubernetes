/*
Copyright 2020 The Kubernetes Authors.

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

package kubesystem

import (
	"fmt"
	"net"
	"strconv"
	"time"

	"k8s.io/kubernetes/test/e2e/framework"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
)

// RestartControllerManager restarts the kube-controller-manager.
func RestartControllerManager() error {
	// TODO: Make it work for all providers and distros.
	if !framework.ProviderIs("gce", "aws") {
		return fmt.Errorf("unsupported provider for RestartControllerManager: %s", framework.TestContext.Provider)
	}
	if framework.ProviderIs("gce") && !framework.MasterOSDistroIs("gci") {
		return fmt.Errorf("unsupported master OS distro: %s", framework.TestContext.MasterOSDistro)
	}
	cmd := "pidof kube-controller-manager | xargs sudo kill"
	framework.Logf("Restarting controller-manager via ssh, running: %v", cmd)
	result, err := e2essh.SSH(cmd, net.JoinHostPort(framework.GetMasterHost(), e2essh.SSHPort), framework.TestContext.Provider)
	if err != nil || result.Code != 0 {
		e2essh.LogResult(result)
		return fmt.Errorf("couldn't restart controller-manager: %v", err)
	}
	return nil
}

// WaitForControllerManagerUp waits for the kube-controller-manager to be up.
func WaitForControllerManagerUp() error {
	cmd := "curl http://localhost:" + strconv.Itoa(framework.InsecureKubeControllerManagerPort) + "/healthz"
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		result, err := e2essh.SSH(cmd, net.JoinHostPort(framework.GetMasterHost(), e2essh.SSHPort), framework.TestContext.Provider)
		if err != nil || result.Code != 0 {
			e2essh.LogResult(result)
		}
		if result.Stdout == "ok" {
			return nil
		}
	}
	return fmt.Errorf("waiting for controller-manager timed out")
}
