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

package remote

import (
	"fmt"
	"os"
	"os/exec"
	"time"

	"k8s.io/klog"
	"k8s.io/kubernetes/test/utils"
)

// CAdvisorE2ERemote contains the specific functions in the cadvisor e2e test suite.
type CAdvisorE2ERemote struct{}

// InitCAdvisorE2ERemote performs initialization for cadvisor remote testing
func InitCAdvisorE2ERemote() TestSuite {
	return &CAdvisorE2ERemote{}
}

// SetupTestPackage implements TestSuite.SetupTestPackage
func (n *CAdvisorE2ERemote) SetupTestPackage(tardir, systemSpecName string) error {
	cadvisorRootDir, err := utils.GetCAdvisorRootDir()
	if err != nil {
		return err
	}
	// build the cadvisor binary and tests
	if err := runCommand(fmt.Sprintf("%s/build/prow_e2e.sh", cadvisorRootDir)); err != nil {
		return err
	}
	// transfer the entire directory to each node
	if err := runCommand("cp", "-R", cadvisorRootDir, fmt.Sprintf("%s/", tardir)); err != nil {
		return err
	}
	return nil
}

func runCommand(command string, args ...string) error {
	cmd := exec.Command(command, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		return fmt.Errorf("failed to run command %s. error: %v", command, err)
	}
	return nil
}

// RunTest implements TestSuite.RunTest
func (n *CAdvisorE2ERemote) RunTest(host, workspace, results, imageDesc, junitFilePrefix, testArgs, ginkgoArgs, systemSpecName, extraEnvs string, timeout time.Duration) (string, error) {
	// Kill any running node processes
	cleanupNodeProcesses(host)

	klog.V(2).Infof("Starting tests on %q", host)
	return SSH(host, "sh", "-c", getSSHCommand(" && ",
		fmt.Sprintf("cd %s/cadvisor", workspace),
		fmt.Sprintf("timeout -k 30s %fs ./build/integration.sh ../results/cadvisor.log",
			timeout.Seconds()),
	))
}
