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

package framework

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path"
	"strings"
	"sync"
	"time"

	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
)

const (
	// DefaultCPUProfileSeconds is default value for how long the CPU profile is gathered for.
	DefaultCPUProfileSeconds = 30
)

func getProfilesDirectoryPath() string {
	return path.Join(TestContext.ReportDir, "profiles")
}

func createProfilesDirectoryIfNeeded() error {
	profileDirPath := getProfilesDirectoryPath()
	if _, err := os.Stat(profileDirPath); os.IsNotExist(err) {
		if mkdirErr := os.Mkdir(profileDirPath, 0777); mkdirErr != nil {
			return fmt.Errorf("Failed to create profiles dir: %v", mkdirErr)
		}
	} else if err != nil {
		return fmt.Errorf("Failed to check existence of profiles dir: %v", err)
	}
	return nil
}

func checkProfileGatheringPrerequisites() error {
	if !TestContext.AllowGatheringProfiles {
		return fmt.Errorf("Can't gather profiles as --allow-gathering-profiles is false")
	}
	if TestContext.ReportDir == "" {
		return fmt.Errorf("Can't gather profiles as --report-dir is empty")
	}
	if err := createProfilesDirectoryIfNeeded(); err != nil {
		return fmt.Errorf("Failed to ensure profiles dir: %v", err)
	}
	return nil
}

func getPortForComponent(componentName string) (int, error) {
	switch componentName {
	case "kube-apiserver":
		return 8080, nil
	case "kube-scheduler":
		return 10251, nil
	case "kube-controller-manager":
		return 10252, nil
	}
	return -1, fmt.Errorf("Port for component %v unknown", componentName)
}

// Gathers profiles from a master component through SSH. E.g usages:
//   - gatherProfile("kube-apiserver", "someTest", "heap")
//   - gatherProfile("kube-scheduler", "someTest", "profile")
//   - gatherProfile("kube-controller-manager", "someTest", "profile?seconds=20")
//
// We don't export this method but wrappers around it (see below).
func gatherProfile(componentName, profileBaseName, profileKind string) error {
	if err := checkProfileGatheringPrerequisites(); err != nil {
		return fmt.Errorf("Profile gathering pre-requisite failed: %v", err)
	}
	profilePort, err := getPortForComponent(componentName)
	if err != nil {
		return fmt.Errorf("Profile gathering failed finding component port: %v", err)
	}
	if profileBaseName == "" {
		profileBaseName = time.Now().Format(time.RFC3339)
	}

	// Get the profile data over SSH.
	getCommand := fmt.Sprintf("curl -s localhost:%v/debug/pprof/%s", profilePort, profileKind)
	sshResult, err := e2essh.SSH(getCommand, GetMasterHost()+":22", TestContext.Provider)
	if err != nil {
		return fmt.Errorf("Failed to execute curl command on master through SSH: %v", err)
	}

	profilePrefix := componentName
	switch {
	case profileKind == "heap":
		profilePrefix += "_MemoryProfile_"
	case strings.HasPrefix(profileKind, "profile"):
		profilePrefix += "_CPUProfile_"
	default:
		return fmt.Errorf("Unknown profile kind provided: %s", profileKind)
	}

	// Write the profile data to a file.
	rawprofilePath := path.Join(getProfilesDirectoryPath(), profilePrefix+profileBaseName+".pprof")
	rawprofile, err := os.Create(rawprofilePath)
	if err != nil {
		return fmt.Errorf("Failed to create file for the profile graph: %v", err)
	}
	defer rawprofile.Close()

	if _, err := rawprofile.Write([]byte(sshResult.Stdout)); err != nil {
		return fmt.Errorf("Failed to write file with profile data: %v", err)
	}
	if err := rawprofile.Close(); err != nil {
		return fmt.Errorf("Failed to close file: %v", err)
	}
	// Create a graph from the data and write it to a pdf file.
	var cmd *exec.Cmd
	switch {
	// TODO: Support other profile kinds if needed (e.g inuse_space, alloc_objects, mutex, etc)
	case profileKind == "heap":
		cmd = exec.Command("go", "tool", "pprof", "-pdf", "-symbolize=none", "--alloc_space", rawprofile.Name())
	case strings.HasPrefix(profileKind, "profile"):
		cmd = exec.Command("go", "tool", "pprof", "-pdf", "-symbolize=none", rawprofile.Name())
	default:
		return fmt.Errorf("Unknown profile kind provided: %s", profileKind)
	}
	outfilePath := path.Join(getProfilesDirectoryPath(), profilePrefix+profileBaseName+".pdf")
	outfile, err := os.Create(outfilePath)
	if err != nil {
		return fmt.Errorf("Failed to create file for the profile graph: %v", err)
	}
	defer outfile.Close()
	cmd.Stdout = outfile
	stderr := bytes.NewBuffer(nil)
	cmd.Stderr = stderr
	if err := cmd.Run(); nil != err {
		return fmt.Errorf("Failed to run 'go tool pprof': %v, stderr: %#v", err, stderr.String())
	}
	return nil
}

// The below exposed functions can take a while to execute as they SSH to the master,
// collect and copy the profile over and then graph it. To allow waiting for these to
// finish before the parent goroutine itself finishes, we accept a sync.WaitGroup
// argument in these functions. Typically you would use the following pattern:
//
// func TestFoo() {
//		var wg sync.WaitGroup
//		wg.Add(3)
//		go framework.GatherCPUProfile("kube-apiserver", "before_foo", &wg)
//		go framework.GatherMemoryProfile("kube-apiserver", "before_foo", &wg)
//		<<<< some code doing foo >>>>>>
//		go framework.GatherCPUProfile("kube-scheduler", "after_foo", &wg)
//		wg.Wait()
// }
//
// If you do not wish to exercise the waiting logic, pass a nil value for the
// waitgroup argument instead. However, then you would be responsible for ensuring
// that the function finishes. There's also a polling-based gatherer utility for
// CPU profiles available below.

// GatherCPUProfile gathers CPU profile.
func GatherCPUProfile(componentName string, profileBaseName string, wg *sync.WaitGroup) {
	GatherCPUProfileForSeconds(componentName, profileBaseName, DefaultCPUProfileSeconds, wg)
}

// GatherCPUProfileForSeconds gathers CPU profile for specified seconds.
func GatherCPUProfileForSeconds(componentName string, profileBaseName string, seconds int, wg *sync.WaitGroup) {
	if wg != nil {
		defer wg.Done()
	}
	if err := gatherProfile(componentName, profileBaseName, fmt.Sprintf("profile?seconds=%v", seconds)); err != nil {
		e2elog.Logf("Failed to gather %v CPU profile: %v", componentName, err)
	}
}

// GatherMemoryProfile gathers memory profile.
func GatherMemoryProfile(componentName string, profileBaseName string, wg *sync.WaitGroup) {
	if wg != nil {
		defer wg.Done()
	}
	if err := gatherProfile(componentName, profileBaseName, "heap"); err != nil {
		e2elog.Logf("Failed to gather %v memory profile: %v", componentName, err)
	}
}

// StartCPUProfileGatherer performs polling-based gathering of the component's CPU
// profile. It takes the interval b/w consecutive gatherings as an argument and
// starts the gathering goroutine. To stop the gatherer, close the returned channel.
func StartCPUProfileGatherer(componentName string, profileBaseName string, interval time.Duration) chan struct{} {
	stopCh := make(chan struct{})
	go func() {
		for {
			select {
			case <-time.After(interval):
				GatherCPUProfile(componentName, profileBaseName+"_"+time.Now().Format(time.RFC3339), nil)
			case <-stopCh:
				return
			}
		}
	}()
	return stopCh
}
