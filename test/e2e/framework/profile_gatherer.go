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
)

const (
	// Default value for how long the CPU profile is gathered for.
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

func gatherProfileOfKind(profileBaseName, kind string) error {
	// Get the profile data over SSH.
	getCommand := fmt.Sprintf("curl -s localhost:8080/debug/pprof/%s", kind)
	sshResult, err := SSH(getCommand, GetMasterHost()+":22", TestContext.Provider)
	if err != nil {
		return fmt.Errorf("Failed to execute curl command on master through SSH: %v", err)
	}

	var profilePrefix string
	switch {
	case kind == "heap":
		profilePrefix = "ApiserverMemoryProfile_"
	case strings.HasPrefix(kind, "profile"):
		profilePrefix = "ApiserverCPUProfile_"
	default:
		return fmt.Errorf("Unknown profile kind provided: %s", kind)
	}

	// Write the data to a file.
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
	case kind == "heap":
		cmd = exec.Command("go", "tool", "pprof", "-pdf", "-symbolize=none", "--alloc_space", rawprofile.Name())
	case strings.HasPrefix(kind, "profile"):
		cmd = exec.Command("go", "tool", "pprof", "-pdf", "-symbolize=none", rawprofile.Name())
	default:
		return fmt.Errorf("Unknown profile kind provided: %s", kind)
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
// func TestFooBar() {
//		var wg sync.WaitGroup
//		wg.Add(3)
//		go framework.GatherApiserverCPUProfile(&wg, "doing_foo")
//		go framework.GatherApiserverMemoryProfile(&wg, "doing_foo")
//		<<<< some code doing foo >>>>>>
//		go framework.GatherApiserverCPUProfile(&wg, "doing_bar")
//		<<<< some code doing bar >>>>>>
//		wg.Wait()
// }
//
// If you do not wish to exercise the waiting logic, pass a nil value for the
// waitgroup argument instead. However, then you would be responsible for ensuring
// that the function finishes.

func GatherApiserverCPUProfile(wg *sync.WaitGroup, profileBaseName string) {
	GatherApiserverCPUProfileForNSeconds(wg, profileBaseName, DefaultCPUProfileSeconds)
}

func GatherApiserverCPUProfileForNSeconds(wg *sync.WaitGroup, profileBaseName string, n int) {
	if wg != nil {
		defer wg.Done()
	}
	if err := checkProfileGatheringPrerequisites(); err != nil {
		Logf("Profile gathering pre-requisite failed: %v", err)
		return
	}
	if profileBaseName == "" {
		profileBaseName = time.Now().Format(time.RFC3339)
	}
	if err := gatherProfileOfKind(profileBaseName, fmt.Sprintf("profile?seconds=%v", n)); err != nil {
		Logf("Failed to gather apiserver CPU profile: %v", err)
	}
}

func GatherApiserverMemoryProfile(wg *sync.WaitGroup, profileBaseName string) {
	if wg != nil {
		defer wg.Done()
	}
	if err := checkProfileGatheringPrerequisites(); err != nil {
		Logf("Profile gathering pre-requisite failed: %v", err)
		return
	}
	if profileBaseName == "" {
		profileBaseName = time.Now().Format(time.RFC3339)
	}
	if err := gatherProfileOfKind(profileBaseName, "heap"); err != nil {
		Logf("Failed to gather apiserver memory profile: %v", err)
	}
}

// StartApiserverCPUProfileGatherer is a polling-based gatherer of the apiserver's
// CPU profile. It takes the delay b/w consecutive gatherings as an argument and
// starts the gathering goroutine. To stop the gatherer, close the returned channel.
func StartApiserverCPUProfileGatherer(delay time.Duration) chan struct{} {
	stopCh := make(chan struct{})
	go func() {
		for {
			select {
			case <-time.After(delay):
				GatherApiserverCPUProfile(nil, "")
			case <-stopCh:
				return
			}
		}
	}()
	return stopCh
}
