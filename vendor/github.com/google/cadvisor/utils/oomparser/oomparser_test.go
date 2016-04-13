// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package oomparser

import (
	"bufio"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"testing"
	"time"
)

const startLine = "Jan 21 22:01:49 localhost kernel: [62278.816267] ruby invoked oom-killer: gfp_mask=0x201da, order=0, oom_score_adj=0"
const endLine = "Jan 21 22:01:49 localhost kernel: [62279.421192] Killed process 19667 (evilprogram2) total-vm:1460016kB, anon-rss:1414008kB, file-rss:4kB"
const containerLine = "Jan 26 14:10:07 kateknister0.mtv.corp.google.com kernel: [1814368.465205] Task in /mem2 killed as a result of limit of /mem3"
const containerLogFile = "containerOomExampleLog.txt"
const systemLogFile = "systemOomExampleLog.txt"

func createExpectedContainerOomInstance(t *testing.T) *OomInstance {
	const longForm = "Jan _2 15:04:05 2006"
	deathTime, err := time.ParseInLocation(longForm, "Jan  5 15:19:27 2015", time.Local)
	if err != nil {
		t.Fatalf("could not parse expected time when creating expected container oom instance. Had error %v", err)
		return nil
	}
	return &OomInstance{
		Pid:                 13536,
		ProcessName:         "memorymonster",
		TimeOfDeath:         deathTime,
		ContainerName:       "/mem2",
		VictimContainerName: "/mem3",
	}
}

func createExpectedSystemOomInstance(t *testing.T) *OomInstance {
	const longForm = "Jan _2 15:04:05 2006"
	deathTime, err := time.ParseInLocation(longForm, "Jan 28 19:58:45 2015", time.Local)
	if err != nil {
		t.Fatalf("could not parse expected time when creating expected system oom instance. Had error %v", err)
		return nil
	}
	return &OomInstance{
		Pid:                 1532,
		ProcessName:         "badsysprogram",
		TimeOfDeath:         deathTime,
		ContainerName:       "/",
		VictimContainerName: "/",
	}
}

func TestGetContainerName(t *testing.T) {
	currentOomInstance := new(OomInstance)
	err := getContainerName(startLine, currentOomInstance)
	if err != nil {
		t.Errorf("bad line fed to getContainerName should yield no error, but had error %v", err)
	}
	if currentOomInstance.ContainerName != "" {
		t.Errorf("bad line fed to getContainerName yielded no container name but set it to %s", currentOomInstance.ContainerName)
	}
	err = getContainerName(containerLine, currentOomInstance)
	if err != nil {
		t.Errorf("container line fed to getContainerName should yield no error, but had error %v", err)
	}
	if currentOomInstance.ContainerName != "/mem2" {
		t.Errorf("getContainerName should have set containerName to /mem2, not %s", currentOomInstance.ContainerName)
	}
	if currentOomInstance.VictimContainerName != "/mem3" {
		t.Errorf("getContainerName should have set victimContainerName to /mem3, not %s", currentOomInstance.VictimContainerName)
	}
}

func TestGetProcessNamePid(t *testing.T) {
	currentOomInstance := new(OomInstance)
	couldParseLine, err := getProcessNamePid(startLine, currentOomInstance)
	if err != nil {
		t.Errorf("bad line fed to getProcessNamePid should yield no error, but had error %v", err)
	}
	if couldParseLine {
		t.Errorf("bad line fed to getProcessNamePid should return false but returned %v", couldParseLine)
	}

	const longForm = "Jan _2 15:04:05 2006"
	stringYear := strconv.Itoa(time.Now().Year())
	correctTime, err := time.ParseInLocation(longForm, fmt.Sprintf("Jan 21 22:01:49 %s", stringYear), time.Local)
	couldParseLine, err = getProcessNamePid(endLine, currentOomInstance)
	if err != nil {
		t.Errorf("good line fed to getProcessNamePid should yield no error, but had error %v", err)
	}
	if !couldParseLine {
		t.Errorf("good line fed to getProcessNamePid should return true but returned %v", couldParseLine)
	}
	if currentOomInstance.ProcessName != "evilprogram2" {
		t.Errorf("getProcessNamePid should have set processName to evilprogram2, not %s", currentOomInstance.ProcessName)
	}
	if currentOomInstance.Pid != 19667 {
		t.Errorf("getProcessNamePid should have set PID to 19667, not %d", currentOomInstance.Pid)
	}
	if !correctTime.Equal(currentOomInstance.TimeOfDeath) {
		t.Errorf("getProcessNamePid should have set date to %v, not %v", correctTime, currentOomInstance.TimeOfDeath)
	}
}

func TestCheckIfStartOfMessages(t *testing.T) {
	couldParseLine := checkIfStartOfOomMessages(endLine)
	if couldParseLine {
		t.Errorf("bad line fed to checkIfStartOfMessages should return false but returned %v", couldParseLine)
	}
	couldParseLine = checkIfStartOfOomMessages(startLine)
	if !couldParseLine {
		t.Errorf("start line fed to checkIfStartOfMessages should return true but returned %v", couldParseLine)
	}
}

func TestStreamOomsContainer(t *testing.T) {
	expectedContainerOomInstance := createExpectedContainerOomInstance(t)
	helpTestStreamOoms(expectedContainerOomInstance, containerLogFile, t)
}

func TestStreamOomsSystem(t *testing.T) {
	expectedSystemOomInstance := createExpectedSystemOomInstance(t)
	helpTestStreamOoms(expectedSystemOomInstance, systemLogFile, t)
}

func helpTestStreamOoms(oomCheckInstance *OomInstance, sysFile string, t *testing.T) {
	outStream := make(chan *OomInstance)
	oomLog := mockOomParser(sysFile, t)
	timeout := make(chan bool, 1)
	go func() {
		time.Sleep(1 * time.Second)
		timeout <- true
	}()

	go oomLog.StreamOoms(outStream)

	select {
	case oomInstance := <-outStream:
		if reflect.DeepEqual(*oomCheckInstance, *oomInstance) {
			t.Errorf("wrong instance returned. Expected %v and got %v",
				oomCheckInstance, oomInstance)
		}
	case <-timeout:
		t.Error(
			"timeout happened before oomInstance was found in test file")
	}
}

func mockOomParser(sysFile string, t *testing.T) *OomParser {
	file, err := os.Open(sysFile)
	if err != nil {
		t.Errorf("had an error opening file: %v", err)
	}
	return &OomParser{
		ioreader: bufio.NewReader(file),
	}
}
