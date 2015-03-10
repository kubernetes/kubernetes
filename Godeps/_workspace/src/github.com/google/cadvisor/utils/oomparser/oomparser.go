// Copyright 2014 Google Inc. All Rights Reserved.
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
	"io"
	"os"
	"path"
	"regexp"
	"strconv"
	"time"

	"github.com/golang/glog"
	"github.com/google/cadvisor/utils"
)

var containerRegexp *regexp.Regexp = regexp.MustCompile(
	`Task in (.*) killed as a result of limit of `)
var lastLineRegexp *regexp.Regexp = regexp.MustCompile(
	`(^[A-Z]{1}[a-z]{2} .*[0-9]{1,2} [0-9]{1,2}:[0-9]{2}:[0-9]{2}) .* Killed process ([0-9]+) \(([0-9A-Za-z_]+)\)`)
var firstLineRegexp *regexp.Regexp = regexp.MustCompile(
	`invoked oom-killer:`)

// struct to hold file from which we obtain OomInstances
type OomParser struct {
	systemFile string
}

// struct that contains information related to an OOM kill instance
type OomInstance struct {
	// process id of the killed process
	Pid int
	// the name of the killed process
	ProcessName string
	// the time that the process was reported to be killed,
	// accurate to the minute
	TimeOfDeath time.Time
	// the absolute name of the container that OOMed
	ContainerName string
}

// gets the container name from a line and adds it to the oomInstance.
func getContainerName(line string, currentOomInstance *OomInstance) error {
	parsedLine := containerRegexp.FindStringSubmatch(line)
	if parsedLine == nil {
		return nil
	}
	currentOomInstance.ContainerName = path.Join("/", parsedLine[1])
	return nil
}

// gets the pid, name, and date from a line and adds it to oomInstance
func getProcessNamePid(line string, currentOomInstance *OomInstance) (bool, error) {
	reList := lastLineRegexp.FindStringSubmatch(line)
	if reList == nil {
		return false, nil
	}
	const longForm = "Jan _2 15:04:05 2006"
	stringYear := strconv.Itoa(time.Now().Year())
	linetime, err := time.Parse(longForm, reList[1]+" "+stringYear)
	if err != nil {
		return false, err
	}
	currentOomInstance.TimeOfDeath = linetime
	if err != nil {
		return false, err
	}
	pid, err := strconv.Atoi(reList[2])
	if err != nil {
		return false, err
	}
	currentOomInstance.Pid = pid
	currentOomInstance.ProcessName = reList[3]
	return true, nil
}

// uses regex to see if line is the start of a kernel oom log
func checkIfStartOfOomMessages(line string) bool {
	potential_oom_start := firstLineRegexp.MatchString(line)
	if potential_oom_start {
		return true
	}
	return false
}

// opens a reader to grab new messages from the Reader object called outPipe
// opened in PopulateOomInformation.  It reads line by line splitting on
// the "\n" character.  Checks if line might be start or end of an oom message
// log. Then the
// lines are checked against a regexp to check for the pid, process name, etc.
// At the end of an oom message group, AnalyzeLines adds the new oomInstance to
// oomLog
func (self *OomParser) analyzeLines(ioreader *bufio.Reader, outStream chan *OomInstance) {
	var line string
	var err error
	for true {
		for line, err = ioreader.ReadString('\n'); err != nil && err == io.EOF; {
			time.Sleep(100 * time.Millisecond)
		}
		in_oom_kernel_log := checkIfStartOfOomMessages(line)
		if in_oom_kernel_log {
			oomCurrentInstance := &OomInstance{
				ContainerName: "/",
			}
			finished := false
			for err == nil && !finished {
				err = getContainerName(line, oomCurrentInstance)
				if err != nil {
					glog.Errorf("%v", err)
				}
				finished, err = getProcessNamePid(line, oomCurrentInstance)
				if err != nil {
					glog.Errorf("%v", err)
				}
				line, err = ioreader.ReadString('\n')
			}
			in_oom_kernel_log = false
			outStream <- oomCurrentInstance
		}
	}
}

// looks for system files that contain kernel messages and if one is found, sets
// the systemFile attribute of the OomParser object
func getSystemFile() (string, error) {
	const varLogMessages = "/var/log/messages"
	const varLogSyslog = "/var/log/syslog"
	if utils.FileExists(varLogMessages) {
		return varLogMessages, nil
	} else if utils.FileExists(varLogSyslog) {
		return varLogSyslog, nil
	}
	return "", fmt.Errorf("neither %s nor %s exists from which to read kernel errors", varLogMessages, varLogSyslog)
}

// calls a go routine that populates self.OomInstances and fills the argument
// channel with OomInstance objects as they are read from the file.
// opens the OomParser's systemFile which was set in getSystemFile
// to look for OOM messages by calling AnalyzeLines.  Takes in the argument
// outStream, which is passed in by the user and passed to AnalyzeLines.
// OomInstance objects are added to outStream when they are found by
// AnalyzeLines
func (self *OomParser) StreamOoms(outStream chan *OomInstance) error {
	file, err := os.Open(self.systemFile)
	if err != nil {
		return err
	}
	ioreader := bufio.NewReader(file)

	// Process the events received from the kernel.
	go func() {
		self.analyzeLines(ioreader, outStream)
	}()
	return nil
}

// initializes an OomParser object and calls getSystemFile to set the systemFile
// attribute.  Returns and OomParser object and an error
func New() (*OomParser, error) {
	systemFileName, err := getSystemFile()
	if err != nil {
		return nil, err
	}
	return &OomParser{
		systemFile: systemFileName}, nil
}
