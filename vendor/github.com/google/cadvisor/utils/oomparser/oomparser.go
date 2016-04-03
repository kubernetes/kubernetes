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
	"os/exec"
	"path"
	"regexp"
	"strconv"
	"time"

	"github.com/google/cadvisor/utils"

	"github.com/golang/glog"
)

var (
	containerRegexp = regexp.MustCompile(`Task in (.*) killed as a result of limit of (.*)`)
	lastLineRegexp  = regexp.MustCompile(`(^[A-Z][a-z]{2} .*[0-9]{1,2} [0-9]{1,2}:[0-9]{2}:[0-9]{2}) .* Killed process ([0-9]+) \(([\w]+)\)`)
	firstLineRegexp = regexp.MustCompile(`invoked oom-killer:`)
)

// struct to hold file from which we obtain OomInstances
type OomParser struct {
	ioreader *bufio.Reader
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
	// the absolute name of the container that was killed
	// due to the OOM.
	VictimContainerName string
}

// gets the container name from a line and adds it to the oomInstance.
func getContainerName(line string, currentOomInstance *OomInstance) error {
	parsedLine := containerRegexp.FindStringSubmatch(line)
	if parsedLine == nil {
		return nil
	}
	currentOomInstance.ContainerName = path.Join("/", parsedLine[1])
	currentOomInstance.VictimContainerName = path.Join("/", parsedLine[2])
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
	linetime, err := time.ParseInLocation(longForm, reList[1]+" "+stringYear, time.Local)
	if err != nil {
		return false, err
	}

	currentOomInstance.TimeOfDeath = linetime
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

// reads the file and sends only complete lines over a channel to analyzeLines.
// Should prevent EOF errors that occur when lines are read before being fully
// written to the log. It reads line by line splitting on
// the "\n" character.
func readLinesFromFile(lineChannel chan string, ioreader *bufio.Reader) {
	linefragment := ""
	var line string
	var err error
	for true {
		line, err = ioreader.ReadString('\n')
		if err == io.EOF {
			if line != "" {
				linefragment += line
			}
			time.Sleep(100 * time.Millisecond)
		} else if err == nil {
			if linefragment != "" {
				line = linefragment + line
				linefragment = ""
			}
			lineChannel <- line
		} else if err != nil && err != io.EOF {
			glog.Errorf("exiting analyzeLinesHelper with error %v", err)
		}
	}
}

// Calls goroutine for readLinesFromFile, which feeds it complete lines.
// Lines are checked against a regexp to check for the pid, process name, etc.
// At the end of an oom message group, StreamOoms adds the new oomInstance to
// oomLog
func (self *OomParser) StreamOoms(outStream chan *OomInstance) {
	lineChannel := make(chan string, 10)
	go func() {
		readLinesFromFile(lineChannel, self.ioreader)
	}()

	for line := range lineChannel {
		in_oom_kernel_log := checkIfStartOfOomMessages(line)
		if in_oom_kernel_log {
			oomCurrentInstance := &OomInstance{
				ContainerName: "/",
			}
			finished := false
			for !finished {
				err := getContainerName(line, oomCurrentInstance)
				if err != nil {
					glog.Errorf("%v", err)
				}
				finished, err = getProcessNamePid(line, oomCurrentInstance)
				if err != nil {
					glog.Errorf("%v", err)
				}
				line = <-lineChannel
			}
			outStream <- oomCurrentInstance
		}
	}
	glog.Infof("exiting analyzeLines")
}

func callJournalctl() (io.ReadCloser, error) {
	cmd := exec.Command("journalctl", "-k", "-f")
	readcloser, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return readcloser, err
}

func trySystemd() (*OomParser, error) {
	readcloser, err := callJournalctl()
	if err != nil {
		return nil, err
	}
	glog.Infof("oomparser using systemd")
	return &OomParser{
		ioreader: bufio.NewReader(readcloser),
	}, nil

}

// List of possible kernel log files. These are prioritized in order so that
// we will use the first one that is available.
var kernelLogFiles = []string{"/var/log/kern.log", "/var/log/messages", "/var/log/syslog"}

// looks for system files that contain kernel messages and if one is found, sets
// the systemFile attribute of the OomParser object
func getSystemFile() (string, error) {
	for _, logFile := range kernelLogFiles {
		if utils.FileExists(logFile) {
			glog.Infof("OOM parser using kernel log file: %q", logFile)
			return logFile, nil
		}
	}
	return "", fmt.Errorf("unable to find any kernel log file available from our set: %v", kernelLogFiles)
}

// initializes an OomParser object and calls getSystemFile to set the systemFile
// attribute.  Returns and OomParser object and an error
func New() (*OomParser, error) {
	systemFile, err := getSystemFile()
	if err != nil {
		return trySystemd()
	}
	file, err := os.Open(systemFile)
	if err != nil {
		return trySystemd()
	}
	return &OomParser{
		ioreader: bufio.NewReader(file),
	}, nil
}
