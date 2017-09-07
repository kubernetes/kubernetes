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
	"path"
	"regexp"
	"strconv"
	"time"

	"github.com/euank/go-kmsg-parser/kmsgparser"

	"github.com/golang/glog"
)

var (
	containerRegexp = regexp.MustCompile(`Task in (.*) killed as a result of limit of (.*)`)
	lastLineRegexp  = regexp.MustCompile(`Killed process ([0-9]+) \((.+)\)`)
	firstLineRegexp = regexp.MustCompile(`invoked oom-killer:`)
)

// OomParser wraps a kmsgparser in order to extract OOM events from the
// individual kernel ring buffer messages.
type OomParser struct {
	parser kmsgparser.Parser
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

	pid, err := strconv.Atoi(reList[1])
	if err != nil {
		return false, err
	}
	currentOomInstance.Pid = pid
	currentOomInstance.ProcessName = reList[2]
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

// StreamOoms writes to a provided a stream of OomInstance objects representing
// OOM events that are found in the logs.
// It will block and should be called from a goroutine.
func (self *OomParser) StreamOoms(outStream chan<- *OomInstance) {
	kmsgEntries := self.parser.Parse()
	defer self.parser.Close()

	for msg := range kmsgEntries {
		in_oom_kernel_log := checkIfStartOfOomMessages(msg.Message)
		if in_oom_kernel_log {
			oomCurrentInstance := &OomInstance{
				ContainerName: "/",
				TimeOfDeath:   msg.Timestamp,
			}
			for msg := range kmsgEntries {
				err := getContainerName(msg.Message, oomCurrentInstance)
				if err != nil {
					glog.Errorf("%v", err)
				}
				finished, err := getProcessNamePid(msg.Message, oomCurrentInstance)
				if err != nil {
					glog.Errorf("%v", err)
				}
				if finished {
					oomCurrentInstance.TimeOfDeath = msg.Timestamp
					break
				}
			}
			outStream <- oomCurrentInstance
		}
	}
	// Should not happen
	glog.Errorf("exiting analyzeLines. OOM events will not be reported.")
}

// initializes an OomParser object. Returns an OomParser object and an error.
func New() (*OomParser, error) {
	parser, err := kmsgparser.NewParser()
	if err != nil {
		return nil, err
	}
	parser.SetLogger(glogAdapter{})
	return &OomParser{parser: parser}, nil
}

type glogAdapter struct{}

var _ kmsgparser.Logger = glogAdapter{}

func (glogAdapter) Infof(format string, args ...interface{}) {
	glog.V(4).Infof(format, args)
}
func (glogAdapter) Warningf(format string, args ...interface{}) {
	glog.Infof(format, args)
}
func (glogAdapter) Errorf(format string, args ...interface{}) {
	glog.Warningf(format, args)
}
