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

	"k8s.io/klog/v2"
)

var (
	legacyContainerRegexp = regexp.MustCompile(`Task in (.*) killed as a result of limit of (.*)`)
	// Starting in 5.0 linux kernels, the OOM message changed
	containerRegexp = regexp.MustCompile(`oom-kill:constraint=(.*),nodemask=(.*),cpuset=(.*),mems_allowed=(.*),oom_memcg=(.*) (.*),task_memcg=(.*),task=(.*),pid=(.*),uid=(.*)`)
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
	// the constraint that triggered the OOM.  One of CONSTRAINT_NONE,
	// CONSTRAINT_CPUSET, CONSTRAINT_MEMORY_POLICY, CONSTRAINT_MEMCG
	Constraint string
}

// gets the container name from a line and adds it to the oomInstance.
func getLegacyContainerName(line string, currentOomInstance *OomInstance) error {
	parsedLine := legacyContainerRegexp.FindStringSubmatch(line)
	if parsedLine == nil {
		return nil
	}
	currentOomInstance.ContainerName = path.Join("/", parsedLine[1])
	currentOomInstance.VictimContainerName = path.Join("/", parsedLine[2])
	return nil
}

// gets the container name from a line and adds it to the oomInstance.
func getContainerName(line string, currentOomInstance *OomInstance) (bool, error) {
	parsedLine := containerRegexp.FindStringSubmatch(line)
	if parsedLine == nil {
		// Fall back to the legacy format if it isn't found here.
		return false, getLegacyContainerName(line, currentOomInstance)
	}
	currentOomInstance.ContainerName = parsedLine[7]
	currentOomInstance.VictimContainerName = parsedLine[5]
	currentOomInstance.Constraint = parsedLine[1]
	pid, err := strconv.Atoi(parsedLine[9])
	if err != nil {
		return false, err
	}
	currentOomInstance.Pid = pid
	currentOomInstance.ProcessName = parsedLine[8]
	return true, nil
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
	potentialOomStart := firstLineRegexp.MatchString(line)
	return potentialOomStart
}

// StreamOoms writes to a provided a stream of OomInstance objects representing
// OOM events that are found in the logs.
// It will block and should be called from a goroutine.
func (p *OomParser) StreamOoms(outStream chan<- *OomInstance) {
	kmsgEntries := p.parser.Parse()
	defer p.parser.Close()

	for msg := range kmsgEntries {
		isOomMessage := checkIfStartOfOomMessages(msg.Message)
		if isOomMessage {
			oomCurrentInstance := &OomInstance{
				ContainerName:       "/",
				VictimContainerName: "/",
				TimeOfDeath:         msg.Timestamp,
			}
			for msg := range kmsgEntries {
				finished, err := getContainerName(msg.Message, oomCurrentInstance)
				if err != nil {
					klog.Errorf("%v", err)
				}
				if !finished {
					finished, err = getProcessNamePid(msg.Message, oomCurrentInstance)
					if err != nil {
						klog.Errorf("%v", err)
					}
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
	klog.Errorf("exiting analyzeLines. OOM events will not be reported.")
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
	klog.V(4).Infof(format, args...)
}
func (glogAdapter) Warningf(format string, args ...interface{}) {
	klog.V(2).Infof(format, args...)
}
func (glogAdapter) Errorf(format string, args ...interface{}) {
	klog.Warningf(format, args...)
}
