//go:build linux

/*
Copyright 2014 Google Inc. All Rights Reserved.

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

// Package oomparser is forked from github.com/google/cadvisor (utils/oomparser).
// It reads the kernel ring buffer (/dev/kmsg) and extracts OOM kill events. It
// is adapted to take a context.Context, log through klog, and read the kernel
// log through the forked go-kmsg-parser in this tree.
package oomparser

import (
	"context"
	"path"
	"regexp"
	"strconv"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/third_party/forked/go-kmsg-parser/kmsgparser"
)

var (
	legacyContainerRegexp = regexp.MustCompile(`Task in (.*) killed as a result of limit of (.*)`)
	// Starting in 5.0 linux kernels, the OOM message changed.
	containerRegexp = regexp.MustCompile(`oom-kill:constraint=(.*),nodemask=(.*),cpuset=(.*),mems_allowed=(.*),oom_memcg=(.*),task_memcg=(.*),task=(.*),pid=(.*),uid=(.*)`)
	lastLineRegexp  = regexp.MustCompile(`Killed process ([0-9]+) \((.+)\)`)
	firstLineRegexp = regexp.MustCompile(`invoked oom-killer:`)
)

// OomInstance holds the information extracted from a single kernel OOM kill.
type OomInstance struct {
	// process id of the killed process
	Pid int
	// the name of the killed process
	ProcessName string
	// the time the process was reported killed, accurate to the minute
	TimeOfDeath time.Time
	// the absolute name of the container that OOMed
	ContainerName string
	// the absolute name of the container that was killed due to the OOM
	VictimContainerName string
	// the constraint that triggered the OOM (CONSTRAINT_NONE, CONSTRAINT_CPUSET,
	// CONSTRAINT_MEMORY_POLICY, CONSTRAINT_MEMCG)
	Constraint string
}

func getLegacyContainerName(line string, o *OomInstance) {
	parsed := legacyContainerRegexp.FindStringSubmatch(line)
	if parsed == nil {
		return
	}
	o.ContainerName = path.Join("/", parsed[1])
	o.VictimContainerName = path.Join("/", parsed[2])
}

// getContainerName fills the container fields from a line. It returns true when
// the line was the modern oom-kill line that also carries the pid and task name.
func getContainerName(line string, o *OomInstance) (bool, error) {
	parsed := containerRegexp.FindStringSubmatch(line)
	if parsed == nil {
		// Fall back to the pre-5.0 kernel format.
		getLegacyContainerName(line, o)
		return false, nil
	}
	o.ContainerName = parsed[6]
	o.VictimContainerName = parsed[5]
	o.Constraint = parsed[1]
	pid, err := strconv.Atoi(parsed[8])
	if err != nil {
		return false, err
	}
	o.Pid = pid
	o.ProcessName = parsed[7]
	return true, nil
}

// getProcessNamePid fills the pid and process name from the "Killed process" line.
func getProcessNamePid(line string, o *OomInstance) (bool, error) {
	parsed := lastLineRegexp.FindStringSubmatch(line)
	if parsed == nil {
		return false, nil
	}
	pid, err := strconv.Atoi(parsed[1])
	if err != nil {
		return false, err
	}
	o.Pid = pid
	o.ProcessName = parsed[2]
	return true, nil
}

func checkIfStartOfOomMessages(line string) bool {
	return firstLineRegexp.MatchString(line)
}

// OomParser reads the kernel ring buffer (/dev/kmsg) and extracts OOM events.
type OomParser struct {
	parser kmsgparser.Parser
}

// New creates an OomParser backed by the kernel log (/dev/kmsg).
func New() (*OomParser, error) {
	parser, err := kmsgparser.NewParser()
	if err != nil {
		return nil, err
	}
	return &OomParser{parser: parser}, nil
}

// StreamOoms writes an OomInstance to outStream for every OOM event found in the
// kernel log. It blocks and should be called from a goroutine.
func (p *OomParser) StreamOoms(ctx context.Context, outStream chan<- *OomInstance) {
	logger := klog.FromContext(ctx)
	kmsgEntries := make(chan kmsgparser.Message, 32)
	go func() {
		defer func() { _ = p.parser.Close() }()
		if err := p.parser.Parse(kmsgEntries); err != nil {
			logger.Error(err, "Exiting kmsg parse; OOM events will not be reported")
		}
	}()

	for msg := range kmsgEntries {
		if !checkIfStartOfOomMessages(msg.Message) {
			continue
		}
		oom := &OomInstance{
			ContainerName:       "/",
			VictimContainerName: "/",
			TimeOfDeath:         msg.Timestamp,
		}
		for msg := range kmsgEntries {
			finished, err := getContainerName(msg.Message, oom)
			if err != nil {
				logger.Error(err, "Failed to parse OOM container name")
			}
			if !finished {
				finished, err = getProcessNamePid(msg.Message, oom)
				if err != nil {
					logger.Error(err, "Failed to parse OOM process name and pid")
				}
			}
			if finished {
				oom.TimeOfDeath = msg.Timestamp
				break
			}
		}
		outStream <- oom
	}
	logger.Error(nil, "Stopped receiving OOM notifications; OOM events will not be reported")
}
