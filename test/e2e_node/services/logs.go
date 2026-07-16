/*
Copyright 2017 The Kubernetes Authors.

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

package services

import (
	"encoding/json"
	"flag"
	"fmt"
)

// LogFileData holds data about logfiles to fetch with a journalctl command or
// file from a node's file system.
type LogFileData struct {
	// Name of the log file.
	Name string `json:"name"`
	// Files are possible absolute paths of the log file.
	Files []string `json:"files"`
	// JournalctlCommand is the journalctl command to get log.
	JournalctlCommand []string `json:"journalctl"`
}

// logFiles are the type used to collect all log files. The key is the expected
// name of the log file after collected.
type logFiles map[string]LogFileData

// String function of flag.Value
func (l *logFiles) String() string {
	return fmt.Sprint(*l)
}

// Set function of flag.Value
func (l *logFiles) Set(value string) error {
	var log LogFileData
	if err := json.Unmarshal([]byte(value), &log); err != nil {
		return err
	}
	// Note that we assume all white space in flag string is separating fields
	logs := *l
	logs[log.Name] = log
	return nil
}

// extraLogs is the extra logs specified by the test runner.
var extraLogs = make(logFiles)

func init() {
	flag.Var(&extraLogs, "extra-log", "Extra log to collect after test in the json format of LogFile.")
}

// requiredLogs is the required logs to collect after the test.
var requiredLogs = []LogFileData{
	{
		Name:              "kern.log",
		Files:             []string{"/var/log/kern.log"},
		JournalctlCommand: []string{"-k"},
	},
	{
		Name:              "cloud-init.log",
		Files:             []string{"/var/log/cloud-init.log", "/var/log/cloud-init-output.log"},
		JournalctlCommand: []string{"-u", "cloud*"},
	},
	{
		Name:              "containerd.log",
		Files:             []string{"/var/log/containerd.log"},
		JournalctlCommand: []string{"-u", "containerd"},
	},
	{
		Name:              "containerd-installation.log",
		JournalctlCommand: []string{"-u", "containerd-installation"},
	},
}

// getLogFiles get all logs to collect after the test.
func getLogFiles() logFiles {
	logs := make(logFiles)
	for _, l := range requiredLogs {
		logs[l.Name] = l
	}
	for _, l := range extraLogs {
		logs[l.Name] = l
	}
	return logs
}
