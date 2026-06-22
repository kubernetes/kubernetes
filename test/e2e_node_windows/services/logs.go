//go:build windows

/*
Copyright The Kubernetes Authors.

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
	Name              string   `json:"name"`
	Files             []string `json:"files"`
	JournalctlCommand []string `json:"journalctl"`
}

// logFiles are the type used to collect all log files.
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
	logs := *l
	logs[log.Name] = log
	return nil
}

var extraLogs = make(logFiles)

func init() {
	flag.Var(&extraLogs, "extra-log", "Extra log to collect after test in the json format of LogFile.")
}

// getLogFiles returns the log files to collect. Windows has no journald or
// standard Linux log paths, so only extra logs specified via flags are returned.
func getLogFiles() logFiles {
	logs := make(logFiles)
	for _, l := range extraLogs {
		logs[l.Name] = l
	}
	return logs
}
