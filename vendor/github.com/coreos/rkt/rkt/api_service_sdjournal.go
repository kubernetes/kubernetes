// Copyright 2016 The rkt Authors
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

// +build sdjournal

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"time"

	"github.com/coreos/go-systemd/sdjournal"
	"github.com/coreos/rkt/api/v1alpha"
	"github.com/coreos/rkt/common"
	pkgPod "github.com/coreos/rkt/pkg/pod"
)

func (s *v1AlphaAPIServer) constrainedGetLogs(request *v1alpha.GetLogsRequest, server v1alpha.PublicAPI_GetLogsServer) error {
	pod, err := pkgPod.PodFromUUIDString(getDataDir(), request.PodId)
	if err != nil {
		return err
	}
	defer pod.Close()

	path, err := pod.JournalLogPath()
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return fmt.Errorf("logging unsupported for pod %q", request.PodId)
	}

	jconf := sdjournal.JournalReaderConfig{
		Path: path,
	}
	if request.AppName != "" {
		jconf.Matches = []sdjournal.Match{
			{
				Field: sdjournal.SD_JOURNAL_FIELD_SYSLOG_IDENTIFIER,
				Value: request.AppName,
			},
		}
	}
	if request.SinceTime != 0 {
		t := time.Unix(request.SinceTime, 0)
		jconf.Since = -time.Since(t)
	}
	if request.Lines != 0 {
		jconf.NumFromTail = uint64(request.Lines)
	}

	jr, err := sdjournal.NewJournalReader(jconf)
	if err != nil {
		return err
	}
	defer jr.Close()

	if request.Follow {
		return jr.Follow(nil, LogsStreamWriter{server: server})
	}

	data, err := ioutil.ReadAll(jr)
	if err != nil {
		return err
	}

	return server.Send(&v1alpha.GetLogsResponse{Lines: common.RemoveEmptyLines(string(data))})
}
