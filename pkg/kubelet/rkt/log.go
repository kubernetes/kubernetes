/*
Copyright 2015 The Kubernetes Authors.

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

package rkt

import (
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/context"

	rktapi "github.com/coreos/rkt/api/v1alpha"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

const (
	journalTimestampLayout = "2006-01-02 15:04:05 -0700 MST"
)

// processLines write the lines into stdout in the required format.
func processLines(lines []string, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) {
	msgKey := "MESSAGE="

	for _, line := range lines {
		msgStart := strings.Index(line, msgKey)
		if msgStart < 0 {
			glog.Warningf("rkt: Invalid log line %q, missing %q", line, msgKey)
			continue
		}

		tss := strings.TrimSpace(line[:msgStart])
		msg := strings.TrimPrefix(line[msgStart:], msgKey)

		t, err := time.Parse(journalTimestampLayout, tss)
		if err != nil {
			glog.Warningf("rkt: Failed to parse the timestamp %q: %v", tss, err)
			continue
		}

		var result string
		if logOptions.Timestamps {
			result = fmt.Sprintf("%s %s\n", t.Format(time.RFC3339), msg)
		} else {
			result = fmt.Sprintf("%s\n", msg)
		}

		if _, err := io.WriteString(stdout, result); err != nil {
			glog.Warningf("rkt: Cannot write log line %q to output: %v", result, err)
		}
	}
}

// GetContainerLogs uses rkt's GetLogs API to get the logs of the container.
// By default, it returns a snapshot of the container log. Set |follow| to true to
// stream the log. Set |follow| to false and specify the number of lines (e.g.
// "100" or "all") to tail the log.
//
// TODO(yifan): This doesn't work with lkvm stage1 yet.
func (r *Runtime) GetContainerLogs(pod *v1.Pod, containerID kubecontainer.ContainerID, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) error {
	id, err := parseContainerID(containerID)
	if err != nil {
		return err
	}

	var since int64
	if logOptions.SinceSeconds != nil {
		t := metav1.Now().Add(-time.Duration(*logOptions.SinceSeconds) * time.Second)
		since = t.Unix()
	}
	if logOptions.SinceTime != nil {
		since = logOptions.SinceTime.Unix()
	}

	getLogsRequest := &rktapi.GetLogsRequest{
		PodId:     id.uuid,
		AppName:   id.appName,
		Follow:    logOptions.Follow,
		SinceTime: since,
	}

	if logOptions.TailLines != nil {
		getLogsRequest.Lines = int32(*logOptions.TailLines)
	}

	stream, err := r.apisvc.GetLogs(context.Background(), getLogsRequest)
	if err != nil {
		glog.Errorf("rkt: Failed to create log stream for pod %q: %v", format.Pod(pod), err)
		return err
	}

	for {
		log, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			glog.Errorf("rkt: Failed to receive log for pod %q: %v", format.Pod(pod), err)
			return err
		}
		processLines(log.Lines, logOptions, stdout, stderr)
	}

	return nil
}
