/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"sync"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

const (
	// The layout of the time format that satisfies the `--since` option for journalctl.
	// See man journalctl for more details.
	journalSinceLayout = "2006-01-02 15:04:05"
)

// pipeLog reads and parses the journal json object from r,
// and writes the logs line by line to w.
func pipeLog(wg *sync.WaitGroup, logOptions *api.PodLogOptions, r io.ReadCloser, w io.Writer) {
	defer func() {
		r.Close()
		wg.Done()
	}()

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		var data interface{}
		b := scanner.Bytes()

		if err := json.Unmarshal(b, &data); err != nil {
			glog.Warningf("rkt: Cannot unmarshal journal log, skipping line: %v", err)
			continue
		}

		// Decode the json object as a map so we don't have to create a struct for it.
		m := data.(map[string]interface{})
		msg := m["MESSAGE"].(string)
		tss := m["__REALTIME_TIMESTAMP"].(string)

		ts, err := strconv.ParseInt(tss, 0, 64)
		if err != nil {
			glog.Warningf("rkt: Cannot parse timestamp of journal log, skipping line: %v", err)
			continue
		}

		// '_REALTIME_TIMESTAMP' is the microseconds since epoch.
		// http://www.freedesktop.org/software/systemd/man/sd_journal_get_realtime_usec.html#Description
		micros := time.Duration(ts) * time.Microsecond
		t := time.Unix(int64(micros.Seconds()), 0)

		var result string
		if logOptions.Timestamps {
			// Use the same time format as docker.
			result = fmt.Sprintf("%s %s\n", t, msg)
		} else {
			result = fmt.Sprintf("%s\n", msg)
		}

		// When user interrupts the 'kubectl logs $POD -f' with 'Ctrl-C', this write will fail.
		// We need to close the reader to force the journalctl process to exit, also we need to
		// return here to avoid goroutine leak.
		if _, err := io.WriteString(w, result); err != nil {
			glog.Warningf("rkt: Cannot write log to output: %v, data: %s", err, result)
			return
		}
	}

	if err := scanner.Err(); err != nil {
		glog.Warningf("rkt: Cannot read journal logs", err)
	}
}

// GetContainerLogs uses journalctl to get the logs of the container.
// By default, it returns a snapshot of the container log. Set |follow| to true to
// stream the log. Set |follow| to false and specify the number of lines (e.g.
// "100" or "all") to tail the log.
//
// In rkt runtime's implementation, per container log is get via 'journalctl -m _HOSTNAME=[rkt-$UUID] -u [APP_NAME]'.
// See https://github.com/coreos/rkt/blob/master/Documentation/commands.md#logging for more details.
//
// TODO(yifan): If the rkt is using lkvm as the stage1 image, then this function will fail.
func (r *Runtime) GetContainerLogs(pod *api.Pod, containerID kubecontainer.ContainerID, logOptions *api.PodLogOptions, stdout, stderr io.Writer) error {
	id, err := parseContainerID(containerID)
	if err != nil {
		return err
	}

	cmd := exec.Command("journalctl", "-m", fmt.Sprintf("_HOSTNAME=rkt-%s", id.uuid), "-u", id.appName, "-a")

	// Get the json structured logs.
	cmd.Args = append(cmd.Args, "-o", "json")

	if logOptions.Follow {
		cmd.Args = append(cmd.Args, "-f")
	}

	if logOptions.TailLines != nil {
		cmd.Args = append(cmd.Args, "-n", strconv.FormatInt(*logOptions.TailLines, 10))
	}

	var since int64
	if logOptions.SinceSeconds != nil {
		t := unversioned.Now().Add(-time.Duration(*logOptions.SinceSeconds) * time.Second)
		since = t.Unix()
	}
	if logOptions.SinceTime != nil {
		since = logOptions.SinceTime.Unix()
	}

	if since > 0 {
		// Need to add '-r' flag if we include '--since' and '-n' at the both time,
		// see https://github.com/systemd/systemd/issues/1477
		cmd.Args = append(cmd.Args, "--since", time.Unix(since, 0).Format(journalSinceLayout))
		if logOptions.TailLines != nil {
			cmd.Args = append(cmd.Args, "-r")
		}
	}

	outPipe, err := cmd.StdoutPipe()
	if err != nil {
		glog.Errorf("rkt: cannot create pipe for journalctl's stdout", err)
		return err
	}
	errPipe, err := cmd.StderrPipe()
	if err != nil {
		glog.Errorf("rkt: cannot create pipe for journalctl's stderr", err)
		return err
	}

	if err := cmd.Start(); err != nil {
		return err
	}

	var wg sync.WaitGroup

	wg.Add(2)

	go pipeLog(&wg, logOptions, outPipe, stdout)
	go pipeLog(&wg, logOptions, errPipe, stderr)

	// Wait until the logs are fed to stdout, stderr.
	wg.Wait()
	cmd.Wait()

	return nil
}
