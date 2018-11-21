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

package exec

import (
	"bytes"
	"io"

	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/probe"
	"k8s.io/utils/exec"
)

const (
	maxReadBytes = 2 * 1024 // 2kb
)

// New creates a Prober.
func New() Prober {
	return execProber{
		maxReadBytes: maxReadBytes,
	}
}

// Prober is an interface defining the Probe object for container readiness/liveness checks.
type Prober interface {
	Probe(e exec.Cmd) (probe.Result, string, error)
}

type execProber struct {
	maxReadBytes int64
}

// Probe executes a command to check the liveness/readiness of container
// from executing a command. Returns the Result status, command output, and
// errors if any.
func (pr execProber) Probe(e exec.Cmd) (probe.Result, string, error) {
	mr, err := combinedCmdOutput(e)
	if err != nil {
		return probe.Unknown, "", err
	}

	readDone := make(chan *readResult)
	go func() {
		readDone <- readN(mr, pr.maxReadBytes)
	}()

	err = e.Start()
	if err != nil {
		return probe.Unknown, "", err
	}

	output := <-readDone
	if err = output.err; err != nil {
		return probe.Unknown, "", err
	}

	err = e.Wait()
	data := string(output.bytes)

	klog.V(4).Infof("Exec probe response: %q", data)
	if err != nil {
		exit, ok := err.(exec.ExitError)
		if ok {
			if exit.ExitStatus() == 0 {
				return probe.Success, data, nil
			}
			return probe.Failure, data, nil
		}
		return probe.Unknown, "", err
	}
	return probe.Success, data, nil
}

func combinedCmdOutput(cmd exec.Cmd) (io.Reader, error) {
	var err error
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	mr := io.MultiReader(stdout, stderr)
	return mr, nil
}

func readN(r io.Reader, n int64) *readResult {
	buf := &bytes.Buffer{}
	_, err := io.CopyN(buf, r, n)
	if err == io.EOF {
		err = nil
	}
	return &readResult{
		err:   err,
		bytes: buf.Bytes(),
	}
}

type readResult struct {
	err   error
	bytes []byte
}
