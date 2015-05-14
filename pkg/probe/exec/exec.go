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

package exec

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"

	"github.com/golang/glog"
)

func New() ExecProber {
	return execProber{}
}

type ExecProber interface {
	Probe(e exec.Cmd) (probe.Result, string, error)
}

type execProber struct{}

func (pr execProber) Probe(e exec.Cmd) (probe.Result, string, error) {
	data, err := e.CombinedOutput()
	glog.V(4).Infof("Exec probe response: %q", string(data))
	if err != nil {
		exit, ok := err.(exec.ExitError)
		if ok {
			if exit.ExitStatus() == 0 {
				return probe.Success, string(data), nil
			} else {
				return probe.Failure, string(data), nil
			}
		}
		return probe.Unknown, "", err
	}
	return probe.Success, string(data), nil
}
