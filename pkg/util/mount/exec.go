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

package mount

import "k8s.io/utils/exec"

// NewOSExec returns a new Exec interface implementation based on exec()
func NewOSExec() *OSExec {
	return &OSExec{}
}

// OSExec is an implementation of Exec interface that uses simple utils.Exec
type OSExec struct{}

var _ Exec = &OSExec{}

// Run exucutes the given cmd and arges and returns stdout and stderr as a
// combined byte stream
func (e *OSExec) Run(cmd string, args ...string) ([]byte, error) {
	exe := exec.New()
	return exe.Command(cmd, args...).CombinedOutput()
}
