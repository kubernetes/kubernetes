/*
Copyright 2023 The Kubernetes Authors.

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

package coloredwriter

import (
	"fmt"
	"github.com/fatih/color"
	"io"
)

/*
TODO:
- It would be great to highlight pods in "kubectl get pods" where # of ready containers < total # of containers
*/

var (
	yellow = color.New(color.FgYellow).SprintFunc()
	red    = color.New(color.FgRed).SprintFunc()
	green  = color.New(color.FgGreen).SprintFunc()
)

type ColoredTabWriter struct {
	Delegate io.Writer
}

func (c ColoredTabWriter) Write(p []byte) (int, error) {
	n := len(p)
	s := string(p)

	switch s {
	case "Running":
		s = fmt.Sprintf("%s", green(s))
	case "ContainerCreating", "Terminating", "Pending":
		s = fmt.Sprintf("%s", yellow(s))
	case "Error", "CrashLoopBackOff", "ImagePullBackOff":
		s = fmt.Sprintf("%s", red(s))
	}

	_, err := c.Delegate.Write([]byte(s))
	if err != nil {
		return c.Delegate.Write(p)
	}
	return n, nil
}
