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
