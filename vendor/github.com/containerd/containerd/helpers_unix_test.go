// +build !windows

package containerd

import (
	"context"
	"fmt"

	"github.com/containerd/containerd/containers"
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

const newLine = "\n"

func withExitStatus(es int) SpecOpts {
	return func(_ context.Context, _ *Client, _ *containers.Container, s *specs.Spec) error {
		s.Process.Args = []string{"sh", "-c", fmt.Sprintf("exit %d", es)}
		return nil
	}
}

func withProcessArgs(args ...string) SpecOpts {
	return WithProcessArgs(args...)
}

func withCat() SpecOpts {
	return WithProcessArgs("cat")
}

func withTrue() SpecOpts {
	return WithProcessArgs("true")
}

func withExecExitStatus(s *specs.Process, es int) {
	s.Args = []string{"sh", "-c", fmt.Sprintf("exit %d", es)}
}

func withExecArgs(s *specs.Process, args ...string) {
	s.Args = args
}

var (
	withRemappedSnapshot     = WithRemappedSnapshot
	withRemappedSnapshotView = WithRemappedSnapshotView
	withNewSnapshot          = WithNewSnapshot
	withImageConfig          = WithImageConfig
)
