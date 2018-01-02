// +build linux

package seccomp

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/containers"
	"github.com/opencontainers/runtime-spec/specs-go"
)

// WithProfile receives the name of a file stored on disk comprising a json
// formated seccomp profile, as specified by the opencontainers/runtime-spec.
// The profile is read from the file, unmarshaled, and set to the spec.
func WithProfile(profile string) containerd.SpecOpts {
	return func(_ context.Context, _ *containerd.Client, _ *containers.Container, s *specs.Spec) error {
		s.Linux.Seccomp = &specs.LinuxSeccomp{}
		f, err := ioutil.ReadFile(profile)
		if err != nil {
			return fmt.Errorf("Cannot load seccomp profile %q: %v", profile, err)
		}
		if err := json.Unmarshal(f, s.Linux.Seccomp); err != nil {
			return fmt.Errorf("Decoding seccomp profile failed %q: %v", profile, err)
		}
		return nil
	}
}

// WithDefaultProfile sets the default seccomp profile to the spec.
// Note: must follow the setting of process capabilities
func WithDefaultProfile() containerd.SpecOpts {
	return func(_ context.Context, _ *containerd.Client, _ *containers.Container, s *specs.Spec) error {
		s.Linux.Seccomp = DefaultProfile(s)
		return nil
	}
}
