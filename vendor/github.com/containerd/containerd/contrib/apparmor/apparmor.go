// +build linux

package apparmor

import (
	"context"
	"io/ioutil"
	"os"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/containers"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
)

// WithProfile sets the provided apparmor profile to the spec
func WithProfile(profile string) containerd.SpecOpts {
	return func(_ context.Context, _ *containerd.Client, _ *containers.Container, s *specs.Spec) error {
		s.Process.ApparmorProfile = profile
		return nil
	}
}

// WithDefaultProfile will generate a default apparmor profile under the provided name
// for the container.  It is only generated if a profile under that name does not exist.
func WithDefaultProfile(name string) containerd.SpecOpts {
	return func(_ context.Context, _ *containerd.Client, _ *containers.Container, s *specs.Spec) error {
		yes, err := isLoaded(name)
		if err != nil {
			return err
		}
		if yes {
			s.Process.ApparmorProfile = name
			return nil
		}
		p, err := loadData(name)
		if err != nil {
			return err
		}
		f, err := ioutil.TempFile("", p.Name)
		if err != nil {
			return err
		}
		defer f.Close()
		path := f.Name()
		defer os.Remove(path)

		if err := generate(p, f); err != nil {
			return err
		}
		if err := load(path); err != nil {
			return errors.Wrapf(err, "load apparmor profile %s", path)
		}
		s.Process.ApparmorProfile = name
		return nil
	}
}
