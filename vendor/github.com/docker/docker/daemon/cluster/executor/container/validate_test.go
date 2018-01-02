package container

import (
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"github.com/docker/docker/daemon"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/swarmkit/api"
)

func newTestControllerWithMount(m api.Mount) (*controller, error) {
	return newController(&daemon.Daemon{}, &api.Task{
		ID:        stringid.GenerateRandomID(),
		ServiceID: stringid.GenerateRandomID(),
		Spec: api.TaskSpec{
			Runtime: &api.TaskSpec_Container{
				Container: &api.ContainerSpec{
					Image: "image_name",
					Labels: map[string]string{
						"com.docker.swarm.task.id": "id",
					},
					Mounts: []api.Mount{m},
				},
			},
		},
	}, nil)
}

func TestControllerValidateMountBind(t *testing.T) {
	// with improper source
	if _, err := newTestControllerWithMount(api.Mount{
		Type:   api.MountTypeBind,
		Source: "foo",
		Target: testAbsPath,
	}); err == nil || !strings.Contains(err.Error(), "invalid bind mount source") {
		t.Fatalf("expected  error, got: %v", err)
	}

	// with non-existing source
	if _, err := newTestControllerWithMount(api.Mount{
		Type:   api.MountTypeBind,
		Source: testAbsNonExistent,
		Target: testAbsPath,
	}); err != nil {
		t.Fatalf("controller should not error at creation: %v", err)
	}

	// with proper source
	tmpdir, err := ioutil.TempDir("", "TestControllerValidateMountBind")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.Remove(tmpdir)

	if _, err := newTestControllerWithMount(api.Mount{
		Type:   api.MountTypeBind,
		Source: tmpdir,
		Target: testAbsPath,
	}); err != nil {
		t.Fatalf("expected  error, got: %v", err)
	}
}

func TestControllerValidateMountVolume(t *testing.T) {
	// with improper source
	if _, err := newTestControllerWithMount(api.Mount{
		Type:   api.MountTypeVolume,
		Source: testAbsPath,
		Target: testAbsPath,
	}); err == nil || !strings.Contains(err.Error(), "invalid volume mount source") {
		t.Fatalf("expected error, got: %v", err)
	}

	// with proper source
	if _, err := newTestControllerWithMount(api.Mount{
		Type:   api.MountTypeVolume,
		Source: "foo",
		Target: testAbsPath,
	}); err != nil {
		t.Fatalf("expected error, got: %v", err)
	}
}

func TestControllerValidateMountTarget(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "TestControllerValidateMountTarget")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.Remove(tmpdir)

	// with improper target
	if _, err := newTestControllerWithMount(api.Mount{
		Type:   api.MountTypeBind,
		Source: testAbsPath,
		Target: "foo",
	}); err == nil || !strings.Contains(err.Error(), "invalid mount target") {
		t.Fatalf("expected error, got: %v", err)
	}

	// with proper target
	if _, err := newTestControllerWithMount(api.Mount{
		Type:   api.MountTypeBind,
		Source: tmpdir,
		Target: testAbsPath,
	}); err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
}

func TestControllerValidateMountTmpfs(t *testing.T) {
	// with improper target
	if _, err := newTestControllerWithMount(api.Mount{
		Type:   api.MountTypeTmpfs,
		Source: "foo",
		Target: testAbsPath,
	}); err == nil || !strings.Contains(err.Error(), "invalid tmpfs source") {
		t.Fatalf("expected error, got: %v", err)
	}

	// with proper target
	if _, err := newTestControllerWithMount(api.Mount{
		Type:   api.MountTypeTmpfs,
		Target: testAbsPath,
	}); err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
}

func TestControllerValidateMountInvalidType(t *testing.T) {
	// with improper target
	if _, err := newTestControllerWithMount(api.Mount{
		Type:   api.Mount_MountType(9999),
		Source: "foo",
		Target: testAbsPath,
	}); err == nil || !strings.Contains(err.Error(), "invalid mount type") {
		t.Fatalf("expected error, got: %v", err)
	}
}
