package cache

import (
	"github.com/docker/docker/api/types/container"
)

// compare two Config struct. Do not compare the "Image" nor "Hostname" fields
// If OpenStdin is set, then it differs
func compare(a, b *container.Config) bool {
	if a == nil || b == nil ||
		a.OpenStdin || b.OpenStdin {
		return false
	}
	if a.AttachStdout != b.AttachStdout ||
		a.AttachStderr != b.AttachStderr ||
		a.User != b.User ||
		a.OpenStdin != b.OpenStdin ||
		a.Tty != b.Tty {
		return false
	}

	if len(a.Cmd) != len(b.Cmd) ||
		len(a.Env) != len(b.Env) ||
		len(a.Labels) != len(b.Labels) ||
		len(a.ExposedPorts) != len(b.ExposedPorts) ||
		len(a.Entrypoint) != len(b.Entrypoint) ||
		len(a.Volumes) != len(b.Volumes) {
		return false
	}

	for i := 0; i < len(a.Cmd); i++ {
		if a.Cmd[i] != b.Cmd[i] {
			return false
		}
	}
	for i := 0; i < len(a.Env); i++ {
		if a.Env[i] != b.Env[i] {
			return false
		}
	}
	for k, v := range a.Labels {
		if v != b.Labels[k] {
			return false
		}
	}
	for k := range a.ExposedPorts {
		if _, exists := b.ExposedPorts[k]; !exists {
			return false
		}
	}

	for i := 0; i < len(a.Entrypoint); i++ {
		if a.Entrypoint[i] != b.Entrypoint[i] {
			return false
		}
	}
	for key := range a.Volumes {
		if _, exists := b.Volumes[key]; !exists {
			return false
		}
	}
	return true
}
