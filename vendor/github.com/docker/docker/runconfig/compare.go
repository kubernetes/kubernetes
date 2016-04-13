package runconfig

// Compare two Config struct. Do not compare the "Image" nor "Hostname" fields
// If OpenStdin is set, then it differs
func Compare(a, b *Config) bool {
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

	if a.Cmd.Len() != b.Cmd.Len() ||
		len(a.Env) != len(b.Env) ||
		len(a.Labels) != len(b.Labels) ||
		len(a.ExposedPorts) != len(b.ExposedPorts) ||
		a.Entrypoint.Len() != b.Entrypoint.Len() ||
		len(a.Volumes) != len(b.Volumes) {
		return false
	}

	aCmd := a.Cmd.Slice()
	bCmd := b.Cmd.Slice()
	for i := 0; i < len(aCmd); i++ {
		if aCmd[i] != bCmd[i] {
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

	aEntrypoint := a.Entrypoint.Slice()
	bEntrypoint := b.Entrypoint.Slice()
	for i := 0; i < len(aEntrypoint); i++ {
		if aEntrypoint[i] != bEntrypoint[i] {
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
