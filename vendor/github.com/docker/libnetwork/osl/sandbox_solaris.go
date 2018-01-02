package osl

// NewSandbox provides a new sandbox instance created in an os specific way
// provided a key which uniquely identifies the sandbox
func NewSandbox(key string, osCreate, isRestore bool) (Sandbox, error) {
	return nil, nil
}

// GenerateKey generates a sandbox key based on the passed
// container id.
func GenerateKey(containerID string) string {
	maxLen := 12

	if len(containerID) < maxLen {
		maxLen = len(containerID)
	}

	return containerID[:maxLen]
}

// InitOSContext initializes OS context while configuring network resources
func InitOSContext() func() {
	return func() {}
}
