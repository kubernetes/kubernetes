package osl

import "testing"

// GenerateKey generates a sandbox key based on the passed
// container id.
func GenerateKey(containerID string) string {
	maxLen := 12
	if len(containerID) < maxLen {
		maxLen = len(containerID)
	}

	return containerID[:maxLen]
}

// NewSandbox provides a new sandbox instance created in an os specific way
// provided a key which uniquely identifies the sandbox
func NewSandbox(key string, osCreate, isRestore bool) (Sandbox, error) {
	return nil, nil
}

// GetSandboxForExternalKey returns sandbox object for the supplied path
func GetSandboxForExternalKey(path string, key string) (Sandbox, error) {
	return nil, nil
}

// GC triggers garbage collection of namespace path right away
// and waits for it.
func GC() {
}

// InitOSContext initializes OS context while configuring network resources
func InitOSContext() func() {
	return func() {}
}

// SetupTestOSContext sets up a separate test  OS context in which tests will be executed.
func SetupTestOSContext(t *testing.T) func() {
	return func() {}
}

// SetBasePath sets the base url prefix for the ns path
func SetBasePath(path string) {
}
