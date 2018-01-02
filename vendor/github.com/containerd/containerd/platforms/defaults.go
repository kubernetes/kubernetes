package platforms

import (
	"runtime"

	specs "github.com/opencontainers/image-spec/specs-go/v1"
)

// Default returns the default specifier for the platform.
func Default() string {
	return Format(DefaultSpec())
}

// DefaultSpec returns the current platform's default platform specification.
func DefaultSpec() specs.Platform {
	return specs.Platform{
		OS:           runtime.GOOS,
		Architecture: runtime.GOARCH,
		// TODO(stevvooe): Need to resolve GOARM for arm hosts.
	}
}
