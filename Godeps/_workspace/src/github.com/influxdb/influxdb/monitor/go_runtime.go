package monitor

import (
	"runtime"
)

// goRuntime captures Go runtime diagnostics
type goRuntime struct{}

func (g *goRuntime) Diagnostics() (*Diagnostic, error) {
	diagnostics := map[string]interface{}{
		"GOARCH":     runtime.GOARCH,
		"GOOS":       runtime.GOOS,
		"GOMAXPROCS": runtime.GOMAXPROCS(-1),
		"version":    runtime.Version(),
	}

	return DiagnosticFromMap(diagnostics), nil
}
