package monitor

import (
	"os"
	"time"
)

var startTime time.Time

func init() {
	startTime = time.Now().UTC()
}

// system captures system-level diagnostics
type system struct{}

func (s *system) Diagnostics() (*Diagnostic, error) {
	diagnostics := map[string]interface{}{
		"PID":         os.Getpid(),
		"currentTime": time.Now().UTC(),
		"started":     startTime,
		"uptime":      time.Since(startTime).String(),
	}

	return DiagnosticFromMap(diagnostics), nil
}
