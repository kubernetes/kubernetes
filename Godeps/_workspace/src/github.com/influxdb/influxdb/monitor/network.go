package monitor

import (
	"os"
)

// network captures network diagnostics
type network struct{}

func (n *network) Diagnostics() (*Diagnostic, error) {
	h, err := os.Hostname()
	if err != nil {
		return nil, err
	}

	diagnostics := map[string]interface{}{
		"hostname": h,
	}

	return DiagnosticFromMap(diagnostics), nil
}
