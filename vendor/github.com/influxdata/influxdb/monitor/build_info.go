package monitor

import "github.com/influxdata/influxdb/monitor/diagnostics"

// system captures build diagnostics
type build struct {
	Version string
	Commit  string
	Branch  string
	Time    string
}

func (b *build) Diagnostics() (*diagnostics.Diagnostics, error) {
	diagnostics := map[string]interface{}{
		"Version":    b.Version,
		"Commit":     b.Commit,
		"Branch":     b.Branch,
		"Build Time": b.Time,
	}

	return DiagnosticsFromMap(diagnostics), nil
}
