package monitor

// system captures build diagnostics
type build struct {
	Version string
	Commit  string
	Branch  string
	Time    string
}

func (b *build) Diagnostics() (*Diagnostic, error) {
	diagnostics := map[string]interface{}{
		"Version":    b.Version,
		"Commit":     b.Commit,
		"Branch":     b.Branch,
		"Build Time": b.Time,
	}

	return DiagnosticFromMap(diagnostics), nil
}
