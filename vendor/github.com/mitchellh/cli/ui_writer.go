package cli

// UiWriter is an io.Writer implementation that can be used with
// loggers that writes every line of log output data to a Ui at the
// Info level.
type UiWriter struct {
	Ui Ui
}

func (w *UiWriter) Write(p []byte) (n int, err error) {
	n = len(p)
	if n > 0 && p[n-1] == '\n' {
		p = p[:n-1]
	}

	w.Ui.Info(string(p))
	return n, nil
}
