package proxy

type logger interface {
	Printf(format string, args ...interface{})
}

type noopLogger struct{}

func (l *noopLogger) Printf(_ string, _ ...interface{}) {
	// Do nothing :)
}
