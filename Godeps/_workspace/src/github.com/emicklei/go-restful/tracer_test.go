package restful

import "testing"

// Use like this:
//
// 		TraceLogger(testLogger{t})
type testLogger struct {
	t *testing.T
}

func (l testLogger) Print(v ...interface{}) {
	l.t.Log(v...)
}

func (l testLogger) Printf(format string, v ...interface{}) {
	l.t.Logf(format, v...)
}
