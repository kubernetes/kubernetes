package goproxy

type Logger interface {
	Printf(format string, v ...interface{})
}
