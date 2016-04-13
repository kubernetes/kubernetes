package logger

import (
	"errors"
	"io"
	"time"
)

var ReadLogsNotSupported = errors.New("configured logging reader does not support reading")

// Message is datastructure that represents record from some container
type Message struct {
	ContainerID string
	Line        []byte
	Source      string
	Timestamp   time.Time
}

// Logger is interface for docker logging drivers
type Logger interface {
	Log(*Message) error
	Name() string
	Close() error
}

//Reader is an interface for docker logging drivers that support reading
type Reader interface {
	ReadLog(args ...string) (io.Reader, error)
}
