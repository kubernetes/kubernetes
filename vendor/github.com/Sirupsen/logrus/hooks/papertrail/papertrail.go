package logrus_papertrail

import (
	"fmt"
	"net"
	"os"
	"time"

	"github.com/Sirupsen/logrus"
)

const (
	format = "Jan 2 15:04:05"
)

// PapertrailHook to send logs to a logging service compatible with the Papertrail API.
type PapertrailHook struct {
	Host    string
	Port    int
	AppName string
	UDPConn net.Conn
}

// NewPapertrailHook creates a hook to be added to an instance of logger.
func NewPapertrailHook(host string, port int, appName string) (*PapertrailHook, error) {
	conn, err := net.Dial("udp", fmt.Sprintf("%s:%d", host, port))
	return &PapertrailHook{host, port, appName, conn}, err
}

// Fire is called when a log event is fired.
func (hook *PapertrailHook) Fire(entry *logrus.Entry) error {
	date := time.Now().Format(format)
	payload := fmt.Sprintf("<22> %s %s: [%s] %s", date, hook.AppName, entry.Level, entry.Message)

	bytesWritten, err := hook.UDPConn.Write([]byte(payload))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Unable to send log line to Papertrail via UDP. Wrote %d bytes before error: %v", bytesWritten, err)
		return err
	}

	return nil
}

// Levels returns the available logging levels.
func (hook *PapertrailHook) Levels() []logrus.Level {
	return []logrus.Level{
		logrus.PanicLevel,
		logrus.FatalLevel,
		logrus.ErrorLevel,
		logrus.WarnLevel,
		logrus.InfoLevel,
		logrus.DebugLevel,
	}
}
