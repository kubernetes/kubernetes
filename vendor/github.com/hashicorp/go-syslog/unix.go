// +build linux darwin freebsd openbsd solaris

package gsyslog

import (
	"fmt"
	"log/syslog"
	"strings"
)

// builtinLogger wraps the Golang implementation of a
// syslog.Writer to provide the Syslogger interface
type builtinLogger struct {
	*builtinWriter
}

// NewLogger is used to construct a new Syslogger
func NewLogger(p Priority, facility, tag string) (Syslogger, error) {
	fPriority, err := facilityPriority(facility)
	if err != nil {
		return nil, err
	}
	priority := syslog.Priority(p) | fPriority
	l, err := newBuiltin(priority, tag)
	if err != nil {
		return nil, err
	}
	return &builtinLogger{l}, nil
}

// WriteLevel writes out a message at the given priority
func (b *builtinLogger) WriteLevel(p Priority, buf []byte) error {
	var err error
	m := string(buf)
	switch p {
	case LOG_EMERG:
		_, err = b.writeAndRetry(syslog.LOG_EMERG, m)
	case LOG_ALERT:
		_, err = b.writeAndRetry(syslog.LOG_ALERT, m)
	case LOG_CRIT:
		_, err = b.writeAndRetry(syslog.LOG_CRIT, m)
	case LOG_ERR:
		_, err = b.writeAndRetry(syslog.LOG_ERR, m)
	case LOG_WARNING:
		_, err = b.writeAndRetry(syslog.LOG_WARNING, m)
	case LOG_NOTICE:
		_, err = b.writeAndRetry(syslog.LOG_NOTICE, m)
	case LOG_INFO:
		_, err = b.writeAndRetry(syslog.LOG_INFO, m)
	case LOG_DEBUG:
		_, err = b.writeAndRetry(syslog.LOG_DEBUG, m)
	default:
		err = fmt.Errorf("Unknown priority: %v", p)
	}
	return err
}

// facilityPriority converts a facility string into
// an appropriate priority level or returns an error
func facilityPriority(facility string) (syslog.Priority, error) {
	facility = strings.ToUpper(facility)
	switch facility {
	case "KERN":
		return syslog.LOG_KERN, nil
	case "USER":
		return syslog.LOG_USER, nil
	case "MAIL":
		return syslog.LOG_MAIL, nil
	case "DAEMON":
		return syslog.LOG_DAEMON, nil
	case "AUTH":
		return syslog.LOG_AUTH, nil
	case "SYSLOG":
		return syslog.LOG_SYSLOG, nil
	case "LPR":
		return syslog.LOG_LPR, nil
	case "NEWS":
		return syslog.LOG_NEWS, nil
	case "UUCP":
		return syslog.LOG_UUCP, nil
	case "CRON":
		return syslog.LOG_CRON, nil
	case "AUTHPRIV":
		return syslog.LOG_AUTHPRIV, nil
	case "FTP":
		return syslog.LOG_FTP, nil
	case "LOCAL0":
		return syslog.LOG_LOCAL0, nil
	case "LOCAL1":
		return syslog.LOG_LOCAL1, nil
	case "LOCAL2":
		return syslog.LOG_LOCAL2, nil
	case "LOCAL3":
		return syslog.LOG_LOCAL3, nil
	case "LOCAL4":
		return syslog.LOG_LOCAL4, nil
	case "LOCAL5":
		return syslog.LOG_LOCAL5, nil
	case "LOCAL6":
		return syslog.LOG_LOCAL6, nil
	case "LOCAL7":
		return syslog.LOG_LOCAL7, nil
	default:
		return 0, fmt.Errorf("invalid syslog facility: %s", facility)
	}
}
