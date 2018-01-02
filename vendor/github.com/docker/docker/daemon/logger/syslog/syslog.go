// Package syslog provides the logdriver for forwarding server logs to syslog endpoints.
package syslog

import (
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	syslog "github.com/RackSec/srslog"

	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/daemon/logger/loggerutils"
	"github.com/docker/docker/pkg/urlutil"
	"github.com/docker/go-connections/tlsconfig"
	"github.com/sirupsen/logrus"
)

const (
	name        = "syslog"
	secureProto = "tcp+tls"
)

var facilities = map[string]syslog.Priority{
	"kern":     syslog.LOG_KERN,
	"user":     syslog.LOG_USER,
	"mail":     syslog.LOG_MAIL,
	"daemon":   syslog.LOG_DAEMON,
	"auth":     syslog.LOG_AUTH,
	"syslog":   syslog.LOG_SYSLOG,
	"lpr":      syslog.LOG_LPR,
	"news":     syslog.LOG_NEWS,
	"uucp":     syslog.LOG_UUCP,
	"cron":     syslog.LOG_CRON,
	"authpriv": syslog.LOG_AUTHPRIV,
	"ftp":      syslog.LOG_FTP,
	"local0":   syslog.LOG_LOCAL0,
	"local1":   syslog.LOG_LOCAL1,
	"local2":   syslog.LOG_LOCAL2,
	"local3":   syslog.LOG_LOCAL3,
	"local4":   syslog.LOG_LOCAL4,
	"local5":   syslog.LOG_LOCAL5,
	"local6":   syslog.LOG_LOCAL6,
	"local7":   syslog.LOG_LOCAL7,
}

type syslogger struct {
	writer *syslog.Writer
}

func init() {
	if err := logger.RegisterLogDriver(name, New); err != nil {
		logrus.Fatal(err)
	}
	if err := logger.RegisterLogOptValidator(name, ValidateLogOpt); err != nil {
		logrus.Fatal(err)
	}
}

// rsyslog uses appname part of syslog message to fill in an %syslogtag% template
// attribute in rsyslog.conf. In order to be backward compatible to rfc3164
// tag will be also used as an appname
func rfc5424formatterWithAppNameAsTag(p syslog.Priority, hostname, tag, content string) string {
	timestamp := time.Now().Format(time.RFC3339)
	pid := os.Getpid()
	msg := fmt.Sprintf("<%d>%d %s %s %s %d %s - %s",
		p, 1, timestamp, hostname, tag, pid, tag, content)
	return msg
}

// The timestamp field in rfc5424 is derived from rfc3339. Whereas rfc3339 makes allowances
// for multiple syntaxes, there are further restrictions in rfc5424, i.e., the maximum
// resolution is limited to "TIME-SECFRAC" which is 6 (microsecond resolution)
func rfc5424microformatterWithAppNameAsTag(p syslog.Priority, hostname, tag, content string) string {
	timestamp := time.Now().Format("2006-01-02T15:04:05.999999Z07:00")
	pid := os.Getpid()
	msg := fmt.Sprintf("<%d>%d %s %s %s %d %s - %s",
		p, 1, timestamp, hostname, tag, pid, tag, content)
	return msg
}

// New creates a syslog logger using the configuration passed in on
// the context. Supported context configuration variables are
// syslog-address, syslog-facility, syslog-format.
func New(info logger.Info) (logger.Logger, error) {
	tag, err := loggerutils.ParseLogTag(info, loggerutils.DefaultTemplate)
	if err != nil {
		return nil, err
	}

	proto, address, err := parseAddress(info.Config["syslog-address"])
	if err != nil {
		return nil, err
	}

	facility, err := parseFacility(info.Config["syslog-facility"])
	if err != nil {
		return nil, err
	}

	syslogFormatter, syslogFramer, err := parseLogFormat(info.Config["syslog-format"], proto)
	if err != nil {
		return nil, err
	}

	var log *syslog.Writer
	if proto == secureProto {
		tlsConfig, tlsErr := parseTLSConfig(info.Config)
		if tlsErr != nil {
			return nil, tlsErr
		}
		log, err = syslog.DialWithTLSConfig(proto, address, facility, tag, tlsConfig)
	} else {
		log, err = syslog.Dial(proto, address, facility, tag)
	}

	if err != nil {
		return nil, err
	}

	log.SetFormatter(syslogFormatter)
	log.SetFramer(syslogFramer)

	return &syslogger{
		writer: log,
	}, nil
}

func (s *syslogger) Log(msg *logger.Message) error {
	line := string(msg.Line)
	source := msg.Source
	logger.PutMessage(msg)
	if source == "stderr" {
		return s.writer.Err(line)
	}
	return s.writer.Info(line)
}

func (s *syslogger) Close() error {
	return s.writer.Close()
}

func (s *syslogger) Name() string {
	return name
}

func parseAddress(address string) (string, string, error) {
	if address == "" {
		return "", "", nil
	}
	if !urlutil.IsTransportURL(address) {
		return "", "", fmt.Errorf("syslog-address should be in form proto://address, got %v", address)
	}
	url, err := url.Parse(address)
	if err != nil {
		return "", "", err
	}

	// unix and unixgram socket validation
	if url.Scheme == "unix" || url.Scheme == "unixgram" {
		if _, err := os.Stat(url.Path); err != nil {
			return "", "", err
		}
		return url.Scheme, url.Path, nil
	}

	// here we process tcp|udp
	host := url.Host
	if _, _, err := net.SplitHostPort(host); err != nil {
		if !strings.Contains(err.Error(), "missing port in address") {
			return "", "", err
		}
		host = host + ":514"
	}

	return url.Scheme, host, nil
}

// ValidateLogOpt looks for syslog specific log options
// syslog-address, syslog-facility.
func ValidateLogOpt(cfg map[string]string) error {
	for key := range cfg {
		switch key {
		case "env":
		case "env-regex":
		case "labels":
		case "syslog-address":
		case "syslog-facility":
		case "syslog-tls-ca-cert":
		case "syslog-tls-cert":
		case "syslog-tls-key":
		case "syslog-tls-skip-verify":
		case "tag":
		case "syslog-format":
		default:
			return fmt.Errorf("unknown log opt '%s' for syslog log driver", key)
		}
	}
	if _, _, err := parseAddress(cfg["syslog-address"]); err != nil {
		return err
	}
	if _, err := parseFacility(cfg["syslog-facility"]); err != nil {
		return err
	}
	if _, _, err := parseLogFormat(cfg["syslog-format"], ""); err != nil {
		return err
	}
	return nil
}

func parseFacility(facility string) (syslog.Priority, error) {
	if facility == "" {
		return syslog.LOG_DAEMON, nil
	}

	if syslogFacility, valid := facilities[facility]; valid {
		return syslogFacility, nil
	}

	fInt, err := strconv.Atoi(facility)
	if err == nil && 0 <= fInt && fInt <= 23 {
		return syslog.Priority(fInt << 3), nil
	}

	return syslog.Priority(0), errors.New("invalid syslog facility")
}

func parseTLSConfig(cfg map[string]string) (*tls.Config, error) {
	_, skipVerify := cfg["syslog-tls-skip-verify"]

	opts := tlsconfig.Options{
		CAFile:             cfg["syslog-tls-ca-cert"],
		CertFile:           cfg["syslog-tls-cert"],
		KeyFile:            cfg["syslog-tls-key"],
		InsecureSkipVerify: skipVerify,
	}

	return tlsconfig.Client(opts)
}

func parseLogFormat(logFormat, proto string) (syslog.Formatter, syslog.Framer, error) {
	switch logFormat {
	case "":
		return syslog.UnixFormatter, syslog.DefaultFramer, nil
	case "rfc3164":
		return syslog.RFC3164Formatter, syslog.DefaultFramer, nil
	case "rfc5424":
		if proto == secureProto {
			return rfc5424formatterWithAppNameAsTag, syslog.RFC5425MessageLengthFramer, nil
		}
		return rfc5424formatterWithAppNameAsTag, syslog.DefaultFramer, nil
	case "rfc5424micro":
		if proto == secureProto {
			return rfc5424microformatterWithAppNameAsTag, syslog.RFC5425MessageLengthFramer, nil
		}
		return rfc5424microformatterWithAppNameAsTag, syslog.DefaultFramer, nil
	default:
		return nil, nil, errors.New("Invalid syslog format")
	}

}
