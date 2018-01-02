// Package fluentd provides the log driver for forwarding server logs
// to fluentd endpoints.
package fluentd

import (
	"fmt"
	"math"
	"net"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/daemon/logger/loggerutils"
	"github.com/docker/docker/pkg/urlutil"
	"github.com/docker/go-units"
	"github.com/fluent/fluent-logger-golang/fluent"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

type fluentd struct {
	tag           string
	containerID   string
	containerName string
	writer        *fluent.Fluent
	extra         map[string]string
}

type location struct {
	protocol string
	host     string
	port     int
	path     string
}

const (
	name = "fluentd"

	defaultProtocol    = "tcp"
	defaultHost        = "127.0.0.1"
	defaultPort        = 24224
	defaultBufferLimit = 1024 * 1024

	// logger tries to reconnect 2**32 - 1 times
	// failed (and panic) after 204 years [ 1.5 ** (2**32 - 1) - 1 seconds]
	defaultRetryWait  = 1000
	defaultMaxRetries = math.MaxInt32

	addressKey      = "fluentd-address"
	bufferLimitKey  = "fluentd-buffer-limit"
	retryWaitKey    = "fluentd-retry-wait"
	maxRetriesKey   = "fluentd-max-retries"
	asyncConnectKey = "fluentd-async-connect"
)

func init() {
	if err := logger.RegisterLogDriver(name, New); err != nil {
		logrus.Fatal(err)
	}
	if err := logger.RegisterLogOptValidator(name, ValidateLogOpt); err != nil {
		logrus.Fatal(err)
	}
}

// New creates a fluentd logger using the configuration passed in on
// the context. The supported context configuration variable is
// fluentd-address.
func New(info logger.Info) (logger.Logger, error) {
	loc, err := parseAddress(info.Config[addressKey])
	if err != nil {
		return nil, err
	}

	tag, err := loggerutils.ParseLogTag(info, loggerutils.DefaultTemplate)
	if err != nil {
		return nil, err
	}

	extra, err := info.ExtraAttributes(nil)
	if err != nil {
		return nil, err
	}

	bufferLimit := defaultBufferLimit
	if info.Config[bufferLimitKey] != "" {
		bl64, err := units.RAMInBytes(info.Config[bufferLimitKey])
		if err != nil {
			return nil, err
		}
		bufferLimit = int(bl64)
	}

	retryWait := defaultRetryWait
	if info.Config[retryWaitKey] != "" {
		rwd, err := time.ParseDuration(info.Config[retryWaitKey])
		if err != nil {
			return nil, err
		}
		retryWait = int(rwd.Seconds() * 1000)
	}

	maxRetries := defaultMaxRetries
	if info.Config[maxRetriesKey] != "" {
		mr64, err := strconv.ParseUint(info.Config[maxRetriesKey], 10, strconv.IntSize)
		if err != nil {
			return nil, err
		}
		maxRetries = int(mr64)
	}

	asyncConnect := false
	if info.Config[asyncConnectKey] != "" {
		if asyncConnect, err = strconv.ParseBool(info.Config[asyncConnectKey]); err != nil {
			return nil, err
		}
	}

	fluentConfig := fluent.Config{
		FluentPort:       loc.port,
		FluentHost:       loc.host,
		FluentNetwork:    loc.protocol,
		FluentSocketPath: loc.path,
		BufferLimit:      bufferLimit,
		RetryWait:        retryWait,
		MaxRetry:         maxRetries,
		AsyncConnect:     asyncConnect,
	}

	logrus.WithField("container", info.ContainerID).WithField("config", fluentConfig).
		Debug("logging driver fluentd configured")

	log, err := fluent.New(fluentConfig)
	if err != nil {
		return nil, err
	}
	return &fluentd{
		tag:           tag,
		containerID:   info.ContainerID,
		containerName: info.ContainerName,
		writer:        log,
		extra:         extra,
	}, nil
}

func (f *fluentd) Log(msg *logger.Message) error {
	data := map[string]string{
		"container_id":   f.containerID,
		"container_name": f.containerName,
		"source":         msg.Source,
		"log":            string(msg.Line),
	}
	for k, v := range f.extra {
		data[k] = v
	}

	ts := msg.Timestamp
	logger.PutMessage(msg)
	// fluent-logger-golang buffers logs from failures and disconnections,
	// and these are transferred again automatically.
	return f.writer.PostWithTime(f.tag, ts, data)
}

func (f *fluentd) Close() error {
	return f.writer.Close()
}

func (f *fluentd) Name() string {
	return name
}

// ValidateLogOpt looks for fluentd specific log option fluentd-address.
func ValidateLogOpt(cfg map[string]string) error {
	for key := range cfg {
		switch key {
		case "env":
		case "env-regex":
		case "labels":
		case "tag":
		case addressKey:
		case bufferLimitKey:
		case retryWaitKey:
		case maxRetriesKey:
		case asyncConnectKey:
			// Accepted
		default:
			return fmt.Errorf("unknown log opt '%s' for fluentd log driver", key)
		}
	}

	_, err := parseAddress(cfg[addressKey])
	return err
}

func parseAddress(address string) (*location, error) {
	if address == "" {
		return &location{
			protocol: defaultProtocol,
			host:     defaultHost,
			port:     defaultPort,
			path:     "",
		}, nil
	}

	protocol := defaultProtocol
	givenAddress := address
	if urlutil.IsTransportURL(address) {
		url, err := url.Parse(address)
		if err != nil {
			return nil, errors.Wrapf(err, "invalid fluentd-address %s", givenAddress)
		}
		// unix and unixgram socket
		if url.Scheme == "unix" || url.Scheme == "unixgram" {
			return &location{
				protocol: url.Scheme,
				host:     "",
				port:     0,
				path:     url.Path,
			}, nil
		}
		// tcp|udp
		protocol = url.Scheme
		address = url.Host
	}

	host, port, err := net.SplitHostPort(address)
	if err != nil {
		if !strings.Contains(err.Error(), "missing port in address") {
			return nil, errors.Wrapf(err, "invalid fluentd-address %s", givenAddress)
		}
		return &location{
			protocol: protocol,
			host:     host,
			port:     defaultPort,
			path:     "",
		}, nil
	}

	portnum, err := strconv.Atoi(port)
	if err != nil {
		return nil, errors.Wrapf(err, "invalid fluentd-address %s", givenAddress)
	}
	return &location{
		protocol: protocol,
		host:     host,
		port:     portnum,
		path:     "",
	}, nil
}
