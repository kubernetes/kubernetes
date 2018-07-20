// Package gelf provides the log driver for forwarding server logs to
// endpoints that support the Graylog Extended Log Format.
package gelf

import (
	"compress/flate"
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"strconv"
	"time"

	"github.com/Graylog2/go-gelf/gelf"
	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/daemon/logger/loggerutils"
	"github.com/docker/docker/pkg/urlutil"
	"github.com/sirupsen/logrus"
)

const name = "gelf"

type gelfLogger struct {
	writer   gelf.Writer
	info     logger.Info
	hostname string
	rawExtra json.RawMessage
}

func init() {
	if err := logger.RegisterLogDriver(name, New); err != nil {
		logrus.Fatal(err)
	}
	if err := logger.RegisterLogOptValidator(name, ValidateLogOpt); err != nil {
		logrus.Fatal(err)
	}
}

// New creates a gelf logger using the configuration passed in on the
// context. The supported context configuration variable is gelf-address.
func New(info logger.Info) (logger.Logger, error) {
	// parse gelf address
	address, err := parseAddress(info.Config["gelf-address"])
	if err != nil {
		return nil, err
	}

	// collect extra data for GELF message
	hostname, err := info.Hostname()
	if err != nil {
		return nil, fmt.Errorf("gelf: cannot access hostname to set source field")
	}

	// parse log tag
	tag, err := loggerutils.ParseLogTag(info, loggerutils.DefaultTemplate)
	if err != nil {
		return nil, err
	}

	extra := map[string]interface{}{
		"_container_id":   info.ContainerID,
		"_container_name": info.Name(),
		"_image_id":       info.ContainerImageID,
		"_image_name":     info.ContainerImageName,
		"_command":        info.Command(),
		"_tag":            tag,
		"_created":        info.ContainerCreated,
	}

	extraAttrs, err := info.ExtraAttributes(func(key string) string {
		if key[0] == '_' {
			return key
		}
		return "_" + key
	})

	if err != nil {
		return nil, err
	}

	for k, v := range extraAttrs {
		extra[k] = v
	}

	rawExtra, err := json.Marshal(extra)
	if err != nil {
		return nil, err
	}

	var gelfWriter gelf.Writer
	if address.Scheme == "udp" {
		gelfWriter, err = newGELFUDPWriter(address.Host, info)
		if err != nil {
			return nil, err
		}
	} else if address.Scheme == "tcp" {
		gelfWriter, err = newGELFTCPWriter(address.Host, info)
		if err != nil {
			return nil, err
		}
	}

	return &gelfLogger{
		writer:   gelfWriter,
		info:     info,
		hostname: hostname,
		rawExtra: rawExtra,
	}, nil
}

// create new TCP gelfWriter
func newGELFTCPWriter(address string, info logger.Info) (gelf.Writer, error) {
	gelfWriter, err := gelf.NewTCPWriter(address)
	if err != nil {
		return nil, fmt.Errorf("gelf: cannot connect to GELF endpoint: %s %v", address, err)
	}

	if v, ok := info.Config["gelf-tcp-max-reconnect"]; ok {
		i, err := strconv.Atoi(v)
		if err != nil || i < 0 {
			return nil, fmt.Errorf("gelf-tcp-max-reconnect must be a positive integer")
		}
		gelfWriter.MaxReconnect = i
	}

	if v, ok := info.Config["gelf-tcp-reconnect-delay"]; ok {
		i, err := strconv.Atoi(v)
		if err != nil || i < 0 {
			return nil, fmt.Errorf("gelf-tcp-reconnect-delay must be a positive integer")
		}
		gelfWriter.ReconnectDelay = time.Duration(i)
	}

	return gelfWriter, nil
}

// create new UDP gelfWriter
func newGELFUDPWriter(address string, info logger.Info) (gelf.Writer, error) {
	gelfWriter, err := gelf.NewUDPWriter(address)
	if err != nil {
		return nil, fmt.Errorf("gelf: cannot connect to GELF endpoint: %s %v", address, err)
	}

	if v, ok := info.Config["gelf-compression-type"]; ok {
		switch v {
		case "gzip":
			gelfWriter.CompressionType = gelf.CompressGzip
		case "zlib":
			gelfWriter.CompressionType = gelf.CompressZlib
		case "none":
			gelfWriter.CompressionType = gelf.CompressNone
		default:
			return nil, fmt.Errorf("gelf: invalid compression type %q", v)
		}
	}

	if v, ok := info.Config["gelf-compression-level"]; ok {
		val, err := strconv.Atoi(v)
		if err != nil {
			return nil, fmt.Errorf("gelf: invalid compression level %s, err %v", v, err)
		}
		gelfWriter.CompressionLevel = val
	}

	return gelfWriter, nil
}

func (s *gelfLogger) Log(msg *logger.Message) error {
	level := gelf.LOG_INFO
	if msg.Source == "stderr" {
		level = gelf.LOG_ERR
	}

	m := gelf.Message{
		Version:  "1.1",
		Host:     s.hostname,
		Short:    string(msg.Line),
		TimeUnix: float64(msg.Timestamp.UnixNano()/int64(time.Millisecond)) / 1000.0,
		Level:    int32(level),
		RawExtra: s.rawExtra,
	}
	logger.PutMessage(msg)

	if err := s.writer.WriteMessage(&m); err != nil {
		return fmt.Errorf("gelf: cannot send GELF message: %v", err)
	}
	return nil
}

func (s *gelfLogger) Close() error {
	return s.writer.Close()
}

func (s *gelfLogger) Name() string {
	return name
}

// ValidateLogOpt looks for gelf specific log option gelf-address.
func ValidateLogOpt(cfg map[string]string) error {
	address, err := parseAddress(cfg["gelf-address"])
	if err != nil {
		return err
	}

	for key, val := range cfg {
		switch key {
		case "gelf-address":
		case "tag":
		case "labels":
		case "env":
		case "env-regex":
		case "gelf-compression-level":
			if address.Scheme != "udp" {
				return fmt.Errorf("compression is only supported on UDP")
			}
			i, err := strconv.Atoi(val)
			if err != nil || i < flate.DefaultCompression || i > flate.BestCompression {
				return fmt.Errorf("unknown value %q for log opt %q for gelf log driver", val, key)
			}
		case "gelf-compression-type":
			if address.Scheme != "udp" {
				return fmt.Errorf("compression is only supported on UDP")
			}
			switch val {
			case "gzip", "zlib", "none":
			default:
				return fmt.Errorf("unknown value %q for log opt %q for gelf log driver", val, key)
			}
		case "gelf-tcp-max-reconnect", "gelf-tcp-reconnect-delay":
			if address.Scheme != "tcp" {
				return fmt.Errorf("%q is only valid for TCP", key)
			}
			i, err := strconv.Atoi(val)
			if err != nil || i < 0 {
				return fmt.Errorf("%q must be a positive integer", key)
			}
		default:
			return fmt.Errorf("unknown log opt %q for gelf log driver", key)
		}
	}

	return nil
}

func parseAddress(address string) (*url.URL, error) {
	if address == "" {
		return nil, fmt.Errorf("gelf-address is a required parameter")
	}
	if !urlutil.IsTransportURL(address) {
		return nil, fmt.Errorf("gelf-address should be in form proto://address, got %v", address)
	}
	url, err := url.Parse(address)
	if err != nil {
		return nil, err
	}

	// we support only udp
	if url.Scheme != "udp" && url.Scheme != "tcp" {
		return nil, fmt.Errorf("gelf: endpoint needs to be TCP or UDP")
	}

	// get host and port
	if _, _, err = net.SplitHostPort(url.Host); err != nil {
		return nil, fmt.Errorf("gelf: please provide gelf-address as proto://host:port")
	}

	return url, nil
}
