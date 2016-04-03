// +build linux

package gelf

import (
	"bytes"
	"fmt"
	"net"
	"net/url"
	"time"

	"github.com/Graylog2/go-gelf/gelf"
	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/logger"
	"github.com/docker/docker/pkg/urlutil"
)

const name = "gelf"

type GelfLogger struct {
	writer *gelf.Writer
	ctx    logger.Context
	fields GelfFields
}

type GelfFields struct {
	hostname      string
	containerId   string
	containerName string
	imageId       string
	imageName     string
	command       string
	tag           string
	created       time.Time
}

func init() {
	if err := logger.RegisterLogDriver(name, New); err != nil {
		logrus.Fatal(err)
	}
	if err := logger.RegisterLogOptValidator(name, ValidateLogOpt); err != nil {
		logrus.Fatal(err)
	}
}

func New(ctx logger.Context) (logger.Logger, error) {
	// parse gelf address
	address, err := parseAddress(ctx.Config["gelf-address"])
	if err != nil {
		return nil, err
	}

	// collect extra data for GELF message
	hostname, err := ctx.Hostname()
	if err != nil {
		return nil, fmt.Errorf("gelf: cannot access hostname to set source field")
	}

	// remove trailing slash from container name
	containerName := bytes.TrimLeft([]byte(ctx.ContainerName), "/")

	fields := GelfFields{
		hostname:      hostname,
		containerId:   ctx.ContainerID,
		containerName: string(containerName),
		imageId:       ctx.ContainerImageID,
		imageName:     ctx.ContainerImageName,
		command:       ctx.Command(),
		tag:           ctx.Config["gelf-tag"],
		created:       ctx.ContainerCreated,
	}

	// create new gelfWriter
	gelfWriter, err := gelf.NewWriter(address)
	if err != nil {
		return nil, fmt.Errorf("gelf: cannot connect to GELF endpoint: %s %v", address, err)
	}

	return &GelfLogger{
		writer: gelfWriter,
		ctx:    ctx,
		fields: fields,
	}, nil
}

func (s *GelfLogger) Log(msg *logger.Message) error {
	// remove trailing and leading whitespace
	short := bytes.TrimSpace([]byte(msg.Line))

	level := gelf.LOG_INFO
	if msg.Source == "stderr" {
		level = gelf.LOG_ERR
	}

	m := gelf.Message{
		Version:  "1.1",
		Host:     s.fields.hostname,
		Short:    string(short),
		TimeUnix: float64(msg.Timestamp.UnixNano()/int64(time.Millisecond)) / 1000.0,
		Level:    level,
		Extra: map[string]interface{}{
			"_container_id":   s.fields.containerId,
			"_container_name": s.fields.containerName,
			"_image_id":       s.fields.imageId,
			"_image_name":     s.fields.imageName,
			"_command":        s.fields.command,
			"_tag":            s.fields.tag,
			"_created":        s.fields.created,
		},
	}

	if err := s.writer.WriteMessage(&m); err != nil {
		return fmt.Errorf("gelf: cannot send GELF message: %v", err)
	}
	return nil
}

func (s *GelfLogger) Close() error {
	return s.writer.Close()
}

func (s *GelfLogger) Name() string {
	return name
}

func ValidateLogOpt(cfg map[string]string) error {
	for key := range cfg {
		switch key {
		case "gelf-address":
		case "gelf-tag":
		default:
			return fmt.Errorf("unknown log opt '%s' for gelf log driver", key)
		}
	}
	return nil
}

func parseAddress(address string) (string, error) {
	if urlutil.IsTransportURL(address) {
		url, err := url.Parse(address)
		if err != nil {
			return "", err
		}

		// we support only udp
		if url.Scheme != "udp" {
			return "", fmt.Errorf("gelf: endpoint needs to be UDP")
		}

		// get host and port
		if _, _, err = net.SplitHostPort(url.Host); err != nil {
			return "", fmt.Errorf("gelf: please provide gelf-address as udp://host:port")
		}

		return url.Host, nil
	}

	return "", nil
}
