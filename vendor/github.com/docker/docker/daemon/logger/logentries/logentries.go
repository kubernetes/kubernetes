// Package logentries provides the log driver for forwarding server logs
// to logentries endpoints.
package logentries

import (
	"fmt"

	"github.com/bsphere/le_go"
	"github.com/docker/docker/daemon/logger"
	"github.com/sirupsen/logrus"
)

type logentries struct {
	tag           string
	containerID   string
	containerName string
	writer        *le_go.Logger
	extra         map[string]string
}

const (
	name  = "logentries"
	token = "logentries-token"
)

func init() {
	if err := logger.RegisterLogDriver(name, New); err != nil {
		logrus.Fatal(err)
	}
	if err := logger.RegisterLogOptValidator(name, ValidateLogOpt); err != nil {
		logrus.Fatal(err)
	}
}

// New creates a logentries logger using the configuration passed in on
// the context. The supported context configuration variable is
// logentries-token.
func New(info logger.Info) (logger.Logger, error) {
	logrus.WithField("container", info.ContainerID).
		WithField("token", info.Config[token]).
		Debug("logging driver logentries configured")

	log, err := le_go.Connect(info.Config[token])
	if err != nil {
		return nil, err
	}
	return &logentries{
		containerID:   info.ContainerID,
		containerName: info.ContainerName,
		writer:        log,
	}, nil
}

func (f *logentries) Log(msg *logger.Message) error {
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
	f.writer.Println(f.tag, ts, data)
	return nil
}

func (f *logentries) Close() error {
	return f.writer.Close()
}

func (f *logentries) Name() string {
	return name
}

// ValidateLogOpt looks for logentries specific log option logentries-address.
func ValidateLogOpt(cfg map[string]string) error {
	for key := range cfg {
		switch key {
		case "env":
		case "env-regex":
		case "labels":
		case "tag":
		case key:
		default:
			return fmt.Errorf("unknown log opt '%s' for logentries log driver", key)
		}
	}

	if cfg[token] == "" {
		return fmt.Errorf("Missing logentries token")
	}

	return nil
}
