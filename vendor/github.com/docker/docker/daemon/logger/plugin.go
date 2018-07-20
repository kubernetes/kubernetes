package logger

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/docker/docker/api/types/plugins/logdriver"
	getter "github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/docker/pkg/stringid"
	"github.com/pkg/errors"
)

var pluginGetter getter.PluginGetter

const extName = "LogDriver"

// logPlugin defines the available functions that logging plugins must implement.
type logPlugin interface {
	StartLogging(streamPath string, info Info) (err error)
	StopLogging(streamPath string) (err error)
	Capabilities() (cap Capability, err error)
	ReadLogs(info Info, config ReadConfig) (stream io.ReadCloser, err error)
}

// RegisterPluginGetter sets the plugingetter
func RegisterPluginGetter(plugingetter getter.PluginGetter) {
	pluginGetter = plugingetter
}

// GetDriver returns a logging driver by its name.
// If the driver is empty, it looks for the local driver.
func getPlugin(name string, mode int) (Creator, error) {
	p, err := pluginGetter.Get(name, extName, mode)
	if err != nil {
		return nil, fmt.Errorf("error looking up logging plugin %s: %v", name, err)
	}

	d := &logPluginProxy{p.Client()}
	return makePluginCreator(name, d, p.BasePath()), nil
}

func makePluginCreator(name string, l *logPluginProxy, basePath string) Creator {
	return func(logCtx Info) (logger Logger, err error) {
		defer func() {
			if err != nil {
				pluginGetter.Get(name, extName, getter.Release)
			}
		}()
		root := filepath.Join(basePath, "run", "docker", "logging")
		if err := os.MkdirAll(root, 0700); err != nil {
			return nil, err
		}

		id := stringid.GenerateNonCryptoID()
		a := &pluginAdapter{
			driverName: name,
			id:         id,
			plugin:     l,
			basePath:   basePath,
			fifoPath:   filepath.Join(root, id),
			logInfo:    logCtx,
		}

		cap, err := a.plugin.Capabilities()
		if err == nil {
			a.capabilities = cap
		}

		stream, err := openPluginStream(a)
		if err != nil {
			return nil, err
		}

		a.stream = stream
		a.enc = logdriver.NewLogEntryEncoder(a.stream)

		if err := l.StartLogging(strings.TrimPrefix(a.fifoPath, basePath), logCtx); err != nil {
			return nil, errors.Wrapf(err, "error creating logger")
		}

		if cap.ReadLogs {
			return &pluginAdapterWithRead{a}, nil
		}

		return a, nil
	}
}
