// +build !windows

package daemon

import (
	"net"
	"net/http"
	"os"
	"path/filepath"

	"github.com/docker/docker/pkg/mount"
	"github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/docker/pkg/plugins"
	metrics "github.com/docker/go-metrics"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

func (daemon *Daemon) listenMetricsSock() (string, error) {
	path := filepath.Join(daemon.configStore.ExecRoot, "metrics.sock")
	unix.Unlink(path)
	l, err := net.Listen("unix", path)
	if err != nil {
		return "", errors.Wrap(err, "error setting up metrics plugin listener")
	}

	mux := http.NewServeMux()
	mux.Handle("/metrics", metrics.Handler())
	go func() {
		http.Serve(l, mux)
	}()
	daemon.metricsPluginListener = l
	return path, nil
}

func registerMetricsPluginCallback(getter plugingetter.PluginGetter, sockPath string) {
	getter.Handle(metricsPluginType, func(name string, client *plugins.Client) {
		// Use lookup since nothing in the system can really reference it, no need
		// to protect against removal
		p, err := getter.Get(name, metricsPluginType, plugingetter.Lookup)
		if err != nil {
			return
		}

		mp := metricsPlugin{p}
		sockBase := mp.sockBase()
		if err := os.MkdirAll(sockBase, 0755); err != nil {
			logrus.WithError(err).WithField("name", name).WithField("path", sockBase).Error("error creating metrics plugin base path")
			return
		}

		defer func() {
			if err != nil {
				os.RemoveAll(sockBase)
			}
		}()

		pluginSockPath := filepath.Join(sockBase, mp.sock())
		_, err = os.Stat(pluginSockPath)
		if err == nil {
			mount.Unmount(pluginSockPath)
		} else {
			logrus.WithField("path", pluginSockPath).Debugf("creating plugin socket")
			f, err := os.OpenFile(pluginSockPath, os.O_CREATE, 0600)
			if err != nil {
				return
			}
			f.Close()
		}

		if err := mount.Mount(sockPath, pluginSockPath, "none", "bind,ro"); err != nil {
			logrus.WithError(err).WithField("name", name).Error("could not mount metrics socket to plugin")
			return
		}

		if err := pluginStartMetricsCollection(p); err != nil {
			if err := mount.Unmount(pluginSockPath); err != nil {
				if mounted, _ := mount.Mounted(pluginSockPath); mounted {
					logrus.WithError(err).WithField("sock_path", pluginSockPath).Error("error unmounting metrics socket from plugin during cleanup")
				}
			}
			logrus.WithError(err).WithField("name", name).Error("error while initializing metrics plugin")
		}
	})
}
