/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package pluginwatcher

import (
	"fmt"
	"net"
	"os"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/golang/glog"
	"github.com/pkg/errors"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	registerapi "k8s.io/kubernetes/pkg/kubelet/apis/pluginregistration/v1alpha1"
	utilfs "k8s.io/kubernetes/pkg/util/filesystem"
)

// RegisterCallbackFn is the type of the callback function that handlers will provide
type RegisterCallbackFn func(pluginName string, endpoint string, versions []string, socketPath string) (chan bool, error)

// Watcher is the plugin watcher
type Watcher struct {
	path      string
	handlers  map[string]RegisterCallbackFn
	stopCh    chan interface{}
	fs        utilfs.Filesystem
	fsWatcher *fsnotify.Watcher
	wg        sync.WaitGroup
	mutex     sync.Mutex
}

// NewWatcher provides a new watcher
func NewWatcher(sockDir string) Watcher {
	return Watcher{
		path:     sockDir,
		handlers: make(map[string]RegisterCallbackFn),
		fs:       &utilfs.DefaultFs{},
	}
}

// AddHandler registers a callback to be invoked for a particular type of plugin
func (w *Watcher) AddHandler(pluginType string, handlerCbkFn RegisterCallbackFn) {
	w.mutex.Lock()
	defer w.mutex.Unlock()
	w.handlers[pluginType] = handlerCbkFn
}

// Creates the plugin directory, if it doesn't already exist.
func (w *Watcher) createPluginDir() error {
	glog.V(4).Infof("Ensuring Plugin directory at %s ", w.path)
	if err := w.fs.MkdirAll(w.path, 0755); err != nil {
		return fmt.Errorf("error (re-)creating root %s: %v", w.path, err)
	}

	return nil
}

// Walks through the plugin directory discover any existing plugin sockets.
func (w *Watcher) traversePluginDir(dir string) error {
	return w.fs.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return fmt.Errorf("error accessing path: %s error: %v", path, err)
		}

		switch mode := info.Mode(); {
		case mode.IsDir():
			if err := w.fsWatcher.Add(path); err != nil {
				return fmt.Errorf("failed to watch %s, err: %v", path, err)
			}
		case mode&os.ModeSocket != 0:
			go func() {
				w.fsWatcher.Events <- fsnotify.Event{
					Name: path,
					Op:   fsnotify.Create,
				}
			}()
		}

		return nil
	})
}

func (w *Watcher) init() error {
	if err := w.createPluginDir(); err != nil {
		return err
	}
	return nil
}

func (w *Watcher) registerPlugin(socketPath string) error {
	//TODO: Implement rate limiting to mitigate any DOS kind of attacks.
	client, conn, err := dial(socketPath)
	if err != nil {
		return fmt.Errorf("dial failed at socket %s, err: %v", socketPath, err)
	}
	defer conn.Close()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	infoResp, err := client.GetInfo(ctx, &registerapi.InfoRequest{})
	if err != nil {
		return fmt.Errorf("failed to get plugin info using RPC GetInfo at socket %s, err: %v", socketPath, err)
	}

	return w.invokeRegistrationCallbackAtHandler(ctx, client, infoResp, socketPath)
}

func (w *Watcher) invokeRegistrationCallbackAtHandler(ctx context.Context, client registerapi.RegistrationClient, infoResp *registerapi.PluginInfo, socketPath string) error {
	var handlerCbkFn RegisterCallbackFn
	var ok bool
	handlerCbkFn, ok = w.handlers[infoResp.Type]
	if !ok {
		errStr := fmt.Sprintf("no handler registered for plugin type: %s at socket %s", infoResp.Type, socketPath)
		if _, err := client.NotifyRegistrationStatus(ctx, &registerapi.RegistrationStatus{
			PluginRegistered: false,
			Error:            errStr,
		}); err != nil {
			return errors.Wrap(err, errStr)
		}
		return errors.New(errStr)
	}

	var versions []string
	for _, version := range infoResp.SupportedVersions {
		versions = append(versions, version)
	}
	// calls handler callback to verify registration request
	chanForAckOfNotification, err := handlerCbkFn(infoResp.Name, infoResp.Endpoint, versions, socketPath)
	if err != nil {
		errStr := fmt.Sprintf("plugin registration failed with err: %v", err)
		if _, err := client.NotifyRegistrationStatus(ctx, &registerapi.RegistrationStatus{
			PluginRegistered: false,
			Error:            errStr,
		}); err != nil {
			return errors.Wrap(err, errStr)
		}
		return errors.New(errStr)
	}

	if _, err := client.NotifyRegistrationStatus(ctx, &registerapi.RegistrationStatus{
		PluginRegistered: true,
	}); err != nil {
		chanForAckOfNotification <- false
		return fmt.Errorf("failed to send registration status at socket %s, err: %v", socketPath, err)
	}

	chanForAckOfNotification <- true
	return nil
}

// Handle filesystem notify event.
func (w *Watcher) handleFsNotifyEvent(event fsnotify.Event) error {
	if event.Op&fsnotify.Create != fsnotify.Create {
		return nil
	}

	fi, err := os.Stat(event.Name)
	if err != nil {
		return fmt.Errorf("stat file %s failed: %v", event.Name, err)
	}

	if !fi.IsDir() {
		return w.registerPlugin(event.Name)
	}

	if err := w.traversePluginDir(event.Name); err != nil {
		return fmt.Errorf("failed to traverse plugin path %s, err: %v", event.Name, err)
	}

	return nil
}

// Start watches for the creation of plugin sockets at the path
func (w *Watcher) Start() error {
	glog.V(2).Infof("Plugin Watcher Start at %s", w.path)
	w.stopCh = make(chan interface{})

	// Creating the directory to be watched if it doesn't exist yet,
	// and walks through the directory to discover the existing plugins.
	if err := w.init(); err != nil {
		return err
	}

	fsWatcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("failed to start plugin fsWatcher, err: %v", err)
	}
	w.fsWatcher = fsWatcher

	if err := w.traversePluginDir(w.path); err != nil {
		fsWatcher.Close()
		return fmt.Errorf("failed to traverse plugin socket path, err: %v", err)
	}

	w.wg.Add(1)
	go func(fsWatcher *fsnotify.Watcher) {
		defer w.wg.Done()
		for {
			select {
			case event := <-fsWatcher.Events:
				//TODO: Handle errors by taking corrective measures
				go func() {
					err := w.handleFsNotifyEvent(event)
					if err != nil {
						glog.Errorf("error %v when handle event: %s", err, event)
					}
				}()
				continue
			case err := <-fsWatcher.Errors:
				if err != nil {
					glog.Errorf("fsWatcher received error: %v", err)
				}
				continue
			case <-w.stopCh:
				fsWatcher.Close()
				return
			}
		}
	}(fsWatcher)
	return nil
}

// Stop stops probing the creation of plugin sockets at the path
func (w *Watcher) Stop() error {
	close(w.stopCh)
	c := make(chan struct{})
	go func() {
		defer close(c)
		w.wg.Wait()
	}()
	select {
	case <-c:
	case <-time.After(10 * time.Second):
		return fmt.Errorf("timeout on stopping watcher")
	}
	return nil
}

// Cleanup cleans the path by removing sockets
func (w *Watcher) Cleanup() error {
	return os.RemoveAll(w.path)
}

// Dial establishes the gRPC communication with the picked up plugin socket. https://godoc.org/google.golang.org/grpc#Dial
func dial(unixSocketPath string) (registerapi.RegistrationClient, *grpc.ClientConn, error) {
	c, err := grpc.Dial(unixSocketPath, grpc.WithInsecure(), grpc.WithBlock(),
		grpc.WithTimeout(10*time.Second),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	if err != nil {
		return nil, nil, fmt.Errorf("failed to dial socket %s, err: %v", unixSocketPath, err)
	}

	return registerapi.NewRegistrationClient(c), c, nil
}
