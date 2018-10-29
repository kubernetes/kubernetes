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
	"strings"
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

// Watcher is the plugin watcher
type Watcher struct {
	path      string
	stopCh    chan interface{}
	fs        utilfs.Filesystem
	fsWatcher *fsnotify.Watcher
	wg        sync.WaitGroup

	mutex       sync.Mutex
	handlers    map[string]PluginHandler
	plugins     map[string]pathInfo
	pluginsPool map[string]map[string]*sync.Mutex // map[pluginType][pluginName]
}

type pathInfo struct {
	pluginType string
	pluginName string
}

// NewWatcher provides a new watcher
func NewWatcher(sockDir string) *Watcher {
	return &Watcher{
		path: sockDir,
		fs:   &utilfs.DefaultFs{},

		handlers:    make(map[string]PluginHandler),
		plugins:     make(map[string]pathInfo),
		pluginsPool: make(map[string]map[string]*sync.Mutex),
	}
}

func (w *Watcher) AddHandler(pluginType string, handler PluginHandler) {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	w.handlers[pluginType] = handler
}

func (w *Watcher) getHandler(pluginType string) (PluginHandler, bool) {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	h, ok := w.handlers[pluginType]
	return h, ok
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

	w.wg.Add(1)
	go func(fsWatcher *fsnotify.Watcher) {
		defer w.wg.Done()
		for {
			select {
			case event := <-fsWatcher.Events:
				//TODO: Handle errors by taking corrective measures

				w.wg.Add(1)
				go func() {
					defer w.wg.Done()

					if event.Op&fsnotify.Create == fsnotify.Create {
						err := w.handleCreateEvent(event)
						if err != nil {
							glog.Errorf("error %v when handling create event: %s", err, event)
						}
					} else if event.Op&fsnotify.Remove == fsnotify.Remove {
						err := w.handleDeleteEvent(event)
						if err != nil {
							glog.Errorf("error %v when handling delete event: %s", err, event)
						}
					}
					return
				}()
				continue
			case err := <-fsWatcher.Errors:
				if err != nil {
					glog.Errorf("fsWatcher received error: %v", err)
				}
				continue
			case <-w.stopCh:
				return
			}
		}
	}(fsWatcher)

	// Traverse plugin dir after starting the plugin processing goroutine
	if err := w.traversePluginDir(w.path); err != nil {
		w.Stop()
		return fmt.Errorf("failed to traverse plugin socket path, err: %v", err)
	}

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
	case <-time.After(11 * time.Second):
		return fmt.Errorf("timeout on stopping watcher")
	}

	w.fsWatcher.Close()

	return nil
}

func (w *Watcher) init() error {
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
		default:
			glog.V(5).Infof("Ignoring file %s with mode %v", path, mode)
		}

		return nil
	})
}

// Handle filesystem notify event.
func (w *Watcher) handleCreateEvent(event fsnotify.Event) error {
	glog.V(6).Infof("Handling create event: %v", event)

	fi, err := os.Stat(event.Name)
	if err != nil {
		return fmt.Errorf("stat file %s failed: %v", event.Name, err)
	}

	if strings.HasPrefix(fi.Name(), ".") {
		glog.Errorf("Ignoring file: %s", fi.Name())
		return nil
	}

	if !fi.IsDir() {
		return w.handlePluginRegistration(event.Name)
	}

	return w.traversePluginDir(event.Name)
}

func (w *Watcher) handlePluginRegistration(socketPath string) error {
	//TODO: Implement rate limiting to mitigate any DOS kind of attacks.
	client, conn, err := dial(socketPath, 10*time.Second)
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

	handler, ok := w.handlers[infoResp.Type]
	if !ok {
		return w.notifyPlugin(client, false, fmt.Sprintf("no handler registered for plugin type: %s at socket %s", infoResp.Type, socketPath))
	}

	// ReRegistration: We want to handle multiple plugins registering at the same time with the same name sequentially.
	// See the state machine for more information.
	// This is done by using a Lock for each plugin with the same name and type
	pool := w.getPluginPool(infoResp.Type, infoResp.Name)

	pool.Lock()
	defer pool.Unlock()

	if infoResp.Endpoint == "" {
		infoResp.Endpoint = socketPath
	}

	// calls handler callback to verify registration request
	if err := handler.ValidatePlugin(infoResp.Name, infoResp.Endpoint, infoResp.SupportedVersions); err != nil {
		return w.notifyPlugin(client, false, fmt.Sprintf("plugin validation failed with err: %v", err))
	}

	// We add the plugin to the pluginwatcher's map before calling a plugin consumer's Register handle
	// so that if we receive a delete event during Register Plugin, we can process it as a DeRegister call.
	w.registerPlugin(socketPath, infoResp.Type, infoResp.Name)

	if err := handler.RegisterPlugin(infoResp.Name, infoResp.Endpoint); err != nil {
		return w.notifyPlugin(client, false, fmt.Sprintf("plugin registration failed with err: %v", err))
	}

	// Notify is called after register to guarantee that even if notify throws an error Register will always be called after validate
	if err := w.notifyPlugin(client, true, ""); err != nil {
		return fmt.Errorf("failed to send registration status at socket %s, err: %v", socketPath, err)
	}

	return nil
}

func (w *Watcher) handleDeleteEvent(event fsnotify.Event) error {
	glog.V(6).Infof("Handling delete event: %v", event)

	plugin, ok := w.getPlugin(event.Name)
	if !ok {
		return fmt.Errorf("could not find plugin for deleted file %s", event.Name)
	}

	// You should not get a Deregister call while registering a plugin
	pool := w.getPluginPool(plugin.pluginType, plugin.pluginName)

	pool.Lock()
	defer pool.Unlock()

	// ReRegisteration: When waiting for the lock a plugin with the same name (not socketPath) could have registered
	// In that case, we don't want to issue a DeRegister call for that plugin
	// When ReRegistering, the new plugin will have removed the current mapping (map[socketPath] = plugin) and replaced
	// it with it's own socketPath.
	if _, ok = w.getPlugin(event.Name); !ok {
		glog.V(2).Infof("A newer plugin watcher has been registered for plugin %v, dropping DeRegister call", plugin)
		return nil
	}

	h, ok := w.getHandler(plugin.pluginType)
	if !ok {
		return fmt.Errorf("could not find handler %s for plugin %s at path %s", plugin.pluginType, plugin.pluginName, event.Name)
	}

	glog.V(2).Infof("DeRegistering plugin %v at path %s", plugin, event.Name)
	w.deRegisterPlugin(event.Name, plugin.pluginType, plugin.pluginName)
	h.DeRegisterPlugin(plugin.pluginName)

	return nil
}

func (w *Watcher) registerPlugin(socketPath, pluginType, pluginName string) {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	// Reregistration case, if this plugin is already in the map, remove it
	// This will prevent handleDeleteEvent to issue a DeRegister call
	for path, info := range w.plugins {
		if info.pluginType != pluginType || info.pluginName != pluginName {
			continue
		}

		delete(w.plugins, path)
		break
	}

	w.plugins[socketPath] = pathInfo{
		pluginType: pluginType,
		pluginName: pluginName,
	}
}

func (w *Watcher) deRegisterPlugin(socketPath, pluginType, pluginName string) {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	delete(w.plugins, socketPath)
	delete(w.pluginsPool[pluginType], pluginName)
}

func (w *Watcher) getPlugin(socketPath string) (pathInfo, bool) {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	plugin, ok := w.plugins[socketPath]
	return plugin, ok
}

func (w *Watcher) getPluginPool(pluginType, pluginName string) *sync.Mutex {
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if _, ok := w.pluginsPool[pluginType]; !ok {
		w.pluginsPool[pluginType] = make(map[string]*sync.Mutex)
	}

	if _, ok := w.pluginsPool[pluginType][pluginName]; !ok {
		w.pluginsPool[pluginType][pluginName] = &sync.Mutex{}
	}

	return w.pluginsPool[pluginType][pluginName]
}

func (w *Watcher) notifyPlugin(client registerapi.RegistrationClient, registered bool, errStr string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	status := &registerapi.RegistrationStatus{
		PluginRegistered: registered,
		Error:            errStr,
	}

	if _, err := client.NotifyRegistrationStatus(ctx, status); err != nil {
		return errors.Wrap(err, errStr)
	}

	if errStr != "" {
		return errors.New(errStr)
	}

	return nil
}

// Dial establishes the gRPC communication with the picked up plugin socket. https://godoc.org/google.golang.org/grpc#Dial
func dial(unixSocketPath string, timeout time.Duration) (registerapi.RegistrationClient, *grpc.ClientConn, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	c, err := grpc.DialContext(ctx, unixSocketPath, grpc.WithInsecure(), grpc.WithBlock(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	if err != nil {
		return nil, nil, fmt.Errorf("failed to dial socket %s, err: %v", unixSocketPath, err)
	}

	return registerapi.NewRegistrationClient(c), c, nil
}
