/*
Copyright 2019 The Kubernetes Authors.

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

package options

import (
	"fmt"

	"k8s.io/kubernetes/pkg/util/filesystem"

	"github.com/fsnotify/fsnotify"
)

// Creates a new filesystem watcher and adds watches for the config file.
func (o *KubeControllerManagerOptions) initWatcher() error {
	o.ErrCh = make(chan error)
	fswatcher := filesystem.NewFsnotifyWatcher()
	err := fswatcher.Init(o.eventHandler, o.errorHandler)
	if err != nil {
		return err
	}
	err = fswatcher.AddWatch(o.ConfigFile)
	if err != nil {
		return err
	}
	o.Watcher = fswatcher
	return nil
}

func (o *KubeControllerManagerOptions) eventHandler(ent fsnotify.Event) {
	eventOpIs := func(Op fsnotify.Op) bool {
		return ent.Op&Op == Op
	}
	if eventOpIs(fsnotify.Write) || eventOpIs(fsnotify.Rename) {
		// error out when ConfigFile is updated
		o.ErrCh <- fmt.Errorf("content of the kub-controller manager server's configuration file was updated")
	}
	o.ErrCh <- nil
}

func (o *KubeControllerManagerOptions) errorHandler(err error) {
	o.ErrCh <- err
}
