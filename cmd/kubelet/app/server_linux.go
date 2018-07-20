/*
Copyright 2015 The Kubernetes Authors.

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

package app

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strconv"

	"github.com/golang/glog"
	"golang.org/x/exp/inotify"
)

func watchForLockfileContention(path string, done chan struct{}) error {
	watcher, err := inotify.NewWatcher()
	if err != nil {
		glog.Errorf("unable to create watcher for lockfile: %v", err)
		return err
	}
	if err = watcher.AddWatch(path, inotify.IN_OPEN|inotify.IN_DELETE_SELF); err != nil {
		glog.Errorf("unable to watch lockfile: %v", err)
		return err
	}
	go func() {
		select {
		case ev := <-watcher.Event:
			glog.Infof("inotify event: %v", ev)
		case err = <-watcher.Error:
			glog.Errorf("inotify watcher error: %v", err)
		}
		close(done)
	}()
	return nil
}

func checkKubeletInInitialNS(pid int) error {
	file, err := os.Open("/proc/" + strconv.Itoa(pid) + "/uid_map")
	if err != nil {
		return err
	}
	defer file.Close()

	buf := bufio.NewReader(file)
	l, _, err := buf.ReadLine()
	if err != nil {
		return err
	}

	line := string(l)
	var a, b, c int64
	fmt.Sscanf(line, "%d %d %d", &a, &b, &c)

	if a == 0 && b == 0 && c == 4294967295 {
		return nil
	}
	return errors.New("kubelet is not running in the `initial` user namespace")
}
