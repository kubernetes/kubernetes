/*
Copyright 2014 The Kubernetes Authors.

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

package tokenfile

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/fsnotify/fsnotify"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/util/filewatcher"
)

type TokenAuthenticator struct {
	path   string
	mutex  *sync.Mutex
	tokens map[string]*user.DefaultInfo
}

// NewCSV returns a TokenAuthenticator, populated from a CSV file.
// The CSV file must contain records in the format "token,username,useruid"
func NewCSV(path string) (*TokenAuthenticator, error) {
	tokens, err := ReadCSV(path)
	if err != nil {
		return nil, err
	}
	authenticator := &TokenAuthenticator{
		path:   path,
		mutex:  &sync.Mutex{},
		tokens: tokens,
	}
	watcher, err := filewatcher.CreateFileWatcher(path)
	if err != nil {
		glog.Errorf("failed to add file watcher on %s", path)
	} else {
		go filewatcher.StartFileEventLoop(watcher, authenticator.HandleEvent, authenticator.HandleError)
	}
	return authenticator, nil
}

func ReadCSV(path string) (map[string]*user.DefaultInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	tokens := make(map[string]*user.DefaultInfo)
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if len(record) < 3 {
			return nil, fmt.Errorf("token file '%s' must have at least 3 columns (token, user name, user uid), found %d", file.Name(), len(record))
		}
		obj := &user.DefaultInfo{
			Name: record[1],
			UID:  record[2],
		}
		tokens[record[0]] = obj

		if len(record) >= 4 {
			obj.Groups = strings.Split(record[3], ",")
		}
	}
	return tokens, nil
}

func (a *TokenAuthenticator) Update(tokens map[string]*user.DefaultInfo) {
	a.mutex.Lock()
	a.tokens = tokens
	a.mutex.Unlock()
}

func (a *TokenAuthenticator) HandleEvent(watcher *fsnotify.Watcher, event fsnotify.Event) {
	if event.Op&fsnotify.Write == fsnotify.Write || event.Op&fsnotify.Remove == fsnotify.Remove {
		glog.Infof("file event caught: %s", event)
		tokens, err := ReadCSV(a.path)
		if err != nil {
			glog.Infof("update token failed, %s", err)
		} else {
			glog.Infof("file updated: %s", a.path)
			a.Update(tokens)
		}
	}
	if event.Op&fsnotify.Remove == fsnotify.Remove {
		// Some file editor do remove operations on file, eg. Vim
		// And the file watcher will be invalid, so we add again.
		watcher.Remove(a.path)
		watcher.Add(a.path)
	}
}

func (a *TokenAuthenticator) HandleError(watcher *fsnotify.Watcher, err error) {
	glog.Errorf("file watcher of %s got error %s", a.path, err)
	watcher.Close()
}

func (a *TokenAuthenticator) AuthenticateToken(value string) (user.Info, bool, error) {
	user, ok := a.tokens[value]
	if !ok {
		return nil, false, nil
	}
	return user, true, nil
}
