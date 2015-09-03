/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/golang/glog"
	"io/ioutil"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/fielderrors"
	"path"
	"path/filepath"
	"regexp"
	"sync"
	"time"
)

const pollPluginDuration = time.Minute * 10
const validNameSpaces = `^.+\.node.kubernetes.io/.+$`

var validNameSpacesRegex = regexp.MustCompile(validNameSpaces)

type NodeLabelManager interface {
	Run()
	GetNodeLabels() map[string]string
}

type nodeLabelManager struct {
	labels    map[string]string
	hostname  string
	directory string
	// TODO(pwittrock): Consider using the atom package compareAndSwap methods in place of a Mutex before merging
	mutext  sync.RWMutex
	started bool
}

func NewNodeLabelManager(hostname string, directory string) NodeLabelManager {
	return &nodeLabelManager{
		labels:    map[string]string{"kubernetes.io/hostname": hostname},
		hostname:  hostname,
		directory: directory,
		mutext:    sync.RWMutex{},
	}
}

func (nlm *nodeLabelManager) setRunning() bool {
	nlm.mutext.Lock()
	defer nlm.mutext.Unlock()
	if nlm.started {
		return false
	}
	nlm.started = true
	return true
}

// Run initializes the node labels from the plugin directory, and then schedules periodic
// polling for updates to labels.
func (nlm *nodeLabelManager) Run() {
	if !nlm.setRunning() {
		glog.Errorln("Node labels already running.  Cannot call Run() a second time.")
		return
	}

	// Make sure that the node labels have been updated at least once when this returns
	err := nlm.updateNodeLabels()
	if err != nil {
		glog.Errorln(err)
	}

	go util.Until(func() {
		err := nlm.updateNodeLabels()
		if err != nil {
			glog.Errorln(err)
		}
	}, pollPluginDuration, util.NeverStop)
}

// UpdateNodeLabels updates the map of labels using plugins in the plugin directory.
// Json files are loaded and expected to contain map[string]string.  Sh files
// are executed in serial and expected to emit a Json map[string]string from Stdout.
// If any errors are encountered while reading plugins the error is logged and
// all label updates are ignored.
func (nlm *nodeLabelManager) updateNodeLabels() error {
	// Look for node label plugins
	if nlm.directory == "" {
		return nil
	}
	files, err := ioutil.ReadDir(nlm.directory)
	if err != nil {
		return err
	}

	// Initialize label map from the available plugins
	nl := map[string]string{"kubernetes.io/hostname": nlm.hostname}
	for _, fi := range files {
		f := path.Join(nlm.directory, fi.Name())
		switch filepath.Ext(fi.Name()) {
		case ".json":
			if err := nlm.readJson(f, nl); err != nil {
				return err
			}
		}
	}

	// Update the labels we export to the master
	nlm.mutext.Lock()
	defer nlm.mutext.Unlock()
	nlm.labels = nl
	return nil
}

// GetNodeLabels returns the current map[string]string of labels to be applied to the node.
func (nlm *nodeLabelManager) GetNodeLabels() map[string]string {
	nlm.mutext.RLock()
	defer nlm.mutext.RUnlock()
	return nlm.labels
}

// ReadJson parses f into labels and adds them to ol.
func (nlm *nodeLabelManager) readJson(f string, l map[string]string) error {
	data, err := ioutil.ReadFile(f)
	if err != nil {
		return err
	}
	return nlm.parseJson(data, l)
}

// ParseJson Unmarshals data into a map[string]string and adds each entry to ol.  Returns an error if any
// any of the labels are invalid or not permitted.  Otherwise returns nil.
func (nlm *nodeLabelManager) parseJson(data []byte, l map[string]string) error {
	var nl map[string]string
	err := json.Unmarshal(data, &nl)
	if err != nil {
		return err
	}
	for k := range nl {
		v, ok := nl[k]
		if !ok {
			return errors.New(fmt.Sprintf("Could not cast label value %v to string", nl[k]))
		}
		if err := nlm.checkLabelOk(k, v, l); err != nil {
			return err
		}
		l[k] = v
	}
	return nil
}

// checkLabelOk returns nil if k,v is a valid node label key, value.  Will return an error on any of the following
// - k or v do not conform to permitted label key, value formats
// - k is not in the correct namespace
// - k has already been defined
func (nlm *nodeLabelManager) checkLabelOk(k string, v string, ol map[string]string) error {
	// Verify Key is ok
	if !util.IsQualifiedName(k) {
		return fielderrors.NewFieldInvalid("label key", k, "Invalid label key")
	}
	// Verify Value is ok
	if !util.IsValidLabelValue(v) {
		return fielderrors.NewFieldInvalid("label value", v, "Invalid label value")
	}

	// Verify this label has not already been defined
	if ov, dup := ol[k]; dup {
		err := errors.New(fmt.Sprintf("Duplicate node label specified for %s: %s %s", k, ov, k))
		return err
	}
	// Verify the label is in a permitted namespace
	if !validNameSpacesRegex.MatchString(k) {
		err := errors.New(fmt.Sprintf(
			"Label must %s live in namespace matching regex %s", k, validNameSpaces))
		return err
	}
	return nil
}
