/*
Copyright 2017 The Kubernetes Authors.

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

package checkpoint

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/dchest/safefile"
	"github.com/ghodss/yaml"
	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/volume/util"
)

const (
	// Delimiter used on checkpoints written to disk
	delimiter = "_"
	podPrefix = "Pod"
)

// Manager is the interface used to manage checkpoints
// which involves writing resources to disk to recover
// during restart or failure scenarios.
// https://github.com/kubernetes/community/pull/1241/files
type Manager interface {
	// LoadPods will load checkpointed Pods from disk
	LoadPods() ([]*v1.Pod, error)

	// WritePod will serialize a Pod to disk
	WritePod(pod *v1.Pod) error

	// Deletes the checkpoint of the given pod from disk
	DeletePod(pod *v1.Pod) error
}

var instance Manager
var mutex = &sync.Mutex{}

// fileCheckPointManager - is a checkpointer that writes contents to disk
// The type information of the resource objects are encoded in the name
type fileCheckPointManager struct {
	path string
}

// NewCheckpointManager will create a Manager that points to the following path
func NewCheckpointManager(path string) Manager {
	// NOTE: This is a precaution; current implementation should not run
	// multiple checkpoint managers.
	mutex.Lock()
	defer mutex.Unlock()
	instance = &fileCheckPointManager{path: path}
	return instance
}

// GetInstance will return the current Manager, there should be only one.
func GetInstance() Manager {
	mutex.Lock()
	defer mutex.Unlock()
	return instance
}

// loadPod will load Pod Checkpoint yaml file.
func (fcp *fileCheckPointManager) loadPod(file string) (*v1.Pod, error) {
	return util.LoadPodFromFile(file)
}

// checkAnnotations will validate the checkpoint annotations exist on the Pod
func (fcp *fileCheckPointManager) checkAnnotations(pod *v1.Pod) bool {
	if podAnnotations := pod.GetAnnotations(); podAnnotations != nil {
		if podAnnotations[core.BootstrapCheckpointAnnotationKey] == "true" {
			return true
		}
	}
	return false
}

// getPodPath returns the full qualified path for the pod checkpoint
func (fcp *fileCheckPointManager) getPodPath(pod *v1.Pod) string {
	return fmt.Sprintf("%v/Pod%v%v.yaml", fcp.path, delimiter, pod.GetUID())
}

// LoadPods Loads All Checkpoints from disk
func (fcp *fileCheckPointManager) LoadPods() ([]*v1.Pod, error) {
	checkpoints := make([]*v1.Pod, 0)
	files, err := ioutil.ReadDir(fcp.path)
	if err != nil {
		return nil, err
	}
	for _, f := range files {
		// get just the filename
		_, fname := filepath.Split(f.Name())
		// Get just the Resource from "Resource_Name"
		fnfields := strings.Split(fname, delimiter)
		switch fnfields[0] {
		case podPrefix:
			pod, err := fcp.loadPod(fmt.Sprintf("%s/%s", fcp.path, f.Name()))
			if err != nil {
				return nil, err
			}
			checkpoints = append(checkpoints, pod)
		default:
			glog.Warningf("Unsupported checkpoint file detected %v", f)
		}
	}
	return checkpoints, nil
}

// Writes a checkpoint to a file on disk if annotation is present
func (fcp *fileCheckPointManager) WritePod(pod *v1.Pod) error {
	var err error
	if fcp.checkAnnotations(pod) {
		if blob, err := yaml.Marshal(pod); err == nil {
			err = safefile.WriteFile(fcp.getPodPath(pod), blob, 0644)
		}
	} else {
		// This is to handle an edge where a pod update could remove
		// an annotation and the checkpoint should then be removed.
		err = fcp.DeletePod(pod)
	}
	return err
}

// Deletes a checkpoint from disk if present
func (fcp *fileCheckPointManager) DeletePod(pod *v1.Pod) error {
	podPath := fcp.getPodPath(pod)
	if err := os.Remove(podPath); !os.IsNotExist(err) {
		return err
	}
	return nil
}
