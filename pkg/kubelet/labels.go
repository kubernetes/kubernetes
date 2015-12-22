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
	"bufio"
	"encoding/json"
	"fmt"
	"github.com/golang/glog"
	"io/ioutil"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/yaml"
	"os"
	"path"
	"path/filepath"
)

// labelSet represents a collection of labels managed by the node and configured through static files
// on the host
// TODO(pwittrock): refactor other node label logic out of kubelet and into this file
type labelSet struct {
	// labelDirectory is the directory where the node label files are stored
	labelDirectory string

	// checkpointFile is a file containing the last set of node label keys
	// it is used for deleting labels that are removed from the labelDirectory files
	checkpointFile string

	// nodeLabelsToAdd is a map of labels we add
	nodeLabelsToAdd map[string]string

	// nodeLabelsToClear is a map of labels we remove if not in nodeLabelsToAdd
	nodeLabelsToClear map[string]string

	// checkpointRequiresAppend and checkpointRequiresWrite are used to record if
	// the checkpoint file must be updated
	checkpointRequiresAppend bool
	checkpointRequiresWrite  bool
}

// NewLabelSet instantiates a new LabelSet, reading node labels from the labelDirectory and checkpointFile.
// All files under labelDirectory will be
func NewLabelSet(labelDirectory string, checkpointFile string) (*labelSet, error) {
	if (checkpointFile == "" || labelDirectory == "") && labelDirectory != checkpointFile {
		return nil, fmt.Errorf("must specify checkpointFile and labelDirectory together.")
	}

	ls := &labelSet{
		labelDirectory:           labelDirectory,
		checkpointFile:           checkpointFile,
		checkpointRequiresAppend: false,
		checkpointRequiresWrite:  false,
	}

	// Create the file if needed
	if checkpointFile != "" {
		if _, err := os.Stat(checkpointFile); os.IsNotExist(err) {
			err = ls.writeLabelsFile(checkpointFile, map[string]string{})
			if err != nil {
				return nil, fmt.Errorf("cannot create checkpointFile %s", checkpointFile)
			}
		}
	}

	// Parse new labels that should be added or updated
	a, err := ls.readLabels()
	if err != nil {
		return nil, err
	}
	ls.nodeLabelsToAdd = a

	// Parse old labels that should be cleared, but may not exist in the current files
	c, err := ls.readCheckpoint()
	if err != nil {
		return nil, err
	}
	ls.nodeLabelsToClear = c

	for k := range ls.nodeLabelsToAdd {
		if _, f := ls.nodeLabelsToClear[k]; !f {
			ls.checkpointRequiresAppend = true
			ls.checkpointRequiresWrite = true
		}
	}

	return ls, nil
}

// AppendCheckpoint writes the new node labels to the checkpoint file, keeping the labels already read
// from the checkpoint file.  Will not write the checkpoint file if it has already been appended.
func (ls *labelSet) AppendLabelCheckpoint() error {
	if !ls.checkpointRequiresAppend {
		return nil
	}

	// Write both new and old labels to the checkpoint
	clearLabels := map[string]string{}
	for k := range ls.nodeLabelsToClear {
		clearLabels[k] = ""
	}
	for k := range ls.nodeLabelsToAdd {
		clearLabels[k] = ""
	}
	err := ls.writeLabelsFile(ls.checkpointFile, clearLabels)
	if err != nil {
		return fmt.Errorf("Could not append checkpoint file %v", err)
	}

	ls.checkpointRequiresAppend = false
	return nil
}

// WriteCheckpoint writes the new node labels to the checkpoint file, and clears the old checkpoin labels.
// Will not write the checkpoint file if has already been written.
func (ls *labelSet) WriteLabelCheckpoint() error {
	if !ls.checkpointRequiresWrite {
		return nil
	}

	// Write new labels to the checkpoint
	clearLabels := map[string]string{}
	for k := range ls.nodeLabelsToAdd {
		clearLabels[k] = ""
	}
	err := ls.writeLabelsFile(ls.checkpointFile, clearLabels)
	if err != nil {
		return fmt.Errorf("Could not write checkpoint file %v", err)
	}

	ls.checkpointRequiresWrite = false
	return nil
}

// WriteLabels adds / updates labels read from the label directory to the map.  Removes labels from the map
// that were in the label directory, but have been removed.
func (ls *labelSet) WriteLabelsToNodeMeta(node *api.Node) {
	labels := node.ObjectMeta.Labels
	for k := range ls.nodeLabelsToClear {
		delete(labels, k)
	}
	for k, v := range ls.nodeLabelsToAdd {
		labels[k] = v
	}
}

// readLabels reads labels from the label directory and returns a map of the results.
func (ls *labelSet) readLabels() (map[string]string, error) {
	dstLabels := map[string]string{}
	if ls.labelDirectory == "" {
		return dstLabels, nil
	}
	files, err := ioutil.ReadDir(ls.labelDirectory)
	if err != nil {
		return nil, err
	}

	// Load each file
	for _, fi := range files {
		f := path.Join(ls.labelDirectory, fi.Name())
		if ext := filepath.Ext(fi.Name()); ext != ".json" && ext != ".yaml" {
			glog.Warningf("Unsupported file type (must be json or yaml) in node labels directory: %s", f)
			continue
		}
		srcLabels, err := ls.readLabelsFile(f)
		if err != nil {
			return nil, err
		}
		for k, v := range srcLabels {
			if _, dk := dstLabels[k]; dk {
				return nil, fmt.Errorf(
					"Multiple values found for label key %s.  Check %s for files defining the label.", ls.labelDirectory, k)
			}
			dstLabels[k] = v
		}
	}
	return dstLabels, nil
}

// readCheckpoint reads labels from the checkpoint file and returns a map of the results.
func (ls *labelSet) readCheckpoint() (map[string]string, error) {
	if ls.checkpointFile == "" {
		return map[string]string{}, nil
	}
	return ls.readLabelsFile(ls.checkpointFile)
}

// writeLabelsFile writes the map of labels to a file as json
func (ls *labelSet) writeLabelsFile(path string, labels map[string]string) error {
	b, err := json.Marshal(labels)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, b, 0644)
}

// readLabelsFile reads in and parses the yaml or json node labels file
func (ls *labelSet) readLabelsFile(path string) (map[string]string, error) {
	labels := make(map[string]string, 0)
	kps := make(map[string]interface{}, 0)

	fd, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fd.Close()

	err = yaml.NewYAMLOrJSONDecoder(bufio.NewReader(fd), 12).Decode(&kps)
	if err != nil {
		return nil, fmt.Errorf("the node label file %s content is invalid, %s", path, err)
	}

	for k, v := range kps {
		// we ONLY accept key=value pairs, no complex types
		switch v.(type) {
		case string:
			labels[k] = v.(string)
		case float64:
			labels[k] = fmt.Sprintf("%f", v.(float64))
		default:
			return nil, fmt.Errorf("node label files (%s) only support key:string, not complex values e.g arrays, maps", path)
		}
	}

	return labels, nil
}
