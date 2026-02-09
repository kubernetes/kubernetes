/*
Copyright 2016 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
	api "k8s.io/kubernetes/pkg/apis/core"
	utilio "k8s.io/utils/io"
)

type podEventType int

const (
	podAdd podEventType = iota
	podModify
	podDelete

	eventBufferLen = 10
)

type watchEvent struct {
	fileName  string
	eventType podEventType
}

type sourceFile struct {
	path           string
	nodeName       types.NodeName
	period         time.Duration
	store          cache.Store
	fileKeyMapping map[string]string
	updates        chan<- sourceUpdate
	watchEvents    chan *watchEvent
}

// NewSourceFile watches a config file for changes.
func NewSourceFile(logger klog.Logger, path string, nodeName types.NodeName, period time.Duration, updates chan<- sourceUpdate) {
	// "github.com/sigma/go-inotify" requires a path without trailing "/"
	path = strings.TrimRight(path, string(os.PathSeparator))

	config := newSourceFile(path, nodeName, period, updates)
	logger.V(1).Info("Watching path", "path", path)
	config.run(logger)
}

func newSourceFile(path string, nodeName types.NodeName, period time.Duration, updates chan<- sourceUpdate) *sourceFile {
	send := func(objs []interface{}) {
		var pods []*v1.Pod
		for _, o := range objs {
			pods = append(pods, o.(*v1.Pod))
		}
		updates <- sourceUpdate{Pods: pods}
	}
	store := cache.NewUndeltaStore(send, cache.MetaNamespaceKeyFunc)
	return &sourceFile{
		path:           path,
		nodeName:       nodeName,
		period:         period,
		store:          store,
		fileKeyMapping: map[string]string{},
		updates:        updates,
		watchEvents:    make(chan *watchEvent, eventBufferLen),
	}
}

func (s *sourceFile) run(logger klog.Logger) {
	listTicker := time.NewTicker(s.period)

	go func() {
		// Read path immediately to speed up startup.
		if err := s.listConfig(logger); err != nil {
			logger.Error(err, "Unable to read config path", "path", s.path)
		}
		for {
			select {
			case <-listTicker.C:
				if err := s.listConfig(logger); err != nil {
					logger.Error(err, "Unable to read config path", "path", s.path)
				}
			case e := <-s.watchEvents:
				if err := s.consumeWatchEvent(logger, e); err != nil {
					logger.Error(err, "Unable to process watch event")
				}
			}
		}
	}()

	s.startWatch(logger)
}

func (s *sourceFile) applyDefaults(logger klog.Logger, pod *api.Pod, source string) error {
	return applyDefaults(logger, pod, source, true, s.nodeName)
}

func (s *sourceFile) listConfig(logger klog.Logger) error {
	path := s.path
	statInfo, err := os.Stat(path)
	if err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		// Emit an update with an empty PodList to allow FileSource to be marked as seen
		s.updates <- sourceUpdate{Pods: []*v1.Pod{}}
		return fmt.Errorf("path does not exist, ignoring")
	}

	switch {
	case statInfo.Mode().IsDir():
		pods, err := s.extractFromDir(logger, path)
		if err != nil {
			return err
		}
		if len(pods) == 0 {
			// Emit an update with an empty PodList to allow FileSource to be marked as seen
			s.updates <- sourceUpdate{Pods: pods}
			return nil
		}
		return s.replaceStore(pods...)

	case statInfo.Mode().IsRegular():
		pod, err := s.extractFromFile(logger, path)
		if err != nil {
			return err
		}
		return s.replaceStore(pod)

	default:
		return fmt.Errorf("path is not a directory or file")
	}
}

// Get as many pod manifests as we can from a directory. Return an error if and only if something
// prevented us from reading anything at all. Do not return an error if only some files
// were problematic.
func (s *sourceFile) extractFromDir(logger klog.Logger, name string) ([]*v1.Pod, error) {
	dirents, err := filepath.Glob(filepath.Join(name, "[^.]*"))
	if err != nil {
		return nil, fmt.Errorf("glob failed: %v", err)
	}

	pods := make([]*v1.Pod, 0, len(dirents))
	if len(dirents) == 0 {
		return pods, nil
	}

	sort.Strings(dirents)
	for _, path := range dirents {
		statInfo, err := os.Stat(path)
		if err != nil {
			logger.Error(err, "Could not get metadata", "path", path)
			continue
		}

		switch {
		case statInfo.Mode().IsDir():
			logger.Error(nil, "Provided manifest path is a directory, not recursing into manifest path", "path", path)
		case statInfo.Mode().IsRegular():
			pod, err := s.extractFromFile(logger, path)
			if err != nil {
				if !os.IsNotExist(err) {
					logger.Error(err, "Could not process manifest file", "path", path)
				}
			} else {
				pods = append(pods, pod)
			}
		default:
			logger.Error(nil, "Manifest path is not a directory or file", "path", path, "mode", statInfo.Mode())
		}
	}
	return pods, nil
}

// extractFromFile parses a file for Pod configuration information.
func (s *sourceFile) extractFromFile(logger klog.Logger, filename string) (pod *v1.Pod, err error) {
	logger.V(3).Info("Reading config file", "path", filename)
	defer func() {
		if err == nil && pod != nil {
			objKey, keyErr := cache.MetaNamespaceKeyFunc(pod)
			if keyErr != nil {
				err = keyErr
				return
			}
			s.fileKeyMapping[filename] = objKey
		}
	}()

	file, err := os.Open(filename)
	if err != nil {
		return pod, err
	}
	defer file.Close()

	data, err := utilio.ReadAtMost(file, maxConfigLength)
	if err != nil {
		return pod, err
	}

	defaultFn := func(logger klog.Logger, pod *api.Pod) error {
		return s.applyDefaults(logger, pod, filename)
	}

	parsed, pod, podErr := tryDecodeSinglePod(logger, data, defaultFn)
	if parsed {
		if podErr != nil {
			return pod, podErr
		}
		return pod, nil
	}

	return pod, fmt.Errorf("%v: couldn't parse as pod(%v), please check config file", filename, podErr)
}

func (s *sourceFile) replaceStore(pods ...*v1.Pod) (err error) {
	objs := []interface{}{}
	for _, pod := range pods {
		objs = append(objs, pod)
	}
	return s.store.Replace(objs, "")
}
