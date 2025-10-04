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

package oom

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"time"
)

type state struct {
	LastProcessedTimestamp time.Time
	storer                 storer
}

func (s *state) store() error {
	return s.storer.store(*s)
}

func (s *state) load() error {
	return s.storer.load(s)
}

func (s *state) isNewEvent(newTime time.Time) bool {
	return newTime.After(s.LastProcessedTimestamp)
}

type storer interface {
	store(data state) error
	load(data *state) error
}

var _ storer = &FileStorer{}

type FileStorer struct {
	filePath string
}

func (fs *FileStorer) store(data state) error {
	file, err := os.OpenFile(fs.filePath, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Errorf("unable to create oom watcher state file at %v: %w", fs.filePath, err)
	}
	defer func() {
		err = file.Close()
	}()
	err = json.NewEncoder(file).Encode(data)
	if err != nil {
		return fmt.Errorf("unable to encode oom watcher state file at %v: %w", fs.filePath, err)
	}
	return nil
}

func (fs *FileStorer) load(data *state) error {
	file, err := os.OpenFile(fs.filePath, os.O_CREATE|os.O_RDONLY, 0644)
	if err != nil {
		return fmt.Errorf("unable to create oom watcher state file at %v: %w", fs.filePath, err)
	}
	defer func() {
		err = file.Close()
	}()
	err = json.NewDecoder(file).Decode(data)
	if err != nil && !errors.Is(err, io.EOF) {
		return fmt.Errorf("unable to decode oom watcher state file at %v: %w", fs.filePath, err)
	}
	return nil
}
