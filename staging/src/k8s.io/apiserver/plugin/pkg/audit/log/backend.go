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

package log

import (
	"fmt"
	"io"
	"strings"
	"sync"

	"gopkg.in/natefinch/lumberjack.v2"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
)

type backend struct {
	out           io.Writer
	path          string
	maxAge        int
	maxBackups    int
	maxSize       int
	groupedByUser bool
	writers       map[string]io.Writer
	sync.Mutex
	sink chan *auditinternal.Event
}

var _ audit.Backend = &backend{}

func NewBackend(out io.Writer, path string, maxAge, maxBackups, maxSize int, groupedByUser bool) *backend {
	b := &backend{
		path:          path,
		maxAge:        maxAge,
		maxBackups:    maxBackups,
		maxSize:       maxSize,
		groupedByUser: groupedByUser,
		writers:       make(map[string]io.Writer),
		sink:          make(chan *auditinternal.Event, 100),
	}
	if path == "-" || !groupedByUser {
		b.out = out
	}
	return b
}

func (b *backend) ProcessEvents(events ...*auditinternal.Event) {
	for _, ev := range events {
		b.logEvent(ev)
	}
}

func (b *backend) logEvent(ev *auditinternal.Event) {
	out := b.getWriter(ev.User.Username)
	line := audit.EventString(ev)
	if _, err := fmt.Fprintln(out, line); err != nil {
		audit.HandlePluginError("log", err, ev)
	}
}

func (b *backend) getWriter(username string) (out io.Writer) {
	if b.path == "-" || !b.groupedByUser {
		return b.out
	}

	path := b.path + "-user-" + username
	if out, ok := b.writers[path]; ok {
		return out
	}
	b.Lock()
	defer b.Unlock()
	if out, ok := b.writers[path]; ok {
		return out
	}
	b.writers[path] = &lumberjack.Logger{
		Filename:   path,
		MaxAge:     b.maxAge,
		MaxBackups: b.maxBackups,
		MaxSize:    b.maxSize,
	}
	return b.writers[path]
}

func (b *backend) Run(stopCh <-chan struct{}) error {
	return nil
}

func auditStringSlice(inList []string) string {
	quotedElements := make([]string, len(inList))
	for i, in := range inList {
		quotedElements[i] = fmt.Sprintf("%q", in)
	}
	return strings.Join(quotedElements, ",")
}
