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

package auditor

import (
	"fmt"
	"io"
	"os"
	"sync"

	"gopkg.in/natefinch/lumberjack.v2"
)

type GroupedAuditor struct {
	path               string
	maxAge             int
	maxBackups         int
	maxSize            int
	groupedByUser      bool
	groupedByNamespace bool
	auditors           map[string]io.Writer
	sync.Mutex
}

type Auditor interface {
	Audit(user, namespace, context string) (int, error)
}

func NewGroupedAuditor(path string, maxAge, maxBackups, maxSize int, groupedByUser, groupedByNamespace bool) *GroupedAuditor {
	return &GroupedAuditor{
		path:               path,
		maxAge:             maxAge,
		maxBackups:         maxBackups,
		maxSize:            maxSize,
		groupedByUser:      groupedByUser,
		groupedByNamespace: groupedByNamespace,
		auditors:           make(map[string]io.Writer),
	}
}

// Audit audits access logs to the suitable destination
func (a *GroupedAuditor) Audit(user, namespace, context string) (n int, err error) {
	out := a.getWriter(user, namespace)
	return fmt.Fprintf(out, context)
}
func (a *GroupedAuditor) getWriter(user, namespace string) (out io.Writer) {
	if a.path == "-" {
		return os.Stdout
	}
	path := a.path
	if a.groupedByUser {
		path += "-user-" + user
	} else if a.groupedByNamespace {
		path += "-namespace-" + namespace
	}
	return a.getWriterSync(path)
}
func (a *GroupedAuditor) getWriterSync(path string) (out io.Writer) {
	if out, ok := a.auditors[path]; ok {
		return out
	}
	a.Lock()
	defer a.Unlock()
	if out, ok := a.auditors[path]; ok {
		return out
	}
	a.auditors[path] = &lumberjack.Logger{
		Filename:   path,
		MaxAge:     a.maxAge,
		MaxBackups: a.maxBackups,
		MaxSize:    a.maxSize,
	}
	return a.auditors[path]
}
