/*
Copyright 2018 The Kubernetes Authors.

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
	"io"
	"os"
	"strings"

	"github.com/golang/glog"
	"github.com/hashicorp/golang-lru"
	"gopkg.in/natefinch/lumberjack.v2"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
)

const (
	// hold 400 file handlers at max
	maxLoggerNum = 400
)

var AllowedSyntax []string = []string{
	"{username}",  // user.username
	"{namespace}", // objectRef.namespace
}

type EventLogger interface {
	GetWriter(ev *auditinternal.Event) io.Writer
	Close()
}

type eventLogger struct {
	path       string
	maxAge     int
	maxBackups int
	maxSize    int

	// staticLogger is used when there is no {username} or {namespace} syntax at path
	staticLogger io.Writer
	// loggers is a least-recently-used cache, the key is the filename, the value is lumberjack logger
	loggers *lru.Cache
}

func (el *eventLogger) GetWriter(ev *auditinternal.Event) io.Writer {
	if el.staticLogger != nil {
		return el.staticLogger
	}

	path := preparePath(ev, el.path)
	out, found := el.loggers.Get(path)
	if found {
		return out.(*lumberjack.Logger)
	}

	glog.V(5).Info("About to emit some audit events to file:", path)
	logger := &lumberjack.Logger{
		Filename:   path,
		MaxAge:     el.maxAge,
		MaxBackups: el.maxBackups,
		MaxSize:    el.maxSize,
	}
	el.loggers.Add(path, logger)
	return logger
}

// Close closes all opening Loggers.
func (el *eventLogger) Close() {
	if el.loggers != nil {
		el.loggers.Purge()
	}
}

// preparePath replaces "{namespace}","{username}" in the path with the info from audit event.
func preparePath(ev *auditinternal.Event, path string) string {
	username := "none"
	if len(ev.User.Username) > 0 {
		username = ev.User.Username
	}

	namespace := "none"
	if ev.ObjectRef != nil && len(ev.ObjectRef.Namespace) != 0 {
		namespace = ev.ObjectRef.Namespace
	}

	path = strings.Replace(path, "{username}", username, -1)
	path = strings.Replace(path, "{namespace}", namespace, -1)
	return path
}

func NewEventLogger(path string, maxAge, maxBackups, maxSize int) (EventLogger, error) {
	var useStaticLogger bool = true
	for _, syntax := range AllowedSyntax {
		if strings.Contains(path, syntax) {
			useStaticLogger = false
		}
	}
	if useStaticLogger {
		var staticLogger io.Writer = os.Stdout
		if path != "-" {
			staticLogger = &lumberjack.Logger{
				Filename:   path,
				MaxAge:     maxAge,
				MaxBackups: maxBackups,
				MaxSize:    maxSize,
			}
		}
		return &eventLogger{
			staticLogger: staticLogger,
		}, nil
	} else {
		loggers, err := lru.NewWithEvict(maxLoggerNum, onEvicted)
		if err != nil {
			return nil, err
		}
		return &eventLogger{
			path:       path,
			maxAge:     maxAge,
			maxBackups: maxBackups,
			maxSize:    maxSize,
			loggers:    loggers,
		}, nil
	}
}

//onEvicated is lru eviction hook, it closes the Logger
func onEvicted(key interface{}, value interface{}) {
	glog.V(5).Info("Closing audit logger in path: ", key.(string))
	logger := value.(*lumberjack.Logger)
	// What will happen if another goroutine tries to write to the logger after closed?
	// lumberjack.Logger will reopen the file, so the openned file handlers may be larger than maxLoggerNum.
	// But it's still thread-safe in this case. This will seldom (or never) happen, because the loggers
	// are storred in a lru cache.
	if err := logger.Close(); err != nil {
		glog.Errorf("Failed to close audit events logger in path: %q, %s", logger.Filename, err.Error())
	}
}
