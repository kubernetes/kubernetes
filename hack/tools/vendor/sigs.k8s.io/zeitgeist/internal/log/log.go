/*
Copyright 2021 The Kubernetes Authors.

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
	"os"
	"strings"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"

	"sigs.k8s.io/zeitgeist/internal/command"
)

// SetupGlobalLogger uses to provided log level string and applies it globally.
func SetupGlobalLogger(level string) error {
	logrus.SetFormatter(&logrus.TextFormatter{
		DisableTimestamp: true,
		ForceColors:      true,
	})

	lvl, err := logrus.ParseLevel(level)
	if err != nil {
		return errors.Wrapf(err, "setting log level to %s", level)
	}
	logrus.SetLevel(lvl)
	if lvl >= logrus.DebugLevel {
		logrus.Debug("Setting commands globally into verbose mode")
		command.SetGlobalVerbose(true)
	}
	logrus.AddHook(NewFilenameHook())
	logrus.Debugf("Using log level %q", lvl)
	return nil
}

// ToFile adds a file destination to the global logger.
func ToFile(fileName string) error {
	file, err := os.OpenFile(fileName, os.O_WRONLY|os.O_CREATE, 0755)
	if err != nil {
		return errors.Wrap(err, "open log file")
	}

	writer := io.MultiWriter(logrus.StandardLogger().Out, file)
	logrus.SetOutput(writer)

	return nil
}

// LevelNames returns a comma separated list of available levels.
func LevelNames() string {
	levels := []string{}
	for _, level := range logrus.AllLevels {
		levels = append(levels, fmt.Sprintf("'%s'", level.String()))
	}
	return strings.Join(levels, ", ")
}
