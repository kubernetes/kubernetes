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

	"github.com/sirupsen/logrus"
)

// StepLogger is a step counting logger implementation.
type StepLogger struct {
	*logrus.Logger
	steps       uint
	currentStep uint
}

// NewStepLogger creates a new logger
func NewStepLogger(steps uint) *StepLogger {
	return &StepLogger{
		Logger:      logrus.StandardLogger(),
		steps:       steps,
		currentStep: 0,
	}
}

// WithStep increments the internal step counter and adds the output to the
// field.
func (l *StepLogger) WithStep() *logrus.Entry {
	l.currentStep++
	return l.WithField(
		"step",
		fmt.Sprintf("%d/%d", l.currentStep, l.steps),
	)
}
