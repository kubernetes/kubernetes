// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"context"
	"fmt"
	"os"
	"os/user"
	"path/filepath"
	"runtime"

	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

type pidProvider func() int
type executablePathProvider func() (string, error)
type commandArgsProvider func() []string
type ownerProvider func() (*user.User, error)
type runtimeNameProvider func() string
type runtimeVersionProvider func() string
type runtimeOSProvider func() string
type runtimeArchProvider func() string

var (
	defaultPidProvider            pidProvider            = os.Getpid
	defaultExecutablePathProvider executablePathProvider = os.Executable
	defaultCommandArgsProvider    commandArgsProvider    = func() []string { return os.Args }
	defaultOwnerProvider          ownerProvider          = user.Current
	defaultRuntimeNameProvider    runtimeNameProvider    = func() string {
		if runtime.Compiler == "gc" {
			return "go"
		}
		return runtime.Compiler
	}
	defaultRuntimeVersionProvider runtimeVersionProvider = runtime.Version
	defaultRuntimeOSProvider      runtimeOSProvider      = func() string { return runtime.GOOS }
	defaultRuntimeArchProvider    runtimeArchProvider    = func() string { return runtime.GOARCH }
)

var (
	pid            = defaultPidProvider
	executablePath = defaultExecutablePathProvider
	commandArgs    = defaultCommandArgsProvider
	owner          = defaultOwnerProvider
	runtimeName    = defaultRuntimeNameProvider
	runtimeVersion = defaultRuntimeVersionProvider
	runtimeOS      = defaultRuntimeOSProvider
	runtimeArch    = defaultRuntimeArchProvider
)

func setDefaultOSProviders() {
	setOSProviders(
		defaultPidProvider,
		defaultExecutablePathProvider,
		defaultCommandArgsProvider,
	)
}

func setOSProviders(
	pidProvider pidProvider,
	executablePathProvider executablePathProvider,
	commandArgsProvider commandArgsProvider,
) {
	pid = pidProvider
	executablePath = executablePathProvider
	commandArgs = commandArgsProvider
}

func setDefaultRuntimeProviders() {
	setRuntimeProviders(
		defaultRuntimeNameProvider,
		defaultRuntimeVersionProvider,
		defaultRuntimeOSProvider,
		defaultRuntimeArchProvider,
	)
}

func setRuntimeProviders(
	runtimeNameProvider runtimeNameProvider,
	runtimeVersionProvider runtimeVersionProvider,
	runtimeOSProvider runtimeOSProvider,
	runtimeArchProvider runtimeArchProvider,
) {
	runtimeName = runtimeNameProvider
	runtimeVersion = runtimeVersionProvider
	runtimeOS = runtimeOSProvider
	runtimeArch = runtimeArchProvider
}

func setDefaultUserProviders() {
	setUserProviders(defaultOwnerProvider)
}

func setUserProviders(ownerProvider ownerProvider) {
	owner = ownerProvider
}

type processPIDDetector struct{}
type processExecutableNameDetector struct{}
type processExecutablePathDetector struct{}
type processCommandArgsDetector struct{}
type processOwnerDetector struct{}
type processRuntimeNameDetector struct{}
type processRuntimeVersionDetector struct{}
type processRuntimeDescriptionDetector struct{}

// Detect returns a *Resource that describes the process identifier (PID) of the
// executing process.
func (processPIDDetector) Detect(ctx context.Context) (*Resource, error) {
	return NewWithAttributes(semconv.SchemaURL, semconv.ProcessPID(pid())), nil
}

// Detect returns a *Resource that describes the name of the process executable.
func (processExecutableNameDetector) Detect(ctx context.Context) (*Resource, error) {
	executableName := filepath.Base(commandArgs()[0])

	return NewWithAttributes(semconv.SchemaURL, semconv.ProcessExecutableName(executableName)), nil
}

// Detect returns a *Resource that describes the full path of the process executable.
func (processExecutablePathDetector) Detect(ctx context.Context) (*Resource, error) {
	executablePath, err := executablePath()
	if err != nil {
		return nil, err
	}

	return NewWithAttributes(semconv.SchemaURL, semconv.ProcessExecutablePath(executablePath)), nil
}

// Detect returns a *Resource that describes all the command arguments as received
// by the process.
func (processCommandArgsDetector) Detect(ctx context.Context) (*Resource, error) {
	return NewWithAttributes(semconv.SchemaURL, semconv.ProcessCommandArgs(commandArgs()...)), nil
}

// Detect returns a *Resource that describes the username of the user that owns the
// process.
func (processOwnerDetector) Detect(ctx context.Context) (*Resource, error) {
	owner, err := owner()
	if err != nil {
		return nil, err
	}

	return NewWithAttributes(semconv.SchemaURL, semconv.ProcessOwner(owner.Username)), nil
}

// Detect returns a *Resource that describes the name of the compiler used to compile
// this process image.
func (processRuntimeNameDetector) Detect(ctx context.Context) (*Resource, error) {
	return NewWithAttributes(semconv.SchemaURL, semconv.ProcessRuntimeName(runtimeName())), nil
}

// Detect returns a *Resource that describes the version of the runtime of this process.
func (processRuntimeVersionDetector) Detect(ctx context.Context) (*Resource, error) {
	return NewWithAttributes(semconv.SchemaURL, semconv.ProcessRuntimeVersion(runtimeVersion())), nil
}

// Detect returns a *Resource that describes the runtime of this process.
func (processRuntimeDescriptionDetector) Detect(ctx context.Context) (*Resource, error) {
	runtimeDescription := fmt.Sprintf(
		"go version %s %s/%s", runtimeVersion(), runtimeOS(), runtimeArch())

	return NewWithAttributes(
		semconv.SchemaURL,
		semconv.ProcessRuntimeDescription(runtimeDescription),
	), nil
}
