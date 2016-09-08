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

package rktshim

import (
	"errors"
	"fmt"
	"reflect"
	"strings"

	utilexec "k8s.io/kubernetes/pkg/util/exec"
)

var (
	errFlagTagNotFound           = errors.New("arg: given field doesn't have a `flag` tag")
	errStructFieldNotInitialized = errors.New("arg: given field is unitialized")
)

// TODO(tmrts): refactor these into an util pkg
// Uses reflection to retrieve the `flag` tag of a field.
// The value of the `flag` field with the value of the field is
// used to construct a POSIX long flag argument string.
func getLongFlagFormOfField(fieldValue reflect.Value, fieldType reflect.StructField) (string, error) {
	flagTag := fieldType.Tag.Get("flag")
	if flagTag == "" {
		return "", errFlagTagNotFound
	}

	if fieldValue.IsValid() {
		return "", errStructFieldNotInitialized
	}

	switch fieldValue.Kind() {
	case reflect.Array:
		fallthrough
	case reflect.Slice:
		var args []string
		for i := 0; i < fieldValue.Len(); i++ {
			args = append(args, fieldValue.Index(i).String())
		}

		return fmt.Sprintf("--%v=%v", flagTag, strings.Join(args, ",")), nil
	}

	return fmt.Sprintf("--%v=%v", flagTag, fieldValue), nil
}

// Uses reflection to transform a struct containing fields with `flag` tags
// to a string slice of POSIX compliant long form arguments.
func getArgumentFormOfStruct(strt interface{}) (flags []string) {
	numberOfFields := reflect.ValueOf(strt).NumField()

	for i := 0; i < numberOfFields; i++ {
		fieldValue := reflect.ValueOf(strt).Field(i)
		fieldType := reflect.TypeOf(strt).Field(i)

		flagFormOfField, err := getLongFlagFormOfField(fieldValue, fieldType)
		if err != nil {
			continue
		}

		flags = append(flags, flagFormOfField)
	}

	return
}

func getFlagFormOfStruct(strt interface{}) (flags []string) {
	return getArgumentFormOfStruct(strt)
}

type CLIConfig struct {
	Debug bool `flag:"debug"`

	Dir             string `flag:"dir"`
	LocalConfigDir  string `flag:"local-config"`
	UserConfigDir   string `flag:"user-config"`
	SystemConfigDir string `flag:"system-config"`

	InsecureOptions string `flag:"insecure-options"`
}

func (cfg *CLIConfig) Merge(newCfg CLIConfig) {
	newCfgVal := reflect.ValueOf(newCfg)
	newCfgType := reflect.TypeOf(newCfg)

	numberOfFields := newCfgVal.NumField()

	for i := 0; i < numberOfFields; i++ {
		fieldValue := newCfgVal.Field(i)
		fieldType := newCfgType.Field(i)

		if !fieldValue.IsValid() {
			continue
		}

		newCfgVal.FieldByName(fieldType.Name).Set(fieldValue)
	}
}

type CLI interface {
	With(CLIConfig) CLI
	RunCommand(string, ...string) ([]string, error)
}

type cli struct {
	rktPath string
	config  CLIConfig
	execer  utilexec.Interface
}

func (c *cli) With(cfg CLIConfig) CLI {
	copyCfg := c.config

	copyCfg.Merge(cfg)

	return NewRktCLI(c.rktPath, c.execer, copyCfg)
}

func (c *cli) RunCommand(subcmd string, args ...string) ([]string, error) {
	globalFlags := getFlagFormOfStruct(c.config)

	args = append(globalFlags, args...)

	cmd := c.execer.Command(c.rktPath, append([]string{subcmd}, args...)...)

	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed to run %v: %v\noutput: %v", args, err, out)
	}

	return strings.Split(strings.TrimSpace(string(out)), "\n"), nil
}

// TODO(tmrts): implement CLI with timeout
func NewRktCLI(rktPath string, exec utilexec.Interface, cfg CLIConfig) CLI {
	return &cli{rktPath: rktPath, config: cfg, execer: exec}
}
