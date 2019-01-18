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

package configmapandsecret

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/types"
)

const (
	defaultCommandTimeout = 5 * time.Second
)

// SecretFactory makes Secrets.
type SecretFactory struct {
	fSys fs.FileSystem
	wd   string
}

// NewSecretFactory returns a new SecretFactory.
func NewSecretFactory(fSys fs.FileSystem, wd string) *SecretFactory {
	return &SecretFactory{fSys: fSys, wd: wd}
}

func (f *SecretFactory) makeFreshSecret(args *types.SecretArgs) *corev1.Secret {
	s := &corev1.Secret{}
	s.APIVersion = "v1"
	s.Kind = "Secret"
	s.Name = args.Name
	s.Type = corev1.SecretType(args.Type)
	if s.Type == "" {
		s.Type = corev1.SecretTypeOpaque
	}
	s.Data = map[string][]byte{}
	return s
}

// MakeSecret returns a new secret.
func (f *SecretFactory) MakeSecret(args *types.SecretArgs, options *types.GeneratorOptions) (*corev1.Secret, error) {
	var all []kvPair
	var err error
	s := f.makeFreshSecret(args)

	timeout := defaultCommandTimeout
	if args.TimeoutSeconds != nil {
		log.Println("SecretArgs.TimeoutSeconds will be deprected in next release. Please use GeneratorOptions.TimeoutSeconds instread.")
		timeout = time.Duration(*args.TimeoutSeconds) * time.Second
	}

	pairs, err := f.keyValuesFromEnvFileCommand(args.EnvCommand, timeout, options)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf(
			"env source file: %s",
			args.EnvCommand))
	}
	all = append(all, pairs...)

	pairs, err = f.keyValuesFromCommands(args.Commands, timeout, options)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf(
			"commands %v", args.Commands))
	}
	all = append(all, pairs...)

	for _, kv := range all {
		err = addKvToSecret(s, kv.key, kv.value)
		if err != nil {
			return nil, err
		}
	}
	if options != nil {
		s.SetLabels(options.Labels)
		s.SetAnnotations(options.Annotations)
	}
	return s, nil
}

func addKvToSecret(secret *corev1.Secret, keyName, data string) error {
	// Note, the rules for SecretKeys  keys are the exact same as the ones for ConfigMap.
	if errs := validation.IsConfigMapKey(keyName); len(errs) != 0 {
		return fmt.Errorf("%q is not a valid key name for a Secret: %s", keyName, strings.Join(errs, ";"))
	}
	if _, entryExists := secret.Data[keyName]; entryExists {
		return fmt.Errorf("cannot add key %s, another key by that name already exists", keyName)
	}
	secret.Data[keyName] = []byte(data)
	return nil
}

func (f *SecretFactory) keyValuesFromEnvFileCommand(cmd string, timeout time.Duration, options *types.GeneratorOptions) ([]kvPair, error) {
	content, err := f.createSecretKey(cmd, timeout, options)
	if err != nil {
		return nil, err
	}
	return keyValuesFromLines(content)
}

func (f *SecretFactory) keyValuesFromCommands(sources map[string]string, timeout time.Duration, options *types.GeneratorOptions) ([]kvPair, error) {
	var kvs []kvPair
	for k, cmd := range sources {
		content, err := f.createSecretKey(cmd, timeout, options)
		if err != nil {
			return nil, err
		}
		kvs = append(kvs, kvPair{key: k, value: string(content)})
	}
	return kvs, nil
}

// Run a command, return its output as the secret.
func (f *SecretFactory) createSecretKey(command string, timeout time.Duration, options *types.GeneratorOptions) ([]byte, error) {
	if !f.fSys.IsDir(f.wd) {
		f.wd = filepath.Dir(f.wd)
		if !f.fSys.IsDir(f.wd) {
			return nil, errors.New("not a directory: " + f.wd)
		}
	}

	if options != nil && options.TimeoutSeconds != nil {
		t := time.Duration(*options.TimeoutSeconds) * time.Second
		if t > timeout {
			timeout = t
		}
	}

	var commands []string
	if options == nil || len(options.Shell) == 0 {
		commands = []string{"sh", "-c", command}
	} else {
		commands = append(options.Shell, command)
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, commands[0], commands[1:]...)
	cmd.Dir = f.wd
	return cmd.Output()
}
