/*
Copyright 2020 The Kubernetes Authors.

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

package proxy

import (
	"encoding/base64"
	"fmt"
	"io"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/drivers/csi-test/mock/service"
)

type PodDirIO struct {
	F             *framework.Framework
	Namespace     string
	PodName       string
	ContainerName string
	Logger        *klog.Logger
}

var _ service.DirIO = PodDirIO{}

func (p PodDirIO) DirExists(path string) (bool, error) {
	stdout, stderr, err := p.execute([]string{
		"sh",
		"-c",
		fmt.Sprintf("if ! [ -e '%s' ]; then echo notexist; elif [ -d '%s' ]; then echo dir; else echo nodir; fi", path, path),
	}, nil)
	if err != nil {
		return false, fmt.Errorf("error executing dir test commands: stderr=%q, %v", stderr, err)
	}
	switch stdout {
	case "notexist":
		return false, nil
	case "nodir":
		return false, fmt.Errorf("%s: not a directory", path)
	case "dir":
		return true, nil
	default:
		return false, fmt.Errorf("unexpected output from dir test commands: %q", stdout)
	}
}

func (p PodDirIO) Mkdir(path string) error {
	_, stderr, err := p.execute([]string{"mkdir", path}, nil)
	if err != nil {
		return fmt.Errorf("mkdir %q: stderr=%q, %v", path, stderr, err)
	}
	return nil
}

func (p PodDirIO) CreateFile(path string, content io.Reader) error {
	// Piping the content into dd via stdin turned out to be unreliable.
	// Sometimes dd would stop after writing zero bytes, without an error
	// from ExecWithOptions (reported as
	// https://github.com/kubernetes/kubernetes/issues/112834).
	//
	// Therefore the content is now encoded inside the command itself.
	data, err := io.ReadAll(content)
	if err != nil {
		return fmt.Errorf("read content: %w", err)
	}
	encoded := make([]byte, base64.StdEncoding.EncodedLen(len(data)))
	base64.StdEncoding.Encode(encoded, data)
	_, stderr, err := p.execute([]string{"sh", "-c", fmt.Sprintf(`base64 -d >'%s' <<EOF
%s
EOF`, path, string(encoded))}, nil)
	if err != nil {
		return fmt.Errorf("decoding into %q: stderr=%q, %v", path, stderr, err)
	}
	return nil
}

func (p PodDirIO) Rename(oldPath, newPath string) error {
	_, stderr, err := p.execute([]string{"mv", oldPath, newPath}, nil)
	if err != nil {
		return fmt.Errorf("rename %q -> %q: stderr=%q, %v", oldPath, newPath, stderr, err)
	}
	return nil
}

func (p PodDirIO) RemoveAll(path string) error {
	_, stderr, err := p.execute([]string{"rm", "-rf", path}, nil)
	if err != nil {
		return fmt.Errorf("rm -rf %q: stderr=%q, %v", path, stderr, err)
	}
	return nil
}

func (p PodDirIO) execute(command []string, stdin io.Reader) (string, string, error) {
	stdout, stderr, err := e2epod.ExecWithOptions(p.F, e2epod.ExecOptions{
		Command:       command,
		Namespace:     p.Namespace,
		PodName:       p.PodName,
		ContainerName: p.ContainerName,
		Stdin:         stdin,
		CaptureStdout: true,
		CaptureStderr: true,
		Quiet:         true,
	})
	if p.Logger != nil {
		p.Logger.Info("Command completed", "command", command, "stdout", stdout, "stderr", stderr, "err", err)

	}
	return stdout, stderr, err

}
