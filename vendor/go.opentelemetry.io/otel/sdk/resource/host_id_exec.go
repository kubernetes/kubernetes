// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build darwin || dragonfly || freebsd || netbsd || openbsd || solaris

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"context"
	"os/exec"
)

func execCommand(name string, arg ...string) (string, error) {
	cmd := exec.CommandContext(context.Background(), name, arg...)
	b, err := cmd.Output()
	if err != nil {
		return "", err
	}

	return string(b), nil
}
