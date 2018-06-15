// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"context"
	"os/exec"
)

type cmd struct {
	*exec.Cmd
}

func commandContext(ctx context.Context, name string, arg ...string) cmd {
	return cmd{Cmd: exec.CommandContext(ctx, name, arg...)}
}
