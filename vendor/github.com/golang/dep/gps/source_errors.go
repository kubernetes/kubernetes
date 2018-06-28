// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"github.com/Masterminds/vcs"
	"github.com/pkg/errors"
)

// unwrapVcsErr recognizes *vcs.LocalError and *vsc.RemoteError, and returns a form
// preserving the actual vcs command output and error, in addition to the message.
// All other types pass through unchanged.
func unwrapVcsErr(err error) error {
	var cause error
	var out, msg string

	switch t := err.(type) {
	case *vcs.LocalError:
		cause, out, msg = t.Original(), t.Out(), t.Error()
	case *vcs.RemoteError:
		cause, out, msg = t.Original(), t.Out(), t.Error()

	default:
		return err
	}

	if cause == nil {
		cause = errors.New(out)
	} else {
		cause = errors.Wrap(cause, out)
	}
	return errors.Wrap(cause, msg)
}
