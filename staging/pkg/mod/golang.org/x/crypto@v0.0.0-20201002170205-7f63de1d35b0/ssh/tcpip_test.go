// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"testing"
)

func TestAutoPortListenBroken(t *testing.T) {
	broken := "SSH-2.0-OpenSSH_5.9hh11"
	works := "SSH-2.0-OpenSSH_6.1"
	if !isBrokenOpenSSHVersion(broken) {
		t.Errorf("version %q not marked as broken", broken)
	}
	if isBrokenOpenSSHVersion(works) {
		t.Errorf("version %q marked as broken", works)
	}
}
