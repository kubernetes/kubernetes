package remotecommand

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Verify that [exec.ExitError] can be returned directly by Executor implementations.
var _ ExitError = (*exec.ExitError)(nil)

type testExitError struct {
	error
	code int
}

func (e testExitError) ExitCode() int { return e.code }

type testLegacyExitError struct {
	error
	code   int
	exited bool
}

func (e testLegacyExitError) Exited() bool    { return e.exited }
func (e testLegacyExitError) ExitStatus() int { return e.code }

type testBothExitError struct {
	error
	exitCode   int
	exitStatus int
	exited     bool
}

func (e testBothExitError) ExitCode() int   { return e.exitCode }
func (e testBothExitError) Exited() bool    { return e.exited }
func (e testBothExitError) ExitStatus() int { return e.exitStatus }

func TestExitCode(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want int
		ok   bool
	}{
		{
			name: "exit error",
			err:  testExitError{code: 42},
			want: 42,
			ok:   true,
		},
		{
			name: "wrapped exit error",
			err:  fmt.Errorf("wrapped: %w", testExitError{code: 42}),
			want: 42,
			ok:   true,
		},
		{
			name: "legacy exit error",
			err:  testLegacyExitError{code: 42, exited: true},
			want: 42,
			ok:   true,
		},
		{
			name: "legacy not exited",
			err:  testLegacyExitError{code: 42},
		},
		{
			// Use a zero-value ProcessState to avoid spawning a process just to obtain an
			// *exec.ExitError. Its exit code is not meaningful for this test.
			name: "stdlib ExitError",
			err:  &exec.ExitError{ProcessState: &os.ProcessState{}},
			want: 0,
			ok:   true,
		},
		{
			// Verify that wrapped stdlib ExitErrors are detected without executing a command.
			name: "wrapped stdlib ExitError",
			err:  fmt.Errorf("wrapped: %w", &exec.ExitError{ProcessState: &os.ProcessState{}}),
			want: 0,
			ok:   true,
		},
		{
			name: "regular error",
			err:  errors.New("boom"),
		},
		{
			name: "nil",
		},
		{
			name: "prefer ExitError",
			err:  testBothExitError{exitCode: 42, exitStatus: 13},
			want: 42,
			ok:   true,
		},
		{
			name: "negative exit code",
			err:  testExitError{code: -1},
		},
		{
			name: "negative exit code falls back to legacy",
			err: testBothExitError{
				exitCode:   -1,
				exitStatus: 42,
				exited:     true,
			},
			want: 42,
			ok:   true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := exitCode(tc.err)
			assert.Equal(t, tc.want, got)
			assert.Equal(t, tc.ok, ok)
		})
	}
}
