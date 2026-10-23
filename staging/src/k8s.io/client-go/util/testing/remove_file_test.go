/*
Copyright The Kubernetes Authors.

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

package testing

import (
	"errors"
	"os"
	"strings"
	"testing"
)

func TestFatalToError(t *testing.T) {
	for name, tc := range map[string]struct {
		initialErr error
		calls      []string
		wantErr    string
	}{
		"no-calls": {
			calls:   nil,
			wantErr: "",
		},
		"single-call": {
			calls:   []string{"boom"},
			wantErr: "boom",
		},
		"multiple-calls-get-joined": {
			calls:   []string{"first", "second"},
			wantErr: "first\nsecond",
		},
		"initial-error-gets-joined": {
			initialErr: errors.New("pre-existing"),
			calls:      []string{"boom"},
			wantErr:    "pre-existing\nboom",
		},
	} {
		t.Run(name, func(t *testing.T) {
			finalErr := tc.initialErr
			tb := FatalToError(&finalErr)
			for _, msg := range tc.calls {
				tb.Helper()
				tb.Fatalf("%s", msg)
			}
			if tc.wantErr == "" {
				if finalErr != nil {
					t.Fatalf("expected no error, got: %v", finalErr)
				}
				return
			}
			if finalErr == nil {
				t.Fatalf("expected error %q, got nil", tc.wantErr)
			}
			if finalErr.Error() != tc.wantErr {
				t.Errorf("expected error %q, got %q", tc.wantErr, finalErr.Error())
			}
		})
	}
}

func TestCloseAndRemoveWithFatalToError(t *testing.T) {
	for name, tc := range map[string]struct {
		setup       func(t *testing.T) *os.File
		wantErr     bool
		wantErrText string
		wantRemoved bool
	}{
		"closes-and-removes-valid-file": {
			setup: func(t *testing.T) *os.File {
				f, err := os.CreateTemp(t.TempDir(), "close-and-remove")
				if err != nil {
					t.Fatalf("failed to create temp file: %v", err)
				}
				return f
			},
			wantErr:     false,
			wantRemoved: true,
		},
		"ignores-nil-file": {
			setup: func(t *testing.T) *os.File {
				return nil
			},
			wantErr: false,
		},
		"reports-error-for-already-closed-file": {
			setup: func(t *testing.T) *os.File {
				f, err := os.CreateTemp(t.TempDir(), "close-and-remove")
				if err != nil {
					t.Fatalf("failed to create temp file: %v", err)
				}
				if err := f.Close(); err != nil {
					t.Fatalf("failed to pre-close temp file: %v", err)
				}
				return f
			},
			wantErr:     true,
			wantErrText: "Error closing",
			wantRemoved: true,
		},
		"reports-error-for-missing-file": {
			setup: func(t *testing.T) *os.File {
				f, err := os.CreateTemp(t.TempDir(), "close-and-remove")
				if err != nil {
					t.Fatalf("failed to create temp file: %v", err)
				}
				if err := os.Remove(f.Name()); err != nil {
					t.Fatalf("failed to pre-remove temp file: %v", err)
				}
				return f
			},
			wantErr:     true,
			wantErrText: "Error removing",
			wantRemoved: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			f := tc.setup(t)

			var finalErr error
			CloseAndRemove(FatalToError(&finalErr), f)

			if tc.wantErr {
				if finalErr == nil {
					t.Fatalf("expected an error, got nil")
				}
				if !strings.Contains(finalErr.Error(), tc.wantErrText) {
					t.Errorf("expected error to contain %q, got: %v", tc.wantErrText, finalErr)
				}
			} else if finalErr != nil {
				t.Fatalf("expected no error, got: %v", finalErr)
			}

			if f == nil {
				return
			}
			_, err := os.Stat(f.Name())
			removed := errors.Is(err, os.ErrNotExist)
			if removed != tc.wantRemoved {
				t.Errorf("file %s: expected removed=%v, got removed=%v (stat err: %v)", f.Name(), tc.wantRemoved, removed, err)
			}
		})
	}
}

func TestCloseAndRemoveWithRealT(t *testing.T) {
	// CloseAndRemove must keep working with a real *testing.T, which is the
	// common case in production code.
	f, err := os.CreateTemp(t.TempDir(), "close-and-remove")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	CloseAndRemove(t, f)
	if _, err := os.Stat(f.Name()); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected file to be removed")
	}
}
