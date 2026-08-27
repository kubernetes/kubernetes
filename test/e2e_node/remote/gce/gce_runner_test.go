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

package gce

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/test/e2e_node/remote"
)

func TestWaitForGCEInstanceReady(t *testing.T) {
	const instanceName = "test-instance"
	ready := &gceInstance{Name: instanceName, Status: "RUNNING"}
	describeErr := errors.New("last describe failure")
	probeErr := errors.New("last runtime failure")

	type observation struct {
		instance            *gceInstance
		describeErr         error
		probeErr            error
		cancelAfterDescribe bool
		cancelAfterProbe    bool
	}
	tests := []struct {
		name               string
		observations       []observation
		cancelBefore       bool
		timeout            bool
		wantInstance       *gceInstance
		wantErr            error
		wantLastErr        error
		wantErrContains    string
		wantErrNotContains string
	}{
		{
			name:         "describe failures are retried",
			observations: []observation{{describeErr: errors.New("attempt 1")}, {describeErr: errors.New("attempt 2")}, {instance: ready}},
			wantInstance: ready,
		},
		{
			name:         "canceled context stops before polling",
			cancelBefore: true,
			wantErr:      context.Canceled,
		},
		{
			name:         "cancellation after describe skips the runtime probe",
			observations: []observation{{instance: ready, cancelAfterDescribe: true}},
			wantErr:      context.Canceled,
		},
		{
			name:               "a newer successful observation clears an older failure",
			observations:       []observation{{instance: ready, probeErr: errors.New("stale runtime failure")}, {instance: ready, cancelAfterProbe: true}},
			wantErr:            context.Canceled,
			wantErrNotContains: "stale runtime failure",
		},
		{
			name:            "non-running observations preserve the last status",
			observations:    []observation{{instance: &gceInstance{Name: instanceName, Status: "PROVISIONING"}}, {instance: &gceInstance{Name: instanceName, Status: "STAGING"}}, {instance: &gceInstance{Name: instanceName, Status: "STOPPING"}, cancelAfterDescribe: true}},
			wantErr:         context.Canceled,
			wantErrContains: `last observation: instance "test-instance" not RUNNING, status="STOPPING"`,
		},
		{
			name:            "timeout preserves the last describe error",
			observations:    []observation{{describeErr: describeErr}},
			timeout:         true,
			wantErr:         context.DeadlineExceeded,
			wantLastErr:     describeErr,
			wantErrContains: `last observation: describe instance "test-instance": last describe failure`,
		},
		{
			name:            "runtime failures preserve the last probe error",
			observations:    []observation{{instance: ready, probeErr: errors.New("attempt 1")}, {instance: ready, probeErr: errors.New("attempt 2")}, {instance: ready, probeErr: probeErr, cancelAfterProbe: true}},
			wantErr:         context.Canceled,
			wantLastErr:     probeErr,
			wantErrContains: `last observation: probe runtime on instance "test-instance": last runtime failure`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if tc.cancelBefore {
				cancel()
			}

			pollTimeout := time.Second
			pollInterval := time.Millisecond
			if tc.timeout {
				pollInterval = time.Second
				pollTimeout = 10 * time.Millisecond
			}
			describeCalls := 0
			got, err := waitForGCEInstanceReady(
				ctx,
				instanceName,
				pollInterval,
				pollTimeout,
				func(_ context.Context, _ string) (*gceInstance, error) {
					if describeCalls >= len(tc.observations) {
						t.Fatalf("unexpected describe call %d", describeCalls+1)
					}
					current := tc.observations[describeCalls]
					describeCalls++
					if current.cancelAfterDescribe {
						cancel()
					}
					return current.instance, current.describeErr
				},
				func(_ context.Context, _ *gceInstance) error {
					current := tc.observations[describeCalls-1]
					if current.cancelAfterProbe {
						cancel()
					}
					return current.probeErr
				},
			)
			if got != tc.wantInstance {
				t.Errorf("waitForGCEInstanceReady() instance = %p, want %p", got, tc.wantInstance)
			}
			if tc.wantErr == nil && err != nil {
				t.Fatalf("waitForGCEInstanceReady() unexpected error: %v", err)
			}
			if tc.wantErr != nil && !errors.Is(err, tc.wantErr) {
				t.Errorf("waitForGCEInstanceReady() error = %v, want it to wrap %v", err, tc.wantErr)
			}
			if tc.wantLastErr != nil && !errors.Is(err, tc.wantLastErr) {
				t.Errorf("waitForGCEInstanceReady() error = %v, want it to wrap %v", err, tc.wantLastErr)
			}
			if tc.wantErrContains != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErrContains)) {
				t.Errorf("waitForGCEInstanceReady() error = %q, want it to contain %q", err, tc.wantErrContains)
			}
			if tc.wantErrNotContains != "" && err != nil && strings.Contains(err.Error(), tc.wantErrNotContains) {
				t.Errorf("waitForGCEInstanceReady() error = %q, must not contain %q", err, tc.wantErrNotContains)
			}
		})
	}
}

func TestWaitForGCEInstanceReadyBoundsOperations(t *testing.T) {
	t.Run("before first observation", func(t *testing.T) {
		instance, err := waitForGCEInstanceReady(
			t.Context(), "test-instance", time.Second, 0,
			func(context.Context, string) (*gceInstance, error) {
				t.Fatal("describe called after the deadline")
				return nil, nil
			},
			func(context.Context, *gceInstance) error {
				t.Fatal("runtime probe called after the deadline")
				return nil
			},
		)
		if instance != nil || !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("waitForGCEInstanceReady() = (%v, %v), want (nil, context deadline exceeded)", instance, err)
		}
	})

	for _, stage := range []string{"describe", "runtime probe"} {
		t.Run(stage, func(t *testing.T) {
			instance, err := waitForGCEInstanceReady(
				t.Context(),
				"test-instance",
				time.Millisecond,
				10*time.Millisecond,
				func(ctx context.Context, _ string) (*gceInstance, error) {
					if stage == "describe" {
						<-ctx.Done()
						return nil, ctx.Err()
					}
					return &gceInstance{Name: "test-instance", Status: "RUNNING"}, nil
				},
				func(ctx context.Context, _ *gceInstance) error {
					if stage == "runtime probe" {
						<-ctx.Done()
						return ctx.Err()
					}
					return nil
				},
			)
			if instance != nil || !errors.Is(err, context.DeadlineExceeded) {
				t.Errorf("waitForGCEInstanceReady() = (%v, %v), want (nil, context deadline exceeded)", instance, err)
			}
		})
	}
}

func TestProbeGCEInstanceRuntime(t *testing.T) {
	sshErr := errors.New("ssh failed")
	tests := []struct {
		name        string
		output      string
		sshErr      error
		wantErr     string
		wantWrapped error
	}{
		{
			name:    "substring matches are not accepted",
			output:  "\nfoo-containerd.service loaded active running helper for containerd.service",
			wantErr: "is not running containerd or CRI-O",
		},
		{name: "systemd status marker before CRI-O is accepted", output: "● crio.service loaded active running Container Runtime Interface for OCI"},
		{
			name:        "SSH failure preserves command error and output",
			output:      "connection reset",
			sshErr:      sshErr,
			wantErr:     `output: "connection reset"`,
			wantWrapped: sshErr,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			instance := &gceInstance{Name: "test-instance", Status: "RUNNING"}
			var gotCommand []string
			err := probeGCEInstanceRuntime(t.Context(), instance, func(_ context.Context, _ string, command ...string) (string, error) {
				gotCommand = command
				return tc.output, tc.sshErr
			})
			if wantCommand := []string{"systemctl", "list-units", "--type=service", "--state=running", "--no-legend", "--plain", "containerd.service", "crio.service"}; !reflect.DeepEqual(gotCommand, wantCommand) {
				t.Errorf("SSH command = %q, want %q", gotCommand, wantCommand)
			}
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("probeGCEInstanceRuntime() unexpected error: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("probeGCEInstanceRuntime() error = %v, want it to contain %q", err, tc.wantErr)
			}
			if tc.wantWrapped != nil && !errors.Is(err, tc.wantWrapped) {
				t.Errorf("probeGCEInstanceRuntime() error = %v, want it to wrap %v", err, tc.wantWrapped)
			}
		})
	}
}

func TestCreateGCEInstanceRechecksReadinessAfterPostSetup(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("requires executable shell scripts")
	}

	tests := []struct {
		name        string
		imageConfig internalGCEImage
		wantSSHCall string
	}{
		{
			name:        "kernel update",
			imageConfig: internalGCEImage{image: "ubuntu-image", project: "image-project", kernelArguments: []string{"test-argument=1"}},
			wantSSHCall: "update-grub",
		},
		{
			name: "cloud-init",
			imageConfig: internalGCEImage{
				image: "cloud-image", project: "image-project",
				metadata: &gceMetadata{Items: []gceMetadataItems{{Key: "user-data", Value: "#cloud-config\n"}}},
			},
			wantSSHCall: "/var/lib/cloud/instance/boot-finished",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			instanceName := "test-prefix-" + tc.imageConfig.image
			instanceJSON := fmt.Sprintf(`{"name":%q,"status":"RUNNING"}`, instanceName)
			binDir := t.TempDir()
			sshLog := filepath.Join(binDir, "ssh.log")
			gcloudScript := fmt.Sprintf(`#!/bin/sh
case "$*" in
  *"project-info describe"*) printf '%%s\n' '{"defaultServiceAccount":"test@example.invalid"}' ;;
  *"instances describe"*) printf '%%s\n' '%s' ;;
  *) exit 99 ;;
esac
`, instanceJSON)
			sshScript := fmt.Sprintf(`#!/bin/sh
printf '%%s\n' "$*" >> %q
case "$*" in
  *"systemctl list-units"*) printf '%%s\n' 'containerd.service loaded active running containerd container runtime' ;;
  *" reboot") exit 1 ;;
  *) exit 0 ;;
esac
`, sshLog)
			for name, content := range map[string]string{"gcloud": gcloudScript, "ssh": sshScript} {
				if err := os.WriteFile(filepath.Join(binDir, name), []byte(content), 0o755); err != nil {
					t.Fatalf("write fake %s: %v", name, err)
				}
			}
			t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))

			oldProject, oldZone := *project, *zone
			t.Cleanup(func() { *project, *zone = oldProject, oldZone })
			*project, *zone = "test-project", ""

			runner := &GCERunner{cfg: remote.Config{InstanceNamePrefix: "test-prefix"}}
			name, err := runner.createGCEInstance(&tc.imageConfig)
			if err != nil {
				t.Fatalf("createGCEInstance() unexpected error: %v", err)
			}
			if name != instanceName {
				t.Errorf("createGCEInstance() name = %q, want %q", name, instanceName)
			}
			sshCalls, err := os.ReadFile(sshLog)
			if err != nil {
				t.Fatalf("read SSH log: %v", err)
			}
			if got := strings.Count(string(sshCalls), "systemctl list-units"); got != 2 {
				t.Errorf("runtime probe calls = %d, want 2; log:\n%s", got, sshCalls)
			}
			if !strings.Contains(string(sshCalls), tc.wantSSHCall) {
				t.Errorf("SSH call %q not found; log:\n%s", tc.wantSSHCall, sshCalls)
			}
		})
	}
}

func TestPickNewestImage(t *testing.T) {
	img := func(name, family, ts string) gceImage {
		return gceImage{Name: name, Family: family, CreationTimestamp: ts}
	}
	tests := []struct {
		name        string
		images      []gceImage
		imageRegex  string
		imageFamily string
		want        string
		wantErr     string
	}{
		{
			name: "newest of the family wins",
			images: []gceImage{
				img("img-old", "fam", "2026-08-01T10:00:00Z"),
				img("img-new", "fam", "2026-08-03T10:00:00Z"),
				img("img-mid", "fam", "2026-08-02T10:00:00Z"),
			},
			imageFamily: "fam",
			want:        "img-new",
		},
		{
			name: "images of other families are ignored",
			images: []gceImage{
				img("other-newer", "other", "2026-08-09T10:00:00Z"),
				img("fam-new", "fam", "2026-08-03T10:00:00Z"),
				img("fam-old", "fam", "2026-08-01T10:00:00Z"),
			},
			imageFamily: "fam",
			want:        "fam-new",
		},
		{
			name: "regex keeps only matching names, even a newer non-match is skipped",
			images: []gceImage{
				img("keep-v2", "fam", "2026-08-03T10:00:00Z"),
				img("skip-v1", "fam", "2026-08-09T10:00:00Z"),
			},
			imageFamily: "fam",
			imageRegex:  "keep-.*",
			want:        "keep-v2",
		},
		{
			name: "regex without a family",
			images: []gceImage{
				img("keep-old", "", "2026-08-01T10:00:00Z"),
				img("skip-newest", "", "2026-08-09T10:00:00Z"),
				img("keep-new", "", "2026-08-03T10:00:00Z"),
			},
			imageRegex: "keep-.*",
			want:       "keep-new",
		},
		{
			name: "no match returns an error",
			images: []gceImage{
				img("other", "other", "2026-08-01T10:00:00Z"),
			},
			imageFamily: "fam",
			wantErr:     "found zero images",
		},
		{
			name: "a malformed timestamp returns an error",
			images: []gceImage{
				img("fam-x", "fam", "not-a-timestamp"),
			},
			imageFamily: "fam",
			wantErr:     "failed to parse instance creation timestamp",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := pickNewestImage(tc.images, tc.imageRegex, tc.imageFamily, "proj")
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("pickNewestImage() error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("pickNewestImage() unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("pickNewestImage() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestGCEImageListArgs(t *testing.T) {
	tests := []struct {
		name        string
		project     string
		imageFamily string
		want        []string
	}{
		{
			name:        "a family adds a server-side filter",
			project:     "proj",
			imageFamily: "fam",
			want:        []string{"compute", "images", "list", "--format=json", "--project=proj", "--filter=family=fam"},
		},
		{
			name:    "no family adds no filter",
			project: "proj",
			want:    []string{"compute", "images", "list", "--format=json", "--project=proj"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := gceImageListArgs(tc.project, tc.imageFamily); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("gceImageListArgs() = %q, want %q", got, tc.want)
			}
		})
	}
}
