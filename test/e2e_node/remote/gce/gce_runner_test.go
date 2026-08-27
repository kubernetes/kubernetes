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
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"sigs.k8s.io/yaml"

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
		name              string
		images            []gceImage
		imageRegex        string
		imageExcludeRegex string
		imageFamily       string
		want              string
		wantErr           string
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
		{
			name: "exclude drops the -cgroupsv1 suffix but keeps a newer name ending in 1",
			images: []gceImage{
				img("ubuntu-v20260801", "fam", "2026-08-03T10:00:00Z"),
				img("ubuntu-v20260801-cgroupsv1", "fam", "2026-08-09T10:00:00Z"),
			},
			imageFamily:       "fam",
			imageExcludeRegex: "-cgroupsv1$",
			want:              "ubuntu-v20260801",
		},
		{
			name: "exclude is applied together with the include regex",
			images: []gceImage{
				img("keep-a", "fam", "2026-08-03T10:00:00Z"),
				img("keep-b-cgroupsv1", "fam", "2026-08-09T10:00:00Z"),
				img("skip-c", "fam", "2026-08-05T10:00:00Z"),
			},
			imageFamily:       "fam",
			imageRegex:        "keep-.*",
			imageExcludeRegex: "-cgroupsv1$",
			want:              "keep-a",
		},
		{
			name: "exclude removes every candidate",
			images: []gceImage{
				img("ubuntu-a-cgroupsv1", "fam", "2026-08-03T10:00:00Z"),
				img("ubuntu-b-cgroupsv1", "fam", "2026-08-09T10:00:00Z"),
			},
			imageFamily:       "fam",
			imageExcludeRegex: "-cgroupsv1$",
			wantErr:           `found zero images based on regex "", exclude regex "-cgroupsv1$"`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			selector, err := compileImageSelector(tc.imageRegex, tc.imageExcludeRegex, tc.imageFamily)
			if err != nil {
				t.Fatalf("compileImageSelector() unexpected error: %v", err)
			}
			got, err := pickNewestImage(tc.images, selector, "proj")
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
			want:        []string{"compute", "images", "list", "--format=json(name,family,creationTimestamp)", "--project=proj", "--filter=family=fam"},
		},
		{
			name:    "no family adds no filter",
			project: "proj",
			want:    []string{"compute", "images", "list", "--format=json(name,family,creationTimestamp)", "--project=proj"},
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

func TestProjectedGCEImageJSON(t *testing.T) {
	// The --format keys in gceImageListArgs must match gceImage's json tags,
	// or the runner would decode empty image data.
	data := []byte(`[{"name":"ubuntu-x","family":"pipeline-1-34-amd64","creationTimestamp":"2026-08-27T00:00:00Z"}]`)
	var images []gceImage
	if err := json.Unmarshal(data, &images); err != nil {
		t.Fatalf("json.Unmarshal() unexpected error: %v", err)
	}
	if len(images) != 1 {
		t.Fatalf("decoded %d images, want 1", len(images))
	}
	got := images[0]
	if got.Name != "ubuntu-x" || got.Family != "pipeline-1-34-amd64" || got.CreationTimestamp != "2026-08-27T00:00:00Z" {
		t.Errorf("decoded %+v, want name, family, and creationTimestamp populated", got)
	}
}

func TestCompileImageSelector(t *testing.T) {
	tests := []struct {
		name              string
		imageRegex        string
		imageExcludeRegex string
		wantInclude       bool
		wantExclude       bool
		wantErr           string
	}{
		{name: "both empty leaves both filters unset"},
		{name: "include only", imageRegex: "keep-.*", wantInclude: true},
		{name: "exclude only", imageExcludeRegex: "-cgroupsv1$", wantExclude: true},
		{name: "a bad include regex returns an error", imageRegex: "[", wantErr: "failed to compile image_regex"},
		{name: "a bad exclude regex returns an error", imageExcludeRegex: "[", wantErr: "failed to compile image_exclude_regex"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s, err := compileImageSelector(tc.imageRegex, tc.imageExcludeRegex, "fam")
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("compileImageSelector() error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("compileImageSelector() unexpected error: %v", err)
			}
			if (s.include != nil) != tc.wantInclude {
				t.Errorf("include non-nil = %v, want %v", s.include != nil, tc.wantInclude)
			}
			if (s.exclude != nil) != tc.wantExclude {
				t.Errorf("exclude non-nil = %v, want %v", s.exclude != nil, tc.wantExclude)
			}
		})
	}
}

func TestGetGCEImage(t *testing.T) {
	const twoImages = `[
		{"name":"ubuntu-v2","family":"fam","creationTimestamp":"2026-08-03T10:00:00Z"},
		{"name":"ubuntu-v2-cgroupsv1","family":"fam","creationTimestamp":"2026-08-09T10:00:00Z"}
	]`
	tests := []struct {
		name              string
		imageRegex        string
		imageExcludeRegex string
		listOut           string
		listErr           error
		wantCalls         int
		want              string
		wantErr           string
	}{
		{
			name:       "a bad include regex fails before any gcloud call",
			imageRegex: "[",
			wantCalls:  0,
			wantErr:    "failed to compile image_regex",
		},
		{
			name:              "a bad exclude regex fails before any gcloud call",
			imageExcludeRegex: "[",
			wantCalls:         0,
			wantErr:           "failed to compile image_exclude_regex",
		},
		{
			name:      "a listing error is wrapped",
			listErr:   errors.New("gcloud boom"),
			wantCalls: 1,
			wantErr:   "failed to list images",
		},
		{
			name:              "the newest non-excluded image is returned",
			imageExcludeRegex: "-cgroupsv1$",
			listOut:           twoImages,
			wantCalls:         1,
			want:              "ubuntu-v2",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			orig := gceImageLister
			gceImageLister = func(args ...string) ([]byte, error) {
				calls++
				return []byte(tc.listOut), tc.listErr
			}
			defer func() { gceImageLister = orig }()

			got, err := (&GCERunner{}).getGCEImage(tc.imageRegex, tc.imageExcludeRegex, "fam", "proj")
			if calls != tc.wantCalls {
				t.Errorf("gcloud invocation count = %d, want %d", calls, tc.wantCalls)
			}
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("getGCEImage() error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("getGCEImage() unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("getGCEImage() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestPrepareGceImagesExplicitImage(t *testing.T) {
	tests := []struct {
		name      string
		config    string
		wantCalls int
		want      string
		wantErr   string
	}{
		{
			name: "an explicit image is used without compiling or listing anything",
			config: `images:
  cos-example:
    image: pinned-image
    image_regex: "["
    image_exclude_regex: "["
    project: proj
`,
			wantCalls: 0,
			want:      "pinned-image",
		},
		{
			name: "without an explicit image the same selectors are compiled and rejected",
			config: `images:
  cos-example:
    image_regex: "["
    image_exclude_regex: "["
    project: proj
`,
			wantCalls: 0,
			wantErr:   "failed to compile image_regex",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			configPath := filepath.Join(t.TempDir(), "images.yaml")
			if err := os.WriteFile(configPath, []byte(tc.config), 0o600); err != nil {
				t.Fatal(err)
			}
			for _, f := range []*string{zone, project} {
				orig := *f
				*f = "test"
				defer func() { *f = orig }()
			}
			calls := 0
			origLister := gceImageLister
			gceImageLister = func(args ...string) ([]byte, error) {
				calls++
				return nil, errors.New("must not be called")
			}
			defer func() { gceImageLister = origLister }()

			g := &GCERunner{cfg: remote.Config{ImageConfigFile: configPath}}
			images, err := g.prepareGceImages()
			if calls != tc.wantCalls {
				t.Errorf("gcloud invocation count = %d, want %d", calls, tc.wantCalls)
			}
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("prepareGceImages() error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("prepareGceImages() unexpected error: %v", err)
			}
			if got := images.images["cos-example"].image; got != tc.want {
				t.Errorf("prepareGceImages() image = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestValidateImageSelector(t *testing.T) {
	tests := []struct {
		name    string
		config  GCEImage
		wantErr string
	}{
		{
			name:   "explicit image is allowed",
			config: GCEImage{Image: "ubuntu-x"},
		},
		{
			name:   "image regex is a valid selector",
			config: GCEImage{ImageRegex: "keep-.*"},
		},
		{
			name:   "image family is a valid selector",
			config: GCEImage{ImageFamily: "fam"},
		},
		{
			name:   "exclude with a family is allowed",
			config: GCEImage{ImageFamily: "fam", ImageExcludeRegex: "-cgroupsv1$"},
		},
		{
			name:   "exclude with an include regex is allowed",
			config: GCEImage{ImageRegex: "keep-.*", ImageExcludeRegex: "-cgroupsv1$"},
		},
		{
			name:    "exclude alone is rejected",
			config:  GCEImage{ImageExcludeRegex: "-cgroupsv1$"},
			wantErr: "image_exclude_regex requires image_regex or image_family",
		},
		{
			name: "explicit image takes precedence and ignores dynamic selectors",
			config: GCEImage{
				Image:             "ubuntu-x",
				ImageRegex:        "keep-.*",
				ImageExcludeRegex: "-cgroupsv1$",
				ImageFamily:       "fam",
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := validateImageSelector("cos-example", tc.config)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("validateImageSelector() error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("validateImageSelector() unexpected error: %v", err)
			}
		})
	}
}

func TestGCEImageConfigUnmarshalImageExcludeRegex(t *testing.T) {
	const data = `
images:
  ubuntu-example:
    image_regex: "ubuntu-.*"
    image_exclude_regex: "-cgroupsv1$"
    image_family: "ubuntu-fam"
    project: "ubuntu-proj"
`
	cfg := GCEImageConfig{Images: make(map[string]GCEImage)}
	if err := yaml.Unmarshal([]byte(data), &cfg); err != nil {
		t.Fatalf("yaml.Unmarshal() unexpected error: %v", err)
	}
	got, ok := cfg.Images["ubuntu-example"]
	if !ok {
		t.Fatalf("image %q missing from decoded config", "ubuntu-example")
	}
	if got.ImageExcludeRegex != "-cgroupsv1$" {
		t.Errorf("ImageExcludeRegex = %q, want %q", got.ImageExcludeRegex, "-cgroupsv1$")
	}
}
