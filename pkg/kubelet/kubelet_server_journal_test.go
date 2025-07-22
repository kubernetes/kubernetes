/*
Copyright 2022 The Kubernetes Authors.

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

package kubelet

import (
	"bytes"
	"context"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	"k8s.io/utils/ptr"
)

func Test_getLoggingCmd(t *testing.T) {
	var emptyCmdEnv []string
	tests := []struct {
		name              string
		args              nodeLogQuery
		services          []string
		wantLinux         []string
		wantWindows       []string
		wantLinuxCmdEnv   []string
		wantWindowsCmdEnv []string
	}{
		{
			name:              "basic",
			args:              nodeLogQuery{},
			services:          []string{},
			wantLinux:         []string{"--utc", "--no-pager", "--output=short-precise"},
			wantLinuxCmdEnv:   emptyCmdEnv,
			wantWindows:       []string{"-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", "Get-WinEvent -FilterHashtable @{LogName='Application'} | Sort-Object TimeCreated | Format-Table -AutoSize -Wrap"},
			wantWindowsCmdEnv: emptyCmdEnv,
		},
		{
			name:              "two providers",
			args:              nodeLogQuery{},
			services:          []string{"p1", "p2"},
			wantLinux:         []string{"--utc", "--no-pager", "--output=short-precise", "--unit=p1", "--unit=p2"},
			wantLinuxCmdEnv:   emptyCmdEnv,
			wantWindows:       []string{"-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", "Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName=$Env:kubelet_provider0,$Env:kubelet_provider1} | Sort-Object TimeCreated | Format-Table -AutoSize -Wrap"},
			wantWindowsCmdEnv: []string{"kubelet_provider0=p1", "kubelet_provider1=p2"},
		},
		{
			name:              "empty provider",
			args:              nodeLogQuery{},
			services:          []string{"p1", "", "p2"},
			wantLinux:         []string{"--utc", "--no-pager", "--output=short-precise", "--unit=p1", "--unit=p2"},
			wantLinuxCmdEnv:   emptyCmdEnv,
			wantWindows:       []string{"-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", "Get-WinEvent -FilterHashtable @{LogName='Application'; ProviderName=$Env:kubelet_provider0,$Env:kubelet_provider2} | Sort-Object TimeCreated | Format-Table -AutoSize -Wrap"},
			wantWindowsCmdEnv: []string{"kubelet_provider0=p1", "kubelet_provider2=p2"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, got, gotCmdEnv, err := getLoggingCmd(&tt.args, tt.services)
			switch os := runtime.GOOS; os {
			case "linux":
				if !reflect.DeepEqual(got, tt.wantLinux) {
					t.Errorf("getLoggingCmd() = %v, want %v", got, tt.wantLinux)
				}
				if !reflect.DeepEqual(gotCmdEnv, tt.wantLinuxCmdEnv) {
					t.Errorf("gotCmdEnv %v, wantLinuxCmdEnv %v", gotCmdEnv, tt.wantLinuxCmdEnv)
				}
			case "windows":
				if !reflect.DeepEqual(got, tt.wantWindows) {
					t.Errorf("getLoggingCmd() = %v, want %v", got, tt.wantWindows)
				}
				if !reflect.DeepEqual(gotCmdEnv, tt.wantWindowsCmdEnv) {
					t.Errorf("gotCmdEnv %v, wantWindowsCmdEnv %v", gotCmdEnv, tt.wantWindowsCmdEnv)
				}
			default:
				if err == nil {
					t.Errorf("getLoggingCmd() = %v, want err", got)
				}
			}
		})
	}
}

func Test_newNodeLogQuery(t *testing.T) {
	validTimeValue := "2019-12-04T02:00:00Z"
	validT, _ := time.Parse(time.RFC3339, validTimeValue)
	tests := []struct {
		name    string
		query   url.Values
		want    *nodeLogQuery
		wantErr bool
	}{
		{name: "empty", query: url.Values{}, want: nil},
		{query: url.Values{"unknown": []string{"true"}}, want: nil},

		{query: url.Values{"sinceTime": []string{""}}, want: nil},
		{query: url.Values{"sinceTime": []string{"2019-12-04 02:00:00"}}, wantErr: true},
		{query: url.Values{"sinceTime": []string{"2019-12-04 02:00:00.000"}}, wantErr: true},
		{query: url.Values{"sinceTime": []string{"2019-12-04 02"}}, wantErr: true},
		{query: url.Values{"sinceTime": []string{"2019-12-04 02:00"}}, wantErr: true},
		{query: url.Values{"sinceTime": []string{validTimeValue}},
			want: &nodeLogQuery{options: options{SinceTime: &validT}}},

		{query: url.Values{"untilTime": []string{""}}, want: nil},
		{query: url.Values{"untilTime": []string{"2019-12-04 02:00:00"}}, wantErr: true},
		{query: url.Values{"untilTime": []string{"2019-12-04 02:00:00.000"}}, wantErr: true},
		{query: url.Values{"untilTime": []string{"2019-12-04 02"}}, wantErr: true},
		{query: url.Values{"untilTime": []string{"2019-12-04 02:00"}}, wantErr: true},
		{query: url.Values{"untilTime": []string{validTimeValue}},
			want: &nodeLogQuery{options: options{UntilTime: &validT}}},

		{query: url.Values{"tailLines": []string{"100"}}, want: &nodeLogQuery{options: options{TailLines: ptr.To(100)}}},
		{query: url.Values{"tailLines": []string{"foo"}}, wantErr: true},
		{query: url.Values{"tailLines": []string{" "}}, wantErr: true},

		{query: url.Values{"pattern": []string{"foo"}}, want: &nodeLogQuery{options: options{Pattern: "foo"}}},

		{query: url.Values{"boot": []string{""}}, want: nil},
		{query: url.Values{"boot": []string{"0"}}, want: &nodeLogQuery{options: options{Boot: ptr.To(0)}}},
		{query: url.Values{"boot": []string{"-23"}}, want: &nodeLogQuery{options: options{Boot: ptr.To(-23)}}},
		{query: url.Values{"boot": []string{"foo"}}, wantErr: true},
		{query: url.Values{"boot": []string{" "}}, wantErr: true},

		{query: url.Values{"query": []string{""}}, wantErr: true},
		{query: url.Values{"query": []string{"   ", "    "}}, wantErr: true},
		{query: url.Values{"query": []string{"foo"}}, want: &nodeLogQuery{Services: []string{"foo"}}},
		{query: url.Values{"query": []string{"foo", "bar"}}, want: &nodeLogQuery{Services: []string{"foo", "bar"}}},
		{query: url.Values{"query": []string{"foo", "/bar"}}, want: &nodeLogQuery{Services: []string{"foo"},
			Files: []string{"/bar"}}},
		{query: url.Values{"query": []string{"/foo", `\bar`}}, want: &nodeLogQuery{Files: []string{"/foo", `\bar`}}},
	}
	for _, tt := range tests {
		t.Run(tt.query.Encode(), func(t *testing.T) {
			got, err := newNodeLogQuery(tt.query)
			if len(err) > 0 != tt.wantErr {
				t.Errorf("newNodeLogQuery() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("different: %s", cmp.Diff(tt.want, got, cmpopts.IgnoreUnexported(nodeLogQuery{})))
			}
		})
	}
}

func Test_validateServices(t *testing.T) {
	var (
		service1                 = "svc1"
		service2                 = "svc2"
		serviceDot               = "svc.foo"
		serviceUnderscore        = "svc_foo"
		serviceAt                = "svc@foo"
		serviceColon             = "svc:foo"
		invalidServiceNewline    = "svc\n"
		invalidServiceNewlineDot = "svc.foo\n"
		invalidServiceSlash      = "svc/foo"
	)
	tests := []struct {
		name     string
		services []string
		wantErr  bool
	}{
		{name: "one service", services: []string{service1}},
		{name: "two services", services: []string{service1, service2}},
		{name: "dot service", services: []string{serviceDot}},
		{name: "underscore service", services: []string{serviceUnderscore}},
		{name: "at service", services: []string{serviceAt}},
		{name: "colon service", services: []string{serviceColon}},
		{name: "invalid service new line", services: []string{invalidServiceNewline}, wantErr: true},
		{name: "invalid service with dot", services: []string{invalidServiceNewlineDot}, wantErr: true},
		{name: "invalid service with slash", services: []string{invalidServiceSlash}, wantErr: true},
		{name: "long service", services: []string{strings.Repeat(service1, 100)}, wantErr: true},
		{name: "max number of services", services: []string{service1, service2, serviceDot, serviceUnderscore, serviceAt}, wantErr: true},
	}
	for _, tt := range tests {
		errs := validateServices(tt.services)
		t.Run(tt.name, func(t *testing.T) {
			if len(errs) > 0 != tt.wantErr {
				t.Errorf("validateServices() error = %v, wantErr %v", errs, tt.wantErr)
				return
			}
		})
	}
}

func Test_nodeLogQuery_validate(t *testing.T) {
	var (
		service1 = "svc1"
		service2 = "svc2"
		file1    = "/test1.log"
		file2    = "/test2.log"
		pattern  = "foo"
		invalid  = "foo\\"
	)
	since, err := time.Parse(time.RFC3339, "2023-01-04T02:00:00Z")
	assert.NoError(t, err)
	until, err := time.Parse(time.RFC3339, "2023-02-04T02:00:00Z")
	assert.NoError(t, err)

	tests := []struct {
		name     string
		Services []string
		Files    []string
		options  options
		wantErr  bool
	}{
		{name: "empty", wantErr: true},
		{name: "empty with options", options: options{SinceTime: &since}, wantErr: true},
		{name: "one service", Services: []string{service1}},
		{name: "two services", Services: []string{service1, service2}},
		{name: "one service one file", Services: []string{service1}, Files: []string{file1}, wantErr: true},
		{name: "two files", Files: []string{file1, file2}, wantErr: true},
		{name: "one file options", Files: []string{file1}, options: options{Pattern: pattern}, wantErr: true},
		{name: "invalid pattern", Services: []string{service1}, options: options{Pattern: invalid}, wantErr: true},
		{name: "since", Services: []string{service1}, options: options{SinceTime: &since}},
		{name: "until", Services: []string{service1}, options: options{UntilTime: &until}},
		{name: "since until", Services: []string{service1}, options: options{SinceTime: &until, UntilTime: &since},
			wantErr: true},
		// boot is not supported on Windows.
		{name: "boot", Services: []string{service1}, options: options{Boot: ptr.To(-1)}, wantErr: runtime.GOOS == "windows"},
		{name: "boot out of range", Services: []string{service1}, options: options{Boot: ptr.To(1)}, wantErr: true},
		{name: "tailLines", Services: []string{service1}, options: options{TailLines: ptr.To(100)}},
		{name: "tailLines out of range", Services: []string{service1}, options: options{TailLines: ptr.To(100000)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := &nodeLogQuery{
				Services: tt.Services,
				Files:    tt.Files,
				options:  tt.options,
			}
			errs := n.validate()
			if len(errs) > 0 != tt.wantErr {
				t.Errorf("nodeLogQuery.validate() error = %v, wantErr %v", errs, tt.wantErr)
				return
			}
		})
	}
}

func Test_heuristicsCopyFileLogs(t *testing.T) {
	ctx := context.TODO()
	buf := &bytes.Buffer{}

	dir, err := os.MkdirTemp("", "logs")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()

	// Check missing logs
	heuristicsCopyFileLogs(ctx, buf, dir, "service.log")
	if !strings.Contains(buf.String(), "log not found for service.log") {
		t.Fail()
	}
	buf.Reset()

	// Check missing service logs
	heuristicsCopyFileLogs(ctx, buf, dir, "service")
	if !strings.Contains(buf.String(), "log not found for service") {
		t.Fail()
	}
	buf.Reset()

	// Check explicitly-named files
	if err := os.WriteFile(filepath.Join(dir, "service.log"), []byte("valid logs"), 0o444); err != nil {
		t.Fatal(err)
	}
	heuristicsCopyFileLogs(ctx, buf, dir, "service.log")
	if buf.String() != "valid logs" {
		t.Fail()
	}
	buf.Reset()

	// Check service logs
	heuristicsCopyFileLogs(ctx, buf, dir, "service")
	if buf.String() != "valid logs" {
		t.Fail()
	}
	buf.Reset()

	// Check that a directory doesn't cause errors
	if err := os.Mkdir(filepath.Join(dir, "service"), 0o755); err != nil {
		t.Fatal(err)
	}
	heuristicsCopyFileLogs(ctx, buf, dir, "service")
	if buf.String() != "valid logs" {
		t.Fail()
	}
	buf.Reset()

	// Check that service logs return the first matching file
	if err := os.WriteFile(filepath.Join(dir, "service", "service.log"), []byte("error"), 0o444); err != nil {
		t.Fatal(err)
	}
	heuristicsCopyFileLogs(ctx, buf, dir, "service")
	if buf.String() != "valid logs" {
		t.Fail()
	}
}
