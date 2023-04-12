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
	"net/url"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
)

func Test_getLoggingCmd(t *testing.T) {
	tests := []struct {
		name        string
		args        nodeLogQuery
		wantLinux   []string
		wantWindows []string
		wantOtherOS []string
	}{
		{
			args:        nodeLogQuery{},
			wantLinux:   []string{"--utc", "--no-pager", "--output=short-precise"},
			wantWindows: []string{"-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", "Get-WinEvent -FilterHashtable @{LogName='Application'} | Sort-Object TimeCreated | Format-Table -AutoSize -Wrap"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, got, err := getLoggingCmd(&tt.args, []string{})
			switch os := runtime.GOOS; os {
			case "linux":
				if !reflect.DeepEqual(got, tt.wantLinux) {
					t.Errorf("getLoggingCmd() = %v, want %v", got, tt.wantLinux)
				}
			case "windows":
				if !reflect.DeepEqual(got, tt.wantWindows) {
					t.Errorf("getLoggingCmd() = %v, want %v", got, tt.wantWindows)
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
		{name: "empty", query: url.Values{}, want: &nodeLogQuery{}},
		{query: url.Values{"unknown": []string{"true"}}, want: &nodeLogQuery{}},

		{query: url.Values{"sinceTime": []string{""}}, want: &nodeLogQuery{}},
		{query: url.Values{"sinceTime": []string{"2019-12-04 02:00:00"}}, wantErr: true},
		{query: url.Values{"sinceTime": []string{"2019-12-04 02:00:00.000"}}, wantErr: true},
		{query: url.Values{"sinceTime": []string{"2019-12-04 02"}}, wantErr: true},
		{query: url.Values{"sinceTime": []string{"2019-12-04 02:00"}}, wantErr: true},
		{query: url.Values{"sinceTime": []string{validTimeValue}},
			want: &nodeLogQuery{options: options{SinceTime: &validT}}},

		{query: url.Values{"untilTime": []string{""}}, want: &nodeLogQuery{}},
		{query: url.Values{"untilTime": []string{"2019-12-04 02:00:00"}}, wantErr: true},
		{query: url.Values{"untilTime": []string{"2019-12-04 02:00:00.000"}}, wantErr: true},
		{query: url.Values{"untilTime": []string{"2019-12-04 02"}}, wantErr: true},
		{query: url.Values{"untilTime": []string{"2019-12-04 02:00"}}, wantErr: true},
		{query: url.Values{"untilTime": []string{validTimeValue}},
			want: &nodeLogQuery{options: options{UntilTime: &validT}}},

		{query: url.Values{"tailLines": []string{"100"}}, want: &nodeLogQuery{options: options{TailLines: intPtr(100)}}},
		{query: url.Values{"tailLines": []string{"foo"}}, wantErr: true},
		{query: url.Values{"tailLines": []string{" "}}, wantErr: true},

		{query: url.Values{"pattern": []string{"foo"}}, want: &nodeLogQuery{options: options{Pattern: "foo"}}},

		{query: url.Values{"boot": []string{""}}, want: &nodeLogQuery{}},
		{query: url.Values{"boot": []string{"0"}}, want: &nodeLogQuery{options: options{Boot: intPtr(0)}}},
		{query: url.Values{"boot": []string{"-23"}}, want: &nodeLogQuery{options: options{Boot: intPtr(-23)}}},
		{query: url.Values{"boot": []string{"foo"}}, wantErr: true},
		{query: url.Values{"boot": []string{" "}}, wantErr: true},

		{query: url.Values{"query": []string{""}}, wantErr: true},
		{query: url.Values{"query": []string{"   ", "    "}}, wantErr: true},
		{query: url.Values{"query": []string{"foo"}}, want: &nodeLogQuery{Services: []string{"foo"}}},
		{query: url.Values{"query": []string{"foo", "bar"}}, want: &nodeLogQuery{Services: []string{"foo", "bar"}}},
		{query: url.Values{"query": []string{"foo", "/bar"}}, want: &nodeLogQuery{Services: []string{"foo"},
			Files: []string{"/bar"}}},
		{query: url.Values{"query": []string{"/foo", `\bar`}}, want: &nodeLogQuery{Files: []string{"/foo", `\bar`}}},
		{query: url.Values{"unit": []string{""}}, wantErr: true},
		{query: url.Values{"unit": []string{"   ", "    "}}, wantErr: true},
		{query: url.Values{"unit": []string{"foo"}}, want: &nodeLogQuery{Services: []string{"foo"}}},
		{query: url.Values{"unit": []string{"foo", "bar"}}, want: &nodeLogQuery{Services: []string{"foo", "bar"}}},
		{query: url.Values{"unit": []string{"foo", "/bar"}}, want: &nodeLogQuery{Services: []string{"foo", "/bar"}}},
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
		service1 = "svc1"
		service2 = "svc2"
		service3 = "svc.foo"
		service4 = "svc_foo"
		service5 = "svc@foo"
		service6 = "svc:foo"
		invalid1 = "svc\n"
		invalid2 = "svc.foo\n"
	)
	tests := []struct {
		name     string
		services []string
		wantErr  bool
	}{
		{name: "one service", services: []string{service1}},
		{name: "two services", services: []string{service1, service2}},
		{name: "dot service", services: []string{service3}},
		{name: "underscore service", services: []string{service4}},
		{name: "at service", services: []string{service5}},
		{name: "colon service", services: []string{service6}},
		{name: "invalid service new line", services: []string{invalid1}, wantErr: true},
		{name: "invalid service with dot", services: []string{invalid2}, wantErr: true},
		{name: "long service", services: []string{strings.Repeat(service1, 100)}, wantErr: true},
		{name: "max number of services", services: []string{service1, service2, service3, service4, service5}, wantErr: true},
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
	sinceTime, err := time.Parse(time.RFC3339, "2023-01-04T02:00:00Z")
	assert.NoError(t, err)
	untilTime, err := time.Parse(time.RFC3339, "2023-02-04T02:00:00Z")
	assert.NoError(t, err)
	since := "2019-12-04 02:00:00"
	until := "2019-12-04 03:00:00"

	tests := []struct {
		name     string
		Services []string
		Files    []string
		options  options
		wantErr  bool
	}{
		{name: "empty"},
		{name: "empty with options", options: options{SinceTime: &sinceTime}},
		{name: "one service", Services: []string{service1}},
		{name: "two services", Services: []string{service1, service2}},
		{name: "one service one file", Services: []string{service1}, Files: []string{file1}, wantErr: true},
		{name: "two files", Files: []string{file1, file2}, wantErr: true},
		{name: "one file options", Files: []string{file1}, options: options{Pattern: pattern}, wantErr: true},
		{name: "invalid pattern", Services: []string{service1}, options: options{Pattern: invalid}, wantErr: true},
		{name: "sinceTime", Services: []string{service1}, options: options{SinceTime: &sinceTime}},
		{name: "untilTime", Services: []string{service1}, options: options{UntilTime: &untilTime}},
		{name: "sinceTime untilTime", Services: []string{service1}, options: options{SinceTime: &untilTime,
			UntilTime: &sinceTime}, wantErr: true},
		{name: "boot", Services: []string{service1}, options: options{Boot: intPtr(-1)}},
		{name: "boot out of range", Services: []string{service1}, options: options{Boot: intPtr(1)}, wantErr: true},
		{name: "tailLines", Services: []string{service1}, options: options{TailLines: intPtr(100)}},
		{name: "tailLines out of range", Services: []string{service1}, options: options{TailLines: intPtr(100000)}},
		{name: "since", Services: []string{service1}, options: options{ocAdm: ocAdm{Since: since}}},
		{name: "since RFC3339", Services: []string{service1}, options: options{ocAdm: ocAdm{Since: sinceTime.String()}}, wantErr: true},
		{name: "until", Services: []string{service1}, options: options{ocAdm: ocAdm{Until: until}}},
		{name: "until RFC3339", Services: []string{service1}, options: options{ocAdm: ocAdm{Until: untilTime.String()}}, wantErr: true},
		{name: "since sinceTime", Services: []string{service1}, options: options{SinceTime: &sinceTime,
			ocAdm: ocAdm{Since: since}}, wantErr: true},
		{name: "until sinceTime", Services: []string{service1}, options: options{SinceTime: &sinceTime,
			ocAdm: ocAdm{Until: until}}, wantErr: true},
		{name: "since untilTime", Services: []string{service1}, options: options{UntilTime: &untilTime,
			ocAdm: ocAdm{Since: since}}, wantErr: true},
		{name: "until untilTime", Services: []string{service1}, options: options{UntilTime: &untilTime,
			ocAdm: ocAdm{Until: until}}, wantErr: true},
		{name: "format", Services: []string{service1}, options: options{ocAdm: ocAdm{Format: "cat"}}},
		{name: "format invalid", Services: []string{service1}, options: options{ocAdm: ocAdm{Format: "foo"}},
			wantErr: true},
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

func intPtr(i int) *int {
	return &i
}
