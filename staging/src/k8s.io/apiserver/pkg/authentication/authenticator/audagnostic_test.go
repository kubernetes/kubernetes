/*
Copyright 2018 The Kubernetes Authors.

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

package authenticator

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apiserver/pkg/authentication/user"
)

func TestAuthenticate(t *testing.T) {
	type treq struct {
		resp          *Response
		authenticated bool
		err           error

		wantResp          *Response
		wantAuthenticated bool
		wantErr           bool
	}
	type taudcfg struct {
		auds         Audiences
		implicitAuds Audiences
	}
	cs := []struct {
		name string

		taudcfgs []taudcfg
		treqs    []treq
	}{
		{
			name: "good audience",

			taudcfgs: []taudcfg{
				{
					implicitAuds: Audiences{"api"},
					auds:         Audiences{"api"},
				},
				{
					implicitAuds: Audiences{"api", "other"},
					auds:         Audiences{"api"},
				},
				{
					implicitAuds: Audiences{"api"},
					auds:         Audiences{"api", "other"},
				},
				{
					implicitAuds: Audiences{"api", "other"},
					auds:         Audiences{"api", "other_other"},
				},
			},

			treqs: []treq{
				{
					resp: &Response{
						User: &user.DefaultInfo{
							Name: "test_user",
						},
					},
					authenticated: true,

					wantResp: &Response{
						User: &user.DefaultInfo{
							Name: "test_user",
						},
						Audiences: Audiences{"api"},
					},
					wantAuthenticated: true,
				},
				{
					err:     errors.New("uhoh"),
					wantErr: true,
				},
				{
					authenticated:     false,
					wantAuthenticated: false,
				},
			},
		},
		{
			name: "multiple good audiences",

			taudcfgs: []taudcfg{
				{
					implicitAuds: Audiences{"api", "other_api"},
					auds:         Audiences{"api", "other_api"},
				},
				{
					implicitAuds: Audiences{"api", "other_api", "other"},
					auds:         Audiences{"api", "other_api"},
				},
				{
					implicitAuds: Audiences{"api", "other_api"},
					auds:         Audiences{"api", "other_api", "other"},
				},
				{
					implicitAuds: Audiences{"api", "other_api", "other"},
					auds:         Audiences{"api", "other_api", "other_other"},
				},
			},

			treqs: []treq{
				{
					resp: &Response{
						User: &user.DefaultInfo{
							Name: "test_user",
						},
					},
					authenticated: true,

					wantResp: &Response{
						User: &user.DefaultInfo{
							Name: "test_user",
						},
						Audiences: Audiences{"api", "other_api"},
					},
					wantAuthenticated: true,
				},
				{
					err: errors.New("uhoh"),

					wantErr: true,
				},
				{
					authenticated: false,

					wantAuthenticated: false,
				},
			},
		},
		{
			name: "bad audience(s)",

			taudcfgs: []taudcfg{
				{
					implicitAuds: Audiences{"api"},
					auds:         Audiences{"other_api"},
				},
				{
					implicitAuds: Audiences{},
					auds:         Audiences{"other_api"},
				},
				{
					implicitAuds: Audiences{"api"},
					auds:         Audiences{},
				},
				{
					implicitAuds: Audiences{"api", "other"},
					auds:         Audiences{},
				},
				{
					implicitAuds: Audiences{},
					auds:         Audiences{"api", "other"},
				},
			},

			treqs: []treq{
				{
					resp: &Response{
						User: &user.DefaultInfo{
							Name: "test_user",
						},
					},
					authenticated: true,

					wantAuthenticated: false,
				},
				{
					err: errors.New("uhoh"),

					wantAuthenticated: false,
				},
				{
					authenticated: false,

					wantAuthenticated: false,
				},
			},
		},
	}

	for _, c := range cs {
		t.Run(c.name, func(t *testing.T) {
			for _, taudcfg := range c.taudcfgs {
				for _, treq := range c.treqs {
					t.Run(fmt.Sprintf("auds=%q,implicit=%q", taudcfg.auds, taudcfg.implicitAuds), func(t *testing.T) {
						ctx := context.Background()
						ctx = WithAudiences(ctx, taudcfg.auds)
						resp, ok, err := authenticate(ctx, taudcfg.implicitAuds, func() (*Response, bool, error) {
							if treq.resp != nil {
								resp := *treq.resp
								return &resp, treq.authenticated, treq.err
							}
							return nil, treq.authenticated, treq.err
						})
						if got, want := (err != nil), treq.wantErr; got != want {
							t.Errorf("Unexpected error. got=%v, want=%v, err=%v", got, want, err)
						}
						if got, want := ok, treq.wantAuthenticated; got != want {
							t.Errorf("Unexpected authentication. got=%v, want=%v", got, want)
						}
						if got, want := resp, treq.wantResp; !reflect.DeepEqual(got, want) {
							t.Errorf("Unexpected response. diff:\n%v", cmp.Diff(got, want))
						}
					})
				}
			}
		})
	}
}
