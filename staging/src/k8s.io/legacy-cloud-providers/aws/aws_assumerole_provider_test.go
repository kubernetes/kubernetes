/*
Copyright 2014 The Kubernetes Authors.

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

package aws

import (
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/credentials"
)

func Test_assumeRoleProviderWithRateLimiting_Retrieve(t *testing.T) {
	type fields struct {
		provider                  credentials.Provider
		invalidateCredsCacheAfter time.Duration
		RWMutex                   sync.RWMutex
		lastError                 error
		lastValue                 credentials.Value
		lastRetrieveTime          time.Time
	}
	tests := []struct {
		name                       string
		fields                     fields
		want                       credentials.Value
		wantProviderCalled         bool
		sleepBeforeCallingProvider time.Duration
		wantErr                    bool
		wantErrString              string
	}{{
		name:               "Call assume role provider and verify access ID returned",
		fields:             fields{provider: &fakeAssumeRoleProvider{accesskeyID: "fakeID"}},
		want:               credentials.Value{AccessKeyID: "fakeID"},
		wantProviderCalled: true,
	}, {
		name: "Immediate call to assume role API, shouldn't call the underlying provider and return the last value",
		fields: fields{
			provider:                  &fakeAssumeRoleProvider{accesskeyID: "fakeID"},
			invalidateCredsCacheAfter: 100 * time.Millisecond,
			lastValue:                 credentials.Value{AccessKeyID: "fakeID1"},
			lastRetrieveTime:          time.Now(),
		},
		want:                       credentials.Value{AccessKeyID: "fakeID1"},
		wantProviderCalled:         false,
		sleepBeforeCallingProvider: 10 * time.Millisecond,
	}, {
		name: "Assume role provider returns an error when trying to assume a role",
		fields: fields{
			provider:                  &fakeAssumeRoleProvider{err: fmt.Errorf("can't assume fake role")},
			invalidateCredsCacheAfter: 10 * time.Millisecond,
			lastRetrieveTime:          time.Now(),
		},
		wantProviderCalled:         true,
		wantErr:                    true,
		wantErrString:              "can't assume fake role",
		sleepBeforeCallingProvider: 15 * time.Millisecond,
	}, {
		name: "Immediate call to assume role API, shouldn't call the underlying provider and return the last error value",
		fields: fields{
			provider:                  &fakeAssumeRoleProvider{},
			invalidateCredsCacheAfter: 100 * time.Millisecond,
			lastRetrieveTime:          time.Now(),
		},
		want:               credentials.Value{},
		wantProviderCalled: false,
		wantErr:            true,
		wantErrString:      "can't assume fake role",
	}, {
		name: "Delayed call to assume role API, should call the underlying provider",
		fields: fields{
			provider:                  &fakeAssumeRoleProvider{accesskeyID: "fakeID2"},
			invalidateCredsCacheAfter: 20 * time.Millisecond,
			lastRetrieveTime:          time.Now(),
		},
		want:                       credentials.Value{AccessKeyID: "fakeID2"},
		wantProviderCalled:         true,
		sleepBeforeCallingProvider: 25 * time.Millisecond,
	}}
	//nolint:govet // ignore copying of sync.RWMutex, it is empty
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := &assumeRoleProviderWithRateLimiting{
				provider:                  tt.fields.provider,
				invalidateCredsCacheAfter: tt.fields.invalidateCredsCacheAfter,
				lastError:                 tt.fields.lastError,
				lastValue:                 tt.fields.lastValue,
				lastRetrieveTime:          tt.fields.lastRetrieveTime,
			}
			time.Sleep(tt.sleepBeforeCallingProvider)
			got, err := l.Retrieve()
			if (err != nil) != tt.wantErr && (tt.wantErr && reflect.DeepEqual(err, tt.wantErrString)) {
				t.Errorf("assumeRoleProviderWithRateLimiting.Retrieve() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("assumeRoleProviderWithRateLimiting.Retrieve() got = %v, want %v", got, tt.want)
				return
			}
			if tt.wantProviderCalled != tt.fields.provider.(*fakeAssumeRoleProvider).providerCalled {
				t.Errorf("provider called %v, want %v", tt.fields.provider.(*fakeAssumeRoleProvider).providerCalled, tt.wantProviderCalled)
			}
		})
	}
}

type fakeAssumeRoleProvider struct {
	accesskeyID    string
	err            error
	providerCalled bool
}

func (f *fakeAssumeRoleProvider) Retrieve() (credentials.Value, error) {
	f.providerCalled = true
	return credentials.Value{AccessKeyID: f.accesskeyID}, f.err
}

func (f *fakeAssumeRoleProvider) IsExpired() bool { return true }
