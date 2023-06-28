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

package storage

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"testing"
)

func encodeContinueOrDie(apiVersion string, resourceVersion int64, nextKey string) string {
	out, err := json.Marshal(&continueToken{APIVersion: apiVersion, ResourceVersion: resourceVersion, ContinueKey: nextKey})
	if err != nil {
		panic(err)
	}
	return base64.RawURLEncoding.EncodeToString(out)
}

func Test_decodeContinue(t *testing.T) {
	type args struct {
		continueValue string
	}
	tests := []struct {
		name        string
		args        args
		wantFromKey string
		wantRv      int64
		wantErr     error
	}{
		{
			name:        "valid",
			args:        args{continueValue: encodeContinueOrDie("meta.k8s.io/v1", 1, "key")},
			wantRv:      1,
			wantFromKey: "key",
		},
		{
			name:        "root path",
			args:        args{continueValue: encodeContinueOrDie("meta.k8s.io/v1", 1, "/")},
			wantRv:      1,
			wantFromKey: "/",
		},
		{
			name:    "empty version",
			args:    args{continueValue: encodeContinueOrDie("", 1, "key")},
			wantErr: ErrUnrecognizedEncodedVersion,
		},
		{
			name:    "invalid version",
			args:    args{continueValue: encodeContinueOrDie("v1", 1, "key")},
			wantErr: ErrUnrecognizedEncodedVersion,
		},
		{
			name:    "invalid RV",
			args:    args{continueValue: encodeContinueOrDie("meta.k8s.io/v1", 0, "key")},
			wantErr: ErrInvalidStartRV,
		},
		{
			name:    "no start Key",
			args:    args{continueValue: encodeContinueOrDie("meta.k8s.io/v1", 1, "")},
			wantErr: ErrEmptyStartKey,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotFromKey, gotRv, err := DecodeContinue(tt.args.continueValue)
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("decodeContinue() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if gotFromKey != tt.wantFromKey {
				t.Errorf("decodeContinue() gotFromKey = %v, want %v", gotFromKey, tt.wantFromKey)
			}
			if gotRv != tt.wantRv {
				t.Errorf("decodeContinue() gotRv = %v, want %v", gotRv, tt.wantRv)
			}
		})
	}
}
