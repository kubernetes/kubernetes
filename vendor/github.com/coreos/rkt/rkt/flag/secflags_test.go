// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package flag

import "testing"

func TestSecFlags(t *testing.T) {
	tests := []struct {
		opts   string
		image  bool
		tls    bool
		onDisk bool
		http   bool
		err    bool
	}{
		{
			opts:   "none",
			image:  false,
			tls:    false,
			onDisk: false,
			http:   false,
		},
		{
			opts:   "image",
			image:  true,
			tls:    false,
			onDisk: false,
			http:   false,
		},
		{
			opts:   "tls",
			image:  false,
			tls:    true,
			onDisk: false,
			http:   false,
		},
		{
			opts:   "onDisk",
			image:  false,
			tls:    false,
			onDisk: true,
			http:   false,
		},
		{
			opts:   "http",
			image:  false,
			tls:    false,
			onDisk: false,
			http:   true,
		},
		{
			opts:   "all",
			image:  true,
			tls:    true,
			onDisk: true,
			http:   true,
		},
		{
			opts:   "image,tls",
			image:  true,
			tls:    true,
			onDisk: false,
			http:   false,
		},
		{
			opts: "i-am-sure-we-will-not-get-this-insecure-flag",
			err:  true,
		},
	}

	for i, tt := range tests {
		sf, err := NewSecFlags(tt.opts)
		if err != nil && !tt.err {
			t.Errorf("test %d: unexpected error in NewSecFlags: %v", i, err)
		} else if err == nil && tt.err {
			t.Errorf("test %d: unexpected success in NewSecFlags for options %q", i, tt.opts)
		}
		if err != nil {
			continue
		}

		if got := sf.SkipImageCheck(); tt.image != got {
			t.Errorf("test %d: expected image skip to be %v, got %v", i, tt.image, got)
		}

		if got := sf.SkipTLSCheck(); tt.tls != got {
			t.Errorf("test %d: expected tls skip to be %v, got %v", i, tt.tls, got)
		}

		if got := sf.SkipOnDiskCheck(); tt.onDisk != got {
			t.Errorf("test %d: expected on disk skip to be %v, got %v", i, tt.onDisk, got)
		}

		if got := sf.AllowHTTP(); tt.http != got {
			t.Errorf("test %d: expected http allowed to be %v, got %v", i, tt.http, got)
		}

		all := tt.http && tt.onDisk && tt.tls && tt.image
		if got := sf.SkipAllSecurityChecks(); all != got {
			t.Errorf("test %d: expected all skip to be %v, got %v", i, all, got)
		}

		any := tt.http || tt.onDisk || tt.tls || tt.image
		if got := sf.SkipAnySecurityChecks(); any != got {
			t.Errorf("test %d: expected all skip to be %v, got %v", i, any, got)
		}
	}
}
