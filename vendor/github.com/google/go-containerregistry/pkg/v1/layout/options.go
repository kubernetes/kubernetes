// Copyright 2019 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package layout

import v1 "github.com/google/go-containerregistry/pkg/v1"

// Option is a functional option for Layout.
type Option func(*options)

type options struct {
	descOpts []descriptorOption
}

func makeOptions(opts ...Option) *options {
	o := &options{
		descOpts: []descriptorOption{},
	}
	for _, apply := range opts {
		apply(o)
	}
	return o
}

type descriptorOption func(*v1.Descriptor)

// WithAnnotations adds annotations to the artifact descriptor.
func WithAnnotations(annotations map[string]string) Option {
	return func(o *options) {
		o.descOpts = append(o.descOpts, func(desc *v1.Descriptor) {
			if desc.Annotations == nil {
				desc.Annotations = make(map[string]string)
			}
			for k, v := range annotations {
				desc.Annotations[k] = v
			}
		})
	}
}

// WithURLs adds urls to the artifact descriptor.
func WithURLs(urls []string) Option {
	return func(o *options) {
		o.descOpts = append(o.descOpts, func(desc *v1.Descriptor) {
			if desc.URLs == nil {
				desc.URLs = []string{}
			}
			desc.URLs = append(desc.URLs, urls...)
		})
	}
}

// WithPlatform sets the platform of the artifact descriptor.
func WithPlatform(platform v1.Platform) Option {
	return func(o *options) {
		o.descOpts = append(o.descOpts, func(desc *v1.Descriptor) {
			desc.Platform = &platform
		})
	}
}
