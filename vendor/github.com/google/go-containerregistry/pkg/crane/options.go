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

package crane

import (
	"context"
	"crypto/tls"
	"net/http"

	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/remote"
)

// Options hold the options that crane uses when calling other packages.
type Options struct {
	Name      []name.Option
	Remote    []remote.Option
	Platform  *v1.Platform
	Keychain  authn.Keychain
	Transport http.RoundTripper

	auth      authn.Authenticator
	insecure  bool
	jobs      int
	noclobber bool
	ctx       context.Context
}

// GetOptions exposes the underlying []remote.Option, []name.Option, and
// platform, based on the passed Option. Generally, you shouldn't need to use
// this unless you've painted yourself into a dependency corner as we have
// with the crane and gcrane cli packages.
func GetOptions(opts ...Option) Options {
	return makeOptions(opts...)
}

func makeOptions(opts ...Option) Options {
	opt := Options{
		Remote: []remote.Option{
			remote.WithAuthFromKeychain(authn.DefaultKeychain),
		},
		Keychain: authn.DefaultKeychain,
		jobs:     4,
		ctx:      context.Background(),
	}

	for _, o := range opts {
		o(&opt)
	}

	// Allow for untrusted certificates if the user
	// passed Insecure but no custom transport.
	if opt.insecure && opt.Transport == nil {
		transport := remote.DefaultTransport.(*http.Transport).Clone()
		transport.TLSClientConfig = &tls.Config{
			InsecureSkipVerify: true, //nolint: gosec
		}

		WithTransport(transport)(&opt)
	} else if opt.Transport == nil {
		opt.Transport = remote.DefaultTransport
	}

	return opt
}

// Option is a functional option for crane.
type Option func(*Options)

// WithTransport is a functional option for overriding the default transport
// for remote operations. Setting a transport will override the Insecure option's
// configuration allowing for image registries to use untrusted certificates.
func WithTransport(t http.RoundTripper) Option {
	return func(o *Options) {
		o.Remote = append(o.Remote, remote.WithTransport(t))
		o.Transport = t
	}
}

// Insecure is an Option that allows image references to be fetched without TLS.
// This will also allow for untrusted (e.g. self-signed) certificates in cases where
// the default transport is used (i.e. when WithTransport is not used).
func Insecure(o *Options) {
	o.Name = append(o.Name, name.Insecure)
	o.insecure = true
}

// WithPlatform is an Option to specify the platform.
func WithPlatform(platform *v1.Platform) Option {
	return func(o *Options) {
		if platform != nil {
			o.Remote = append(o.Remote, remote.WithPlatform(*platform))
		}
		o.Platform = platform
	}
}

// WithAuthFromKeychain is a functional option for overriding the default
// authenticator for remote operations, using an authn.Keychain to find
// credentials.
//
// By default, crane will use authn.DefaultKeychain.
func WithAuthFromKeychain(keys authn.Keychain) Option {
	return func(o *Options) {
		// Replace the default keychain at position 0.
		o.Remote[0] = remote.WithAuthFromKeychain(keys)
		o.Keychain = keys
	}
}

// WithAuth is a functional option for overriding the default authenticator
// for remote operations.
//
// By default, crane will use authn.DefaultKeychain.
func WithAuth(auth authn.Authenticator) Option {
	return func(o *Options) {
		// Replace the default keychain at position 0.
		o.Remote[0] = remote.WithAuth(auth)
		o.auth = auth
	}
}

// WithUserAgent adds the given string to the User-Agent header for any HTTP
// requests.
func WithUserAgent(ua string) Option {
	return func(o *Options) {
		o.Remote = append(o.Remote, remote.WithUserAgent(ua))
	}
}

// WithNondistributable is an option that allows pushing non-distributable
// layers.
func WithNondistributable() Option {
	return func(o *Options) {
		o.Remote = append(o.Remote, remote.WithNondistributable)
	}
}

// WithContext is a functional option for setting the context.
func WithContext(ctx context.Context) Option {
	return func(o *Options) {
		o.ctx = ctx
		o.Remote = append(o.Remote, remote.WithContext(ctx))
	}
}

// WithJobs sets the number of concurrent jobs to run.
//
// The default number of jobs is GOMAXPROCS.
func WithJobs(jobs int) Option {
	return func(o *Options) {
		if jobs > 0 {
			o.jobs = jobs
		}
		o.Remote = append(o.Remote, remote.WithJobs(o.jobs))
	}
}

// WithNoClobber modifies behavior to avoid overwriting existing tags, if possible.
func WithNoClobber(noclobber bool) Option {
	return func(o *Options) {
		o.noclobber = noclobber
	}
}
