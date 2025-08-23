// Copyright 2018 Google LLC All Rights Reserved.
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

package authn

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/docker/cli/cli/config"
	"github.com/docker/cli/cli/config/configfile"
	"github.com/docker/cli/cli/config/types"
	"github.com/google/go-containerregistry/pkg/name"
	"github.com/mitchellh/go-homedir"
)

// Resource represents a registry or repository that can be authenticated against.
type Resource interface {
	// String returns the full string representation of the target, e.g.
	// gcr.io/my-project or just gcr.io.
	String() string

	// RegistryStr returns just the registry portion of the target, e.g. for
	// gcr.io/my-project, this should just return gcr.io. This is needed to
	// pull out an appropriate hostname.
	RegistryStr() string
}

// Keychain is an interface for resolving an image reference to a credential.
type Keychain interface {
	// Resolve looks up the most appropriate credential for the specified target.
	Resolve(Resource) (Authenticator, error)
}

// ContextKeychain is like Keychain, but allows for context to be passed in.
type ContextKeychain interface {
	ResolveContext(context.Context, Resource) (Authenticator, error)
}

// defaultKeychain implements Keychain with the semantics of the standard Docker
// credential keychain.
type defaultKeychain struct {
	mu sync.Mutex
}

var (
	// DefaultKeychain implements Keychain by interpreting the docker config file.
	DefaultKeychain = &defaultKeychain{}
)

const (
	// DefaultAuthKey is the key used for dockerhub in config files, which
	// is hardcoded for historical reasons.
	DefaultAuthKey = "https://" + name.DefaultRegistry + "/v1/"
)

// Resolve calls ResolveContext with ctx if the given [Keychain] implements [ContextKeychain],
// otherwise it calls Resolve with the given [Resource].
func Resolve(ctx context.Context, keychain Keychain, target Resource) (Authenticator, error) {
	if rctx, ok := keychain.(ContextKeychain); ok {
		return rctx.ResolveContext(ctx, target)
	}

	return keychain.Resolve(target)
}

// ResolveContext implements ContextKeychain.
func (dk *defaultKeychain) Resolve(target Resource) (Authenticator, error) {
	return dk.ResolveContext(context.Background(), target)
}

// Resolve implements Keychain.
func (dk *defaultKeychain) ResolveContext(_ context.Context, target Resource) (Authenticator, error) {
	dk.mu.Lock()
	defer dk.mu.Unlock()

	// Podman users may have their container registry auth configured in a
	// different location, that Docker packages aren't aware of.
	// If the Docker config file isn't found, we'll fallback to look where
	// Podman configures it, and parse that as a Docker auth config instead.

	// First, check $HOME/.docker/config.json
	foundDockerConfig := false
	home, err := homedir.Dir()
	if err == nil {
		foundDockerConfig = fileExists(filepath.Join(home, ".docker/config.json"))
	}
	// If $HOME/.docker/config.json isn't found, check $DOCKER_CONFIG (if set)
	if !foundDockerConfig && os.Getenv("DOCKER_CONFIG") != "" {
		foundDockerConfig = fileExists(filepath.Join(os.Getenv("DOCKER_CONFIG"), "config.json"))
	}
	// If either of those locations are found, load it using Docker's
	// config.Load, which may fail if the config can't be parsed.
	//
	// If neither was found, look for Podman's auth at
	// $REGISTRY_AUTH_FILE or $XDG_RUNTIME_DIR/containers/auth.json
	// and attempt to load it as a Docker config.
	//
	// If neither are found, fallback to Anonymous.
	var cf *configfile.ConfigFile
	if foundDockerConfig {
		cf, err = config.Load(os.Getenv("DOCKER_CONFIG"))
		if err != nil {
			return nil, err
		}
	} else if fileExists(os.Getenv("REGISTRY_AUTH_FILE")) {
		f, err := os.Open(os.Getenv("REGISTRY_AUTH_FILE"))
		if err != nil {
			return nil, err
		}
		defer f.Close()
		cf, err = config.LoadFromReader(f)
		if err != nil {
			return nil, err
		}
	} else if fileExists(filepath.Join(os.Getenv("XDG_RUNTIME_DIR"), "containers/auth.json")) {
		f, err := os.Open(filepath.Join(os.Getenv("XDG_RUNTIME_DIR"), "containers/auth.json"))
		if err != nil {
			return nil, err
		}
		defer f.Close()
		cf, err = config.LoadFromReader(f)
		if err != nil {
			return nil, err
		}
	} else {
		return Anonymous, nil
	}

	// See:
	// https://github.com/google/ko/issues/90
	// https://github.com/moby/moby/blob/fc01c2b481097a6057bec3cd1ab2d7b4488c50c4/registry/config.go#L397-L404
	var cfg, empty types.AuthConfig
	for _, key := range []string{
		target.String(),
		target.RegistryStr(),
	} {
		if key == name.DefaultRegistry {
			key = DefaultAuthKey
		}

		cfg, err = cf.GetAuthConfig(key)
		if err != nil {
			return nil, err
		}
		// cf.GetAuthConfig automatically sets the ServerAddress attribute. Since
		// we don't make use of it, clear the value for a proper "is-empty" test.
		// See: https://github.com/google/go-containerregistry/issues/1510
		cfg.ServerAddress = ""
		if cfg != empty {
			break
		}
	}
	if cfg == empty {
		return Anonymous, nil
	}

	return FromConfig(AuthConfig{
		Username:      cfg.Username,
		Password:      cfg.Password,
		Auth:          cfg.Auth,
		IdentityToken: cfg.IdentityToken,
		RegistryToken: cfg.RegistryToken,
	}), nil
}

// fileExists returns true if the given path exists and is not a directory.
func fileExists(path string) bool {
	fi, err := os.Stat(path)
	return err == nil && !fi.IsDir()
}

// Helper is a subset of the Docker credential helper credentials.Helper
// interface used by NewKeychainFromHelper.
//
// See:
// https://pkg.go.dev/github.com/docker/docker-credential-helpers/credentials#Helper
type Helper interface {
	Get(serverURL string) (string, string, error)
}

// NewKeychainFromHelper returns a Keychain based on a Docker credential helper
// implementation that can Get username and password credentials for a given
// server URL.
func NewKeychainFromHelper(h Helper) Keychain { return wrapper{h} }

type wrapper struct{ h Helper }

func (w wrapper) Resolve(r Resource) (Authenticator, error) {
	return w.ResolveContext(context.Background(), r)
}

func (w wrapper) ResolveContext(_ context.Context, r Resource) (Authenticator, error) {
	u, p, err := w.h.Get(r.RegistryStr())
	if err != nil {
		return Anonymous, nil
	}
	// If the secret being stored is an identity token, the Username should be set to <token>
	// ref: https://docs.docker.com/engine/reference/commandline/login/#credential-helper-protocol
	if u == "<token>" {
		return FromConfig(AuthConfig{Username: u, IdentityToken: p}), nil
	}
	return FromConfig(AuthConfig{Username: u, Password: p}), nil
}

func RefreshingKeychain(inner Keychain, duration time.Duration) Keychain {
	return &refreshingKeychain{
		keychain: inner,
		duration: duration,
	}
}

type refreshingKeychain struct {
	keychain Keychain
	duration time.Duration
	clock    func() time.Time
}

func (r *refreshingKeychain) Resolve(target Resource) (Authenticator, error) {
	return r.ResolveContext(context.Background(), target)
}

func (r *refreshingKeychain) ResolveContext(ctx context.Context, target Resource) (Authenticator, error) {
	last := time.Now()
	auth, err := Resolve(ctx, r.keychain, target)
	if err != nil || auth == Anonymous {
		return auth, err
	}
	return &refreshing{
		target:   target,
		keychain: r.keychain,
		last:     last,
		cached:   auth,
		duration: r.duration,
		clock:    r.clock,
	}, nil
}

type refreshing struct {
	sync.Mutex
	target   Resource
	keychain Keychain

	duration time.Duration

	last   time.Time
	cached Authenticator

	// for testing
	clock func() time.Time
}

func (r *refreshing) Authorization() (*AuthConfig, error) {
	return r.AuthorizationContext(context.Background())
}

func (r *refreshing) AuthorizationContext(ctx context.Context) (*AuthConfig, error) {
	r.Lock()
	defer r.Unlock()
	if r.cached == nil || r.expired() {
		r.last = r.now()
		auth, err := Resolve(ctx, r.keychain, r.target)
		if err != nil {
			return nil, err
		}
		r.cached = auth
	}
	return Authorization(ctx, r.cached)
}

func (r *refreshing) now() time.Time {
	if r.clock == nil {
		return time.Now()
	}
	return r.clock()
}

func (r *refreshing) expired() bool {
	return r.now().Sub(r.last) > r.duration
}
