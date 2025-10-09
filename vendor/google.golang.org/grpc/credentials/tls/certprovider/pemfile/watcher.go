/*
 *
 * Copyright 2020 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package pemfile provides a file watching certificate provider plugin
// implementation which works for files with PEM contents.
//
// # Experimental
//
// Notice: All APIs in this package are experimental and may be removed in a
// later release.
package pemfile

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"google.golang.org/grpc/credentials/tls/certprovider"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/credentials/spiffe"
)

const defaultCertRefreshDuration = 1 * time.Hour

var (
	// For overriding from unit tests.
	newDistributor = func() distributor { return certprovider.NewDistributor() }

	logger = grpclog.Component("pemfile")
)

// Options configures a certificate provider plugin that watches a specified set
// of files that contain certificates and keys in PEM format.
type Options struct {
	// CertFile is the file that holds the identity certificate.
	// Optional. If this is set, KeyFile must also be set.
	CertFile string
	// KeyFile is the file that holds identity private key.
	// Optional. If this is set, CertFile must also be set.
	KeyFile string
	// RootFile is the file that holds trusted root certificate(s).
	// Optional.
	RootFile string
	// SPIFFEBundleMapFile is the file that holds the spiffe bundle map.
	// If a given provider configures both the RootFile and the
	// SPIFFEBundleMapFile, the SPIFFEBundleMapFile will be preferred.
	// Optional.
	SPIFFEBundleMapFile string
	// RefreshDuration is the amount of time the plugin waits before checking
	// for updates in the specified files.
	// Optional. If not set, a default value (1 hour) will be used.
	RefreshDuration time.Duration
}

func (o Options) canonical() []byte {
	return []byte(fmt.Sprintf("%s:%s:%s:%s:%s", o.CertFile, o.KeyFile, o.RootFile, o.SPIFFEBundleMapFile, o.RefreshDuration))
}

func (o Options) validate() error {
	if o.CertFile == "" && o.KeyFile == "" && o.RootFile == "" && o.SPIFFEBundleMapFile == "" {
		return fmt.Errorf("pemfile: at least one credential file needs to be specified")
	}
	if keySpecified, certSpecified := o.KeyFile != "", o.CertFile != ""; keySpecified != certSpecified {
		return fmt.Errorf("pemfile: private key file and identity cert file should be both specified or not specified")
	}
	// C-core has a limitation that they cannot verify that a certificate file
	// matches a key file. So, the only way to get around this is to make sure
	// that both files are in the same directory and that they do an atomic
	// read. Even though Java/Go do not have this limitation, we want the
	// overall plugin behavior to be consistent across languages.
	if certDir, keyDir := filepath.Dir(o.CertFile), filepath.Dir(o.KeyFile); certDir != keyDir {
		return errors.New("pemfile: certificate and key file must be in the same directory")
	}
	return nil
}

// NewProvider returns a new certificate provider plugin that is configured to
// watch the PEM files specified in the passed in options.
func NewProvider(o Options) (certprovider.Provider, error) {
	if err := o.validate(); err != nil {
		return nil, err
	}
	return newProvider(o), nil
}

// newProvider is used to create a new certificate provider plugin after
// validating the options, and hence does not return an error.
func newProvider(o Options) certprovider.Provider {
	if o.RefreshDuration == 0 {
		o.RefreshDuration = defaultCertRefreshDuration
	}

	provider := &watcher{opts: o}
	if o.CertFile != "" && o.KeyFile != "" {
		provider.identityDistributor = newDistributor()
	}
	if o.RootFile != "" || o.SPIFFEBundleMapFile != "" {
		provider.rootDistributor = newDistributor()
	}

	ctx, cancel := context.WithCancel(context.Background())
	provider.cancel = cancel
	go provider.run(ctx)
	return provider
}

// watcher is a certificate provider plugin that implements the
// certprovider.Provider interface. It watches a set of certificate and key
// files and provides the most up-to-date key material for consumption by
// credentials implementation.
type watcher struct {
	identityDistributor         distributor
	rootDistributor             distributor
	opts                        Options
	certFileContents            []byte
	keyFileContents             []byte
	rootFileContents            []byte
	spiffeBundleMapFileContents []byte
	cancel                      context.CancelFunc
}

// distributor wraps the methods on certprovider.Distributor which are used by
// the plugin. This is very useful in tests which need to know exactly when the
// plugin updates its key material.
type distributor interface {
	KeyMaterial(ctx context.Context) (*certprovider.KeyMaterial, error)
	Set(km *certprovider.KeyMaterial, err error)
	Stop()
}

// updateIdentityDistributor checks if the cert/key files that the plugin is
// watching have changed, and if so, reads the new contents and updates the
// identityDistributor with the new key material.
//
// Skips updates when file reading or parsing fails.
// TODO(easwars): Retry with limit (on the number of retries or the amount of
// time) upon failures.
func (w *watcher) updateIdentityDistributor() {
	if w.identityDistributor == nil {
		return
	}

	certFileContents, err := os.ReadFile(w.opts.CertFile)
	if err != nil {
		logger.Warningf("certFile (%s) read failed: %v", w.opts.CertFile, err)
		return
	}
	keyFileContents, err := os.ReadFile(w.opts.KeyFile)
	if err != nil {
		logger.Warningf("keyFile (%s) read failed: %v", w.opts.KeyFile, err)
		return
	}
	// If the file contents have not changed, skip updating the distributor.
	if bytes.Equal(w.certFileContents, certFileContents) && bytes.Equal(w.keyFileContents, keyFileContents) {
		return
	}

	cert, err := tls.X509KeyPair(certFileContents, keyFileContents)
	if err != nil {
		logger.Warningf("tls.X509KeyPair(%q, %q) failed: %v", certFileContents, keyFileContents, err)
		return
	}
	w.certFileContents = certFileContents
	w.keyFileContents = keyFileContents
	w.identityDistributor.Set(&certprovider.KeyMaterial{Certs: []tls.Certificate{cert}}, nil)
}

// updateRootDistributor checks if the root cert file that the plugin is
// watching hs changed, and if so, updates the rootDistributor with the new key
// material.
//
// Skips updates when root cert reading or parsing fails.
// TODO(easwars): Retry with limit (on the number of retries or the amount of
// time) upon failures.
func (w *watcher) updateRootDistributor() {
	if w.rootDistributor == nil {
		return
	}

	// If SPIFFEBundleMap is set, use it and DON'T use the RootFile, even if it
	// fails
	if w.opts.SPIFFEBundleMapFile != "" {
		w.maybeUpdateSPIFFEBundleMap()
	} else {
		w.maybeUpdateRootFile()
	}
}

func (w *watcher) maybeUpdateSPIFFEBundleMap() {
	spiffeBundleMapContents, err := os.ReadFile(w.opts.SPIFFEBundleMapFile)
	if err != nil {
		logger.Warningf("spiffeBundleMapFile (%s) read failed: %v", w.opts.SPIFFEBundleMapFile, err)
		return
	}
	// If the file contents have not changed, skip updating the distributor.
	if bytes.Equal(w.spiffeBundleMapFileContents, spiffeBundleMapContents) {
		return
	}
	bundleMap, err := spiffe.BundleMapFromBytes(spiffeBundleMapContents)
	if err != nil {
		logger.Warning("Failed to parse spiffe bundle map")
		return
	}
	w.spiffeBundleMapFileContents = spiffeBundleMapContents
	w.rootDistributor.Set(&certprovider.KeyMaterial{SPIFFEBundleMap: bundleMap}, nil)
}

func (w *watcher) maybeUpdateRootFile() {
	rootFileContents, err := os.ReadFile(w.opts.RootFile)
	if err != nil {
		logger.Warningf("rootFile (%s) read failed: %v", w.opts.RootFile, err)
		return
	}
	trustPool := x509.NewCertPool()
	if !trustPool.AppendCertsFromPEM(rootFileContents) {
		logger.Warning("Failed to parse root certificate")
		return
	}
	// If the file contents have not changed, skip updating the distributor.
	if bytes.Equal(w.rootFileContents, rootFileContents) {
		return
	}

	w.rootFileContents = rootFileContents
	w.rootDistributor.Set(&certprovider.KeyMaterial{Roots: trustPool}, nil)
}

// run is a long running goroutine which watches the configured files for
// changes, and pushes new key material into the appropriate distributors which
// is returned from calls to KeyMaterial().
func (w *watcher) run(ctx context.Context) {
	ticker := time.NewTicker(w.opts.RefreshDuration)
	for {
		w.updateIdentityDistributor()
		w.updateRootDistributor()
		select {
		case <-ctx.Done():
			ticker.Stop()
			if w.identityDistributor != nil {
				w.identityDistributor.Stop()
			}
			if w.rootDistributor != nil {
				w.rootDistributor.Stop()
			}
			return
		case <-ticker.C:
		}
	}
}

// KeyMaterial returns the key material sourced by the watcher.
// Callers are expected to use the returned value as read-only.
func (w *watcher) KeyMaterial(ctx context.Context) (*certprovider.KeyMaterial, error) {
	km := &certprovider.KeyMaterial{}
	if w.identityDistributor != nil {
		identityKM, err := w.identityDistributor.KeyMaterial(ctx)
		if err != nil {
			return nil, err
		}
		km.Certs = identityKM.Certs
	}
	if w.rootDistributor != nil {
		rootKM, err := w.rootDistributor.KeyMaterial(ctx)
		if err != nil {
			return nil, err
		}
		km.SPIFFEBundleMap = rootKM.SPIFFEBundleMap
		km.Roots = rootKM.Roots
	}
	return km, nil
}

// Close cleans up resources allocated by the watcher.
func (w *watcher) Close() {
	w.cancel()
}
