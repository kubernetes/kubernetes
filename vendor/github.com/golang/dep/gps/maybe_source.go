// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"bytes"
	"context"
	"fmt"
	"net/url"
	"path/filepath"

	"github.com/Masterminds/vcs"
	"github.com/pkg/errors"
)

// A maybeSource represents a set of information that, given some
// typically-expensive network effort, could be transformed into a proper source.
//
// Wrapping these up as their own type achieves two goals:
//
// * Allows control over when deduction logic triggers network activity
// * Makes it easy to attempt multiple URLs for a given import path
type maybeSource interface {
	try(ctx context.Context, cachedir string, c singleSourceCache, superv *supervisor) (source, sourceState, error)
	possibleURLs() []*url.URL
}

type errorSlice []error

func (errs *errorSlice) Error() string {
	var buf bytes.Buffer
	for _, err := range *errs {
		fmt.Fprintf(&buf, "\n\t%s", err)
	}
	return buf.String()
}

type maybeSources []maybeSource

func (mbs maybeSources) try(ctx context.Context, cachedir string, c singleSourceCache, superv *supervisor) (source, sourceState, error) {
	var errs errorSlice
	for _, mb := range mbs {
		src, state, err := mb.try(ctx, cachedir, c, superv)
		if err == nil {
			return src, state, nil
		}
		urls := ""
		for _, url := range mb.possibleURLs() {
			urls += url.String() + "\n"
		}
		errs = append(errs, errors.Wrapf(err, "failed to set up sources from the following URLs:\n%s", urls))
	}

	return nil, 0, errors.Wrap(&errs, "no valid source could be created")
}

// This really isn't generally intended to be used - the interface is for
// maybeSources to be able to interrogate its members, not other things to
// interrogate a maybeSources.
func (mbs maybeSources) possibleURLs() []*url.URL {
	urlslice := make([]*url.URL, 0, len(mbs))
	for _, mb := range mbs {
		urlslice = append(urlslice, mb.possibleURLs()...)
	}
	return urlslice
}

// sourceCachePath returns a url-sanitized source cache dir path.
func sourceCachePath(cacheDir, sourceURL string) string {
	return filepath.Join(cacheDir, "sources", sanitizer.Replace(sourceURL))
}

type maybeGitSource struct {
	url *url.URL
}

func (m maybeGitSource) try(ctx context.Context, cachedir string, c singleSourceCache, superv *supervisor) (source, sourceState, error) {
	ustr := m.url.String()

	r, err := newCtxRepo(vcs.Git, ustr, sourceCachePath(cachedir, ustr))
	if err != nil {
		return nil, 0, unwrapVcsErr(err)
	}

	src := &gitSource{
		baseVCSSource: baseVCSSource{
			repo: r,
		},
	}

	// Pinging invokes the same action as calling listVersions, so just do that.
	var vl []PairedVersion
	if err := superv.do(ctx, "git:lv:maybe", ctListVersions, func(ctx context.Context) error {
		var err error
		vl, err = src.listVersions(ctx)
		return errors.Wrapf(err, "remote repository at %s does not exist, or is inaccessible", ustr)
	}); err != nil {
		return nil, 0, err
	}

	state := sourceIsSetUp | sourceExistsUpstream | sourceHasLatestVersionList

	if r.CheckLocal() {
		state |= sourceExistsLocally

		if err := superv.do(ctx, "git", ctValidateLocal, func(ctx context.Context) error {
			// If repository already exists on disk, make a pass to be sure
			// everything's clean.
			return src.ensureClean(ctx)
		}); err != nil {
			return nil, 0, err
		}
	}

	c.setVersionMap(vl)
	return src, state, nil
}

func (m maybeGitSource) possibleURLs() []*url.URL {
	return []*url.URL{m.url}
}

type maybeGopkginSource struct {
	// the original gopkg.in import path. this is used to create the on-disk
	// location to avoid duplicate resource management - e.g., if instances of
	// a gopkg.in project are accessed via different schemes, or if the
	// underlying github repository is accessed directly.
	opath string
	// the actual upstream URL - always github
	url *url.URL
	// the major version to apply for filtering
	major uint64
	// whether or not the source package is "unstable"
	unstable bool
}

func (m maybeGopkginSource) try(ctx context.Context, cachedir string, c singleSourceCache, superv *supervisor) (source, sourceState, error) {
	// We don't actually need a fully consistent transform into the on-disk path
	// - just something that's unique to the particular gopkg.in domain context.
	// So, it's OK to just dumb-join the scheme with the path.
	aliasURL := m.url.Scheme + "://" + m.opath
	path := sourceCachePath(cachedir, aliasURL)
	ustr := m.url.String()

	r, err := newCtxRepo(vcs.Git, ustr, path)
	if err != nil {
		return nil, 0, unwrapVcsErr(err)
	}

	src := &gopkginSource{
		gitSource: gitSource{
			baseVCSSource: baseVCSSource{
				repo: r,
			},
		},
		major:    m.major,
		unstable: m.unstable,
		aliasURL: aliasURL,
	}

	var vl []PairedVersion
	if err := superv.do(ctx, "git:lv:maybe", ctListVersions, func(ctx context.Context) error {
		var err error
		vl, err = src.listVersions(ctx)
		return errors.Wrapf(err, "remote repository at %s does not exist, or is inaccessible", ustr)
	}); err != nil {
		return nil, 0, err
	}

	c.setVersionMap(vl)
	state := sourceIsSetUp | sourceExistsUpstream | sourceHasLatestVersionList

	if r.CheckLocal() {
		state |= sourceExistsLocally
	}

	return src, state, nil
}

func (m maybeGopkginSource) possibleURLs() []*url.URL {
	return []*url.URL{m.url}
}

type maybeBzrSource struct {
	url *url.URL
}

func (m maybeBzrSource) try(ctx context.Context, cachedir string, c singleSourceCache, superv *supervisor) (source, sourceState, error) {
	ustr := m.url.String()

	r, err := newCtxRepo(vcs.Bzr, ustr, sourceCachePath(cachedir, ustr))
	if err != nil {
		return nil, 0, unwrapVcsErr(err)
	}

	if err := superv.do(ctx, "bzr:ping", ctSourcePing, func(ctx context.Context) error {
		if !r.Ping() {
			return fmt.Errorf("remote repository at %s does not exist, or is inaccessible", ustr)
		}
		return nil
	}); err != nil {
		return nil, 0, err
	}

	state := sourceIsSetUp | sourceExistsUpstream
	if r.CheckLocal() {
		state |= sourceExistsLocally
	}

	src := &bzrSource{
		baseVCSSource: baseVCSSource{
			repo: r,
		},
	}

	return src, state, nil
}

func (m maybeBzrSource) possibleURLs() []*url.URL {
	return []*url.URL{m.url}
}

type maybeHgSource struct {
	url *url.URL
}

func (m maybeHgSource) try(ctx context.Context, cachedir string, c singleSourceCache, superv *supervisor) (source, sourceState, error) {
	ustr := m.url.String()

	r, err := newCtxRepo(vcs.Hg, ustr, sourceCachePath(cachedir, ustr))
	if err != nil {
		return nil, 0, unwrapVcsErr(err)
	}

	if err := superv.do(ctx, "hg:ping", ctSourcePing, func(ctx context.Context) error {
		if !r.Ping() {
			return fmt.Errorf("remote repository at %s does not exist, or is inaccessible", ustr)
		}
		return nil
	}); err != nil {
		return nil, 0, err
	}

	state := sourceIsSetUp | sourceExistsUpstream
	if r.CheckLocal() {
		state |= sourceExistsLocally
	}

	src := &hgSource{
		baseVCSSource: baseVCSSource{
			repo: r,
		},
	}

	return src, state, nil
}

func (m maybeHgSource) possibleURLs() []*url.URL {
	return []*url.URL{m.url}
}
