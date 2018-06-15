// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/Masterminds/semver"
	"github.com/golang/dep/gps/pkgtree"
	"github.com/golang/dep/internal/fs"
	"github.com/pkg/errors"
)

type baseVCSSource struct {
	repo ctxRepo
}

func (bs *baseVCSSource) sourceType() string {
	return string(bs.repo.Vcs())
}

func (bs *baseVCSSource) existsLocally(ctx context.Context) bool {
	return bs.repo.CheckLocal()
}

// TODO reimpl for git
func (bs *baseVCSSource) existsUpstream(ctx context.Context) bool {
	return !bs.repo.Ping()
}

func (bs *baseVCSSource) upstreamURL() string {
	return bs.repo.Remote()
}

func (bs *baseVCSSource) disambiguateRevision(ctx context.Context, r Revision) (Revision, error) {
	ci, err := bs.repo.CommitInfo(string(r))
	if err != nil {
		return "", err
	}
	return Revision(ci.Commit), nil
}

func (bs *baseVCSSource) getManifestAndLock(ctx context.Context, pr ProjectRoot, r Revision, an ProjectAnalyzer) (Manifest, Lock, error) {
	err := bs.repo.updateVersion(ctx, r.String())
	if err != nil {
		return nil, nil, unwrapVcsErr(err)
	}

	m, l, err := an.DeriveManifestAndLock(bs.repo.LocalPath(), pr)
	if err != nil {
		return nil, nil, err
	}

	if l != nil && l != Lock(nil) {
		l = prepLock(l)
	}

	return prepManifest(m), l, nil
}

func (bs *baseVCSSource) revisionPresentIn(r Revision) (bool, error) {
	return bs.repo.IsReference(string(r)), nil
}

// initLocal clones/checks out the upstream repository to disk for the first
// time.
func (bs *baseVCSSource) initLocal(ctx context.Context) error {
	err := bs.repo.get(ctx)

	if err != nil {
		return unwrapVcsErr(err)
	}
	return nil
}

// updateLocal ensures the local data (versions and code) we have about the
// source is fully up to date with that of the canonical upstream source.
func (bs *baseVCSSource) updateLocal(ctx context.Context) error {
	err := bs.repo.fetch(ctx)

	if err != nil {
		return unwrapVcsErr(err)
	}
	return nil
}

func (bs *baseVCSSource) listPackages(ctx context.Context, pr ProjectRoot, r Revision) (ptree pkgtree.PackageTree, err error) {
	err = bs.repo.updateVersion(ctx, r.String())

	if err != nil {
		err = unwrapVcsErr(err)
	} else {
		ptree, err = pkgtree.ListPackages(bs.repo.LocalPath(), string(pr))
	}

	return
}

func (bs *baseVCSSource) exportRevisionTo(ctx context.Context, r Revision, to string) error {
	// Only make the parent dir, as CopyDir will balk on trying to write to an
	// empty but existing dir.
	if err := os.MkdirAll(filepath.Dir(to), 0777); err != nil {
		return err
	}

	if err := bs.repo.updateVersion(ctx, r.String()); err != nil {
		return unwrapVcsErr(err)
	}

	return fs.CopyDir(bs.repo.LocalPath(), to)
}

var (
	gitHashRE = regexp.MustCompile(`^[a-f0-9]{40}$`)
)

// gitSource is a generic git repository implementation that should work with
// all standard git remotes.
type gitSource struct {
	baseVCSSource
}

// ensureClean sees to it that a git repository is clean and in working order,
// or returns an error if the adaptive recovery attempts fail.
func (s *gitSource) ensureClean(ctx context.Context) error {
	r := s.repo.(*gitRepo)
	cmd := commandContext(
		ctx,
		"git",
		"status",
		"--porcelain",
	)
	cmd.SetDir(r.LocalPath())

	out, err := cmd.CombinedOutput()
	if err != nil {
		// An error on simple git status indicates some aggressive repository
		// corruption, outside of the purview that we can deal with here.
		return err
	}

	if len(bytes.TrimSpace(out)) == 0 {
		// No output from status indicates a clean tree, without any modified or
		// untracked files - we're in good shape.
		return nil
	}

	// We could be more parsimonious about this, but it's probably not worth it
	// - it's a rare case to have to do any cleanup anyway, so when we do, we
	// might as well just throw the kitchen sink at it.
	cmd = commandContext(
		ctx,
		"git",
		"reset",
		"--hard",
	)
	cmd.SetDir(r.LocalPath())
	_, err = cmd.CombinedOutput()
	if err != nil {
		return err
	}

	// We also need to git clean -df; just reuse defendAgainstSubmodules here,
	// even though it's a bit layer-breaky.
	err = r.defendAgainstSubmodules(ctx)
	if err != nil {
		return err
	}

	// Check status one last time. If it's still not clean, give up.
	cmd = commandContext(
		ctx,
		"git",
		"status",
		"--porcelain",
	)
	cmd.SetDir(r.LocalPath())

	out, err = cmd.CombinedOutput()
	if err != nil {
		return err
	}

	if len(bytes.TrimSpace(out)) != 0 {
		return errors.Errorf("failed to clean up git repository at %s - dirty? corrupted? status output: \n%s", r.LocalPath(), string(out))
	}

	return nil
}

func (s *gitSource) exportRevisionTo(ctx context.Context, rev Revision, to string) error {
	r := s.repo

	if err := os.MkdirAll(to, 0777); err != nil {
		return err
	}

	// Back up original index
	idx, bak := filepath.Join(r.LocalPath(), ".git", "index"), filepath.Join(r.LocalPath(), ".git", "origindex")
	err := fs.RenameWithFallback(idx, bak)
	if err != nil {
		return err
	}

	// could have an err here...but it's hard to imagine how?
	defer fs.RenameWithFallback(bak, idx)

	{
		cmd := commandContext(ctx, "git", "read-tree", rev.String())
		cmd.SetDir(r.LocalPath())
		if out, err := cmd.CombinedOutput(); err != nil {
			return errors.Wrap(err, string(out))
		}
	}

	// Ensure we have exactly one trailing slash
	to = strings.TrimSuffix(to, string(os.PathSeparator)) + string(os.PathSeparator)
	// Checkout from our temporary index to the desired target location on
	// disk; now it's git's job to make it fast.
	//
	// Sadly, this approach *does* also write out vendor dirs. There doesn't
	// appear to be a way to make checkout-index respect sparse checkout
	// rules (-a supersedes it). The alternative is using plain checkout,
	// though we have a bunch of housekeeping to do to set up, then tear
	// down, the sparse checkout controls, as well as restore the original
	// index and HEAD.
	{
		cmd := commandContext(ctx, "git", "checkout-index", "-a", "--prefix="+to)
		cmd.SetDir(r.LocalPath())
		if out, err := cmd.CombinedOutput(); err != nil {
			return errors.Wrap(err, string(out))
		}
	}

	return nil
}

func (s *gitSource) isValidHash(hash []byte) bool {
	return gitHashRE.Match(hash)
}

func (s *gitSource) listVersions(ctx context.Context) (vlist []PairedVersion, err error) {
	r := s.repo

	cmd := commandContext(ctx, "git", "ls-remote", r.Remote())
	// We want to invoke from a place where it's not possible for there to be a
	// .git file instead of a .git directory, as git ls-remote will choke on the
	// former and erroneously quit. However, we can't be sure that the repo
	// exists on disk yet at this point; if it doesn't, then instead use the
	// parent of the local path, as that's still likely a good bet.
	if r.CheckLocal() {
		cmd.SetDir(r.LocalPath())
	} else {
		cmd.SetDir(filepath.Dir(r.LocalPath()))
	}
	// Ensure no prompting for PWs
	cmd.SetEnv(append([]string{"GIT_ASKPASS=", "GIT_TERMINAL_PROMPT=0"}, os.Environ()...))
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, errors.Wrap(err, string(out))
	}

	all := bytes.Split(bytes.TrimSpace(out), []byte("\n"))
	if len(all) == 1 && len(all[0]) == 0 {
		return nil, fmt.Errorf("no data returned from ls-remote")
	}

	// Pull out the HEAD rev (it's always first) so we know what branches to
	// mark as default. This is, perhaps, not the best way to glean this, but it
	// was good enough for git itself until 1.8.5. Also, the alternative is
	// sniffing data out of the pack protocol, which is a separate request, and
	// also waaaay more than we want to do right now.
	//
	// The cost is that we could potentially have multiple branches marked as
	// the default. If that does occur, a later check (again, emulating git
	// <1.8.5 behavior) further narrows the failure mode by choosing master as
	// the sole default branch if a) master exists and b) master is one of the
	// branches marked as a default.
	//
	// This all reduces the failure mode to a very narrow range of
	// circumstances. Nevertheless, if we do end up emitting multiple
	// default branches, it is possible that a user could end up following a
	// non-default branch, IF:
	//
	// * Multiple branches match the HEAD rev
	// * None of them are master
	// * The solver makes it into the branch list in the version queue
	// * The user/tool has provided no constraint (so, anyConstraint)
	// * A branch that is not actually the default, but happens to share the
	//   rev, is lexicographically less than the true default branch
	//
	// If all of those conditions are met, then the user would end up with an
	// erroneous non-default branch in their lock file.
	var headrev Revision
	var onedef, multidef, defmaster bool

	smap := make(map[string]int)
	uniq := 0
	vlist = make([]PairedVersion, len(all))
	for _, pair := range all {
		var v PairedVersion
		// Valid `git ls-remote` output should start with hash, be at least
		// 45 chars long and 40th character should be '\t'
		//
		// See: https://github.com/golang/dep/pull/1160#issuecomment-328843519
		if len(pair) < 45 || pair[40] != '\t' || !s.isValidHash(pair[:40]) {
			continue
		}
		if string(pair[41:]) == "HEAD" {
			// If HEAD is present, it's always first
			headrev = Revision(pair[:40])
		} else if string(pair[46:51]) == "heads" {
			rev := Revision(pair[:40])

			isdef := rev == headrev
			n := string(pair[52:])
			if isdef {
				if onedef {
					multidef = true
				}
				onedef = true
				if n == "master" {
					defmaster = true
				}
			}
			v = branchVersion{
				name:      n,
				isDefault: isdef,
			}.Pair(rev).(PairedVersion)

			vlist[uniq] = v
			uniq++
		} else if string(pair[46:50]) == "tags" {
			vstr := string(pair[51:])
			if strings.HasSuffix(vstr, "^{}") {
				// If the suffix is there, then we *know* this is the rev of
				// the underlying commit object that we actually want
				vstr = strings.TrimSuffix(vstr, "^{}")
				if i, ok := smap[vstr]; ok {
					v = NewVersion(vstr).Pair(Revision(pair[:40]))
					vlist[i] = v
					continue
				}
			} else if _, ok := smap[vstr]; ok {
				// Already saw the deref'd version of this tag, if one
				// exists, so skip this.
				continue
				// Can only hit this branch if we somehow got the deref'd
				// version first. Which should be impossible, but this
				// covers us in case of weirdness, anyway.
			}
			v = NewVersion(vstr).Pair(Revision(pair[:40]))
			smap[vstr] = uniq
			vlist[uniq] = v
			uniq++
		}
	}

	// Trim off excess from the slice
	vlist = vlist[:uniq]

	// There were multiple default branches, but one was master. So, go through
	// and strip the default flag from all the non-master branches.
	if multidef && defmaster {
		for k, v := range vlist {
			pv := v.(PairedVersion)
			if bv, ok := pv.Unpair().(branchVersion); ok {
				if bv.name != "master" && bv.isDefault {
					bv.isDefault = false
					vlist[k] = bv.Pair(pv.Revision())
				}
			}
		}
	}

	return
}

// gopkginSource is a specialized git source that performs additional filtering
// according to the input URL.
type gopkginSource struct {
	gitSource
	major    uint64
	unstable bool
	// The aliased URL we report as being the one we talk to, even though we're
	// actually talking directly to GitHub.
	aliasURL string
}

func (s *gopkginSource) upstreamURL() string {
	return s.aliasURL
}

func (s *gopkginSource) listVersions(ctx context.Context) ([]PairedVersion, error) {
	ovlist, err := s.gitSource.listVersions(ctx)
	if err != nil {
		return nil, err
	}

	// Apply gopkg.in's filtering rules
	vlist := make([]PairedVersion, len(ovlist))
	k := 0
	var dbranch int // index of branch to be marked default
	var bsv semver.Version
	var defaultBranch PairedVersion
	tryDefaultAsV0 := s.major == 0
	for _, v := range ovlist {
		// all git versions will always be paired
		pv := v.(versionPair)
		switch tv := pv.v.(type) {
		case semVersion:
			tryDefaultAsV0 = false
			if tv.sv.Major() == s.major && !s.unstable {
				vlist[k] = v
				k++
			}
		case branchVersion:
			if tv.isDefault && defaultBranch == nil {
				defaultBranch = pv
			}

			// The semver lib isn't exactly the same as gopkg.in's logic, but
			// it's close enough that it's probably fine to use. We can be more
			// exact if real problems crop up.
			sv, err := semver.NewVersion(tv.name)
			if err != nil {
				continue
			}
			tryDefaultAsV0 = false

			if sv.Major() != s.major {
				// not the same major version as specified in the import path constraint
				continue
			}

			// Gopkg.in has a special "-unstable" suffix which we need to handle
			// separately.
			if s.unstable != strings.HasSuffix(tv.name, gopkgUnstableSuffix) {
				continue
			}

			// Turn off the default branch marker unconditionally; we can't know
			// which one to mark as default until we've seen them all
			tv.isDefault = false
			// Figure out if this is the current leader for default branch
			if bsv == (semver.Version{}) || bsv.LessThan(sv) {
				bsv = sv
				dbranch = k
			}
			pv.v = tv
			vlist[k] = pv
			k++
		}
		// The switch skips plainVersions because they cannot possibly meet
		// gopkg.in's requirements
	}

	vlist = vlist[:k]
	if bsv != (semver.Version{}) {
		dbv := vlist[dbranch].(versionPair)
		vlist[dbranch] = branchVersion{
			name:      dbv.v.(branchVersion).name,
			isDefault: true,
		}.Pair(dbv.r)
	}

	// Treat the default branch as v0 only when no other semver branches/tags exist
	// See http://labix.org/gopkg.in#VersionZero
	if tryDefaultAsV0 && defaultBranch != nil {
		vlist = append(vlist, defaultBranch)
	}

	return vlist, nil
}

// bzrSource is a generic bzr repository implementation that should work with
// all standard bazaar remotes.
type bzrSource struct {
	baseVCSSource
}

func (s *bzrSource) exportRevisionTo(ctx context.Context, rev Revision, to string) error {
	if err := s.baseVCSSource.exportRevisionTo(ctx, rev, to); err != nil {
		return err
	}

	return os.RemoveAll(filepath.Join(to, ".bzr"))
}

func (s *bzrSource) listVersions(ctx context.Context) ([]PairedVersion, error) {
	r := s.repo

	// TODO(sdboyer) this should be handled through the gateway's FSM
	if !r.CheckLocal() {
		err := s.initLocal(ctx)
		if err != nil {
			return nil, err
		}
	}

	// Now, list all the tags
	tagsCmd := commandContext(ctx, "bzr", "tags", "--show-ids", "-v")
	tagsCmd.SetDir(r.LocalPath())
	out, err := tagsCmd.CombinedOutput()
	if err != nil {
		return nil, errors.Wrap(err, string(out))
	}

	all := bytes.Split(bytes.TrimSpace(out), []byte("\n"))

	viCmd := commandContext(ctx, "bzr", "version-info", "--custom", "--template={revision_id}", "--revision=branch:.")
	viCmd.SetDir(r.LocalPath())
	branchrev, err := viCmd.CombinedOutput()
	if err != nil {
		return nil, errors.Wrap(err, string(branchrev))
	}

	vlist := make([]PairedVersion, 0, len(all)+1)

	// Now, all the tags.
	for _, line := range all {
		idx := bytes.IndexByte(line, 32) // space
		v := NewVersion(string(line[:idx]))
		r := Revision(bytes.TrimSpace(line[idx:]))
		vlist = append(vlist, v.Pair(r))
	}

	// Last, add the default branch, hardcoding the visual representation of it
	// that bzr uses when operating in the workflow mode we're using.
	v := newDefaultBranch("(default)")
	vlist = append(vlist, v.Pair(Revision(string(branchrev))))

	return vlist, nil
}

func (s *bzrSource) disambiguateRevision(ctx context.Context, r Revision) (Revision, error) {
	// If we used the default baseVCSSource behavior here, we would return the
	// bazaar revision number, which is not a globally unique identifier - it is
	// only unique within a branch. This is just the way that
	// github.com/Masterminds/vcs chooses to handle bazaar. We want a
	// disambiguated unique ID, though, so we need slightly different behavior:
	// check whether r doesn't error when we try to look it up. If so, trust that
	// it's a revision.
	_, err := s.repo.CommitInfo(string(r))
	if err != nil {
		return "", err
	}
	return r, nil
}

// hgSource is a generic hg repository implementation that should work with
// all standard mercurial servers.
type hgSource struct {
	baseVCSSource
}

func (s *hgSource) exportRevisionTo(ctx context.Context, rev Revision, to string) error {
	// TODO: use hg instead of the generic approach in
	// baseVCSSource.exportRevisionTo to make it faster.
	if err := s.baseVCSSource.exportRevisionTo(ctx, rev, to); err != nil {
		return err
	}

	return os.RemoveAll(filepath.Join(to, ".hg"))
}

func (s *hgSource) listVersions(ctx context.Context) ([]PairedVersion, error) {
	var vlist []PairedVersion

	r := s.repo
	// TODO(sdboyer) this should be handled through the gateway's FSM
	if !r.CheckLocal() {
		err := s.initLocal(ctx)
		if err != nil {
			return nil, err
		}
	}

	// Now, list all the tags
	tagsCmd := commandContext(ctx, "hg", "tags", "--debug", "--verbose")
	tagsCmd.SetDir(r.LocalPath())
	out, err := tagsCmd.CombinedOutput()
	if err != nil {
		return nil, errors.Wrap(err, string(out))
	}

	all := bytes.Split(bytes.TrimSpace(out), []byte("\n"))
	lbyt := []byte("local")
	nulrev := []byte("0000000000000000000000000000000000000000")
	for _, line := range all {
		if bytes.Equal(lbyt, line[len(line)-len(lbyt):]) {
			// Skip local tags
			continue
		}

		// tip is magic, don't include it
		if bytes.HasPrefix(line, []byte("tip")) {
			continue
		}

		// Split on colon; this gets us the rev and the tag plus local revno
		pair := bytes.Split(line, []byte(":"))
		if bytes.Equal(nulrev, pair[1]) {
			// null rev indicates this tag is marked for deletion
			continue
		}

		idx := bytes.IndexByte(pair[0], 32) // space
		v := NewVersion(string(pair[0][:idx])).Pair(Revision(pair[1])).(PairedVersion)
		vlist = append(vlist, v)
	}

	// bookmarks next, because the presence of the magic @ bookmark has to
	// determine how we handle the branches
	var magicAt bool
	bookmarksCmd := commandContext(ctx, "hg", "bookmarks", "--debug")
	bookmarksCmd.SetDir(r.LocalPath())
	out, err = bookmarksCmd.CombinedOutput()
	if err != nil {
		// better nothing than partial and misleading
		return nil, errors.Wrap(err, string(out))
	}

	out = bytes.TrimSpace(out)
	if !bytes.Equal(out, []byte("no bookmarks set")) {
		all = bytes.Split(out, []byte("\n"))
		for _, line := range all {
			// Trim leading spaces, and * marker if present
			line = bytes.TrimLeft(line, " *")
			pair := bytes.Split(line, []byte(":"))
			// if this doesn't split exactly once, we have something weird
			if len(pair) != 2 {
				continue
			}

			// Split on colon; this gets us the rev and the branch plus local revno
			idx := bytes.IndexByte(pair[0], 32) // space
			// if it's the magic @ marker, make that the default branch
			str := string(pair[0][:idx])
			var v PairedVersion
			if str == "@" {
				magicAt = true
				v = newDefaultBranch(str).Pair(Revision(pair[1])).(PairedVersion)
			} else {
				v = NewBranch(str).Pair(Revision(pair[1])).(PairedVersion)
			}
			vlist = append(vlist, v)
		}
	}

	cmd := commandContext(ctx, "hg", "branches", "-c", "--debug")
	cmd.SetDir(r.LocalPath())
	out, err = cmd.CombinedOutput()
	if err != nil {
		// better nothing than partial and misleading
		return nil, errors.Wrap(err, string(out))
	}

	all = bytes.Split(bytes.TrimSpace(out), []byte("\n"))
	for _, line := range all {
		// Trim inactive and closed suffixes, if present; we represent these
		// anyway
		line = bytes.TrimSuffix(line, []byte(" (inactive)"))
		line = bytes.TrimSuffix(line, []byte(" (closed)"))

		// Split on colon; this gets us the rev and the branch plus local revno
		pair := bytes.Split(line, []byte(":"))
		idx := bytes.IndexByte(pair[0], 32) // space
		str := string(pair[0][:idx])
		// if there was no magic @ bookmark, and this is mercurial's magic
		// "default" branch, then mark it as default branch
		var v PairedVersion
		if !magicAt && str == "default" {
			v = newDefaultBranch(str).Pair(Revision(pair[1])).(PairedVersion)
		} else {
			v = NewBranch(str).Pair(Revision(pair[1])).(PairedVersion)
		}
		vlist = append(vlist, v)
	}

	return vlist, nil
}
