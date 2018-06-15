// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"context"
	"fmt"
	"log"
	"sync"

	"github.com/golang/dep/gps/pkgtree"
	"github.com/pkg/errors"
)

// sourceState represent the states that a source can be in, depending on how
// much search and discovery work ahs been done by a source's managing gateway.
//
// These are basically used to achieve a cheap approximation of a FSM.
type sourceState int32

const (
	sourceIsSetUp sourceState = 1 << iota
	sourceExistsUpstream
	sourceExistsLocally
	sourceHasLatestVersionList
	sourceHasLatestLocally
)

type srcReturnChans struct {
	ret chan *sourceGateway
	err chan error
}

func (rc srcReturnChans) awaitReturn() (sg *sourceGateway, err error) {
	select {
	case sg = <-rc.ret:
	case err = <-rc.err:
	}
	return
}

type sourceCoordinator struct {
	supervisor *supervisor
	srcmut     sync.RWMutex // guards srcs and nameToURL maps
	srcs       map[string]*sourceGateway
	nameToURL  map[string]string
	psrcmut    sync.Mutex // guards protoSrcs map
	protoSrcs  map[string][]srcReturnChans
	deducer    deducer
	cachedir   string
	logger     *log.Logger
}

func newSourceCoordinator(superv *supervisor, deducer deducer, cachedir string, logger *log.Logger) *sourceCoordinator {
	return &sourceCoordinator{
		supervisor: superv,
		deducer:    deducer,
		cachedir:   cachedir,
		logger:     logger,
		srcs:       make(map[string]*sourceGateway),
		nameToURL:  make(map[string]string),
		protoSrcs:  make(map[string][]srcReturnChans),
	}
}

func (sc *sourceCoordinator) close() {}

func (sc *sourceCoordinator) getSourceGatewayFor(ctx context.Context, id ProjectIdentifier) (*sourceGateway, error) {
	if err := sc.supervisor.ctx.Err(); err != nil {
		return nil, err
	}

	normalizedName := id.normalizedSource()

	sc.srcmut.RLock()
	if url, has := sc.nameToURL[normalizedName]; has {
		srcGate, has := sc.srcs[url]
		sc.srcmut.RUnlock()
		if has {
			return srcGate, nil
		}
		panic(fmt.Sprintf("%q was URL for %q in nameToURL, but no corresponding srcGate in srcs map", url, normalizedName))
	}

	// Without a direct match, we must fold the input name to a generally
	// stable, caseless variant and primarily work from that. This ensures that
	// on case-insensitive filesystems, we do not end up with multiple
	// sourceGateways for paths that vary only by case. We perform folding
	// unconditionally, independent of whether the underlying fs is
	// case-sensitive, in order to ensure uniform behavior.
	//
	// This has significant implications. It is effectively deciding that the
	// ProjectRoot portion of import paths are case-insensitive, which is by no
	// means an invariant maintained by all hosting systems. If this presents a
	// problem in practice, then we can explore expanding the deduction system
	// to include case-sensitivity-for-roots metadata and treat it on a
	// host-by-host basis. Such cases would still be rejected by the Go
	// toolchain's compiler, though, and case-sensitivity in root names is
	// likely to be at least frowned on if not disallowed by most hosting
	// systems. So we follow this path, which is both a vastly simpler solution
	// and one that seems quite likely to work in practice.
	foldedNormalName := toFold(normalizedName)
	notFolded := foldedNormalName != normalizedName
	if notFolded {
		// If the folded name differs from the input name, then there may
		// already be an entry for it in the nameToURL map, so check again.
		if url, has := sc.nameToURL[foldedNormalName]; has {
			// There was a match on the canonical folded variant. Upgrade to a
			// write lock, so that future calls on this name don't need to
			// burn cycles on folding.
			sc.srcmut.RUnlock()
			sc.srcmut.Lock()
			// It may be possible that another goroutine could interleave
			// between the unlock and re-lock. Even if they do, though, they'll
			// only have recorded the same url value as we have here. In other
			// words, these operations commute, so we can safely write here
			// without checking again.
			sc.nameToURL[normalizedName] = url

			srcGate, has := sc.srcs[url]
			sc.srcmut.Unlock()
			if has {
				return srcGate, nil
			}
			panic(fmt.Sprintf("%q was URL for %q in nameToURL, but no corresponding srcGate in srcs map", url, normalizedName))
		}
	}
	sc.srcmut.RUnlock()

	// No gateway exists for this path yet; set up a proto, being careful to fold
	// together simultaneous attempts on the same case-folded path.
	sc.psrcmut.Lock()
	if chans, has := sc.protoSrcs[foldedNormalName]; has {
		// Another goroutine is already working on this normalizedName. Fold
		// in with that work by attaching our return channels to the list.
		rc := srcReturnChans{
			ret: make(chan *sourceGateway, 1),
			err: make(chan error, 1),
		}
		sc.protoSrcs[foldedNormalName] = append(chans, rc)
		sc.psrcmut.Unlock()
		return rc.awaitReturn()
	}

	sc.protoSrcs[foldedNormalName] = []srcReturnChans{}
	sc.psrcmut.Unlock()

	doReturn := func(sg *sourceGateway, err error) {
		sc.psrcmut.Lock()
		if sg != nil {
			for _, rc := range sc.protoSrcs[foldedNormalName] {
				rc.ret <- sg
			}
		} else if err != nil {
			for _, rc := range sc.protoSrcs[foldedNormalName] {
				rc.err <- err
			}
		} else {
			panic("sg and err both nil")
		}

		delete(sc.protoSrcs, foldedNormalName)
		sc.psrcmut.Unlock()
	}

	pd, err := sc.deducer.deduceRootPath(ctx, normalizedName)
	if err != nil {
		// As in the deducer, don't cache errors so that externally-driven retry
		// strategies can be constructed.
		doReturn(nil, err)
		return nil, err
	}

	// It'd be quite the feat - but not impossible - for a gateway
	// corresponding to this normalizedName to have slid into the main
	// sources map after the initial unlock, but before this goroutine got
	// scheduled. Guard against that by checking the main sources map again
	// and bailing out if we find an entry.
	var srcGate *sourceGateway
	sc.srcmut.RLock()
	if url, has := sc.nameToURL[foldedNormalName]; has {
		if srcGate, has := sc.srcs[url]; has {
			sc.srcmut.RUnlock()
			doReturn(srcGate, nil)
			return srcGate, nil
		}
		panic(fmt.Sprintf("%q was URL for %q in nameToURL, but no corresponding srcGate in srcs map", url, normalizedName))
	}
	sc.srcmut.RUnlock()

	srcGate = newSourceGateway(pd.mb, sc.supervisor, sc.cachedir)

	// The normalized name is usually different from the source URL- e.g.
	// github.com/sdboyer/gps vs. https://github.com/sdboyer/gps. But it's
	// possible to arrive here with a full URL as the normalized name - and both
	// paths *must* lead to the same sourceGateway instance in order to ensure
	// disk access is correctly managed.
	//
	// Therefore, we now must query the sourceGateway to get the actual
	// sourceURL it's operating on, and ensure it's *also* registered at
	// that path in the map. This will cause it to actually initiate the
	// maybeSource.try() behavior in order to settle on a URL.
	url, err := srcGate.sourceURL(ctx)
	if err != nil {
		doReturn(nil, err)
		return nil, err
	}

	// If the normalizedName and foldedNormalName differ, then we're pretty well
	// guaranteed that returned URL will also need folding into canonical form.
	var unfoldedURL string
	if notFolded {
		unfoldedURL = url
		url = toFold(url)
	}

	// We know we have a working srcGateway at this point, and need to
	// integrate it back into the main map.
	sc.srcmut.Lock()
	defer sc.srcmut.Unlock()
	// Record the name -> URL mapping, making sure that we also get the
	// self-mapping.
	sc.nameToURL[foldedNormalName] = url
	if url != foldedNormalName {
		sc.nameToURL[url] = url
	}

	// Make sure we have both the folded and unfolded names and URLs recorded in
	// the map, if the input needed folding.
	if notFolded {
		sc.nameToURL[normalizedName] = url
		sc.nameToURL[unfoldedURL] = url
	}

	if sa, has := sc.srcs[url]; has {
		// URL already had an entry in the main map; use that as the result.
		doReturn(sa, nil)
		return sa, nil
	}

	sc.srcs[url] = srcGate
	doReturn(srcGate, nil)
	return srcGate, nil
}

// sourceGateways manage all incoming calls for data from sources, serializing
// and caching them as needed.
type sourceGateway struct {
	cachedir string
	maybe    maybeSource
	srcState sourceState
	src      source
	cache    singleSourceCache
	mu       sync.Mutex // global lock, serializes all behaviors
	suprvsr  *supervisor
}

func newSourceGateway(maybe maybeSource, superv *supervisor, cachedir string) *sourceGateway {
	sg := &sourceGateway{
		maybe:    maybe,
		cachedir: cachedir,
		suprvsr:  superv,
	}
	sg.cache = sg.createSingleSourceCache()

	return sg
}

func (sg *sourceGateway) syncLocal(ctx context.Context) error {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	_, err := sg.require(ctx, sourceIsSetUp|sourceExistsLocally|sourceHasLatestLocally)
	return err
}

func (sg *sourceGateway) existsInCache(ctx context.Context) bool {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	_, err := sg.require(ctx, sourceIsSetUp|sourceExistsLocally)
	if err != nil {
		return false
	}

	return sg.srcState&sourceExistsLocally != 0
}

func (sg *sourceGateway) existsUpstream(ctx context.Context) bool {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	_, err := sg.require(ctx, sourceIsSetUp|sourceExistsUpstream)
	if err != nil {
		return false
	}

	return sg.srcState&sourceExistsUpstream != 0
}

func (sg *sourceGateway) exportVersionTo(ctx context.Context, v Version, to string) error {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	_, err := sg.require(ctx, sourceIsSetUp|sourceExistsLocally)
	if err != nil {
		return err
	}

	r, err := sg.convertToRevision(ctx, v)
	if err != nil {
		return err
	}

	err = sg.suprvsr.do(ctx, sg.src.upstreamURL(), ctExportTree, func(ctx context.Context) error {
		return sg.src.exportRevisionTo(ctx, r, to)
	})

	// It's possible (in git) that we may have tried this against a version that
	// doesn't exist in the repository cache, even though we know it exists in
	// the upstream. If it looks like that might be the case, update the local
	// and retry.
	// TODO(sdboyer) It'd be better if we could check the error to see if this
	// actually was the cause of the problem.
	if err != nil && sg.srcState&sourceHasLatestLocally == 0 {
		if _, err = sg.require(ctx, sourceHasLatestLocally); err == nil {
			err = sg.suprvsr.do(ctx, sg.src.upstreamURL(), ctExportTree, func(ctx context.Context) error {
				return sg.src.exportRevisionTo(ctx, r, to)
			})
		}
	}

	return err
}

func (sg *sourceGateway) getManifestAndLock(ctx context.Context, pr ProjectRoot, v Version, an ProjectAnalyzer) (Manifest, Lock, error) {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	r, err := sg.convertToRevision(ctx, v)
	if err != nil {
		return nil, nil, err
	}

	m, l, has := sg.cache.getManifestAndLock(r, an.Info())
	if has {
		return m, l, nil
	}

	_, err = sg.require(ctx, sourceIsSetUp|sourceExistsLocally)
	if err != nil {
		return nil, nil, err
	}

	label := fmt.Sprintf("%s:%s", sg.src.upstreamURL(), an.Info())
	err = sg.suprvsr.do(ctx, label, ctGetManifestAndLock, func(ctx context.Context) error {
		m, l, err = sg.src.getManifestAndLock(ctx, pr, r, an)
		return err
	})

	// It's possible (in git) that we may have tried this against a version that
	// doesn't exist in the repository cache, even though we know it exists in
	// the upstream. If it looks like that might be the case, update the local
	// and retry.
	// TODO(sdboyer) It'd be better if we could check the error to see if this
	// actually was the cause of the problem.
	if err != nil && sg.srcState&sourceHasLatestLocally == 0 {
		// TODO(sdboyer) we should warn/log/something in adaptive recovery
		// situations like this
		_, err = sg.require(ctx, sourceHasLatestLocally)
		if err != nil {
			return nil, nil, err
		}

		err = sg.suprvsr.do(ctx, label, ctGetManifestAndLock, func(ctx context.Context) error {
			m, l, err = sg.src.getManifestAndLock(ctx, pr, r, an)
			return err
		})
	}

	if err != nil {
		return nil, nil, err
	}

	sg.cache.setManifestAndLock(r, an.Info(), m, l)
	return m, l, nil
}

// FIXME ProjectRoot input either needs to parameterize the cache, or be
// incorporated on the fly on egress...?
func (sg *sourceGateway) listPackages(ctx context.Context, pr ProjectRoot, v Version) (pkgtree.PackageTree, error) {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	r, err := sg.convertToRevision(ctx, v)
	if err != nil {
		return pkgtree.PackageTree{}, err
	}

	ptree, has := sg.cache.getPackageTree(r)
	if has {
		return ptree, nil
	}

	_, err = sg.require(ctx, sourceIsSetUp|sourceExistsLocally)
	if err != nil {
		return pkgtree.PackageTree{}, err
	}

	label := fmt.Sprintf("%s:%s", pr, sg.src.upstreamURL())
	err = sg.suprvsr.do(ctx, label, ctListPackages, func(ctx context.Context) error {
		ptree, err = sg.src.listPackages(ctx, pr, r)
		return err
	})

	// It's possible (in git) that we may have tried this against a version that
	// doesn't exist in the repository cache, even though we know it exists in
	// the upstream. If it looks like that might be the case, update the local
	// and retry.
	// TODO(sdboyer) It'd be better if we could check the error to see if this
	// actually was the cause of the problem.
	if err != nil && sg.srcState&sourceHasLatestLocally == 0 {
		// TODO(sdboyer) we should warn/log/something in adaptive recovery
		// situations like this
		_, err = sg.require(ctx, sourceHasLatestLocally)
		if err != nil {
			return pkgtree.PackageTree{}, err
		}

		err = sg.suprvsr.do(ctx, label, ctListPackages, func(ctx context.Context) error {
			ptree, err = sg.src.listPackages(ctx, pr, r)
			return err
		})
	}

	if err != nil {
		return pkgtree.PackageTree{}, err
	}

	sg.cache.setPackageTree(r, ptree)
	return ptree, nil
}

func (sg *sourceGateway) convertToRevision(ctx context.Context, v Version) (Revision, error) {
	// When looking up by Version, there are four states that may have
	// differing opinions about version->revision mappings:
	//
	//   1. The upstream source/repo (canonical)
	//   2. The local source/repo
	//   3. The local cache
	//   4. The input (params to this method)
	//
	// If the input differs from any of the above, it's likely because some lock
	// got written somewhere with a version/rev pair that has since changed or
	// been removed. But correct operation dictates that such a mis-mapping be
	// respected; if the mis-mapping is to be corrected, it has to be done
	// intentionally by the caller, not automatically here.
	r, has := sg.cache.toRevision(v)
	if has {
		return r, nil
	}

	if sg.srcState&sourceHasLatestVersionList != 0 {
		// We have the latest version list already and didn't get a match, so
		// this is definitely a failure case.
		return "", fmt.Errorf("version %q does not exist in source", v)
	}

	// The version list is out of date; it's possible this version might
	// show up after loading it.
	_, err := sg.require(ctx, sourceIsSetUp|sourceHasLatestVersionList)
	if err != nil {
		return "", err
	}

	r, has = sg.cache.toRevision(v)
	if !has {
		return "", fmt.Errorf("version %q does not exist in source", v)
	}

	return r, nil
}

func (sg *sourceGateway) listVersions(ctx context.Context) ([]PairedVersion, error) {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	// TODO(sdboyer) The problem here is that sourceExistsUpstream may not be
	// sufficient (e.g. bzr, hg), but we don't want to force local b/c git
	// doesn't need it
	_, err := sg.require(ctx, sourceIsSetUp|sourceExistsUpstream|sourceHasLatestVersionList)
	if err != nil {
		return nil, err
	}
	if pvs, ok := sg.cache.getAllVersions(); ok {
		return pvs, nil
	}
	return nil, nil
}

func (sg *sourceGateway) revisionPresentIn(ctx context.Context, r Revision) (bool, error) {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	_, err := sg.require(ctx, sourceIsSetUp|sourceExistsLocally)
	if err != nil {
		return false, err
	}

	if _, exists := sg.cache.getVersionsFor(r); exists {
		return true, nil
	}

	present, err := sg.src.revisionPresentIn(r)
	if err == nil && present {
		sg.cache.markRevisionExists(r)
	}
	return present, err
}

func (sg *sourceGateway) disambiguateRevision(ctx context.Context, r Revision) (Revision, error) {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	_, err := sg.require(ctx, sourceIsSetUp|sourceExistsLocally)
	if err != nil {
		return "", err
	}

	return sg.src.disambiguateRevision(ctx, r)
}

func (sg *sourceGateway) sourceURL(ctx context.Context) (string, error) {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	_, err := sg.require(ctx, sourceIsSetUp)
	if err != nil {
		return "", err
	}

	return sg.src.upstreamURL(), nil
}

// createSingleSourceCache creates a singleSourceCache instance for use by
// the encapsulated source.
func (sg *sourceGateway) createSingleSourceCache() singleSourceCache {
	// TODO(sdboyer) when persistent caching is ready, just drop in the creation
	// of a source-specific handle here
	return newMemoryCache()
}

func (sg *sourceGateway) require(ctx context.Context, wanted sourceState) (errState sourceState, err error) {
	todo := (^sg.srcState) & wanted
	var flag sourceState = 1

	for todo != 0 {
		if todo&flag != 0 {
			// Assign the currently visited bit to errState so that we can
			// return easily later.
			//
			// Also set up addlState so that individual ops can easily attach
			// more states that were incidentally satisfied by the op.
			errState = flag
			var addlState sourceState

			switch flag {
			case sourceIsSetUp:
				sg.src, addlState, err = sg.maybe.try(ctx, sg.cachedir, sg.cache, sg.suprvsr)
			case sourceExistsUpstream:
				err = sg.suprvsr.do(ctx, sg.src.sourceType(), ctSourcePing, func(ctx context.Context) error {
					if !sg.src.existsUpstream(ctx) {
						return fmt.Errorf("%s does not exist upstream", sg.src.upstreamURL())
					}
					return nil
				})
			case sourceExistsLocally:
				if !sg.src.existsLocally(ctx) {
					err = sg.suprvsr.do(ctx, sg.src.sourceType(), ctSourceInit, func(ctx context.Context) error {
						return sg.src.initLocal(ctx)
					})

					if err == nil {
						addlState |= sourceHasLatestLocally
					} else {
						err = errors.Wrapf(err, "%s does not exist in the local cache and fetching failed", sg.src.upstreamURL())
					}
				}
			case sourceHasLatestVersionList:
				var pvl []PairedVersion
				err = sg.suprvsr.do(ctx, sg.src.sourceType(), ctListVersions, func(ctx context.Context) error {
					pvl, err = sg.src.listVersions(ctx)
					return err
				})

				if err == nil {
					sg.cache.setVersionMap(pvl)
				}
			case sourceHasLatestLocally:
				err = sg.suprvsr.do(ctx, sg.src.sourceType(), ctSourceFetch, func(ctx context.Context) error {
					return sg.src.updateLocal(ctx)
				})
			}

			if err != nil {
				return
			}

			checked := flag | addlState
			sg.srcState |= checked
			todo &= ^checked
		}

		flag <<= 1
	}

	return 0, nil
}

// source is an abstraction around the different underlying types (git, bzr, hg,
// svn, maybe raw on-disk code, and maybe eventually a registry) that can
// provide versioned project source trees.
type source interface {
	existsLocally(context.Context) bool
	existsUpstream(context.Context) bool
	upstreamURL() string
	initLocal(context.Context) error
	updateLocal(context.Context) error
	listVersions(context.Context) ([]PairedVersion, error)
	getManifestAndLock(context.Context, ProjectRoot, Revision, ProjectAnalyzer) (Manifest, Lock, error)
	listPackages(context.Context, ProjectRoot, Revision) (pkgtree.PackageTree, error)
	revisionPresentIn(Revision) (bool, error)
	disambiguateRevision(context.Context, Revision) (Revision, error)
	exportRevisionTo(context.Context, Revision, string) error
	sourceType() string
}
