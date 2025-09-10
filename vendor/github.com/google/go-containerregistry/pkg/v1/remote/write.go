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

package remote

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"

	"github.com/google/go-containerregistry/internal/redact"
	"github.com/google/go-containerregistry/internal/retry"
	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/logs"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/remote/transport"
	"github.com/google/go-containerregistry/pkg/v1/stream"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

// Taggable is an interface that enables a manifest PUT (e.g. for tagging).
type Taggable interface {
	RawManifest() ([]byte, error)
}

// Write pushes the provided img to the specified image reference.
func Write(ref name.Reference, img v1.Image, options ...Option) (rerr error) {
	return Push(ref, img, options...)
}

// writer writes the elements of an image to a remote image reference.
type writer struct {
	repo      name.Repository
	auth      authn.Authenticator
	transport http.RoundTripper

	client *http.Client

	progress  *progress
	backoff   Backoff
	predicate retry.Predicate

	scopeLock sync.Mutex
	// Keep track of scopes that we have already requested.
	scopeSet map[string]struct{}
	scopes   []string
}

func makeWriter(ctx context.Context, repo name.Repository, ls []v1.Layer, o *options) (*writer, error) {
	auth := o.auth
	if o.keychain != nil {
		kauth, err := authn.Resolve(ctx, o.keychain, repo)
		if err != nil {
			return nil, err
		}
		auth = kauth
	}
	scopes := scopesForUploadingImage(repo, ls)
	tr, err := transport.NewWithContext(ctx, repo.Registry, auth, o.transport, scopes)
	if err != nil {
		return nil, err
	}

	scopeSet := map[string]struct{}{}
	for _, scope := range scopes {
		scopeSet[scope] = struct{}{}
	}
	return &writer{
		repo:      repo,
		client:    &http.Client{Transport: tr},
		auth:      auth,
		transport: o.transport,
		progress:  o.progress,
		backoff:   o.retryBackoff,
		predicate: o.retryPredicate,
		scopes:    scopes,
		scopeSet:  scopeSet,
	}, nil
}

// url returns a url.Url for the specified path in the context of this remote image reference.
func (w *writer) url(path string) url.URL {
	return url.URL{
		Scheme: w.repo.Scheme(),
		Host:   w.repo.RegistryStr(),
		Path:   path,
	}
}

func (w *writer) maybeUpdateScopes(ctx context.Context, ml *MountableLayer) error {
	if ml.Reference.Context().String() == w.repo.String() {
		return nil
	}
	if ml.Reference.Context().Registry.String() != w.repo.Registry.String() {
		return nil
	}

	scope := ml.Reference.Scope(transport.PullScope)

	w.scopeLock.Lock()
	defer w.scopeLock.Unlock()

	if _, ok := w.scopeSet[scope]; !ok {
		w.scopeSet[scope] = struct{}{}
		w.scopes = append(w.scopes, scope)

		logs.Debug.Printf("Refreshing token to add scope %q", scope)
		wt, err := transport.NewWithContext(ctx, w.repo.Registry, w.auth, w.transport, w.scopes)
		if err != nil {
			return err
		}
		w.client = &http.Client{Transport: wt}
	}

	return nil
}

// nextLocation extracts the fully-qualified URL to which we should send the next request in an upload sequence.
func (w *writer) nextLocation(resp *http.Response) (string, error) {
	loc := resp.Header.Get("Location")
	if len(loc) == 0 {
		return "", errors.New("missing Location header")
	}
	u, err := url.Parse(loc)
	if err != nil {
		return "", err
	}

	// If the location header returned is just a url path, then fully qualify it.
	// We cannot simply call w.url, since there might be an embedded query string.
	return resp.Request.URL.ResolveReference(u).String(), nil
}

// checkExistingBlob checks if a blob exists already in the repository by making a
// HEAD request to the blob store API.  GCR performs an existence check on the
// initiation if "mount" is specified, even if no "from" sources are specified.
// However, this is not broadly applicable to all registries, e.g. ECR.
func (w *writer) checkExistingBlob(ctx context.Context, h v1.Hash) (bool, error) {
	u := w.url(fmt.Sprintf("/v2/%s/blobs/%s", w.repo.RepositoryStr(), h.String()))

	req, err := http.NewRequest(http.MethodHead, u.String(), nil)
	if err != nil {
		return false, err
	}

	resp, err := w.client.Do(req.WithContext(ctx))
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	if err := transport.CheckError(resp, http.StatusOK, http.StatusNotFound); err != nil {
		return false, err
	}

	return resp.StatusCode == http.StatusOK, nil
}

// initiateUpload initiates the blob upload, which starts with a POST that can
// optionally include the hash of the layer and a list of repositories from
// which that layer might be read. On failure, an error is returned.
// On success, the layer was either mounted (nothing more to do) or a blob
// upload was initiated and the body of that blob should be sent to the returned
// location.
func (w *writer) initiateUpload(ctx context.Context, from, mount, origin string) (location string, mounted bool, err error) {
	u := w.url(fmt.Sprintf("/v2/%s/blobs/uploads/", w.repo.RepositoryStr()))
	uv := url.Values{}
	if mount != "" && from != "" {
		// Quay will fail if we specify a "mount" without a "from".
		uv.Set("mount", mount)
		uv.Set("from", from)
		if origin != "" {
			uv.Set("origin", origin)
		}
	}
	u.RawQuery = uv.Encode()

	// Make the request to initiate the blob upload.
	req, err := http.NewRequest(http.MethodPost, u.String(), nil)
	if err != nil {
		return "", false, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := w.client.Do(req.WithContext(ctx))
	if err != nil {
		if from != "" {
			// https://github.com/google/go-containerregistry/issues/1679
			logs.Warn.Printf("retrying without mount: %v", err)
			return w.initiateUpload(ctx, "", "", "")
		}
		return "", false, err
	}
	defer resp.Body.Close()

	if err := transport.CheckError(resp, http.StatusCreated, http.StatusAccepted); err != nil {
		if from != "" {
			// https://github.com/google/go-containerregistry/issues/1404
			logs.Warn.Printf("retrying without mount: %v", err)
			return w.initiateUpload(ctx, "", "", "")
		}
		return "", false, err
	}

	// Check the response code to determine the result.
	switch resp.StatusCode {
	case http.StatusCreated:
		// We're done, we were able to fast-path.
		return "", true, nil
	case http.StatusAccepted:
		// Proceed to PATCH, upload has begun.
		loc, err := w.nextLocation(resp)
		return loc, false, err
	default:
		panic("Unreachable: initiateUpload")
	}
}

// streamBlob streams the contents of the blob to the specified location.
// On failure, this will return an error.  On success, this will return the location
// header indicating how to commit the streamed blob.
func (w *writer) streamBlob(ctx context.Context, layer v1.Layer, streamLocation string) (commitLocation string, rerr error) {
	reset := func() {}
	defer func() {
		if rerr != nil {
			reset()
		}
	}()
	blob, err := layer.Compressed()
	if err != nil {
		return "", err
	}

	getBody := layer.Compressed
	if w.progress != nil {
		var count int64
		blob = &progressReader{rc: blob, progress: w.progress, count: &count}
		getBody = func() (io.ReadCloser, error) {
			blob, err := layer.Compressed()
			if err != nil {
				return nil, err
			}
			return &progressReader{rc: blob, progress: w.progress, count: &count}, nil
		}
		reset = func() {
			w.progress.complete(-count)
		}
	}

	req, err := http.NewRequest(http.MethodPatch, streamLocation, blob)
	if err != nil {
		return "", err
	}
	if _, ok := layer.(*stream.Layer); !ok {
		// We can't retry streaming layers.
		req.GetBody = getBody

		// If we know the size, set it.
		if size, err := layer.Size(); err == nil {
			req.ContentLength = size
		}
	}
	req.Header.Set("Content-Type", "application/octet-stream")

	resp, err := w.client.Do(req.WithContext(ctx))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if err := transport.CheckError(resp, http.StatusNoContent, http.StatusAccepted, http.StatusCreated); err != nil {
		return "", err
	}

	// The blob has been uploaded, return the location header indicating
	// how to commit this layer.
	return w.nextLocation(resp)
}

// commitBlob commits this blob by sending a PUT to the location returned from
// streaming the blob.
func (w *writer) commitBlob(ctx context.Context, location, digest string) error {
	u, err := url.Parse(location)
	if err != nil {
		return err
	}
	v := u.Query()
	v.Set("digest", digest)
	u.RawQuery = v.Encode()

	req, err := http.NewRequest(http.MethodPut, u.String(), nil)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/octet-stream")

	resp, err := w.client.Do(req.WithContext(ctx))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return transport.CheckError(resp, http.StatusCreated)
}

// incrProgress increments and sends a progress update, if WithProgress is used.
func (w *writer) incrProgress(written int64) {
	if w.progress == nil {
		return
	}
	w.progress.complete(written)
}

// uploadOne performs a complete upload of a single layer.
func (w *writer) uploadOne(ctx context.Context, l v1.Layer) error {
	tryUpload := func() error {
		ctx := retry.Never(ctx)
		var from, mount, origin string
		if h, err := l.Digest(); err == nil {
			// If we know the digest, this isn't a streaming layer. Do an existence
			// check so we can skip uploading the layer if possible.
			existing, err := w.checkExistingBlob(ctx, h)
			if err != nil {
				return err
			}
			if existing {
				size, err := l.Size()
				if err != nil {
					return err
				}
				w.incrProgress(size)
				logs.Progress.Printf("existing blob: %v", h)
				return nil
			}

			mount = h.String()
		}
		if ml, ok := l.(*MountableLayer); ok {
			if err := w.maybeUpdateScopes(ctx, ml); err != nil {
				return err
			}

			from = ml.Reference.Context().RepositoryStr()
			origin = ml.Reference.Context().RegistryStr()

			// This keeps breaking with DockerHub.
			// https://github.com/google/go-containerregistry/issues/1741
			if w.repo.RegistryStr() == name.DefaultRegistry && origin != w.repo.RegistryStr() {
				from = ""
				origin = ""
			}
		}

		location, mounted, err := w.initiateUpload(ctx, from, mount, origin)
		if err != nil {
			return err
		} else if mounted {
			size, err := l.Size()
			if err != nil {
				return err
			}
			w.incrProgress(size)
			h, err := l.Digest()
			if err != nil {
				return err
			}
			logs.Progress.Printf("mounted blob: %s", h.String())
			return nil
		}

		// Only log layers with +json or +yaml. We can let through other stuff if it becomes popular.
		// TODO(opencontainers/image-spec#791): Would be great to have an actual parser.
		mt, err := l.MediaType()
		if err != nil {
			return err
		}
		smt := string(mt)
		if !strings.HasSuffix(smt, "+json") && !strings.HasSuffix(smt, "+yaml") {
			ctx = redact.NewContext(ctx, "omitting binary blobs from logs")
		}

		location, err = w.streamBlob(ctx, l, location)
		if err != nil {
			return err
		}

		h, err := l.Digest()
		if err != nil {
			return err
		}
		digest := h.String()

		if err := w.commitBlob(ctx, location, digest); err != nil {
			return err
		}
		logs.Progress.Printf("pushed blob: %s", digest)
		return nil
	}

	return retry.Retry(tryUpload, w.predicate, w.backoff)
}

type withMediaType interface {
	MediaType() (types.MediaType, error)
}

// This is really silly, but go interfaces don't let me satisfy remote.Taggable
// with remote.Descriptor because of name collisions between method names and
// struct fields.
//
// Use reflection to either pull the v1.Descriptor out of remote.Descriptor or
// create a descriptor based on the RawManifest and (optionally) MediaType.
func unpackTaggable(t Taggable) ([]byte, *v1.Descriptor, error) {
	if d, ok := t.(*Descriptor); ok {
		return d.Manifest, &d.Descriptor, nil
	}
	b, err := t.RawManifest()
	if err != nil {
		return nil, nil, err
	}

	// A reasonable default if Taggable doesn't implement MediaType.
	mt := types.DockerManifestSchema2

	if wmt, ok := t.(withMediaType); ok {
		m, err := wmt.MediaType()
		if err != nil {
			return nil, nil, err
		}
		mt = m
	}

	h, sz, err := v1.SHA256(bytes.NewReader(b))
	if err != nil {
		return nil, nil, err
	}

	return b, &v1.Descriptor{
		MediaType: mt,
		Size:      sz,
		Digest:    h,
	}, nil
}

// commitSubjectReferrers is responsible for updating the fallback tag manifest to track descriptors referring to a subject for registries that don't yet support the Referrers API.
// TODO: use conditional requests to avoid race conditions
func (w *writer) commitSubjectReferrers(ctx context.Context, sub name.Digest, add v1.Descriptor) error {
	// Check if the registry supports Referrers API.
	// TODO: This should be done once per registry, not once per subject.
	u := w.url(fmt.Sprintf("/v2/%s/referrers/%s", w.repo.RepositoryStr(), sub.DigestStr()))
	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", string(types.OCIImageIndex))
	resp, err := w.client.Do(req.WithContext(ctx))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if err := transport.CheckError(resp, http.StatusOK, http.StatusNotFound, http.StatusBadRequest); err != nil {
		return err
	}
	if resp.StatusCode == http.StatusOK {
		// The registry supports Referrers API. The registry is responsible for updating the referrers list.
		return nil
	}

	// The registry doesn't support Referrers API, we need to update the manifest tagged with the fallback tag.
	// Make the request to GET the current manifest.
	t := fallbackTag(sub)
	u = w.url(fmt.Sprintf("/v2/%s/manifests/%s", w.repo.RepositoryStr(), t.Identifier()))
	req, err = http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", string(types.OCIImageIndex))
	resp, err = w.client.Do(req.WithContext(ctx))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	var im v1.IndexManifest
	if err := transport.CheckError(resp, http.StatusOK, http.StatusNotFound); err != nil {
		return err
	} else if resp.StatusCode == http.StatusNotFound {
		// Not found just means there are no attachments. Start with an empty index.
		im = v1.IndexManifest{
			SchemaVersion: 2,
			MediaType:     types.OCIImageIndex,
			Manifests:     []v1.Descriptor{add},
		}
	} else {
		if err := json.NewDecoder(resp.Body).Decode(&im); err != nil {
			return err
		}
		if im.SchemaVersion != 2 {
			return fmt.Errorf("fallback tag manifest is not a schema version 2: %d", im.SchemaVersion)
		}
		if im.MediaType != types.OCIImageIndex {
			return fmt.Errorf("fallback tag manifest is not an OCI image index: %s", im.MediaType)
		}
		for _, desc := range im.Manifests {
			if desc.Digest == add.Digest {
				// The digest is already attached, nothing to do.
				logs.Progress.Printf("fallback tag %s already had referrer", t.Identifier())
				return nil
			}
		}
		// Append the new descriptor to the index.
		im.Manifests = append(im.Manifests, add)
	}

	// Sort the manifests for reproducibility.
	sort.Slice(im.Manifests, func(i, j int) bool {
		return im.Manifests[i].Digest.String() < im.Manifests[j].Digest.String()
	})
	logs.Progress.Printf("updating fallback tag %s with new referrer", t.Identifier())
	return w.commitManifest(ctx, fallbackTaggable{im}, t)
}

type fallbackTaggable struct {
	im v1.IndexManifest
}

func (f fallbackTaggable) RawManifest() ([]byte, error)        { return json.Marshal(f.im) }
func (f fallbackTaggable) MediaType() (types.MediaType, error) { return types.OCIImageIndex, nil }

// commitManifest does a PUT of the image's manifest.
func (w *writer) commitManifest(ctx context.Context, t Taggable, ref name.Reference) error {
	// If the manifest refers to a subject, we need to check whether we need to update the fallback tag manifest.
	raw, err := t.RawManifest()
	if err != nil {
		return err
	}
	var mf struct {
		MediaType types.MediaType `json:"mediaType"`
		Subject   *v1.Descriptor  `json:"subject,omitempty"`
		Config    struct {
			MediaType types.MediaType `json:"mediaType"`
		} `json:"config"`
	}
	if err := json.Unmarshal(raw, &mf); err != nil {
		return err
	}

	tryUpload := func() error {
		ctx := retry.Never(ctx)
		raw, desc, err := unpackTaggable(t)
		if err != nil {
			return err
		}

		u := w.url(fmt.Sprintf("/v2/%s/manifests/%s", w.repo.RepositoryStr(), ref.Identifier()))

		// Make the request to PUT the serialized manifest
		req, err := http.NewRequest(http.MethodPut, u.String(), bytes.NewBuffer(raw))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", string(desc.MediaType))

		resp, err := w.client.Do(req.WithContext(ctx))
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		if err := transport.CheckError(resp, http.StatusOK, http.StatusCreated, http.StatusAccepted); err != nil {
			return err
		}

		// If the manifest referred to a subject, we may need to update the fallback tag manifest.
		// TODO: If this fails, we'll retry the whole upload. We should retry just this part.
		if mf.Subject != nil {
			h, size, err := v1.SHA256(bytes.NewReader(raw))
			if err != nil {
				return err
			}
			desc := v1.Descriptor{
				ArtifactType: string(mf.Config.MediaType),
				MediaType:    mf.MediaType,
				Digest:       h,
				Size:         size,
			}
			if err := w.commitSubjectReferrers(ctx,
				ref.Context().Digest(mf.Subject.Digest.String()),
				desc); err != nil {
				return err
			}
		}

		// The image was successfully pushed!
		logs.Progress.Printf("%v: digest: %v size: %d", ref, desc.Digest, desc.Size)
		w.incrProgress(int64(len(raw)))
		return nil
	}

	return retry.Retry(tryUpload, w.predicate, w.backoff)
}

func scopesForUploadingImage(repo name.Repository, layers []v1.Layer) []string {
	// use a map as set to remove duplicates scope strings
	scopeSet := map[string]struct{}{}

	for _, l := range layers {
		if ml, ok := l.(*MountableLayer); ok {
			// we will add push scope for ref.Context() after the loop.
			// for now we ask pull scope for references of the same registry
			if ml.Reference.Context().String() != repo.String() && ml.Reference.Context().Registry.String() == repo.Registry.String() {
				scopeSet[ml.Reference.Scope(transport.PullScope)] = struct{}{}
			}
		}
	}

	scopes := make([]string, 0)
	// Push scope should be the first element because a few registries just look at the first scope to determine access.
	scopes = append(scopes, repo.Scope(transport.PushScope))

	for scope := range scopeSet {
		scopes = append(scopes, scope)
	}

	return scopes
}

// WriteIndex pushes the provided ImageIndex to the specified image reference.
// WriteIndex will attempt to push all of the referenced manifests before
// attempting to push the ImageIndex, to retain referential integrity.
func WriteIndex(ref name.Reference, ii v1.ImageIndex, options ...Option) (rerr error) {
	return Push(ref, ii, options...)
}

// WriteLayer uploads the provided Layer to the specified repo.
func WriteLayer(repo name.Repository, layer v1.Layer, options ...Option) (rerr error) {
	o, err := makeOptions(options...)
	if err != nil {
		return err
	}
	if o.progress != nil {
		defer func() { o.progress.Close(rerr) }()
	}
	return newPusher(o).Upload(o.context, repo, layer)
}

// Tag adds a tag to the given Taggable via PUT /v2/.../manifests/<tag>
//
// Notable implementations of Taggable are v1.Image, v1.ImageIndex, and
// remote.Descriptor.
//
// If t implements MediaType, we will use that for the Content-Type, otherwise
// we will default to types.DockerManifestSchema2.
//
// Tag does not attempt to write anything other than the manifest, so callers
// should ensure that all blobs or manifests that are referenced by t exist
// in the target registry.
func Tag(tag name.Tag, t Taggable, options ...Option) error {
	return Put(tag, t, options...)
}

// Put adds a manifest from the given Taggable via PUT /v1/.../manifest/<ref>
//
// Notable implementations of Taggable are v1.Image, v1.ImageIndex, and
// remote.Descriptor.
//
// If t implements MediaType, we will use that for the Content-Type, otherwise
// we will default to types.DockerManifestSchema2.
//
// Put does not attempt to write anything other than the manifest, so callers
// should ensure that all blobs or manifests that are referenced by t exist
// in the target registry.
func Put(ref name.Reference, t Taggable, options ...Option) error {
	o, err := makeOptions(options...)
	if err != nil {
		return err
	}
	return newPusher(o).Put(o.context, ref, t)
}

// Push uploads the given Taggable to the specified reference.
func Push(ref name.Reference, t Taggable, options ...Option) (rerr error) {
	o, err := makeOptions(options...)
	if err != nil {
		return err
	}
	if o.progress != nil {
		defer func() { o.progress.Close(rerr) }()
	}
	return newPusher(o).Push(o.context, ref, t)
}
