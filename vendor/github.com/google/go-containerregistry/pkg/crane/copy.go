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

package crane

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/google/go-containerregistry/pkg/logs"
	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/v1/remote"
	"github.com/google/go-containerregistry/pkg/v1/remote/transport"
	"golang.org/x/sync/errgroup"
)

// Copy copies a remote image or index from src to dst.
func Copy(src, dst string, opt ...Option) error {
	o := makeOptions(opt...)
	srcRef, err := name.ParseReference(src, o.Name...)
	if err != nil {
		return fmt.Errorf("parsing reference %q: %w", src, err)
	}

	dstRef, err := name.ParseReference(dst, o.Name...)
	if err != nil {
		return fmt.Errorf("parsing reference for %q: %w", dst, err)
	}

	puller, err := remote.NewPuller(o.Remote...)
	if err != nil {
		return err
	}

	if tag, ok := dstRef.(name.Tag); ok {
		if o.noclobber {
			logs.Progress.Printf("Checking existing tag %v", tag)
			head, err := puller.Head(o.ctx, tag)
			var terr *transport.Error
			if errors.As(err, &terr) {
				if terr.StatusCode != http.StatusNotFound && terr.StatusCode != http.StatusForbidden {
					return err
				}
			} else if err != nil {
				return err
			}

			if head != nil {
				return fmt.Errorf("refusing to clobber existing tag %s@%s", tag, head.Digest)
			}
		}
	}

	pusher, err := remote.NewPusher(o.Remote...)
	if err != nil {
		return err
	}

	logs.Progress.Printf("Copying from %v to %v", srcRef, dstRef)
	desc, err := puller.Get(o.ctx, srcRef)
	if err != nil {
		return fmt.Errorf("fetching %q: %w", src, err)
	}

	if o.Platform == nil {
		return pusher.Push(o.ctx, dstRef, desc)
	}

	// If platform is explicitly set, don't copy the whole index, just the appropriate image.
	img, err := desc.Image()
	if err != nil {
		return err
	}
	return pusher.Push(o.ctx, dstRef, img)
}

// CopyRepository copies every tag from src to dst.
func CopyRepository(src, dst string, opt ...Option) error {
	o := makeOptions(opt...)

	srcRepo, err := name.NewRepository(src, o.Name...)
	if err != nil {
		return err
	}

	dstRepo, err := name.NewRepository(dst, o.Name...)
	if err != nil {
		return fmt.Errorf("parsing reference for %q: %w", dst, err)
	}

	puller, err := remote.NewPuller(o.Remote...)
	if err != nil {
		return err
	}

	ignoredTags := map[string]struct{}{}
	if o.noclobber {
		// TODO: It would be good to propagate noclobber down into remote so we can use Etags.
		have, err := puller.List(o.ctx, dstRepo)
		if err != nil {
			var terr *transport.Error
			if errors.As(err, &terr) {
				// Some registries create repository on first push, so listing tags will fail.
				// If we see 404 or 403, assume we failed because the repository hasn't been created yet.
				if terr.StatusCode != http.StatusNotFound && terr.StatusCode != http.StatusForbidden {
					return err
				}
			} else {
				return err
			}
		}
		for _, tag := range have {
			ignoredTags[tag] = struct{}{}
		}
	}

	pusher, err := remote.NewPusher(o.Remote...)
	if err != nil {
		return err
	}

	lister, err := puller.Lister(o.ctx, srcRepo)
	if err != nil {
		return err
	}

	g, ctx := errgroup.WithContext(o.ctx)
	g.SetLimit(o.jobs)

	for lister.HasNext() {
		tags, err := lister.Next(ctx)
		if err != nil {
			return err
		}

		for _, tag := range tags.Tags {
			tag := tag

			if o.noclobber {
				if _, ok := ignoredTags[tag]; ok {
					logs.Progress.Printf("Skipping %s due to no-clobber", tag)
					continue
				}
			}

			g.Go(func() error {
				srcTag, err := name.ParseReference(src+":"+tag, o.Name...)
				if err != nil {
					return fmt.Errorf("failed to parse tag: %w", err)
				}
				dstTag, err := name.ParseReference(dst+":"+tag, o.Name...)
				if err != nil {
					return fmt.Errorf("failed to parse tag: %w", err)
				}

				logs.Progress.Printf("Fetching %s", srcTag)
				desc, err := puller.Get(ctx, srcTag)
				if err != nil {
					return err
				}

				logs.Progress.Printf("Pushing %s", dstTag)
				return pusher.Push(ctx, dstTag, desc)
			})
		}
	}

	return g.Wait()
}
