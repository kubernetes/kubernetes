// Copyright 2020 Google LLC All Rights Reserved.
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
	"github.com/google/go-containerregistry/pkg/name"
	"golang.org/x/sync/errgroup"
)

// MultiWrite writes the given Images or ImageIndexes to the given refs, as
// efficiently as possible, by deduping shared layer blobs while uploading them
// in parallel.
func MultiWrite(todo map[name.Reference]Taggable, options ...Option) (rerr error) {
	o, err := makeOptions(options...)
	if err != nil {
		return err
	}
	if o.progress != nil {
		defer func() { o.progress.Close(rerr) }()
	}
	p := newPusher(o)

	g, ctx := errgroup.WithContext(o.context)
	g.SetLimit(o.jobs)

	for ref, t := range todo {
		ref, t := ref, t
		g.Go(func() error {
			return p.Push(ctx, ref, t)
		})
	}

	return g.Wait()
}
