// Copyright 2022 Google LLC All Rights Reserved.
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
	"io"
	"sync"
	"sync/atomic"

	v1 "github.com/google/go-containerregistry/pkg/v1"
)

type progress struct {
	sync.Mutex
	updates    chan<- v1.Update
	lastUpdate *v1.Update
}

func (p *progress) total(delta int64) {
	p.Lock()
	defer p.Unlock()
	atomic.AddInt64(&p.lastUpdate.Total, delta)
}

func (p *progress) complete(delta int64) {
	p.Lock()
	defer p.Unlock()
	p.updates <- v1.Update{
		Total:    p.lastUpdate.Total,
		Complete: atomic.AddInt64(&p.lastUpdate.Complete, delta),
	}
}

func (p *progress) err(err error) error {
	if err != nil && p.updates != nil {
		p.updates <- v1.Update{Error: err}
	}
	return err
}

func (p *progress) Close(err error) {
	_ = p.err(err)
	close(p.updates)
}

type progressReader struct {
	rc io.ReadCloser

	count    *int64 // number of bytes this reader has read, to support resetting on retry.
	progress *progress
}

func (r *progressReader) Read(b []byte) (int, error) {
	n, err := r.rc.Read(b)
	if err != nil {
		return n, err
	}
	atomic.AddInt64(r.count, int64(n))
	// TODO: warn/debug log if sending takes too long, or if sending is blocked while context is canceled.
	r.progress.complete(int64(n))
	return n, nil
}

func (r *progressReader) Close() error { return r.rc.Close() }
