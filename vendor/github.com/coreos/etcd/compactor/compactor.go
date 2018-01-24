// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package compactor

import (
	"context"
	"fmt"
	"time"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"

	"github.com/coreos/pkg/capnslog"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "compactor")
)

const (
	checkCompactionInterval = 5 * time.Minute

	ModePeriodic = "periodic"
	ModeRevision = "revision"
)

// Compactor purges old log from the storage periodically.
type Compactor interface {
	// Run starts the main loop of the compactor in background.
	// Use Stop() to halt the loop and release the resource.
	Run()
	// Stop halts the main loop of the compactor.
	Stop()
	// Pause temporally suspend the compactor not to run compaction. Resume() to unpose.
	Pause()
	// Resume restarts the compactor suspended by Pause().
	Resume()
}

type Compactable interface {
	Compact(ctx context.Context, r *pb.CompactionRequest) (*pb.CompactionResponse, error)
}

type RevGetter interface {
	Rev() int64
}

func New(mode string, retention time.Duration, rg RevGetter, c Compactable) (Compactor, error) {
	switch mode {
	case ModePeriodic:
		return NewPeriodic(retention, rg, c), nil
	case ModeRevision:
		return NewRevision(int64(retention), rg, c), nil
	default:
		return nil, fmt.Errorf("unsupported compaction mode %s", mode)
	}
}
