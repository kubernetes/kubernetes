// Copyright 2015 The etcd Authors
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
	"reflect"
	"testing"
	"time"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/pkg/testutil"
	"github.com/jonboulle/clockwork"
	"golang.org/x/net/context"
)

func TestPeriodic(t *testing.T) {
	fc := clockwork.NewFakeClock()
	rg := &fakeRevGetter{testutil.NewRecorderStream(), 0}
	compactable := &fakeCompactable{testutil.NewRecorderStream()}
	tb := &Periodic{
		clock:        fc,
		periodInHour: 1,
		rg:           rg,
		c:            compactable,
	}

	tb.Run()
	defer tb.Stop()

	n := int(time.Hour / checkCompactionInterval)
	// collect 3 hours of revisions
	for i := 0; i < 3; i++ {
		// advance one hour, one revision for each interval
		for j := 0; j < n; j++ {
			fc.Advance(checkCompactionInterval)
			rg.Wait(1)
		}
		// ready to acknowledge hour "i"
		// block until compactor calls clock.After()
		fc.BlockUntil(1)
		// unblock the After()
		fc.Advance(checkCompactionInterval)
		a, err := compactable.Wait(1)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(a[0].Params[0], &pb.CompactionRequest{Revision: int64(i*n) + 1}) {
			t.Errorf("compact request = %v, want %v", a[0].Params[0], &pb.CompactionRequest{Revision: int64(i*n) + 1})
		}
	}
}

func TestPeriodicPause(t *testing.T) {
	fc := clockwork.NewFakeClock()
	compactable := &fakeCompactable{testutil.NewRecorderStream()}
	rg := &fakeRevGetter{testutil.NewRecorderStream(), 0}
	tb := &Periodic{
		clock:        fc,
		periodInHour: 1,
		rg:           rg,
		c:            compactable,
	}

	tb.Run()
	tb.Pause()

	// tb will collect 3 hours of revisions but not compact since paused
	n := int(time.Hour / checkCompactionInterval)
	for i := 0; i < 3*n; i++ {
		fc.Advance(checkCompactionInterval)
		rg.Wait(1)
	}
	// tb ends up waiting for the clock

	select {
	case a := <-compactable.Chan():
		t.Fatalf("unexpected action %v", a)
	case <-time.After(10 * time.Millisecond):
	}

	// tb resumes to being blocked on the clock
	tb.Resume()

	// unblock clock, will kick off a compaction at hour 3
	fc.Advance(checkCompactionInterval)
	a, err := compactable.Wait(1)
	if err != nil {
		t.Fatal(err)
	}
	// compact the revision from hour 2
	wreq := &pb.CompactionRequest{Revision: int64(2*n + 1)}
	if !reflect.DeepEqual(a[0].Params[0], wreq) {
		t.Errorf("compact request = %v, want %v", a[0].Params[0], wreq.Revision)
	}
}

type fakeCompactable struct {
	testutil.Recorder
}

func (fc *fakeCompactable) Compact(ctx context.Context, r *pb.CompactionRequest) (*pb.CompactionResponse, error) {
	fc.Record(testutil.Action{Name: "c", Params: []interface{}{r}})
	return &pb.CompactionResponse{}, nil
}

type fakeRevGetter struct {
	testutil.Recorder
	rev int64
}

func (fr *fakeRevGetter) Rev() int64 {
	fr.Record(testutil.Action{Name: "g"})
	fr.rev++
	return fr.rev
}
