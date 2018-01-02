package scheduler

import (
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/storage/driver/inmemory"
)

func testRefs(t *testing.T) (reference.Reference, reference.Reference, reference.Reference) {
	ref1, err := reference.Parse("testrepo@sha256:aaaaeaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}

	ref2, err := reference.Parse("testrepo@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}

	ref3, err := reference.Parse("testrepo@sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}

	return ref1, ref2, ref3
}

func TestSchedule(t *testing.T) {
	ref1, ref2, ref3 := testRefs(t)
	timeUnit := time.Millisecond
	remainingRepos := map[string]bool{
		ref1.String(): true,
		ref2.String(): true,
		ref3.String(): true,
	}

	var mu sync.Mutex
	s := New(context.Background(), inmemory.New(), "/ttl")
	deleteFunc := func(repoName reference.Reference) error {
		if len(remainingRepos) == 0 {
			t.Fatalf("Incorrect expiry count")
		}
		_, ok := remainingRepos[repoName.String()]
		if !ok {
			t.Fatalf("Trying to remove nonexistent repo: %s", repoName)
		}
		t.Log("removing", repoName)
		mu.Lock()
		delete(remainingRepos, repoName.String())
		mu.Unlock()

		return nil
	}
	s.onBlobExpire = deleteFunc
	err := s.Start()
	if err != nil {
		t.Fatalf("Error starting ttlExpirationScheduler: %s", err)
	}

	s.add(ref1, 3*timeUnit, entryTypeBlob)
	s.add(ref2, 1*timeUnit, entryTypeBlob)

	func() {
		s.Lock()
		s.add(ref3, 1*timeUnit, entryTypeBlob)
		s.Unlock()

	}()

	// Ensure all repos are deleted
	<-time.After(50 * timeUnit)

	mu.Lock()
	defer mu.Unlock()
	if len(remainingRepos) != 0 {
		t.Fatalf("Repositories remaining: %#v", remainingRepos)
	}
}

func TestRestoreOld(t *testing.T) {
	ref1, ref2, _ := testRefs(t)
	remainingRepos := map[string]bool{
		ref1.String(): true,
		ref2.String(): true,
	}

	var wg sync.WaitGroup
	wg.Add(len(remainingRepos))
	var mu sync.Mutex
	deleteFunc := func(r reference.Reference) error {
		mu.Lock()
		defer mu.Unlock()
		if r.String() == ref1.String() && len(remainingRepos) == 2 {
			t.Errorf("ref1 should not be removed first")
		}
		_, ok := remainingRepos[r.String()]
		if !ok {
			t.Fatalf("Trying to remove nonexistent repo: %s", r)
		}
		delete(remainingRepos, r.String())
		wg.Done()
		return nil
	}

	timeUnit := time.Millisecond
	serialized, err := json.Marshal(&map[string]schedulerEntry{
		ref1.String(): {
			Expiry:    time.Now().Add(10 * timeUnit),
			Key:       ref1.String(),
			EntryType: 0,
		},
		ref2.String(): {
			Expiry:    time.Now().Add(-3 * timeUnit), // TTL passed, should be removed first
			Key:       ref2.String(),
			EntryType: 0,
		},
	})
	if err != nil {
		t.Fatalf("Error serializing test data: %s", err.Error())
	}

	ctx := context.Background()
	pathToStatFile := "/ttl"
	fs := inmemory.New()
	err = fs.PutContent(ctx, pathToStatFile, serialized)
	if err != nil {
		t.Fatal("Unable to write serialized data to fs")
	}
	s := New(context.Background(), fs, "/ttl")
	s.OnBlobExpire(deleteFunc)
	err = s.Start()
	if err != nil {
		t.Fatalf("Error starting ttlExpirationScheduler: %s", err)
	}
	defer s.Stop()

	wg.Wait()
	mu.Lock()
	defer mu.Unlock()
	if len(remainingRepos) != 0 {
		t.Fatalf("Repositories remaining: %#v", remainingRepos)
	}
}

func TestStopRestore(t *testing.T) {
	ref1, ref2, _ := testRefs(t)

	timeUnit := time.Millisecond
	remainingRepos := map[string]bool{
		ref1.String(): true,
		ref2.String(): true,
	}

	var mu sync.Mutex
	deleteFunc := func(r reference.Reference) error {
		mu.Lock()
		delete(remainingRepos, r.String())
		mu.Unlock()
		return nil
	}

	fs := inmemory.New()
	pathToStateFile := "/ttl"
	s := New(context.Background(), fs, pathToStateFile)
	s.onBlobExpire = deleteFunc

	err := s.Start()
	if err != nil {
		t.Fatalf(err.Error())
	}
	s.add(ref1, 300*timeUnit, entryTypeBlob)
	s.add(ref2, 100*timeUnit, entryTypeBlob)

	// Start and stop before all operations complete
	// state will be written to fs
	s.Stop()
	time.Sleep(10 * time.Millisecond)

	// v2 will restore state from fs
	s2 := New(context.Background(), fs, pathToStateFile)
	s2.onBlobExpire = deleteFunc
	err = s2.Start()
	if err != nil {
		t.Fatalf("Error starting v2: %s", err.Error())
	}

	<-time.After(500 * timeUnit)
	mu.Lock()
	defer mu.Unlock()
	if len(remainingRepos) != 0 {
		t.Fatalf("Repositories remaining: %#v", remainingRepos)
	}

}

func TestDoubleStart(t *testing.T) {
	s := New(context.Background(), inmemory.New(), "/ttl")
	err := s.Start()
	if err != nil {
		t.Fatalf("Unable to start scheduler")
	}
	err = s.Start()
	if err == nil {
		t.Fatalf("Scheduler started twice without error")
	}
}
