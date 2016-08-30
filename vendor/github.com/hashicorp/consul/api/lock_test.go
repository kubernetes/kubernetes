package api

import (
	"log"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestLock_LockUnlock(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	lock, err := c.LockKey("test/lock")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Initial unlock should fail
	err = lock.Unlock()
	if err != ErrLockNotHeld {
		t.Fatalf("err: %v", err)
	}

	// Should work
	leaderCh, err := lock.Lock(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if leaderCh == nil {
		t.Fatalf("not leader")
	}

	// Double lock should fail
	_, err = lock.Lock(nil)
	if err != ErrLockHeld {
		t.Fatalf("err: %v", err)
	}

	// Should be leader
	select {
	case <-leaderCh:
		t.Fatalf("should be leader")
	default:
	}

	// Initial unlock should work
	err = lock.Unlock()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Double unlock should fail
	err = lock.Unlock()
	if err != ErrLockNotHeld {
		t.Fatalf("err: %v", err)
	}

	// Should lose leadership
	select {
	case <-leaderCh:
	case <-time.After(time.Second):
		t.Fatalf("should not be leader")
	}
}

func TestLock_ForceInvalidate(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	lock, err := c.LockKey("test/lock")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should work
	leaderCh, err := lock.Lock(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if leaderCh == nil {
		t.Fatalf("not leader")
	}
	defer lock.Unlock()

	go func() {
		// Nuke the session, simulator an operator invalidation
		// or a health check failure
		session := c.Session()
		session.Destroy(lock.lockSession, nil)
	}()

	// Should loose leadership
	select {
	case <-leaderCh:
	case <-time.After(time.Second):
		t.Fatalf("should not be leader")
	}
}

func TestLock_DeleteKey(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	// This uncovered some issues around special-case handling of low index
	// numbers where it would work with a low number but fail for higher
	// ones, so we loop this a bit to sweep the index up out of that
	// territory.
	for i := 0; i < 10; i++ {
		func() {
			lock, err := c.LockKey("test/lock")
			if err != nil {
				t.Fatalf("err: %v", err)
			}

			// Should work
			leaderCh, err := lock.Lock(nil)
			if err != nil {
				t.Fatalf("err: %v", err)
			}
			if leaderCh == nil {
				t.Fatalf("not leader")
			}
			defer lock.Unlock()

			go func() {
				// Nuke the key, simulate an operator intervention
				kv := c.KV()
				kv.Delete("test/lock", nil)
			}()

			// Should loose leadership
			select {
			case <-leaderCh:
			case <-time.After(time.Second):
				t.Fatalf("should not be leader")
			}
		}()
	}
}

func TestLock_Contend(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	wg := &sync.WaitGroup{}
	acquired := make([]bool, 3)
	for idx := range acquired {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			lock, err := c.LockKey("test/lock")
			if err != nil {
				t.Fatalf("err: %v", err)
			}

			// Should work eventually, will contend
			leaderCh, err := lock.Lock(nil)
			if err != nil {
				t.Fatalf("err: %v", err)
			}
			if leaderCh == nil {
				t.Fatalf("not leader")
			}
			defer lock.Unlock()
			log.Printf("Contender %d acquired", idx)

			// Set acquired and then leave
			acquired[idx] = true
		}(idx)
	}

	// Wait for termination
	doneCh := make(chan struct{})
	go func() {
		wg.Wait()
		close(doneCh)
	}()

	// Wait for everybody to get a turn
	select {
	case <-doneCh:
	case <-time.After(3 * DefaultLockRetryTime):
		t.Fatalf("timeout")
	}

	for idx, did := range acquired {
		if !did {
			t.Fatalf("contender %d never acquired", idx)
		}
	}
}

func TestLock_Destroy(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	lock, err := c.LockKey("test/lock")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should work
	leaderCh, err := lock.Lock(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if leaderCh == nil {
		t.Fatalf("not leader")
	}

	// Destroy should fail
	if err := lock.Destroy(); err != ErrLockHeld {
		t.Fatalf("err: %v", err)
	}

	// Should be able to release
	err = lock.Unlock()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Acquire with a different lock
	l2, err := c.LockKey("test/lock")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should work
	leaderCh, err = l2.Lock(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if leaderCh == nil {
		t.Fatalf("not leader")
	}

	// Destroy should still fail
	if err := lock.Destroy(); err != ErrLockInUse {
		t.Fatalf("err: %v", err)
	}

	// Should release
	err = l2.Unlock()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Destroy should work
	err = lock.Destroy()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Double destroy should work
	err = l2.Destroy()
	if err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestLock_Conflict(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	sema, err := c.SemaphorePrefix("test/lock/", 2)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should work
	lockCh, err := sema.Acquire(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if lockCh == nil {
		t.Fatalf("not hold")
	}
	defer sema.Release()

	lock, err := c.LockKey("test/lock/.lock")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should conflict with semaphore
	_, err = lock.Lock(nil)
	if err != ErrLockConflict {
		t.Fatalf("err: %v", err)
	}

	// Should conflict with semaphore
	err = lock.Destroy()
	if err != ErrLockConflict {
		t.Fatalf("err: %v", err)
	}
}

func TestLock_ReclaimLock(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	session, _, err := c.Session().Create(&SessionEntry{}, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	lock, err := c.LockOpts(&LockOptions{Key: "test/lock", Session: session})
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should work
	leaderCh, err := lock.Lock(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if leaderCh == nil {
		t.Fatalf("not leader")
	}
	defer lock.Unlock()

	l2, err := c.LockOpts(&LockOptions{Key: "test/lock", Session: session})
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	reclaimed := make(chan (<-chan struct{}), 1)
	go func() {
		l2Ch, err := l2.Lock(nil)
		if err != nil {
			t.Fatalf("not locked: %v", err)
		}
		reclaimed <- l2Ch
	}()

	// Should reclaim the lock
	var leader2Ch <-chan struct{}

	select {
	case leader2Ch = <-reclaimed:
	case <-time.After(time.Second):
		t.Fatalf("should have locked")
	}

	// unlock should work
	err = l2.Unlock()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	//Both locks should see the unlock
	select {
	case <-leader2Ch:
	case <-time.After(time.Second):
		t.Fatalf("should not be leader")
	}

	select {
	case <-leaderCh:
	case <-time.After(time.Second):
		t.Fatalf("should not be leader")
	}
}

func TestLock_MonitorRetry(t *testing.T) {
	t.Parallel()
	raw, s := makeClient(t)
	defer s.Stop()

	// Set up a server that always responds with 500 errors.
	failer := func(w http.ResponseWriter, req *http.Request) {
		w.WriteHeader(500)
	}
	outage := httptest.NewServer(http.HandlerFunc(failer))
	defer outage.Close()

	// Set up a reverse proxy that will send some requests to the
	// 500 server and pass everything else through to the real Consul
	// server.
	var mutex sync.Mutex
	errors := 0
	director := func(req *http.Request) {
		mutex.Lock()
		defer mutex.Unlock()

		req.URL.Scheme = "http"
		if errors > 0 && req.Method == "GET" && strings.Contains(req.URL.Path, "/v1/kv/test/lock") {
			req.URL.Host = outage.URL[7:] // Strip off "http://".
			errors--
		} else {
			req.URL.Host = raw.config.Address
		}
	}
	proxy := httptest.NewServer(&httputil.ReverseProxy{Director: director})
	defer proxy.Close()

	// Make another client that points at the proxy instead of the real
	// Consul server.
	config := raw.config
	config.Address = proxy.URL[7:] // Strip off "http://".
	c, err := NewClient(&config)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Set up a lock with retries enabled.
	opts := &LockOptions{
		Key:            "test/lock",
		SessionTTL:     "60s",
		MonitorRetries: 3,
	}
	lock, err := c.LockOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Make sure the default got set.
	if lock.opts.MonitorRetryTime != DefaultMonitorRetryTime {
		t.Fatalf("bad: %d", lock.opts.MonitorRetryTime)
	}

	// Now set a custom time for the test.
	opts.MonitorRetryTime = 250 * time.Millisecond
	lock, err = c.LockOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if lock.opts.MonitorRetryTime != 250*time.Millisecond {
		t.Fatalf("bad: %d", lock.opts.MonitorRetryTime)
	}

	// Should get the lock.
	leaderCh, err := lock.Lock(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if leaderCh == nil {
		t.Fatalf("not leader")
	}

	// Poke the key using the raw client to force the monitor to wake up
	// and check the lock again. This time we will return errors for some
	// of the responses.
	mutex.Lock()
	errors = 2
	mutex.Unlock()
	pair, _, err := raw.KV().Get("test/lock", &QueryOptions{})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if _, err := raw.KV().Put(pair, &WriteOptions{}); err != nil {
		t.Fatalf("err: %v", err)
	}
	time.Sleep(5 * opts.MonitorRetryTime)

	// Should still be the leader.
	select {
	case <-leaderCh:
		t.Fatalf("should be leader")
	default:
	}

	// Now return an overwhelming number of errors.
	mutex.Lock()
	errors = 10
	mutex.Unlock()
	if _, err := raw.KV().Put(pair, &WriteOptions{}); err != nil {
		t.Fatalf("err: %v", err)
	}
	time.Sleep(5 * opts.MonitorRetryTime)

	// Should lose leadership.
	select {
	case <-leaderCh:
	case <-time.After(time.Second):
		t.Fatalf("should not be leader")
	}
}

func TestLock_OneShot(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	// Set up a lock as a one-shot.
	opts := &LockOptions{
		Key:         "test/lock",
		LockTryOnce: true,
	}
	lock, err := c.LockOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Make sure the default got set.
	if lock.opts.LockWaitTime != DefaultLockWaitTime {
		t.Fatalf("bad: %d", lock.opts.LockWaitTime)
	}

	// Now set a custom time for the test.
	opts.LockWaitTime = 250 * time.Millisecond
	lock, err = c.LockOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if lock.opts.LockWaitTime != 250*time.Millisecond {
		t.Fatalf("bad: %d", lock.opts.LockWaitTime)
	}

	// Should get the lock.
	ch, err := lock.Lock(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if ch == nil {
		t.Fatalf("not leader")
	}

	// Now try with another session.
	contender, err := c.LockOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	start := time.Now()
	ch, err = contender.Lock(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if ch != nil {
		t.Fatalf("should not be leader")
	}
	diff := time.Now().Sub(start)
	if diff < contender.opts.LockWaitTime || diff > 2*contender.opts.LockWaitTime {
		t.Fatalf("time out of bounds: %9.6f", diff.Seconds())
	}

	// Unlock and then make sure the contender can get it.
	if err := lock.Unlock(); err != nil {
		t.Fatalf("err: %v", err)
	}
	ch, err = contender.Lock(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if ch == nil {
		t.Fatalf("should be leader")
	}
}
