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

func TestSemaphore_AcquireRelease(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	sema, err := c.SemaphorePrefix("test/semaphore", 2)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Initial release should fail
	err = sema.Release()
	if err != ErrSemaphoreNotHeld {
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

	// Double lock should fail
	_, err = sema.Acquire(nil)
	if err != ErrSemaphoreHeld {
		t.Fatalf("err: %v", err)
	}

	// Should be held
	select {
	case <-lockCh:
		t.Fatalf("should be held")
	default:
	}

	// Initial release should work
	err = sema.Release()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Double unlock should fail
	err = sema.Release()
	if err != ErrSemaphoreNotHeld {
		t.Fatalf("err: %v", err)
	}

	// Should lose resource
	select {
	case <-lockCh:
	case <-time.After(time.Second):
		t.Fatalf("should not be held")
	}
}

func TestSemaphore_ForceInvalidate(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	sema, err := c.SemaphorePrefix("test/semaphore", 2)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should work
	lockCh, err := sema.Acquire(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if lockCh == nil {
		t.Fatalf("not acquired")
	}
	defer sema.Release()

	go func() {
		// Nuke the session, simulator an operator invalidation
		// or a health check failure
		session := c.Session()
		session.Destroy(sema.lockSession, nil)
	}()

	// Should loose slot
	select {
	case <-lockCh:
	case <-time.After(time.Second):
		t.Fatalf("should not be locked")
	}
}

func TestSemaphore_DeleteKey(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	sema, err := c.SemaphorePrefix("test/semaphore", 2)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should work
	lockCh, err := sema.Acquire(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if lockCh == nil {
		t.Fatalf("not locked")
	}
	defer sema.Release()

	go func() {
		// Nuke the key, simulate an operator intervention
		kv := c.KV()
		kv.DeleteTree("test/semaphore", nil)
	}()

	// Should loose leadership
	select {
	case <-lockCh:
	case <-time.After(time.Second):
		t.Fatalf("should not be locked")
	}
}

func TestSemaphore_Contend(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	wg := &sync.WaitGroup{}
	acquired := make([]bool, 4)
	for idx := range acquired {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			sema, err := c.SemaphorePrefix("test/semaphore", 2)
			if err != nil {
				t.Fatalf("err: %v", err)
			}

			// Should work eventually, will contend
			lockCh, err := sema.Acquire(nil)
			if err != nil {
				t.Fatalf("err: %v", err)
			}
			if lockCh == nil {
				t.Fatalf("not locked")
			}
			defer sema.Release()
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

func TestSemaphore_BadLimit(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	sema, err := c.SemaphorePrefix("test/semaphore", 0)
	if err == nil {
		t.Fatalf("should error")
	}

	sema, err = c.SemaphorePrefix("test/semaphore", 1)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	_, err = sema.Acquire(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	sema2, err := c.SemaphorePrefix("test/semaphore", 2)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	_, err = sema2.Acquire(nil)
	if err.Error() != "semaphore limit conflict (lock: 1, local: 2)" {
		t.Fatalf("err: %v", err)
	}
}

func TestSemaphore_Destroy(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	sema, err := c.SemaphorePrefix("test/semaphore", 2)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	sema2, err := c.SemaphorePrefix("test/semaphore", 2)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	_, err = sema.Acquire(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	_, err = sema2.Acquire(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Destroy should fail, still held
	if err := sema.Destroy(); err != ErrSemaphoreHeld {
		t.Fatalf("err: %v", err)
	}

	err = sema.Release()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Destroy should fail, still in use
	if err := sema.Destroy(); err != ErrSemaphoreInUse {
		t.Fatalf("err: %v", err)
	}

	err = sema2.Release()
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Destroy should work
	if err := sema.Destroy(); err != nil {
		t.Fatalf("err: %v", err)
	}

	// Destroy should work
	if err := sema2.Destroy(); err != nil {
		t.Fatalf("err: %v", err)
	}
}

func TestSemaphore_Conflict(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	lock, err := c.LockKey("test/sema/.lock")
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

	sema, err := c.SemaphorePrefix("test/sema/", 2)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Should conflict with lock
	_, err = sema.Acquire(nil)
	if err != ErrSemaphoreConflict {
		t.Fatalf("err: %v", err)
	}

	// Should conflict with lock
	err = sema.Destroy()
	if err != ErrSemaphoreConflict {
		t.Fatalf("err: %v", err)
	}
}

func TestSemaphore_MonitorRetry(t *testing.T) {
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
		if errors > 0 && req.Method == "GET" && strings.Contains(req.URL.Path, "/v1/kv/test/sema/.lock") {
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
	opts := &SemaphoreOptions{
		Prefix:         "test/sema/.lock",
		Limit:          2,
		SessionTTL:     "60s",
		MonitorRetries: 3,
	}
	sema, err := c.SemaphoreOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Make sure the default got set.
	if sema.opts.MonitorRetryTime != DefaultMonitorRetryTime {
		t.Fatalf("bad: %d", sema.opts.MonitorRetryTime)
	}

	// Now set a custom time for the test.
	opts.MonitorRetryTime = 250 * time.Millisecond
	sema, err = c.SemaphoreOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if sema.opts.MonitorRetryTime != 250*time.Millisecond {
		t.Fatalf("bad: %d", sema.opts.MonitorRetryTime)
	}

	// Should get the lock.
	ch, err := sema.Acquire(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if ch == nil {
		t.Fatalf("didn't acquire")
	}

	// Take the semaphore using the raw client to force the monitor to wake
	// up and check the lock again. This time we will return errors for some
	// of the responses.
	mutex.Lock()
	errors = 2
	mutex.Unlock()
	another, err := raw.SemaphoreOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if _, err := another.Acquire(nil); err != nil {
		t.Fatalf("err: %v", err)
	}
	time.Sleep(5 * opts.MonitorRetryTime)

	// Should still have the semaphore.
	select {
	case <-ch:
		t.Fatalf("lost the semaphore")
	default:
	}

	// Now return an overwhelming number of errors, using the raw client to
	// poke the key and get the monitor to run again.
	mutex.Lock()
	errors = 10
	mutex.Unlock()
	if err := another.Release(); err != nil {
		t.Fatalf("err: %v", err)
	}
	time.Sleep(5 * opts.MonitorRetryTime)

	// Should lose the semaphore.
	select {
	case <-ch:
	case <-time.After(time.Second):
		t.Fatalf("should not have the semaphore")
	}
}

func TestSemaphore_OneShot(t *testing.T) {
	t.Parallel()
	c, s := makeClient(t)
	defer s.Stop()

	// Set up a semaphore as a one-shot.
	opts := &SemaphoreOptions{
		Prefix:           "test/sema/.lock",
		Limit:            2,
		SemaphoreTryOnce: true,
	}
	sema, err := c.SemaphoreOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	// Make sure the default got set.
	if sema.opts.SemaphoreWaitTime != DefaultSemaphoreWaitTime {
		t.Fatalf("bad: %d", sema.opts.SemaphoreWaitTime)
	}

	// Now set a custom time for the test.
	opts.SemaphoreWaitTime = 250 * time.Millisecond
	sema, err = c.SemaphoreOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if sema.opts.SemaphoreWaitTime != 250*time.Millisecond {
		t.Fatalf("bad: %d", sema.opts.SemaphoreWaitTime)
	}

	// Should acquire the semaphore.
	ch, err := sema.Acquire(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if ch == nil {
		t.Fatalf("should have acquired the semaphore")
	}

	// Try with another session.
	another, err := c.SemaphoreOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	ch, err = another.Acquire(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if ch == nil {
		t.Fatalf("should have acquired the semaphore")
	}

	// Try with a third one that shouldn't get it.
	contender, err := c.SemaphoreOpts(opts)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	start := time.Now()
	ch, err = contender.Acquire(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if ch != nil {
		t.Fatalf("should not have acquired the semaphore")
	}
	diff := time.Now().Sub(start)
	if diff < contender.opts.SemaphoreWaitTime || diff > 2*contender.opts.SemaphoreWaitTime {
		t.Fatalf("time out of bounds: %9.6f", diff.Seconds())
	}

	// Give up a slot and make sure the third one can get it.
	if err := another.Release(); err != nil {
		t.Fatalf("err: %v", err)
	}
	ch, err = contender.Acquire(nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if ch == nil {
		t.Fatalf("should have acquired the semaphore")
	}
}
