// Copyright 2011 Gary Burd
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package redis_test

import (
	"errors"
	"io"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/garyburd/redigo/internal/redistest"
	"github.com/garyburd/redigo/redis"
)

type poolTestConn struct {
	d   *poolDialer
	err error
	redis.Conn
}

func (c *poolTestConn) Close() error { c.d.open -= 1; return nil }
func (c *poolTestConn) Err() error   { return c.err }

func (c *poolTestConn) Do(commandName string, args ...interface{}) (reply interface{}, err error) {
	if commandName == "ERR" {
		c.err = args[0].(error)
		commandName = "PING"
	}
	if commandName != "" {
		c.d.commands = append(c.d.commands, commandName)
	}
	return c.Conn.Do(commandName, args...)
}

func (c *poolTestConn) Send(commandName string, args ...interface{}) error {
	c.d.commands = append(c.d.commands, commandName)
	return c.Conn.Send(commandName, args...)
}

type poolDialer struct {
	t        *testing.T
	dialed   int
	open     int
	commands []string
	dialErr  error
}

func (d *poolDialer) dial() (redis.Conn, error) {
	d.dialed += 1
	if d.dialErr != nil {
		return nil, d.dialErr
	}
	c, err := redistest.Dial()
	if err != nil {
		return nil, err
	}
	d.open += 1
	return &poolTestConn{d: d, Conn: c}, nil
}

func (d *poolDialer) check(message string, p *redis.Pool, dialed, open int) {
	if d.dialed != dialed {
		d.t.Errorf("%s: dialed=%d, want %d", message, d.dialed, dialed)
	}
	if d.open != open {
		d.t.Errorf("%s: open=%d, want %d", message, d.open, open)
	}
	if active := p.ActiveCount(); active != open {
		d.t.Errorf("%s: active=%d, want %d", message, active, open)
	}
}

func TestPoolReuse(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle: 2,
		Dial:    d.dial,
	}

	for i := 0; i < 10; i++ {
		c1 := p.Get()
		c1.Do("PING")
		c2 := p.Get()
		c2.Do("PING")
		c1.Close()
		c2.Close()
	}

	d.check("before close", p, 2, 2)
	p.Close()
	d.check("after close", p, 2, 0)
}

func TestPoolMaxIdle(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle: 2,
		Dial:    d.dial,
	}
	for i := 0; i < 10; i++ {
		c1 := p.Get()
		c1.Do("PING")
		c2 := p.Get()
		c2.Do("PING")
		c3 := p.Get()
		c3.Do("PING")
		c1.Close()
		c2.Close()
		c3.Close()
	}
	d.check("before close", p, 12, 2)
	p.Close()
	d.check("after close", p, 12, 0)
}

func TestPoolError(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle: 2,
		Dial:    d.dial,
	}

	c := p.Get()
	c.Do("ERR", io.EOF)
	if c.Err() == nil {
		t.Errorf("expected c.Err() != nil")
	}
	c.Close()

	c = p.Get()
	c.Do("ERR", io.EOF)
	c.Close()

	d.check(".", p, 2, 0)
}

func TestPoolClose(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle: 2,
		Dial:    d.dial,
	}

	c1 := p.Get()
	c1.Do("PING")
	c2 := p.Get()
	c2.Do("PING")
	c3 := p.Get()
	c3.Do("PING")

	c1.Close()
	if _, err := c1.Do("PING"); err == nil {
		t.Errorf("expected error after connection closed")
	}

	c2.Close()
	c2.Close()

	p.Close()

	d.check("after pool close", p, 3, 1)

	if _, err := c1.Do("PING"); err == nil {
		t.Errorf("expected error after connection and pool closed")
	}

	c3.Close()

	d.check("after conn close", p, 3, 0)

	c1 = p.Get()
	if _, err := c1.Do("PING"); err == nil {
		t.Errorf("expected error after pool closed")
	}
}

func TestPoolTimeout(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle:     2,
		IdleTimeout: 300 * time.Second,
		Dial:        d.dial,
	}

	now := time.Now()
	redis.SetNowFunc(func() time.Time { return now })
	defer redis.SetNowFunc(time.Now)

	c := p.Get()
	c.Do("PING")
	c.Close()

	d.check("1", p, 1, 1)

	now = now.Add(p.IdleTimeout)

	c = p.Get()
	c.Do("PING")
	c.Close()

	d.check("2", p, 2, 1)

	p.Close()
}

func TestPoolConcurrenSendReceive(t *testing.T) {
	p := &redis.Pool{
		Dial: redistest.Dial,
	}
	c := p.Get()
	done := make(chan error, 1)
	go func() {
		_, err := c.Receive()
		done <- err
	}()
	c.Send("PING")
	c.Flush()
	err := <-done
	if err != nil {
		t.Fatalf("Receive() returned error %v", err)
	}
	_, err = c.Do("")
	if err != nil {
		t.Fatalf("Do() returned error %v", err)
	}
	c.Close()
	p.Close()
}

func TestPoolBorrowCheck(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle:      2,
		Dial:         d.dial,
		TestOnBorrow: func(redis.Conn, time.Time) error { return redis.Error("BLAH") },
	}

	for i := 0; i < 10; i++ {
		c := p.Get()
		c.Do("PING")
		c.Close()
	}
	d.check("1", p, 10, 1)
	p.Close()
}

func TestPoolMaxActive(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle:   2,
		MaxActive: 2,
		Dial:      d.dial,
	}
	c1 := p.Get()
	c1.Do("PING")
	c2 := p.Get()
	c2.Do("PING")

	d.check("1", p, 2, 2)

	c3 := p.Get()
	if _, err := c3.Do("PING"); err != redis.ErrPoolExhausted {
		t.Errorf("expected pool exhausted")
	}

	c3.Close()
	d.check("2", p, 2, 2)
	c2.Close()
	d.check("3", p, 2, 2)

	c3 = p.Get()
	if _, err := c3.Do("PING"); err != nil {
		t.Errorf("expected good channel, err=%v", err)
	}
	c3.Close()

	d.check("4", p, 2, 2)
	p.Close()
}

func TestPoolMonitorCleanup(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle:   2,
		MaxActive: 2,
		Dial:      d.dial,
	}
	c := p.Get()
	c.Send("MONITOR")
	c.Close()

	d.check("", p, 1, 0)
	p.Close()
}

func TestPoolPubSubCleanup(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle:   2,
		MaxActive: 2,
		Dial:      d.dial,
	}

	c := p.Get()
	c.Send("SUBSCRIBE", "x")
	c.Close()

	want := []string{"SUBSCRIBE", "UNSUBSCRIBE", "PUNSUBSCRIBE", "ECHO"}
	if !reflect.DeepEqual(d.commands, want) {
		t.Errorf("got commands %v, want %v", d.commands, want)
	}
	d.commands = nil

	c = p.Get()
	c.Send("PSUBSCRIBE", "x*")
	c.Close()

	want = []string{"PSUBSCRIBE", "UNSUBSCRIBE", "PUNSUBSCRIBE", "ECHO"}
	if !reflect.DeepEqual(d.commands, want) {
		t.Errorf("got commands %v, want %v", d.commands, want)
	}
	d.commands = nil

	p.Close()
}

func TestPoolTransactionCleanup(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle:   2,
		MaxActive: 2,
		Dial:      d.dial,
	}

	c := p.Get()
	c.Do("WATCH", "key")
	c.Do("PING")
	c.Close()

	want := []string{"WATCH", "PING", "UNWATCH"}
	if !reflect.DeepEqual(d.commands, want) {
		t.Errorf("got commands %v, want %v", d.commands, want)
	}
	d.commands = nil

	c = p.Get()
	c.Do("WATCH", "key")
	c.Do("UNWATCH")
	c.Do("PING")
	c.Close()

	want = []string{"WATCH", "UNWATCH", "PING"}
	if !reflect.DeepEqual(d.commands, want) {
		t.Errorf("got commands %v, want %v", d.commands, want)
	}
	d.commands = nil

	c = p.Get()
	c.Do("WATCH", "key")
	c.Do("MULTI")
	c.Do("PING")
	c.Close()

	want = []string{"WATCH", "MULTI", "PING", "DISCARD"}
	if !reflect.DeepEqual(d.commands, want) {
		t.Errorf("got commands %v, want %v", d.commands, want)
	}
	d.commands = nil

	c = p.Get()
	c.Do("WATCH", "key")
	c.Do("MULTI")
	c.Do("DISCARD")
	c.Do("PING")
	c.Close()

	want = []string{"WATCH", "MULTI", "DISCARD", "PING"}
	if !reflect.DeepEqual(d.commands, want) {
		t.Errorf("got commands %v, want %v", d.commands, want)
	}
	d.commands = nil

	c = p.Get()
	c.Do("WATCH", "key")
	c.Do("MULTI")
	c.Do("EXEC")
	c.Do("PING")
	c.Close()

	want = []string{"WATCH", "MULTI", "EXEC", "PING"}
	if !reflect.DeepEqual(d.commands, want) {
		t.Errorf("got commands %v, want %v", d.commands, want)
	}
	d.commands = nil

	p.Close()
}

func startGoroutines(p *redis.Pool, cmd string, args ...interface{}) chan error {
	errs := make(chan error, 10)
	for i := 0; i < cap(errs); i++ {
		go func() {
			c := p.Get()
			_, err := c.Do(cmd, args...)
			errs <- err
			c.Close()
		}()
	}

	// Wait for goroutines to block.
	time.Sleep(time.Second / 4)

	return errs
}

func TestWaitPool(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle:   1,
		MaxActive: 1,
		Dial:      d.dial,
		Wait:      true,
	}
	defer p.Close()
	c := p.Get()
	errs := startGoroutines(p, "PING")
	d.check("before close", p, 1, 1)
	c.Close()
	timeout := time.After(2 * time.Second)
	for i := 0; i < cap(errs); i++ {
		select {
		case err := <-errs:
			if err != nil {
				t.Fatal(err)
			}
		case <-timeout:
			t.Fatalf("timeout waiting for blocked goroutine %d", i)
		}
	}
	d.check("done", p, 1, 1)
}

func TestWaitPoolClose(t *testing.T) {
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle:   1,
		MaxActive: 1,
		Dial:      d.dial,
		Wait:      true,
	}
	c := p.Get()
	if _, err := c.Do("PING"); err != nil {
		t.Fatal(err)
	}
	errs := startGoroutines(p, "PING")
	d.check("before close", p, 1, 1)
	p.Close()
	timeout := time.After(2 * time.Second)
	for i := 0; i < cap(errs); i++ {
		select {
		case err := <-errs:
			switch err {
			case nil:
				t.Fatal("blocked goroutine did not get error")
			case redis.ErrPoolExhausted:
				t.Fatal("blocked goroutine got pool exhausted error")
			}
		case <-timeout:
			t.Fatal("timeout waiting for blocked goroutine")
		}
	}
	c.Close()
	d.check("done", p, 1, 0)
}

func TestWaitPoolCommandError(t *testing.T) {
	testErr := errors.New("test")
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle:   1,
		MaxActive: 1,
		Dial:      d.dial,
		Wait:      true,
	}
	defer p.Close()
	c := p.Get()
	errs := startGoroutines(p, "ERR", testErr)
	d.check("before close", p, 1, 1)
	c.Close()
	timeout := time.After(2 * time.Second)
	for i := 0; i < cap(errs); i++ {
		select {
		case err := <-errs:
			if err != nil {
				t.Fatal(err)
			}
		case <-timeout:
			t.Fatalf("timeout waiting for blocked goroutine %d", i)
		}
	}
	d.check("done", p, cap(errs), 0)
}

func TestWaitPoolDialError(t *testing.T) {
	testErr := errors.New("test")
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle:   1,
		MaxActive: 1,
		Dial:      d.dial,
		Wait:      true,
	}
	defer p.Close()
	c := p.Get()
	errs := startGoroutines(p, "ERR", testErr)
	d.check("before close", p, 1, 1)

	d.dialErr = errors.New("dial")
	c.Close()

	nilCount := 0
	errCount := 0
	timeout := time.After(2 * time.Second)
	for i := 0; i < cap(errs); i++ {
		select {
		case err := <-errs:
			switch err {
			case nil:
				nilCount++
			case d.dialErr:
				errCount++
			default:
				t.Fatalf("expected dial error or nil, got %v", err)
			}
		case <-timeout:
			t.Fatalf("timeout waiting for blocked goroutine %d", i)
		}
	}
	if nilCount != 1 {
		t.Errorf("expected one nil error, got %d", nilCount)
	}
	if errCount != cap(errs)-1 {
		t.Errorf("expected %d dial erors, got %d", cap(errs)-1, errCount)
	}
	d.check("done", p, cap(errs), 0)
}

// Borrowing requires us to iterate over the idle connections, unlock the pool,
// and perform a blocking operation to check the connection still works. If
// TestOnBorrow fails, we must reacquire the lock and continue iteration. This
// test ensures that iteration will work correctly if multiple threads are
// iterating simultaneously.
func TestLocking_TestOnBorrowFails_PoolDoesntCrash(t *testing.T) {
	count := 100

	// First we'll Create a pool where the pilfering of idle connections fails.
	d := poolDialer{t: t}
	p := &redis.Pool{
		MaxIdle:   count,
		MaxActive: count,
		Dial:      d.dial,
		TestOnBorrow: func(c redis.Conn, t time.Time) error {
			return errors.New("No way back into the real world.")
		},
	}
	defer p.Close()

	// Fill the pool with idle connections.
	b1 := sync.WaitGroup{}
	b1.Add(count)
	b2 := sync.WaitGroup{}
	b2.Add(count)
	for i := 0; i < count; i++ {
		go func() {
			c := p.Get()
			if c.Err() != nil {
				t.Errorf("pool get failed: %v", c.Err())
			}
			b1.Done()
			b1.Wait()
			c.Close()
			b2.Done()
		}()
	}
	b2.Wait()
	if d.dialed != count {
		t.Errorf("Expected %d dials, got %d", count, d.dialed)
	}

	// Spawn a bunch of goroutines to thrash the pool.
	b2.Add(count)
	for i := 0; i < count; i++ {
		go func() {
			c := p.Get()
			if c.Err() != nil {
				t.Errorf("pool get failed: %v", c.Err())
			}
			c.Close()
			b2.Done()
		}()
	}
	b2.Wait()
	if d.dialed != count*2 {
		t.Errorf("Expected %d dials, got %d", count*2, d.dialed)
	}
}

func BenchmarkPoolGet(b *testing.B) {
	b.StopTimer()
	p := redis.Pool{Dial: redistest.Dial, MaxIdle: 2}
	c := p.Get()
	if err := c.Err(); err != nil {
		b.Fatal(err)
	}
	c.Close()
	defer p.Close()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		c = p.Get()
		c.Close()
	}
}

func BenchmarkPoolGetErr(b *testing.B) {
	b.StopTimer()
	p := redis.Pool{Dial: redistest.Dial, MaxIdle: 2}
	c := p.Get()
	if err := c.Err(); err != nil {
		b.Fatal(err)
	}
	c.Close()
	defer p.Close()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		c = p.Get()
		if err := c.Err(); err != nil {
			b.Fatal(err)
		}
		c.Close()
	}
}

func BenchmarkPoolGetPing(b *testing.B) {
	b.StopTimer()
	p := redis.Pool{Dial: redistest.Dial, MaxIdle: 2}
	c := p.Get()
	if err := c.Err(); err != nil {
		b.Fatal(err)
	}
	c.Close()
	defer p.Close()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		c = p.Get()
		if _, err := c.Do("PING"); err != nil {
			b.Fatal(err)
		}
		c.Close()
	}
}
