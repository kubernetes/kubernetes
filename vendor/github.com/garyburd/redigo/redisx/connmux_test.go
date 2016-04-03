// Copyright 2014 Gary Burd
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

package redisx_test

import (
	"net/textproto"
	"sync"
	"testing"

	"github.com/garyburd/redigo/internal/redistest"
	"github.com/garyburd/redigo/redis"
	"github.com/garyburd/redigo/redisx"
)

func TestConnMux(t *testing.T) {
	c, err := redistest.Dial()
	if err != nil {
		t.Fatalf("error connection to database, %v", err)
	}
	m := redisx.NewConnMux(c)
	defer m.Close()

	c1 := m.Get()
	c2 := m.Get()
	c1.Send("ECHO", "hello")
	c2.Send("ECHO", "world")
	c1.Flush()
	c2.Flush()
	s, err := redis.String(c1.Receive())
	if err != nil {
		t.Fatal(err)
	}
	if s != "hello" {
		t.Fatalf("echo returned %q, want %q", s, "hello")
	}
	s, err = redis.String(c2.Receive())
	if err != nil {
		t.Fatal(err)
	}
	if s != "world" {
		t.Fatalf("echo returned %q, want %q", s, "world")
	}
	c1.Close()
	c2.Close()
}

func TestConnMuxClose(t *testing.T) {
	c, err := redistest.Dial()
	if err != nil {
		t.Fatalf("error connection to database, %v", err)
	}
	m := redisx.NewConnMux(c)
	defer m.Close()

	c1 := m.Get()
	c2 := m.Get()

	if err := c1.Send("ECHO", "hello"); err != nil {
		t.Fatal(err)
	}
	if err := c1.Close(); err != nil {
		t.Fatal(err)
	}

	if err := c2.Send("ECHO", "world"); err != nil {
		t.Fatal(err)
	}
	if err := c2.Flush(); err != nil {
		t.Fatal(err)
	}

	s, err := redis.String(c2.Receive())
	if err != nil {
		t.Fatal(err)
	}
	if s != "world" {
		t.Fatalf("echo returned %q, want %q", s, "world")
	}
	c2.Close()
}

func BenchmarkConn(b *testing.B) {
	b.StopTimer()
	c, err := redistest.Dial()
	if err != nil {
		b.Fatalf("error connection to database, %v", err)
	}
	defer c.Close()
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		if _, err := c.Do("PING"); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConnMux(b *testing.B) {
	b.StopTimer()
	c, err := redistest.Dial()
	if err != nil {
		b.Fatalf("error connection to database, %v", err)
	}
	m := redisx.NewConnMux(c)
	defer m.Close()

	b.StartTimer()

	for i := 0; i < b.N; i++ {
		c := m.Get()
		if _, err := c.Do("PING"); err != nil {
			b.Fatal(err)
		}
		c.Close()
	}
}

func BenchmarkPool(b *testing.B) {
	b.StopTimer()

	p := redis.Pool{Dial: redistest.Dial, MaxIdle: 1}
	defer p.Close()

	// Fill the pool.
	c := p.Get()
	if err := c.Err(); err != nil {
		b.Fatal(err)
	}
	c.Close()

	b.StartTimer()

	for i := 0; i < b.N; i++ {
		c := p.Get()
		if _, err := c.Do("PING"); err != nil {
			b.Fatal(err)
		}
		c.Close()
	}
}

const numConcurrent = 10

func BenchmarkConnMuxConcurrent(b *testing.B) {
	b.StopTimer()
	c, err := redistest.Dial()
	if err != nil {
		b.Fatalf("error connection to database, %v", err)
	}
	defer c.Close()

	m := redisx.NewConnMux(c)

	var wg sync.WaitGroup
	wg.Add(numConcurrent)

	b.StartTimer()

	for i := 0; i < numConcurrent; i++ {
		go func() {
			defer wg.Done()
			for i := 0; i < b.N; i++ {
				c := m.Get()
				if _, err := c.Do("PING"); err != nil {
					b.Fatal(err)
				}
				c.Close()
			}
		}()
	}
	wg.Wait()
}

func BenchmarkPoolConcurrent(b *testing.B) {
	b.StopTimer()

	p := redis.Pool{Dial: redistest.Dial, MaxIdle: numConcurrent}
	defer p.Close()

	// Fill the pool.
	conns := make([]redis.Conn, numConcurrent)
	for i := range conns {
		c := p.Get()
		if err := c.Err(); err != nil {
			b.Fatal(err)
		}
		conns[i] = c
	}
	for _, c := range conns {
		c.Close()
	}

	var wg sync.WaitGroup
	wg.Add(numConcurrent)

	b.StartTimer()

	for i := 0; i < numConcurrent; i++ {
		go func() {
			defer wg.Done()
			for i := 0; i < b.N; i++ {
				c := p.Get()
				if _, err := c.Do("PING"); err != nil {
					b.Fatal(err)
				}
				c.Close()
			}
		}()
	}
	wg.Wait()
}

func BenchmarkPipelineConcurrency(b *testing.B) {
	b.StopTimer()
	c, err := redistest.Dial()
	if err != nil {
		b.Fatalf("error connection to database, %v", err)
	}
	defer c.Close()

	var wg sync.WaitGroup
	wg.Add(numConcurrent)

	var pipeline textproto.Pipeline

	b.StartTimer()

	for i := 0; i < numConcurrent; i++ {
		go func() {
			defer wg.Done()
			for i := 0; i < b.N; i++ {
				id := pipeline.Next()
				pipeline.StartRequest(id)
				c.Send("PING")
				c.Flush()
				pipeline.EndRequest(id)
				pipeline.StartResponse(id)
				_, err := c.Receive()
				if err != nil {
					b.Fatal(err)
				}
				pipeline.EndResponse(id)
			}
		}()
	}
	wg.Wait()
}
