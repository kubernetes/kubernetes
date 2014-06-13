// Copyright 2014 Docker authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the DOCKER-LICENSE file.

package engine

import (
	"bufio"
	"container/ring"
	"fmt"
	"io"
	"sync"
)

type Output struct {
	sync.Mutex
	dests []io.Writer
	tasks sync.WaitGroup
}

// NewOutput returns a new Output object with no destinations attached.
// Writing to an empty Output will cause the written data to be discarded.
func NewOutput() *Output {
	return &Output{}
}

// Add attaches a new destination to the Output. Any data subsequently written
// to the output will be written to the new destination in addition to all the others.
// This method is thread-safe.
// FIXME: Add cannot fail
func (o *Output) Add(dst io.Writer) error {
	o.Mutex.Lock()
	defer o.Mutex.Unlock()
	o.dests = append(o.dests, dst)
	return nil
}

// AddPipe creates an in-memory pipe with io.Pipe(), adds its writing end as a destination,
// and returns its reading end for consumption by the caller.
// This is a rough equivalent similar to Cmd.StdoutPipe() in the standard os/exec package.
// This method is thread-safe.
func (o *Output) AddPipe() (io.Reader, error) {
	r, w := io.Pipe()
	o.Add(w)
	return r, nil
}

// AddTail starts a new goroutine which will read all subsequent data written to the output,
// line by line, and append the last `n` lines to `dst`.
func (o *Output) AddTail(dst *[]string, n int) error {
	src, err := o.AddPipe()
	if err != nil {
		return err
	}
	o.tasks.Add(1)
	go func() {
		defer o.tasks.Done()
		Tail(src, n, dst)
	}()
	return nil
}

// AddString starts a new goroutine which will read all subsequent data written to the output,
// line by line, and store the last line into `dst`.
func (o *Output) AddString(dst *string) error {
	src, err := o.AddPipe()
	if err != nil {
		return err
	}
	o.tasks.Add(1)
	go func() {
		defer o.tasks.Done()
		lines := make([]string, 0, 1)
		Tail(src, 1, &lines)
		if len(lines) == 0 {
			*dst = ""
		} else {
			*dst = lines[0]
		}
	}()
	return nil
}

// Write writes the same data to all registered destinations.
// This method is thread-safe.
func (o *Output) Write(p []byte) (n int, err error) {
	o.Mutex.Lock()
	defer o.Mutex.Unlock()
	var firstErr error
	for _, dst := range o.dests {
		_, err := dst.Write(p)
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return len(p), firstErr
}

// Close unregisters all destinations and waits for all background
// AddTail and AddString tasks to complete.
// The Close method of each destination is called if it exists.
func (o *Output) Close() error {
	o.Mutex.Lock()
	defer o.Mutex.Unlock()
	var firstErr error
	for _, dst := range o.dests {
		if closer, ok := dst.(io.WriteCloser); ok {
			err := closer.Close()
			if err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	o.tasks.Wait()
	return firstErr
}

type Input struct {
	src io.Reader
	sync.Mutex
}

// NewInput returns a new Input object with no source attached.
// Reading to an empty Input will return io.EOF.
func NewInput() *Input {
	return &Input{}
}

// Read reads from the input in a thread-safe way.
func (i *Input) Read(p []byte) (n int, err error) {
	i.Mutex.Lock()
	defer i.Mutex.Unlock()
	if i.src == nil {
		return 0, io.EOF
	}
	return i.src.Read(p)
}

// Add attaches a new source to the input.
// Add can only be called once per input. Subsequent calls will
// return an error.
func (i *Input) Add(src io.Reader) error {
	i.Mutex.Lock()
	defer i.Mutex.Unlock()
	if i.src != nil {
		return fmt.Errorf("Maximum number of sources reached: 1")
	}
	i.src = src
	return nil
}

// Tail reads from `src` line per line, and returns the last `n` lines as an array.
// A ring buffer is used to only store `n` lines at any time.
func Tail(src io.Reader, n int, dst *[]string) {
	scanner := bufio.NewScanner(src)
	r := ring.New(n)
	for scanner.Scan() {
		if n == 0 {
			continue
		}
		r.Value = scanner.Text()
		r = r.Next()
	}
	r.Do(func(v interface{}) {
		if v == nil {
			return
		}
		*dst = append(*dst, v.(string))
	})
}

// AddEnv starts a new goroutine which will decode all subsequent data
// as a stream of json-encoded objects, and point `dst` to the last
// decoded object.
// The result `env` can be queried using the type-neutral Env interface.
// It is not safe to query `env` until the Output is closed.
func (o *Output) AddEnv() (dst *Env, err error) {
	src, err := o.AddPipe()
	if err != nil {
		return nil, err
	}
	dst = &Env{}
	o.tasks.Add(1)
	go func() {
		defer o.tasks.Done()
		decoder := NewDecoder(src)
		for {
			env, err := decoder.Decode()
			if err != nil {
				return
			}
			*dst = *env
		}
	}()
	return dst, nil
}
