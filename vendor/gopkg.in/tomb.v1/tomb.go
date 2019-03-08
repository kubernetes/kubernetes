// Copyright (c) 2011 - Gustavo Niemeyer <gustavo@niemeyer.net>
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimer in the documentation
//       and/or other materials provided with the distribution.
//     * Neither the name of the copyright holder nor the names of its
//       contributors may be used to endorse or promote products derived from
//       this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// The tomb package offers a conventional API for clean goroutine termination.
//
// A Tomb tracks the lifecycle of a goroutine as alive, dying or dead,
// and the reason for its death.
//
// The zero value of a Tomb assumes that a goroutine is about to be
// created or already alive. Once Kill or Killf is called with an
// argument that informs the reason for death, the goroutine is in
// a dying state and is expected to terminate soon. Right before the
// goroutine function or method returns, Done must be called to inform
// that the goroutine is indeed dead and about to stop running.
//
// A Tomb exposes Dying and Dead channels. These channels are closed
// when the Tomb state changes in the respective way. They enable
// explicit blocking until the state changes, and also to selectively
// unblock select statements accordingly.
//
// When the tomb state changes to dying and there's still logic going
// on within the goroutine, nested functions and methods may choose to
// return ErrDying as their error value, as this error won't alter the
// tomb state if provied to the Kill method. This is a convenient way to
// follow standard Go practices in the context of a dying tomb.
//
// For background and a detailed example, see the following blog post:
//
//   http://blog.labix.org/2011/10/09/death-of-goroutines-under-control
//
// For a more complex code snippet demonstrating the use of multiple
// goroutines with a single Tomb, see:
//
//   http://play.golang.org/p/Xh7qWsDPZP
//
package tomb

import (
	"errors"
	"fmt"
	"sync"
)

// A Tomb tracks the lifecycle of a goroutine as alive, dying or dead,
// and the reason for its death.
//
// See the package documentation for details.
type Tomb struct {
	m      sync.Mutex
	dying  chan struct{}
	dead   chan struct{}
	reason error
}

var (
	ErrStillAlive = errors.New("tomb: still alive")
	ErrDying = errors.New("tomb: dying")
)

func (t *Tomb) init() {
	t.m.Lock()
	if t.dead == nil {
		t.dead = make(chan struct{})
		t.dying = make(chan struct{})
		t.reason = ErrStillAlive
	}
	t.m.Unlock()
}

// Dead returns the channel that can be used to wait
// until t.Done has been called.
func (t *Tomb) Dead() <-chan struct{} {
	t.init()
	return t.dead
}

// Dying returns the channel that can be used to wait
// until t.Kill or t.Done has been called.
func (t *Tomb) Dying() <-chan struct{} {
	t.init()
	return t.dying
}

// Wait blocks until the goroutine is in a dead state and returns the
// reason for its death.
func (t *Tomb) Wait() error {
	t.init()
	<-t.dead
	t.m.Lock()
	reason := t.reason
	t.m.Unlock()
	return reason
}

// Done flags the goroutine as dead, and should be called a single time
// right before the goroutine function or method returns.
// If the goroutine was not already in a dying state before Done is
// called, it will be flagged as dying and dead at once with no
// error.
func (t *Tomb) Done() {
	t.Kill(nil)
	close(t.dead)
}

// Kill flags the goroutine as dying for the given reason.
// Kill may be called multiple times, but only the first
// non-nil error is recorded as the reason for termination.
//
// If reason is ErrDying, the previous reason isn't replaced
// even if it is nil. It's a runtime error to call Kill with
// ErrDying if t is not in a dying state.
func (t *Tomb) Kill(reason error) {
	t.init()
	t.m.Lock()
	defer t.m.Unlock()
	if reason == ErrDying {
		if t.reason == ErrStillAlive {
			panic("tomb: Kill with ErrDying while still alive")
		}
		return
	}
	if t.reason == nil || t.reason == ErrStillAlive {
		t.reason = reason
	}
	// If the receive on t.dying succeeds, then
	// it can only be because we have already closed it.
	// If it blocks, then we know that it needs to be closed.
	select {
	case <-t.dying:
	default:
		close(t.dying)
	}
}

// Killf works like Kill, but builds the reason providing the received
// arguments to fmt.Errorf. The generated error is also returned.
func (t *Tomb) Killf(f string, a ...interface{}) error {
	err := fmt.Errorf(f, a...)
	t.Kill(err)
	return err
}

// Err returns the reason for the goroutine death provided via Kill
// or Killf, or ErrStillAlive when the goroutine is still alive.
func (t *Tomb) Err() (reason error) {
	t.init()
	t.m.Lock()
	reason = t.reason
	t.m.Unlock()
	return
}
