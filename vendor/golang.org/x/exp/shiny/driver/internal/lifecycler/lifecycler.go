// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lifecycler tracks a window's lifecycle state.
//
// It eliminates sending redundant lifecycle events, ones where the From and To
// stages are equal. For example, moving a window from one part of the screen
// to another should not send multiple events from StageVisible to
// StageVisible, even though the underlying window system's message might only
// hold the new position, and not whether the window was previously visible.
package lifecycler // import "golang.org/x/exp/shiny/driver/internal/lifecycler"

import (
	"sync"

	"golang.org/x/mobile/event/lifecycle"
)

// State is a window's lifecycle state.
type State struct {
	mu      sync.Mutex
	stage   lifecycle.Stage
	dead    bool
	focused bool
	visible bool
}

func (s *State) SetDead(b bool) {
	s.mu.Lock()
	s.dead = b
	s.mu.Unlock()
}

func (s *State) SetFocused(b bool) {
	s.mu.Lock()
	s.focused = b
	s.mu.Unlock()
}

func (s *State) SetVisible(b bool) {
	s.mu.Lock()
	s.visible = b
	s.mu.Unlock()
}

func (s *State) SendEvent(r Sender) {
	s.mu.Lock()
	from, to := s.stage, lifecycle.StageAlive
	// The order of these if's is important. For example, once a window becomes
	// StageDead, it should never change stage again.
	//
	// Similarly, focused trumps visible. It's hard to imagine a situation
	// where a window is focused and not visible on screen, but in that
	// unlikely case, StageFocused seems the most appropriate stage.
	if s.dead {
		to = lifecycle.StageDead
	} else if s.focused {
		to = lifecycle.StageFocused
	} else if s.visible {
		to = lifecycle.StageVisible
	}
	s.stage = to
	s.mu.Unlock()

	if from != to {
		r.Send(lifecycle.Event{
			From: from,
			To:   to,
		})
	}
}

// Sender is who to send the lifecycle event to.
type Sender interface {
	Send(event interface{})
}
