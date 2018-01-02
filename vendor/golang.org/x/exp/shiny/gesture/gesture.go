// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gesture provides gesture events such as long presses and drags.
// These are higher level than underlying mouse and touch events.
package gesture

import (
	"fmt"
	"time"

	"golang.org/x/exp/shiny/screen"
	"golang.org/x/mobile/event/mouse"
)

// TODO: handle touch events, not just mouse events.
//
// TODO: multi-button / multi-touch gestures such as pinch, rotate and tilt?

const (
	// TODO: use a resolution-independent unit such as DIPs or Millimetres?
	dragThreshold = 10 // Pixels.

	doublePressThreshold = 300 * time.Millisecond
	longPressThreshold   = 500 * time.Millisecond
)

// Type describes the type of a touch event.
type Type uint8

const (
	// TypeStart and TypeEnd are the start and end of a gesture. A gesture
	// spans multiple events.
	TypeStart Type = 0
	TypeEnd   Type = 1

	// TypeIsXxx is when the gesture is recognized as a long press, double
	// press or drag. For example, a mouse button press won't generate a
	// TypeIsLongPress immediately, but if a threshold duration passes without
	// the corresponding mouse button release, a TypeIsLongPress event is sent.
	//
	// Once a TypeIsXxx event is sent, the corresponding Event.Xxx bool field
	// is set for this and subsequent events. For example, a TypeTap event by
	// itself doesn't say whether or not it is a single tap or the first tap of
	// a double tap. If the app needs to distinguish these two sorts of taps,
	// it can wait until a TypeEnd or TypeIsDoublePress event is seen. If a
	// TypeEnd is seen before TypeIsDoublePress, or equivalently, if the
	// TypeEnd event's DoublePress field is false, the gesture is a single tap.
	//
	// These attributes aren't exclusive. A long press drag is perfectly valid.
	//
	// The uncommon "double press" instead of "double tap" terminology is
	// because, in this package, taps are associated with button releases, not
	// button presses. Note also that "double" really means "at least two".
	TypeIsLongPress   Type = 10
	TypeIsDoublePress Type = 11
	TypeIsDrag        Type = 12

	// TypeTap and TypeDrag are tap and drag events.
	//
	// For 'flinging' drags, to simulate inertia, look to the Velocity field of
	// the TypeEnd event.
	//
	// TODO: implement velocity.
	TypeTap  Type = 20
	TypeDrag Type = 21

	// All internal types are >= typeInternal.
	typeInternal Type = 100

	// The typeXxxSchedule and typeXxxResolve constants are used for the two
	// step process for sending an event after a timeout, in a separate
	// goroutine. There are two steps so that the spawned goroutine is
	// guaranteed to execute only after any other EventDeque.SendFirst calls
	// are made for the one underlying mouse or touch event.

	typeDoublePressSchedule Type = 100
	typeDoublePressResolve  Type = 101

	typeLongPressSchedule Type = 110
	typeLongPressResolve  Type = 111
)

func (t Type) String() string {
	switch t {
	case TypeStart:
		return "Start"
	case TypeEnd:
		return "End"
	case TypeIsLongPress:
		return "IsLongPress"
	case TypeIsDoublePress:
		return "IsDoublePress"
	case TypeIsDrag:
		return "IsDrag"
	case TypeTap:
		return "Tap"
	case TypeDrag:
		return "Drag"
	default:
		return fmt.Sprintf("gesture.Type(%d)", t)
	}
}

// Point is a mouse or touch location, in pixels.
type Point struct {
	X, Y float32
}

// Event is a gesture event.
type Event struct {
	// Type is the gesture type.
	Type Type

	// Drag, LongPress and DoublePress are set when the gesture is recognized as
	// a drag, etc.
	//
	// Note that these status fields can be lost during a gesture's events over
	// time: LongPress can be set for the first press of a double press, but
	// unset on the second press.
	Drag        bool
	LongPress   bool
	DoublePress bool

	// InitialPos is the initial position of the button press or touch that
	// started this gesture.
	InitialPos Point

	// CurrentPos is the current position of the button or touch event.
	CurrentPos Point

	// TODO: a "Velocity Point" field. See
	//	- frameworks/native/libs/input/VelocityTracker.cpp in AOSP, or
	//	- https://chromium.googlesource.com/chromium/src/+/master/ui/events/gesture_detection/velocity_tracker.cc in Chromium,
	// for some velocity tracking implementations.

	// Time is the event's time.
	Time time.Time

	// TODO: include the mouse Button and key Modifiers?
}

type internalEvent struct {
	eventFilter *EventFilter

	typ  Type
	x, y float32

	// pressCounter is the EventFilter.pressCounter value at the time this
	// internal event was scheduled to be delivered after a timeout. It detects
	// whether there have been other button presses and releases during that
	// timeout, and hence whether this internalEvent is obsolete.
	pressCounter uint32
}

// EventFilter generates gesture events from lower level mouse and touch
// events.
type EventFilter struct {
	EventDeque screen.EventDeque

	inProgress  bool
	drag        bool
	longPress   bool
	doublePress bool

	// initialPos is the initial position of the button press or touch that
	// started this gesture.
	initialPos Point

	// pressButton is the initial button that started this gesture. If
	// button.None, no gesture is in progress.
	pressButton mouse.Button

	// pressCounter is incremented on every button press and release.
	pressCounter uint32
}

func (f *EventFilter) sendFirst(t Type, x, y float32, now time.Time) {
	if t >= typeInternal {
		f.EventDeque.SendFirst(internalEvent{
			eventFilter:  f,
			typ:          t,
			x:            x,
			y:            y,
			pressCounter: f.pressCounter,
		})
		return
	}
	f.EventDeque.SendFirst(Event{
		Type:        t,
		Drag:        f.drag,
		LongPress:   f.longPress,
		DoublePress: f.doublePress,
		InitialPos:  f.initialPos,
		CurrentPos: Point{
			X: x,
			Y: y,
		},
		// TODO: Velocity.
		Time: now,
	})
}

func (f *EventFilter) sendAfter(e internalEvent, sleep time.Duration) {
	time.Sleep(sleep)
	f.EventDeque.SendFirst(e)
}

func (f *EventFilter) end(x, y float32, now time.Time) {
	f.sendFirst(TypeEnd, x, y, now)
	f.inProgress = false
	f.drag = false
	f.longPress = false
	f.doublePress = false
	f.initialPos = Point{}
	f.pressButton = mouse.ButtonNone
}

// Filter filters the event. It can return e, a different event, or nil to
// consume the event. It can also trigger side effects such as pushing new
// events onto its EventDeque.
func (f *EventFilter) Filter(e interface{}) interface{} {
	switch e := e.(type) {
	case internalEvent:
		if e.eventFilter != f {
			break
		}

		now := time.Now()

		switch e.typ {
		case typeDoublePressSchedule:
			e.typ = typeDoublePressResolve
			go f.sendAfter(e, doublePressThreshold)

		case typeDoublePressResolve:
			if e.pressCounter == f.pressCounter {
				// It's a single press only.
				f.end(e.x, e.y, now)
			}

		case typeLongPressSchedule:
			e.typ = typeLongPressResolve
			go f.sendAfter(e, longPressThreshold)

		case typeLongPressResolve:
			if e.pressCounter == f.pressCounter && !f.drag {
				f.longPress = true
				f.sendFirst(TypeIsLongPress, e.x, e.y, now)
			}
		}
		return nil

	case mouse.Event:
		now := time.Now()

		switch e.Direction {
		case mouse.DirNone:
			if f.pressButton == mouse.ButtonNone {
				break
			}
			startDrag := false
			if !f.drag &&
				(abs(e.X-f.initialPos.X) > dragThreshold || abs(e.Y-f.initialPos.Y) > dragThreshold) {
				f.drag = true
				startDrag = true
			}
			if f.drag {
				f.sendFirst(TypeDrag, e.X, e.Y, now)
			}
			if startDrag {
				f.sendFirst(TypeIsDrag, e.X, e.Y, now)
			}

		case mouse.DirPress:
			if f.pressButton != mouse.ButtonNone {
				break
			}

			oldInProgress := f.inProgress
			oldDoublePress := f.doublePress

			f.drag = false
			f.longPress = false
			f.doublePress = f.inProgress
			f.initialPos = Point{e.X, e.Y}
			f.pressButton = e.Button
			f.pressCounter++

			f.inProgress = true

			f.sendFirst(typeLongPressSchedule, e.X, e.Y, now)
			if !oldDoublePress && f.doublePress {
				f.sendFirst(TypeIsDoublePress, e.X, e.Y, now)
			}
			if !oldInProgress {
				f.sendFirst(TypeStart, e.X, e.Y, now)
			}

		case mouse.DirRelease:
			if f.pressButton != e.Button {
				break
			}
			f.pressButton = mouse.ButtonNone
			f.pressCounter++

			if f.drag {
				f.end(e.X, e.Y, now)
				break
			}
			f.sendFirst(typeDoublePressSchedule, e.X, e.Y, now)
			f.sendFirst(TypeTap, e.X, e.Y, now)
		}
	}
	return e
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	} else if x == 0 {
		return 0 // Handle floating point negative zero.
	}
	return x
}
