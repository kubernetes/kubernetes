/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package testing

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"time"
)

// WaitForChannelEvent blocks until the channel receives an event.
// Returns an error if the channel is closed or the context is done.
func WaitForChannelEvent[T any](ctx context.Context, ch <-chan T) (T, error) {
	var t T // zero value
	for {
		select {
		case <-ctx.Done():
			err := ctx.Err()
			switch err {
			case context.DeadlineExceeded:
				return t, fmt.Errorf("timed out waiting for channel to close: %w", err)
			default:
				return t, fmt.Errorf("context cancelled before channel closed: %w", err)
			}
		case event, ok := <-ch:
			if !ok {
				return t, errors.New("channel closed before receiving event")
			}
			return event, nil
		}
	}
}

// WaitForChannelToClose blocks until the channel is closed.
// Returns an error if any events are received or the context is done.
func WaitForChannelToClose[T any](ctx context.Context, ch <-chan T) error {
	for {
		select {
		case <-ctx.Done():
			err := ctx.Err()
			switch err {
			case context.DeadlineExceeded:
				return fmt.Errorf("timed out waiting for channel to close: %w", err)
			default:
				return fmt.Errorf("context cancelled before channel closed: %w", err)
			}
		case event, ok := <-ch:
			if !ok {
				return nil
			}
			return fmt.Errorf("channel received unexpected event: %#v", event)
		}
	}
}

// WaitForAllChannelsToClose blocks until all the channels are closed.
// Returns an error if any events are received or the context is done.
func WaitForAllChannelsToClose[T any](ctx context.Context, channels ...<-chan T) error {
	// Build a list of cases to select from
	cases := make([]reflect.SelectCase, len(channels)+1)
	for i, ch := range channels {
		cases[i] = reflect.SelectCase{
			Dir:  reflect.SelectRecv,
			Chan: reflect.ValueOf(ch),
		}
	}
	// Add the context done channel as the last case
	contextCaseIndex := len(channels)
	cases[contextCaseIndex] = reflect.SelectCase{
		Dir:  reflect.SelectRecv,
		Chan: reflect.ValueOf(ctx.Done()),
	}
	// Select from the cases until all channels are closed, an event is received,
	// or the context is done.
	channelsRemaining := len(cases)
	for channelsRemaining > 1 {
		// Block until one of the channels receives an event or closes
		chosenIndex, value, ok := reflect.Select(cases)
		if !ok {
			// Return error immediately if the context is done
			if chosenIndex == contextCaseIndex {
				err := ctx.Err()
				switch err {
				case context.DeadlineExceeded:
					return fmt.Errorf("timed out waiting for channel to close: %w", err)
				default:
					return fmt.Errorf("context cancelled before channel closed: %w", err)
				}
			}
			// Remove closed channel from case to ignore it going forward
			cases[chosenIndex].Chan = reflect.ValueOf(nil)
			channelsRemaining--
			continue
		}
		// All events received are treated as errors
		return fmt.Errorf("channel %d received unexpected event: %#v", chosenIndex, value.Interface())
	}
	// All channels closed before the context was done
	return nil
}

// WaitForChannelEventWithTimeout blocks until the channel receives an event.
// Returns an error if the channel is closed, the context is done, or the
// timeout is reached
func WaitForChannelEventWithTimeout[T any](ctx context.Context, timeout time.Duration, ch <-chan T) (T, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return WaitForChannelEvent(ctx, ch)
}

// WaitForChannelToClose blocks until the channel is closed.
// Returns an error if any events are received, the context is done, or the
// timeout is reached
func WaitForChannelToCloseWithTimeout[T any](ctx context.Context, timeout time.Duration, ch <-chan T) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return WaitForChannelToClose(ctx, ch)
}

// WaitForAllChannelsToCloseWithTimeout blocks until all the channels are closed.
// Returns an error if any events are received, the context is done, or the
// timeout is reached
func WaitForAllChannelsToCloseWithTimeout[T any](ctx context.Context, timeout time.Duration, channels ...<-chan T) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return WaitForAllChannelsToClose(ctx, channels...)
}
