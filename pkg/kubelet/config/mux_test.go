/*
Copyright 2014 The Kubernetes Authors.

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

package config

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestConfigurationChannels(t *testing.T) {
	ctx := ktesting.Init(t)
	ctx = ktesting.WithCancel(ctx)
	defer ctx.Cancel("TestConfigurationChannels completed")

	mux := newMux(nil)
	channelOne := mux.ChannelWithContext(ctx, "one")
	if channelOne != mux.ChannelWithContext(ctx, "one") {
		t.Error("Didn't get the same muxuration channel back with the same name")
	}
	channelTwo := mux.ChannelWithContext(ctx, "two")
	if channelOne == channelTwo {
		t.Error("Got back the same muxuration channel for different names")
	}
}

func TestMergeInvoked(t *testing.T) {
	ctx := ktesting.Init(t)
	ctx = ktesting.WithCancel(ctx)
	defer ctx.Cancel("TestMergeInvoked completed")

	const expectedSource = "one"
	done := make(chan interface{})
	var merger mergeFunc = func(ctx context.Context, source string, update sourceUpdate) error {
		if expectedSource != source {
			t.Errorf("Expected %s, Got %s", expectedSource, source)
		}
		expectedUpdate := fakeUpdate(expectedSource)
		if !reflect.DeepEqual(expectedUpdate, update) {
			t.Errorf("Expected %v, Got %v", expectedUpdate, update)
		}
		close(done)
		return nil
	}

	mux := newMux(&merger)

	mux.ChannelWithContext(ctx, expectedSource) <- fakeUpdate(expectedSource)

	// Wait for Merge call.
	select {
	case <-done:
		// Test complete.
	case <-ctx.Done():
		t.Fatal("Test context canceled before completion")
	}
}

// mergeFunc implements the Merger interface
type mergeFunc func(ctx context.Context, source string, update sourceUpdate) error

func (f mergeFunc) Merge(ctx context.Context, source string, update sourceUpdate) error {
	return f(ctx, source, update)
}

func TestSimultaneousMerge(t *testing.T) {
	ctx := ktesting.Init(t)
	ctx = ktesting.WithCancel(ctx)
	defer ctx.Cancel("TestSimultaneousMerge completed")

	ch := make(chan bool, 2)
	mux := newMux(mergeFunc(func(ctx context.Context, source string, update sourceUpdate) error {
		if nsSource := update.Pods[0].Namespace; nsSource != source {
			t.Errorf("Expected %s, Got %s", source, nsSource)
		}
		ch <- true
		return nil
	}))
	source := mux.ChannelWithContext(ctx, "one")
	source2 := mux.ChannelWithContext(ctx, "two")
	source <- fakeUpdate("one")
	source2 <- fakeUpdate("two")
	<-ch
	<-ch
}

func fakeUpdate(source string) sourceUpdate {
	return sourceUpdate{[]*v1.Pod{{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-pod", source),
			Namespace: source,
			UID:       types.UID(fmt.Sprintf("%s-pod-uid", source)),
		},
	}}}
}
