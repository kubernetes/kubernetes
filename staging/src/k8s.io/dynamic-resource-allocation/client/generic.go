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

package client

import (
	"context"
	"sync"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	types "k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	watch "k8s.io/apimachinery/pkg/watch"
	drav1beta1 "k8s.io/dynamic-resource-allocation/api/v1beta1"
	drav1beta2 "k8s.io/dynamic-resource-allocation/api/v1beta2"
	"k8s.io/klog/v2"
)

var scheme = runtime.NewScheme()

func init() {
	utilruntime.Must(drav1beta1.AddToScheme(scheme))
	utilruntime.Must(drav1beta2.AddToScheme(scheme))
}

// In all of the following generics:
// - N: the native non-pointer type of the package
// - O: the legacy non-pointer type that we need to convert to or from
// - [NO]P: the corresponding pointer type
// - [NO]L: the corresponding non-pointer list type
// - [NO]AC: the corresponding non-pointer apply-configuration type
//
// More legacy types will get added when reaching GA.

func newConvertingClient[NP objectPtr[N], N, NL, NAC any, OP objectPtr[O], O, OL, OAC any, O2P objectPtr[O2], O2, O2L, O2AC any](c *Client, native funcs[N, NL, NAC], v1beta1 funcs[O, OL, OAC], v1beta2 funcs[O2, O2L, O2AC]) *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC] {
	return &convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]{
		c:       c,
		native:  native,
		v1beta1: v1beta1,
		v1beta2: v1beta2,
	}
}

type convertingClient[NP objectPtr[N], N, NL, NAC any, OP objectPtr[O], O, OL, OAC any, O2P objectPtr[O2], O2, O2L, O2AC any] struct {
	c       *Client
	native  funcs[N, NL, NAC]
	v1beta1 funcs[O, OL, OAC]
	v1beta2 funcs[O2, O2L, O2AC]
}

type funcs[T, TL, TAC any] interface {
	Create(context.Context, *T, metav1.CreateOptions) (*T, error)
	Update(context.Context, *T, metav1.UpdateOptions) (*T, error)
	Delete(context.Context, string, metav1.DeleteOptions) error
	DeleteCollection(context.Context, metav1.DeleteOptions, metav1.ListOptions) error
	Get(context.Context, string, metav1.GetOptions) (*T, error)
	List(context.Context, metav1.ListOptions) (*TL, error)
	Watch(context.Context, metav1.ListOptions) (watch.Interface, error)
	Apply(context.Context, *TAC, metav1.ApplyOptions) (*T, error)
}

type funcsWithStatus[T, TL, TAC any] interface {
	funcs[T, TL, TAC]
	UpdateStatus(context.Context, *T, metav1.UpdateOptions) (*T, error)
	ApplyStatus(context.Context, *TAC, metav1.ApplyOptions) (*T, error)
}

func (t *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]) Create(ctx context.Context, obj *N, opts metav1.CreateOptions) (*N, error) {
	apis := newCall(t.c, func(currentAPI int32) (*N, error) {
		switch currentAPI {
		case useV1beta1API:
			return putWithConversion(obj, func(obj *O) (*O, error) {
				return t.v1beta1.Create(ctx, obj, opts)
			})
		case useV1beta2API:
			return putWithConversion(obj, func(obj *O2) (*O2, error) {
				return t.v1beta2.Create(ctx, obj, opts)
			})
		default:
			return t.native.Create(ctx, obj, opts)
		}
	})
	return apis.run()
}

func (t *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]) Update(ctx context.Context, obj *N, opts metav1.UpdateOptions) (*N, error) {
	apis := newCall(t.c, func(currentAPI int32) (*N, error) {
		switch currentAPI {
		case useV1beta1API:
			return putWithConversion(obj, func(obj *O) (*O, error) {
				return t.v1beta1.Update(ctx, obj, opts)
			})
		case useV1beta2API:
			return putWithConversion(obj, func(obj *O2) (*O2, error) {
				return t.v1beta2.Update(ctx, obj, opts)
			})
		default:
			return t.native.Update(ctx, obj, opts)
		}
	})
	return apis.run()
}

func (t *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]) UpdateStatus(ctx context.Context, obj *N, opts metav1.UpdateOptions) (*N, error) {
	apis := newCall(t.c, func(currentAPI int32) (*N, error) {
		switch currentAPI {
		case useV1beta1API:
			return putWithConversion(obj, func(obj *O) (*O, error) {
				return t.v1beta1.(funcsWithStatus[O, OL, OAC]).UpdateStatus(ctx, obj, opts)
			})
		case useV1beta2API:
			return putWithConversion(obj, func(obj *O2) (*O2, error) {
				return t.v1beta1.(funcsWithStatus[O2, O2L, O2AC]).UpdateStatus(ctx, obj, opts)
			})
		default:
			return t.native.(funcsWithStatus[N, NL, NAC]).UpdateStatus(ctx, obj, opts)
		}
	})
	return apis.run()
}

func (t *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]) Delete(ctx context.Context, name string, opts metav1.DeleteOptions) error {
	apis := newCall(t.c, func(currentAPI int32) (*N, error) {
		switch currentAPI {
		case useV1beta1API:
			return nil, t.v1beta1.Delete(ctx, name, opts)
		case useV1beta2API:
			return nil, t.v1beta2.Delete(ctx, name, opts)
		default:
			return nil, t.native.Delete(ctx, name, opts)
		}
	})
	_, err := apis.run()
	return err
}

func (t *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	apis := newCall(t.c, func(currentAPI int32) (*N, error) {
		switch currentAPI {
		case useV1beta1API:
			return nil, t.v1beta1.DeleteCollection(ctx, opts, listOpts)
		case useV1beta2API:
			return nil, t.v1beta2.DeleteCollection(ctx, opts, listOpts)
		default:
			return nil, t.native.DeleteCollection(ctx, opts, listOpts)
		}
	})
	_, err := apis.run()
	return err
}

func (t *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]) Get(ctx context.Context, name string, opts metav1.GetOptions) (*N, error) {
	apis := newCall(t.c, func(currentAPI int32) (*N, error) {
		switch currentAPI {
		case useV1beta1API:
			return getWithConversion[N, O](func() (*O, error) {
				return t.v1beta1.Get(ctx, name, opts)
			})
		case useV1beta2API:
			return getWithConversion[N, O2](func() (*O2, error) {
				return t.v1beta2.Get(ctx, name, opts)
			})
		default:
			return t.native.Get(ctx, name, opts)
		}
	})
	return apis.run()
}

func (t *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]) List(ctx context.Context, opts metav1.ListOptions) (*NL, error) {
	apis := newCall(t.c, func(currentAPI int32) (*NL, error) {
		switch currentAPI {
		case useV1beta1API:
			return getWithConversion[NL, OL](func() (*OL, error) {
				return t.v1beta1.List(ctx, opts)
			})
		case useV1beta2API:
			return getWithConversion[NL, O2L](func() (*O2L, error) {
				return t.v1beta2.List(ctx, opts)
			})
		default:
			return t.native.List(ctx, opts)
		}
	})
	return apis.run()
}

func (t *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	apis := newCall(t.c, func(currentAPI int32) (watch.Interface, error) {
		switch currentAPI {
		case useV1beta1API:
			return watchWithConversion[NP, N, OP](func() (watch.Interface, error) {
				return t.v1beta1.Watch(ctx, opts)
			})
		case useV1beta2API:
			return watchWithConversion[NP, N, O2P](func() (watch.Interface, error) {
				return t.v1beta2.Watch(ctx, opts)
			})
		default:
			return t.native.Watch(ctx, opts)
		}
	})
	return apis.run()
}

func (t *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (result *N, err error) {
	return nil, ErrNotImplemented
}

func (t *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]) Apply(ctx context.Context, obj *NAC, opts metav1.ApplyOptions) (result *N, err error) {
	return nil, ErrNotImplemented
}

func (t *convertingClient[NP, N, NL, NAC, OP, O, OL, OAC, O2P, O2, O2L, O2AC]) ApplyStatus(ctx context.Context, obj *NAC, opts metav1.ApplyOptions) (result *N, err error) {
	return nil, ErrNotImplemented
}

func newCall[N any](c *Client, call func(currentAPI int32) (N, error)) callSequence[N] {
	seq := callSequence[N]{
		c:    c,
		call: call,
	}
	return seq
}

type callSequence[N any] struct {
	c    *Client
	call func(currentAPI int32) (N, error)
}

func (s *callSequence[N]) run() (N, error) {
	currentAPI := s.c.useAPI.Load()
	var value N
	var err error
	for i := int32(0); i < numAPIs; i++ {
		value, err = s.call(currentAPI)
		// When encountering "not found", try with the next API version.
		// We give up after having gone through all of them.
		//
		// This check has a false positive for genuine "not found" errors
		// when accessing specific objects. The client will then try
		// all API version until it arrives back at the original one - not
		// nice because of the overhead, but not wrong semantically
		// and shouldn't occur often.
		if apierrors.IsNotFound(err) {
			currentAPI = (currentAPI + 1) % numAPIs
			continue
		}
		break
	}
	s.c.useAPI.Store(currentAPI)
	return value, err
}

func putWithConversion[N, O any](obj *N, call func(*O) (*O, error)) (*N, error) {
	out := new(O)
	if err := scheme.Convert(obj, out, nil); err != nil {
		return nil, err
	}
	in, err := call(out)
	if err != nil {
		return nil, err
	}
	value := new(N)
	if err := scheme.Convert(in, value, nil); err != nil {
		return nil, err
	}
	return value, nil
}

func getWithConversion[N, O any](call func() (*O, error)) (*N, error) {
	in, err := call()
	if err != nil {
		return nil, err
	}
	value := new(N)
	if err := scheme.Convert(in, value, nil); err != nil {
		return nil, err
	}
	return value, nil
}

func watchWithConversion[NP objectPtr[N], N any, OP runtime.Object](call func() (watch.Interface, error)) (watch.Interface, error) {
	in, err := call()
	if err != nil {
		return nil, err
	}
	out := &watchSomething[NP, N, OP]{
		upstream:   in,
		resultChan: make(chan watch.Event),
		stopChan:   make(chan struct{}),
	}
	go out.run() // TODO: ctx
	return out, nil
}

// objectPtr is a helper which tells Go that a pointer to T
// implements the runtime.Object interface in watchWithConversion.
type objectPtr[T any] interface {
	*T
	runtime.Object
}

type watchSomething[NP objectPtr[N], N any, OP runtime.Object] struct {
	upstream   watch.Interface
	resultChan chan watch.Event
	stopChan   chan struct{}
	stopOnce   sync.Once
}

func (w *watchSomething[NP, N, OP]) Stop() {
	w.upstream.Stop()
	w.stopOnce.Do(func() {
		close(w.stopChan)
	})
}

func (w *watchSomething[NP, N, OP]) ResultChan() <-chan watch.Event {
	return w.resultChan
}

func (w *watchSomething[NP, N, OP]) run() {
	defer utilruntime.HandleCrash()
	defer close(w.resultChan)
	resultChan := w.upstream.ResultChan()
	for {
		e, ok := <-resultChan
		if !ok {
			// The producer stopped first.
			break
		}
		if in, ok := e.Object.(OP); ok {
			out := new(N)
			if err := scheme.Convert(in, out, nil); err != nil {
				//nolint:logcheck // Shouldn't happen.
				klog.Error(err, "convert ResourceSlice")
			}
			e = watch.Event{
				Type:   e.Type,
				Object: NP(out),
			}
		}
		// This must not get blocked when the consumer stops reading,
		// hence the stopChan.
		select {
		case w.resultChan <- e:
		case <-w.stopChan:
			break
		}
	}
}
