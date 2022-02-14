/*
Copyright 2022 The Kubernetes Authors.

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

package controller

import (
	"context"
	"fmt"
	"testing"
	"time"

	"k8s.io/component-base/metrics"
)

func TestSyncAndRecordWithCtx(t *testing.T) {
	since = sinceFn

	keyFn := keyFuncFor("SyncAndRecordWithCtx")
	exerciseFn := func(tc syncAndRecordTestCase) error {
		sync := SyncAndRecordWithCtx(keyFn(tc.name), tc.fn)
		return sync(context.TODO(), tc.name)
	}

	checkTestCases(t, keyFn, exerciseFn)
}

func TestRunSyncAndRecordWithCtx(t *testing.T) {
	since = sinceFn

	keyFn := keyFuncFor("RunSyncAndRecordWithCtx")
	exerciseFn := func(tc syncAndRecordTestCase) error {
		return RunSyncAndRecordWithCtx(context.TODO(), keyFn(tc.name), tc.name, tc.fn)
	}

	checkTestCases(t, keyFn, exerciseFn)
}

func TestRunSyncAndRecord(t *testing.T) {
	since = sinceFn

	keyFn := keyFuncFor("RunSyncAndRecord")
	exerciseFn := func(tc syncAndRecordTestCase) error {
		return RunSyncAndRecord(keyFn(tc.name), tc.name, tc.fnNoCtx)
	}

	checkTestCases(t, keyFn, exerciseFn)
}

func TestHasSyncedAndRecordedWithCtx(t *testing.T) {
	since = sinceFn

	keyFn := keyFuncFor("HasSyncedAndRecordedWithCtx")
	exerciseFn := func(tc syncAndRecordTestCase) error {
		sync := HasSyncedAndRecordedWithCtx(keyFn(tc.name), tc.fnWithBool)
		_, err := sync(context.TODO(), tc.name)
		return err
	}

	checkTestCases(t, keyFn, exerciseFn)
}

func TestHasSyncedAndRecorded(t *testing.T) {
	since = sinceFn

	keyFn := keyFuncFor("HasSyncedAndRecorded")
	exerciseFn := func(tc syncAndRecordTestCase) error {
		sync := HasSyncedAndRecorded(keyFn(tc.name), tc.fnWithBoolNoCtx)
		_, err := sync(tc.name)
		return err
	}

	checkTestCases(t, keyFn, exerciseFn)
}

func TestRunHasSyncedAndRecordedWithCtx(t *testing.T) {
	since = sinceFn

	keyFn := keyFuncFor("RunHasSyncedAndRecordedWithCtx")
	exerciseFn := func(tc syncAndRecordTestCase) error {
		_, err := RunHasSyncedAndRecordedWithCtx(context.TODO(), keyFn(tc.name), tc.name, tc.fnWithBool)
		return err
	}

	checkTestCases(t, keyFn, exerciseFn)
}

func TestRunHasSyncedAndRecorded(t *testing.T) {
	since = sinceFn

	keyFn := keyFuncFor("RunHasSyncedAndRecorded")
	exerciseFn := func(tc syncAndRecordTestCase) error {
		_, err := RunHasSyncedAndRecorded(keyFn(tc.name), tc.name, tc.fnWithBoolNoCtx)
		return err
	}

	checkTestCases(t, keyFn, exerciseFn)
}

func TestSyncAndRecordAll(t *testing.T) {
	// TODO
}

func RunTestSyncAndRecordAll(t *testing.T) {
	// TODO
}

// functions to wrap or pass

func returnErr(ctx context.Context, key string) error {
	return returnErrNoContext(key)
}

func returnErrNoContext(key string) error {
	return fmt.Errorf("%v", key)
}

func returnErrWithBool(ctx context.Context, key string) (bool, error) {
	return false, returnErrNoContext(key)
}

func returnErrWithBoolNoCtx(key string) (bool, error) {
	return false, returnErrNoContext(key)
}

func returnOk(ctx context.Context, key string) error {
	return nil
}

func returnOkNoContext(key string) error {
	return nil
}

func returnOkWithBool(ctx context.Context, key string) (bool, error) {
	return true, nil
}

func returnOkWithBoolNoCtx(key string) (bool, error) {
	return true, nil
}

type (
	syncAndRecordTestCase struct {
		name            string
		fn              func(context.Context, string) error
		fnNoCtx         func(string) error
		fnWithBool      func(context.Context, string) (bool, error)
		fnWithBoolNoCtx func(string) (bool, error)
		expectedErr     bool
	}
	keyFunc      func(string) string
	exerciseFunc func(syncAndRecordTestCase) error
)

var syncAndRecordCases = []syncAndRecordTestCase{
	{
		name:            "error",
		fn:              returnErr,
		fnNoCtx:         returnErrNoContext,
		fnWithBool:      returnErrWithBool,
		fnWithBoolNoCtx: returnErrWithBoolNoCtx,
		expectedErr:     true,
	},
	{
		name:            "no-error",
		fn:              returnOk,
		fnNoCtx:         returnOkNoContext,
		fnWithBool:      returnOkWithBool,
		fnWithBoolNoCtx: returnOkWithBoolNoCtx,
		expectedErr:     false,
	},
}

func checkTestCases(t *testing.T, keyFn keyFunc, exerciseFn exerciseFunc) {
	for ii := range syncAndRecordCases {
		tc := syncAndRecordCases[ii]
		t.Run(tc.name, func(t *testing.T) {
			checkTestCase(t, tc, keyFn, exerciseFn)
		})
	}
}

func checkTestCase(t *testing.T, tc syncAndRecordTestCase, keyFn keyFunc, exerciseFn exerciseFunc) {
	registry := metrics.NewKubeRegistry()
	registry.MustRegister(ReconciliationDurations)
	registry.Reset()

	err := exerciseFn(tc)

	if e, a := tc.expectedErr, (err != nil); e != a {
		t.Errorf("unexpected behavior: expected err to be %v, got %v", e, a)
	}

	ms, err := registry.Gather()
	if err != nil {
		t.Errorf("error gathering from registry: %v", err)
	}

	if len(ms) != 1 {
		t.Errorf("unexpected metrics count, expected 1, got %v", len(ms))
	}

	mf := ms[0]
	for _, m := range mf.GetMetric() {
		for _, l := range m.Label {
			if *l.Name == "controller" {
				if e, a := keyFn(tc.name), *l.Value; e != a {
					t.Errorf("unexpected 'controller' label; expected %v, got %v", keyFn(tc.name), *l.Value)
				}
			}

			if *l.Name == "status" {
				if e, a := expectedStatus(tc.expectedErr), *l.Value; e != a {
					t.Errorf("unexpected status; expected %v, got %v", e, a)
				}
			}
		}

		if e, a := uint64(1), m.GetHistogram().GetSampleCount(); e != a {
			t.Errorf("unexpected sample count, expected %v, got %v", e, a)
		}

		if e, a := 5.0, m.GetHistogram().GetSampleSum(); e != a {
			t.Errorf("unexpected sample sum; expected %v, got %v", e, a)
		}
	}
}

var sinceFn = func(t time.Time) time.Duration {
	return 5 * time.Second
}

func keyFuncFor(name string) func(string) string {
	return func(s string) string {
		return name + "-" + s
	}
}

func expectedStatus(expectedErr bool) string {
	if expectedErr {
		return "failed"
	}

	return "success"
}
