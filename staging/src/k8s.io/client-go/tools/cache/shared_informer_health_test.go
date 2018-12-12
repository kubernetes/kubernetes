/*
Copyright 2018 The Kubernetes Authors.

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

package cache

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
)

func TestListWatchLockFreeCheckMethod(t *testing.T) {
	errorCh := make(chan error)
	target := newDefaultListWatchLockFreeHealth("", errorCh)
	expectedCheckMsg := "seen 80 errors for 10m0s, the last 2 errors are = [error 1, error 2]"

	tc := clock.NewFakeClock(time.Now())
	tc.Step(time.Duration(-10) * time.Minute)
	target.startingTime = tc.Now().UnixNano()
	target.consistentErrors = 80
	errBuf := target.errCircularBuffer.Load().(errBuffer)
	errBuf.add(fmt.Errorf("error 1"))
	errBuf.add(fmt.Errorf("error 2"))
	target.errCircularBuffer.Store(errBuf)

	result := target.check(nil)
	if result == nil {
		t.Fatal("expected to receive error but none was returned")
	}
	if expectedCheckMsg != result.Error() {
		t.Fatalf("expected to get \"%s\" but received \"%s\"", expectedCheckMsg, result.Error())
	}
}

func TestListWatchLockFreeCheckInvalidatel(t *testing.T) {
	errorCh := make(chan error)
	wg := sync.WaitGroup{}
	heartBeat := make(chan struct{}, 1)
	ctx, ctxCancel := context.WithCancel(context.Background())
	target := newDefaultListWatchLockFreeHealth("", errorCh)
	target.minTimeToFail = time.Duration(1 * time.Second)

	tc := clock.NewFakeClock(time.Now())
	tc.Step(time.Duration(-10) * time.Minute)
	target.startingTime = tc.Now().UnixNano()
	target.consistentErrors = 80
	errBuf := target.errCircularBuffer.Load().(errBuffer)
	errBuf.add(fmt.Errorf("error 1"))
	errBuf.add(fmt.Errorf("error 2"))
	target.errCircularBuffer.Store(errBuf)

	// target should report bad health condition
	result := target.check(nil)
	if result == nil {
		t.Fatal("expected to receive error but none was returned")
	}

	// sending nil error invalidates the steady state
	target.run(ctx.Done())

	wg.Add(1)
	go errorsProducer(ctx.Done(), errorCh, heartBeat, &wg, func() error {
		return nil
	})

	<-heartBeat
	time.Sleep(target.minTimeToFail * 4)
	ctxCancel()
	wg.Wait()

	// target must report good health condition
	result = target.check(nil)
	if result != nil {
		t.Fatalf("got error %v but didn't expect to get one", result)
	}
	if target.consistentErrors != 0 {
		t.Fatalf("target.consistentErrors has unexpected value %v expected 0", target.consistentErrors)
	}
	actualErrBuf := target.errCircularBuffer.Load().(errBuffer)
	if len(actualErrBuf.get()) > 0 {
		t.Fatalf("expected to find errBufer empty but it contains %d elements", len(actualErrBuf.get()))
	}
}

func TestListWatchLockFreeCheckInternalMethod(t *testing.T) {
	type tuple struct {
		errCount int
		errs     []error
	}
	globalCounter := 0
	ctx, ctxCancel := context.WithCancel(context.Background())
	errorCh := make(chan error)
	heartBeat := make(chan struct{}, 2)
	testResults := []tuple{}
	wg := sync.WaitGroup{}

	// act
	target := newDefaultListWatchLockFreeHealth("", errorCh)
	target.minTimeToFail = time.Duration(1 * time.Second)
	target.run(ctx.Done())

	wg.Add(2)
	go errorsProducer(ctx.Done(), errorCh, heartBeat, &wg, func() error {
		time.Sleep(target.minTimeToFail / time.Duration(target.minConsistentErrors*2))
		globalCounter++
		if globalCounter > int(target.minConsistentErrors)*4 {
			globalCounter = 0
		}
		if globalCounter == 0 {
			return nil
		}
		return errors.New(fmt.Sprintf("%v", globalCounter))
	})

	go func() {
		defer wg.Done()
		heartBeat <- struct{}{}
		for {
			select {
			case <-ctx.Done():
				return
			default:
				time.Sleep(target.minTimeToFail / 2)
				if errorsCount, _, errors := target.checkInternal(); errorsCount > 0 {
					testResults = append(testResults, tuple{errorsCount, errors})
				}
			}
		}
	}()

	<-heartBeat
	<-heartBeat
	time.Sleep(target.minTimeToFail * 4)
	ctxCancel()
	wg.Wait()

	// validate
	//
	// The test results must contain at least 2 elements.
	// The test scenario increases the global counter at least 20 (target.minTimeToFail / time.Duration(target.minConsistentErrors * 2)) times per second.
	// It probes the results at least 2 (target.minTimeToFail / 2) times per second and the test runs for 4 second.
	// It means it tries to collect the results at least 8 times.
	// We can rule out 4 of them straight away because we expect to report failure when the number of errors is > 10 for at least 1 minute.
	// We can rule out more because every other second we reset the global counter.
	if len(testResults) < 2 {
		t.Fatalf("expected to receive at least = 2 results , got = %d", len(testResults))
	}
	for _, actualTuple := range testResults {
		if actualTuple.errCount < 20 || actualTuple.errCount > int(target.minConsistentErrors)*4 {
			t.Fatalf("expected that actualErrors will be in (20, 40) range, actual value is %d", actualTuple.errCount)
		}
		for _, errRcv := range actualTuple.errs {
			errValue, err := strconv.Atoi(errRcv.Error())
			if err != nil {
				t.Fatal(err)
			}
			if errValue > actualTuple.errCount {
				t.Fatalf("expected that the error value = %d, will be not be lesser than the total errors count = %d", errValue, actualTuple.errCount)
			}

		}
	}
}

func errorsProducer(stopCh <-chan struct{}, consumer chan<- error, heartBeat chan<- struct{}, wg *sync.WaitGroup, generator func() error) {
	defer wg.Done()
	heartBeat <- struct{}{}
	for {
		select {
		case <-stopCh:
			return
		case consumer <- generator():
		}
	}
}

func TestErrBuff(t *testing.T) {
	strToErrFunc := func(msg ...string) []error {
		ret := []error{}
		for _, s := range msg {
			ret = append(ret, errors.New(s))
		}
		return ret
	}
	scenarios := []struct {
		name       string
		size       int
		itemsToAdd int
		output     []error
	}{

		{
			name:       "scenario 1: adds exactly \"size\" items to the buffer - no wrapping",
			size:       5,
			itemsToAdd: 5,
			output:     strToErrFunc("0", "1", "2", "3", "4"),
		},
		{
			name:       "scenario 2: adds more than \"size\" items to the buffer - wrapping",
			size:       5,
			itemsToAdd: 6,
			output:     strToErrFunc("1", "2", "3", "4", "5"),
		},
		{
			name:       "scenario 3: adds twice \"size\" items to the buffer - wrapping",
			size:       5,
			itemsToAdd: 10,
			output:     strToErrFunc("5", "6", "7", "8", "9"),
		},
		{
			name:       "scenario 4: adds twice + 3 \"size\" items to the buffer - wrapping",
			size:       5,
			itemsToAdd: 13,
			output:     strToErrFunc("8", "9", "10", "11", "12"),
		},
		{
			name:       "scenario 5: adds less than \"size\" items to the buffer - no wrapping",
			size:       5,
			itemsToAdd: 3,
			output:     strToErrFunc("0", "1", "2"),
		},
		{
			name:       "scenario 6: adds 0 items to the buffer - no wrapping",
			size:       5,
			itemsToAdd: 0,
			output:     strToErrFunc(),
		},
		{
			name:       "scenario 7: adds 1984 items to the buffer - wrapping",
			size:       5,
			itemsToAdd: 1984,
			output:     strToErrFunc("1979", "1980", "1981", "1982", "1983"),
		},
		{
			name:       "scenario 8: adds 99 items to the buffer - wrapping - size 1",
			size:       1,
			itemsToAdd: 99,
			output:     strToErrFunc("98"),
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {

			target := newErrBuffer(scenario.size)
			for i := 0; i < scenario.itemsToAdd; i++ {
				target.add(strToErrFunc(fmt.Sprintf("%d", i))[0])
			}

			actualOutput := target.get()
			if !reflect.DeepEqual(actualOutput, scenario.output) {
				t.Fatalf("expected %v got %v", scenario.output, actualOutput)
			}
		})
	}

}

func TestErrBuffZero(t *testing.T) {
	// add some errors
	target := newErrBuffer(5)
	for i := 0; i < 100; i++ {
		target.add(fmt.Errorf("%d", i))
	}

	// validate if errors were added
	{
		output := []error{
			fmt.Errorf("%s", "95"),
			fmt.Errorf("%s", "96"),
			fmt.Errorf("%s", "97"),
			fmt.Errorf("%s", "98"),
			fmt.Errorf("%s", "99"),
		}
		actualOutput := target.get()
		if !reflect.DeepEqual(actualOutput, output) {
			t.Fatalf("expected %v got %v", output, actualOutput)
		}
	}

	// zero buffer and validate if it's empty
	target = target.zero()
	{
		output := []error{}
		actualOutput := target.get()
		if !reflect.DeepEqual(actualOutput, output) {
			t.Fatalf("expected %v got %v", output, actualOutput)
		}

	}
}

func TestErrBuffCopy(t *testing.T) {
	// add some errors
	target := newErrBuffer(5)
	for i := 0; i < 200; i++ {
		target.add(fmt.Errorf("%d", i))
	}

	// validate if errors were added
	{
		output := []error{
			fmt.Errorf("%s", "195"),
			fmt.Errorf("%s", "196"),
			fmt.Errorf("%s", "197"),
			fmt.Errorf("%s", "198"),
			fmt.Errorf("%s", "199"),
		}
		actualOutput := target.get()
		if !reflect.DeepEqual(actualOutput, output) {
			t.Fatalf("expected %v got %v", output, actualOutput)
		}
	}

	// make a copy of the buffer and validate if changing the original buffer doesn't change the copy
	targetCpy := target.copy()
	for i := 0; i < 10; i++ {
		target.add(fmt.Errorf("%d", i))
	}
	{
		output := []error{
			fmt.Errorf("%s", "5"),
			fmt.Errorf("%s", "6"),
			fmt.Errorf("%s", "7"),
			fmt.Errorf("%s", "8"),
			fmt.Errorf("%s", "9"),
		}
		actualOutput := target.get()
		if !reflect.DeepEqual(actualOutput, output) {
			t.Fatalf("expected %v got %v", output, actualOutput)
		}
	}
	{
		output := []error{
			fmt.Errorf("%s", "195"),
			fmt.Errorf("%s", "196"),
			fmt.Errorf("%s", "197"),
			fmt.Errorf("%s", "198"),
			fmt.Errorf("%s", "199"),
		}
		actualOutput := targetCpy.get()
		if !reflect.DeepEqual(actualOutput, output) {
			t.Fatalf("expected %v got %v", output, actualOutput)
		}

	}
}
