package factory

import (
	"context"
	"fmt"
	"testing"

	etcdrpc "go.etcd.io/etcd/api/v3/v3rpc/rpctypes"

	"k8s.io/apiserver/pkg/storage/etcd3/metrics"
)

func getRetryScenarios() []struct {
	name               string
	retryFnError       func() error
	expectedRetries    int
	expectedFinalError error
} {
	return []struct {
		name               string
		retryFnError       func() error
		expectedRetries    int
		expectedFinalError error
	}{
		{
			name: "retry ErrLeaderChanged",
			retryFnError: func() error {
				return etcdrpc.ErrLeaderChanged
			},
			expectedRetries:    5,
			expectedFinalError: etcdrpc.ErrLeaderChanged,
		},
		{
			name: "retry ErrLeaderChanged a few times",
			retryFnError: func() func() error {
				retryCounter := -1
				return func() error {
					retryCounter++
					if retryCounter == 3 {
						return nil
					}
					return etcdrpc.ErrLeaderChanged
				}
			}(),
			expectedRetries: 3,
		},
		{
			name: "no retries",
			retryFnError: func() error {
				return nil
			},
		},
		{
			name: "no retries for a random error",
			retryFnError: func() error {
				return fmt.Errorf("random error")
			},
			expectedFinalError: fmt.Errorf("random error"),
		},
	}
}

func TestEtcd3RetryingProber(t *testing.T) {
	for _, scenario := range getRetryScenarios() {
		t.Run(scenario.name, func(t *testing.T) {
			ctx := context.TODO()
			targetDelegate := &fakeEtcd3RetryingProberMonitor{
				// we set it to -1 to indicate that the first
				// execution is not a retry
				actualRetries: -1,
				probeFn:       scenario.retryFnError,
			}

			target := &etcd3RetryingProberMonitor{delegate: targetDelegate}
			err := target.Probe(ctx)

			if targetDelegate.actualRetries != scenario.expectedRetries {
				t.Errorf("Unexpected number of retries %v, expected %v", targetDelegate.actualRetries, scenario.expectedRetries)
			}
			if (err == nil && scenario.expectedFinalError != nil) || (err != nil && scenario.expectedFinalError == nil) {
				t.Errorf("Expected error %v, got %v", scenario.expectedFinalError, err)
			}
			if err != nil && scenario.expectedFinalError != nil && err.Error() != scenario.expectedFinalError.Error() {
				t.Errorf("Expected error %v, got %v", scenario.expectedFinalError, err)
			}
		})
	}
}

func TestEtcd3RetryingMonitor(t *testing.T) {
	for _, scenario := range getRetryScenarios() {
		t.Run(scenario.name, func(t *testing.T) {
			ctx := context.TODO()
			expectedRetValue := int64(scenario.expectedRetries)
			targetDelegate := &fakeEtcd3RetryingProberMonitor{
				// we set it to -1 to indicate that the first
				// execution is not a retry
				actualRetries: -1,
				monitorFn: func() func() (metrics.StorageMetrics, error) {
					retryCounter := -1
					return func() (metrics.StorageMetrics, error) {
						retryCounter++
						err := scenario.retryFnError()
						ret := metrics.StorageMetrics{int64(retryCounter)}
						return ret, err
					}
				}(),
			}

			target := &etcd3RetryingProberMonitor{delegate: targetDelegate}
			actualRetValue, err := target.Monitor(ctx)

			if targetDelegate.actualRetries != scenario.expectedRetries {
				t.Errorf("Unexpected number of retries %v, expected %v", targetDelegate.actualRetries, scenario.expectedRetries)
			}
			if (err == nil && scenario.expectedFinalError != nil) || (err != nil && scenario.expectedFinalError == nil) {
				t.Errorf("Expected error %v, got %v", scenario.expectedFinalError, err)
			}
			if err != nil && scenario.expectedFinalError != nil && err.Error() != scenario.expectedFinalError.Error() {
				t.Errorf("Expected error %v, got %v", scenario.expectedFinalError, err)
			}
			if actualRetValue.Size != expectedRetValue {
				t.Errorf("Unexpected value returned actual %v, expected %v", actualRetValue.Size, expectedRetValue)
			}
		})
	}
}

type fakeEtcd3RetryingProberMonitor struct {
	actualRetries int
	probeFn       func() error
	monitorFn     func() (metrics.StorageMetrics, error)
}

func (f *fakeEtcd3RetryingProberMonitor) Probe(_ context.Context) error {
	f.actualRetries++
	return f.probeFn()
}

func (f *fakeEtcd3RetryingProberMonitor) Monitor(_ context.Context) (metrics.StorageMetrics, error) {
	f.actualRetries++
	return f.monitorFn()
}

func (f *fakeEtcd3RetryingProberMonitor) Close() error {
	panic("not implemented")
}
