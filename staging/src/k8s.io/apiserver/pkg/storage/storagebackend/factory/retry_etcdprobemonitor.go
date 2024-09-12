package factory

import (
	"context"

	"k8s.io/apiserver/pkg/storage/etcd3/etcd3retry"
	"k8s.io/apiserver/pkg/storage/etcd3/metrics"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

type proberMonitor interface {
	Prober
	metrics.Monitor
}

type etcd3RetryingProberMonitor struct {
	delegate proberMonitor
}

func newRetryingETCD3ProberMonitor(c storagebackend.Config) (*etcd3RetryingProberMonitor, error) {
	delegate, err := newETCD3ProberMonitor(c)
	if err != nil {
		return nil, err
	}
	return &etcd3RetryingProberMonitor{delegate: delegate}, nil
}

func (t *etcd3RetryingProberMonitor) Probe(ctx context.Context) error {
	return etcd3retry.OnError(ctx, etcd3retry.DefaultRetry, etcd3retry.IsRetriableEtcdError, func() error {
		return t.delegate.Probe(ctx)
	})
}

func (t *etcd3RetryingProberMonitor) Monitor(ctx context.Context) (metrics.StorageMetrics, error) {
	var ret metrics.StorageMetrics
	err := etcd3retry.OnError(ctx, etcd3retry.DefaultRetry, etcd3retry.IsRetriableEtcdError, func() error {
		var innerErr error
		ret, innerErr = t.delegate.Monitor(ctx)
		return innerErr
	})
	return ret, err
}

func (t *etcd3RetryingProberMonitor) Close() error {
	return t.delegate.Close()
}
