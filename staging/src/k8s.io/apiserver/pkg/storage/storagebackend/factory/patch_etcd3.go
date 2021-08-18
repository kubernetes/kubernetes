package factory

import (
	"context"
	"fmt"
	"path"
	"sync/atomic"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

func newETCD3HealthCheck(c storagebackend.Config) (func() error, error) {
	// constructing the etcd v3 client blocks and times out if etcd is not available.
	// retry in a loop in the background until we successfully create the client, storing the client or error encountered
	// the health check is run every second and the result is stored for an immediate healthz response

	clientValue := &atomic.Value{}

	clientErrMsg := &atomic.Value{}
	clientErrMsg.Store("etcd client connection not yet established")

	go wait.PollUntil(time.Second, func() (bool, error) {
		client, err := newETCD3Client(c.Transport)
		if err != nil {
			clientErrMsg.Store(err.Error())
			return false, nil
		}
		clientValue.Store(client)
		clientErrMsg.Store("")
		return true, nil
	}, wait.NeverStop)

	healthzErrorMessage := &atomic.Value{}
	healthzErrorMessage.Store("etcd client connection not yet established")
	go wait.Until(func() {
		if errMsg := clientErrMsg.Load().(string); len(errMsg) > 0 {
			healthzErrorMessage.Store(errMsg)
			return
		}
		client := clientValue.Load().(*clientv3.Client)
		healthcheckTimeout := storagebackend.DefaultHealthcheckTimeout
		if c.HealthcheckTimeout != time.Duration(0) {
			healthcheckTimeout = c.HealthcheckTimeout
		}
		ctx, cancel := context.WithTimeout(context.Background(), healthcheckTimeout)
		defer cancel()
		// See https://github.com/etcd-io/etcd/blob/c57f8b3af865d1b531b979889c602ba14377420e/etcdctl/ctlv3/command/ep_command.go#L118
		_, err := client.Get(ctx, path.Join("/", c.Prefix, "health"))
		if err == nil {
			healthzErrorMessage.Store("")
			return
		}
		healthzErrorMessage.Store(err.Error())
	}, time.Second, wait.NeverStop)

	return func() error {
		if errMsg := healthzErrorMessage.Load().(string); len(errMsg) > 0 {
			return fmt.Errorf(errMsg)
		}
		return nil
	}, nil
}
