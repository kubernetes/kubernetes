package manager

import (
	"context"
	"fmt"
	"github.com/golang/glog"
	"google.golang.org/grpc"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/logplugin/v1alpha1"
	"net"
	"sync"
	"time"
)

// endpointStopGracePeriod indicates the grace period after an endpoint is stopped
// because its log plugin fails. LogPluginManager keeps the stopped endpoint in its
// cache during this grace period to cover the time gap for the capacity change to
// take effect.
const endpointStopGracePeriod = time.Duration(5) * time.Minute
const gRPCTimeout = 10 * time.Second

// endpoint maps to a single registered log plugin. It is responsible
// for managing gRPC communications with the log plugin and caching
// states reported by the log plugin.
type pluginEndpoint interface {
	run()
	stop()

	addConfig(config *pluginapi.Config) (*pluginapi.AddConfigResponse, error)
	delConfig(name string) (*pluginapi.DelConfigResponse, error)
	listConfig() (*pluginapi.ListConfigResponse, error)
	getState(name string) (*pluginapi.GetStateResponse, error)
	name() string
}

type pluginEndpointImpl struct {
	client     pluginapi.LogPluginClient
	clientConn *grpc.ClientConn
	stopTime   time.Time
	mutex      sync.Mutex

	socketPath    string
	logPluginName string
}

// NewEndpoint creates a new endpoint for the given resourceName.
// This is to be used during normal log plugin registration.
func newEndpoint(socketPath, logPluginName string) (pluginEndpoint, error) {
	return newEndpointImpl(socketPath, logPluginName)
}

func newEndpointImpl(socketPath, logPluginName string) (*pluginEndpointImpl, error) {
	client, c, err := dial(socketPath)
	if err != nil {
		glog.Errorf("can't create new endpoint with path %s err %v", socketPath, err)
		return nil, err
	}

	return &pluginEndpointImpl{
		client:        client,
		clientConn:    c,
		socketPath:    socketPath,
		logPluginName: logPluginName,
	}, nil
}

// newStoppedEndpointImpl creates a new endpoint for the given logPluginName with stopTime set.
// This is to be used during Kubelet restart, before the actual log plugin re-registers.
func newStoppedEndpointImpl(logPluginName string) *pluginEndpointImpl {
	return &pluginEndpointImpl{
		logPluginName: logPluginName,
		stopTime:      time.Now(),
	}
}

func (e *pluginEndpointImpl) isStopped() bool {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	return !e.stopTime.IsZero()
}

func (e *pluginEndpointImpl) stopGracePeriodExpired() bool {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	return !e.stopTime.IsZero() && time.Since(e.stopTime) > endpointStopGracePeriod
}

// used for testing only
func (e *pluginEndpointImpl) setStopTime(t time.Time) {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	e.stopTime = t
}

func (e *pluginEndpointImpl) run() {

}

func (e *pluginEndpointImpl) stop() {
	e.mutex.Lock()
	defer e.mutex.Unlock()
	if e.clientConn != nil {
		e.clientConn.Close()
	}
	e.stopTime = time.Now()
}

// dial establishes the gRPC communication with the registered log plugin.
func dial(unixSocketPath string) (pluginapi.LogPluginClient, *grpc.ClientConn, error) {
	ctx, cancel := context.WithTimeout(context.Background(), gRPCTimeout)
	defer cancel()
	c, err := grpc.DialContext(ctx, unixSocketPath, grpc.WithInsecure(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("dial to %s error, %v", unixSocketPath, err)
	}

	return pluginapi.NewLogPluginClient(c), c, nil
}

func (e *pluginEndpointImpl) addConfig(config *pluginapi.Config) (*pluginapi.AddConfigResponse, error) {
	if e.isStopped() {
		return nil, fmt.Errorf("endpoint %s is stopped", e.logPluginName)
	}
	ctx, cancel := context.WithTimeout(context.Background(), gRPCTimeout)
	defer cancel()
	return e.client.AddConfig(ctx, &pluginapi.AddConfigRequest{
		Config: config,
	})
}

func (e *pluginEndpointImpl) delConfig(name string) (*pluginapi.DelConfigResponse, error) {
	if e.isStopped() {
		return nil, fmt.Errorf("endpoint %s is stopped", e.logPluginName)
	}
	ctx, cancel := context.WithTimeout(context.Background(), gRPCTimeout)
	defer cancel()
	return e.client.DelConfig(ctx, &pluginapi.DelConfigRequest{
		Name: name,
	})
}

func (e *pluginEndpointImpl) listConfig() (*pluginapi.ListConfigResponse, error) {
	if e.isStopped() {
		return nil, fmt.Errorf("endpoint %s is stopped", e.logPluginName)
	}
	ctx, cancel := context.WithTimeout(context.Background(), gRPCTimeout)
	defer cancel()
	return e.client.ListConfig(ctx, &pluginapi.Empty{})
}

func (e *pluginEndpointImpl) getState(name string) (*pluginapi.GetStateResponse, error) {
	if e.isStopped() {
		return nil, fmt.Errorf("endpoint %s is stopped", e.logPluginName)
	}
	ctx, cancel := context.WithTimeout(context.Background(), gRPCTimeout)
	defer cancel()
	return e.client.GetState(ctx, &pluginapi.GetStateRequest{
		Name: name,
	})
}

func (e *pluginEndpointImpl) name() string {
	return e.logPluginName
}
