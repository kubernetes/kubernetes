package manager

import (
	"context"
	"log"
	"net"
	"os"
	"path"
	"sync"
	"time"

	"google.golang.org/grpc"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/logplugin/v1alpha1"
)

// Stub implementation for LogPlugin.
type LogPluginStub struct {
	socketPath string
	name       string

	server *grpc.Server
	stopCh chan interface{}
	wg     sync.WaitGroup

	configs map[string]*pluginapi.Config
	states  map[string]pluginapi.State
}

// NewLogPluginStub returns an initialized LogPlugin Stub.
func NewLogPluginStub(socketPath string, name string) *LogPluginStub {
	return &LogPluginStub{
		socketPath: socketPath,
		name:       name,
		stopCh:     make(chan interface{}),
		configs:    make(map[string]*pluginapi.Config),
		states:     make(map[string]pluginapi.State),
	}
}

// Start starts the gRPC server of the log plugin. Can only
// be called once.
func (p *LogPluginStub) Start() error {
	err := p.cleanup()
	if err != nil {
		return err
	}

	sock, err := net.Listen("unix", p.socketPath)
	if err != nil {
		return err
	}

	p.wg.Add(1)
	p.server = grpc.NewServer([]grpc.ServerOption{}...)
	pluginapi.RegisterLogPluginServer(p.server, p)

	go func() {
		defer p.wg.Done()
		p.server.Serve(sock)
	}()
	_, conn, err := dial(p.socketPath)
	if err != nil {
		return err
	}
	conn.Close()

	log.Println("starting to serve on", p.socketPath)

	return nil
}

// Stop stops the gRPC server. Can be called without a prior Start
// and more than once. Not safe to be called concurrently by different
// goroutines!
func (p *LogPluginStub) Stop() error {
	if p.server == nil {
		return nil
	}
	p.server.Stop()
	p.wg.Wait()
	p.server = nil
	close(p.stopCh) // This prevents re-starting the server.

	return p.cleanup()
}

// Register registers the log plugin with Kubelet.
func (p *LogPluginStub) Register(kubeletEndpoint, logPluginName string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, kubeletEndpoint, grpc.WithInsecure(), grpc.WithBlock(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}))
	if err != nil {
		return err
	}
	defer conn.Close()
	client := pluginapi.NewRegistrationClient(conn)
	req := &pluginapi.RegisterRequest{
		Version:  pluginapi.Version,
		Endpoint: path.Base(p.socketPath),
		Name:     logPluginName,
	}

	_, err = client.Register(context.Background(), req)
	if err != nil {
		return err
	}
	return nil
}

func (p *LogPluginStub) AddConfig(ctx context.Context, r *pluginapi.AddConfigRequest) (*pluginapi.AddConfigResponse, error) {
	p.configs[r.Config.Metadata.Name] = r.Config
	return &pluginapi.AddConfigResponse{
		Changed: true,
	}, nil
}

func (p *LogPluginStub) DelConfig(ctx context.Context, r *pluginapi.DelConfigRequest) (*pluginapi.DelConfigResponse, error) {
	delete(p.configs, r.Name)
	return &pluginapi.DelConfigResponse{
		Changed: true,
	}, nil
}

func (p *LogPluginStub) setState(name string, state pluginapi.State) {
	p.states[name] = state
}

func (p *LogPluginStub) GetState(ctx context.Context, r *pluginapi.GetStateRequest) (*pluginapi.GetStateResponse, error) {
	state, exists := p.states[r.Name]
	if !exists {
		state = pluginapi.State_NotFound
	}
	return &pluginapi.GetStateResponse{
		State: state,
	}, nil
}

func (p *LogPluginStub) ListConfig(ctx context.Context, r *pluginapi.Empty) (*pluginapi.ListConfigResponse, error) {
	configs := make([]*pluginapi.Config, 0, len(p.configs))
	for _, config := range p.configs {
		configs = append(configs, config)
	}
	return &pluginapi.ListConfigResponse{
		Configs: configs,
	}, nil
}

func (p *LogPluginStub) cleanup() error {
	if err := os.Remove(p.socketPath); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}
