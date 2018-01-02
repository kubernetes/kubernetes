// +build !windows

package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	eventsapi "github.com/containerd/containerd/api/services/events/v1"
	"github.com/containerd/containerd/dialer"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/events"
	"github.com/containerd/containerd/linux/shim"
	shimapi "github.com/containerd/containerd/linux/shim/v1"
	"github.com/containerd/containerd/reaper"
	"github.com/containerd/typeurl"
	google_protobuf "github.com/golang/protobuf/ptypes/empty"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
	"google.golang.org/grpc"
)

const usage = `
                    __        _                     __           __    _
  _________  ____  / /_____ _(_)___  ___  _________/ /     _____/ /_  (_)___ ___
 / ___/ __ \/ __ \/ __/ __ ` + "`" + `/ / __ \/ _ \/ ___/ __  /_____/ ___/ __ \/ / __ ` + "`" + `__ \
/ /__/ /_/ / / / / /_/ /_/ / / / / /  __/ /  / /_/ /_____(__  ) / / / / / / / / /
\___/\____/_/ /_/\__/\__,_/_/_/ /_/\___/_/   \__,_/     /____/_/ /_/_/_/ /_/ /_/

shim for container lifecycle and reconnection
`

var (
	debugFlag         bool
	namespaceFlag     string
	socketFlag        string
	addressFlag       string
	workdirFlag       string
	runtimeRootFlag   string
	criuFlag          string
	systemdCgroupFlag bool
)

func init() {
	flag.BoolVar(&debugFlag, "debug", false, "enable debug output in logs")
	flag.StringVar(&namespaceFlag, "namespace", "", "namespace that owns the shim")
	flag.StringVar(&socketFlag, "socket", "", "abstract socket path to serve")
	flag.StringVar(&addressFlag, "address", "", "grpc address back to main containerd")
	flag.StringVar(&workdirFlag, "workdir", "", "path used to storge large temporary data")
	flag.StringVar(&runtimeRootFlag, "runtime-root", shim.RuncRoot, "root directory for the runtime")
	flag.StringVar(&criuFlag, "criu", "", "path to criu binary")
	flag.BoolVar(&systemdCgroupFlag, "systemd-cgroup", false, "set runtime to use systemd-cgroup")
	flag.Parse()
}

func main() {
	if debugFlag {
		logrus.SetLevel(logrus.DebugLevel)
	}
	if err := executeShim(); err != nil {
		fmt.Fprintf(os.Stderr, "containerd-shim: %s\n", err)
		os.Exit(1)
	}
}

func executeShim() error {
	// start handling signals as soon as possible so that things are properly reaped
	// or if runtime exits before we hit the handler
	signals, err := setupSignals()
	if err != nil {
		return err
	}
	path, err := os.Getwd()
	if err != nil {
		return err
	}
	server := newServer()
	e, err := connectEvents(addressFlag)
	if err != nil {
		return err
	}
	sv, err := shim.NewService(
		shim.Config{
			Path:          path,
			Namespace:     namespaceFlag,
			WorkDir:       workdirFlag,
			Criu:          criuFlag,
			SystemdCgroup: systemdCgroupFlag,
			RuntimeRoot:   runtimeRootFlag,
		},
		&remoteEventsPublisher{client: e},
	)
	if err != nil {
		return err
	}
	logrus.Debug("registering grpc server")
	shimapi.RegisterShimServer(server, sv)
	socket := socketFlag
	if err := serve(server, socket); err != nil {
		return err
	}
	logger := logrus.WithFields(logrus.Fields{
		"pid":       os.Getpid(),
		"path":      path,
		"namespace": namespaceFlag,
	})
	return handleSignals(logger, signals, server, sv)
}

// serve serves the grpc API over a unix socket at the provided path
// this function does not block
func serve(server *grpc.Server, path string) error {
	var (
		l   net.Listener
		err error
	)
	if path == "" {
		l, err = net.FileListener(os.NewFile(3, "socket"))
		path = "[inherited from parent]"
	} else {
		l, err = net.Listen("unix", "\x00"+path)
	}
	if err != nil {
		return err
	}
	logrus.WithField("socket", path).Debug("serving api on unix socket")
	go func() {
		defer l.Close()
		if err := server.Serve(l); err != nil &&
			!strings.Contains(err.Error(), "use of closed network connection") {
			logrus.WithError(err).Fatal("containerd-shim: GRPC server failure")
		}
	}()
	return nil
}

func handleSignals(logger *logrus.Entry, signals chan os.Signal, server *grpc.Server, sv *shim.Service) error {
	var (
		termOnce sync.Once
		done     = make(chan struct{})
	)

	for {
		select {
		case <-done:
			return nil
		case s := <-signals:
			switch s {
			case unix.SIGCHLD:
				if err := reaper.Reap(); err != nil {
					logger.WithError(err).Error("reap exit status")
				}
			case unix.SIGTERM, unix.SIGINT:
				go termOnce.Do(func() {
					server.Stop()
					// Ensure our child is dead if any
					sv.Kill(context.Background(), &shimapi.KillRequest{
						Signal: uint32(syscall.SIGKILL),
						All:    true,
					})
					sv.Delete(context.Background(), &google_protobuf.Empty{})
					close(done)
				})
			case unix.SIGUSR1:
				dumpStacks(logger)
			}
		}
	}
}

func dumpStacks(logger *logrus.Entry) {
	var (
		buf       []byte
		stackSize int
	)
	bufferLen := 16384
	for stackSize == len(buf) {
		buf = make([]byte, bufferLen)
		stackSize = runtime.Stack(buf, true)
		bufferLen *= 2
	}
	buf = buf[:stackSize]
	logger.Infof("=== BEGIN goroutine stack dump ===\n%s\n=== END goroutine stack dump ===", buf)
}

func connectEvents(address string) (eventsapi.EventsClient, error) {
	conn, err := connect(address, dialer.Dialer)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to dial %q", address)
	}
	return eventsapi.NewEventsClient(conn), nil
}

func connect(address string, d func(string, time.Duration) (net.Conn, error)) (*grpc.ClientConn, error) {
	gopts := []grpc.DialOption{
		grpc.WithBlock(),
		grpc.WithInsecure(),
		grpc.WithTimeout(60 * time.Second),
		grpc.WithDialer(d),
		grpc.FailOnNonTempDialError(true),
		grpc.WithBackoffMaxDelay(3 * time.Second),
	}
	conn, err := grpc.Dial(dialer.DialAddress(address), gopts...)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to dial %q", address)
	}
	return conn, nil
}

type remoteEventsPublisher struct {
	client eventsapi.EventsClient
}

func (l *remoteEventsPublisher) Publish(ctx context.Context, topic string, event events.Event) error {
	encoded, err := typeurl.MarshalAny(event)
	if err != nil {
		return err
	}
	if _, err := l.client.Publish(ctx, &eventsapi.PublishRequest{
		Topic: topic,
		Event: encoded,
	}); err != nil {
		return errdefs.FromGRPC(err)
	}
	return nil
}
