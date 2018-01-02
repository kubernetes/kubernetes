package commands

import (
	gocontext "context"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/namespaces"
	"github.com/urfave/cli"
)

// AppContext returns the context for a command. Should only be called once per
// command, near the start.
//
// This will ensure the namespace is picked up and set the timeout, if one is
// defined.
func AppContext(context *cli.Context) (gocontext.Context, gocontext.CancelFunc) {
	var (
		ctx       = gocontext.Background()
		timeout   = context.GlobalDuration("timeout")
		namespace = context.GlobalString("namespace")
		cancel    gocontext.CancelFunc
	)
	ctx = namespaces.WithNamespace(ctx, namespace)
	if timeout > 0 {
		ctx, cancel = gocontext.WithTimeout(ctx, timeout)
	} else {
		ctx, cancel = gocontext.WithCancel(ctx)
	}
	return ctx, cancel
}

// NewClient returns a new containerd client
func NewClient(context *cli.Context) (*containerd.Client, gocontext.Context, gocontext.CancelFunc, error) {
	client, err := containerd.New(context.GlobalString("address"))
	if err != nil {
		return nil, nil, nil, err
	}
	ctx, cancel := AppContext(context)
	return client, ctx, cancel, nil
}
