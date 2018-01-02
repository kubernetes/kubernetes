// +build !windows

package shim

import (
	"fmt"
	"io/ioutil"
	"net"
	"time"

	gocontext "context"

	"google.golang.org/grpc"

	"github.com/containerd/console"
	"github.com/containerd/containerd/cmd/ctr/commands"
	shim "github.com/containerd/containerd/linux/shim/v1"
	"github.com/containerd/typeurl"
	protobuf "github.com/gogo/protobuf/types"
	google_protobuf "github.com/golang/protobuf/ptypes/empty"
	"github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/urfave/cli"
)

var empty = &google_protobuf.Empty{}

var fifoFlags = []cli.Flag{
	cli.StringFlag{
		Name:  "stdin",
		Usage: "specify the path to the stdin fifo",
	},
	cli.StringFlag{
		Name:  "stdout",
		Usage: "specify the path to the stdout fifo",
	},
	cli.StringFlag{
		Name:  "stderr",
		Usage: "specify the path to the stderr fifo",
	},
	cli.BoolFlag{
		Name:  "tty,t",
		Usage: "enable tty support",
	},
}

// Command is the cli command for interacting with a shim
var Command = cli.Command{
	Name:  "shim",
	Usage: "interact with a shim directly",
	Flags: []cli.Flag{
		cli.StringFlag{
			Name:  "socket",
			Usage: "socket on which to connect to the shim",
		},
	},
	Subcommands: []cli.Command{
		deleteCommand,
		execCommand,
		startCommand,
		stateCommand,
	},
}

var startCommand = cli.Command{
	Name:  "start",
	Usage: "start a container with a shim",
	Action: func(context *cli.Context) error {
		service, err := getShimService(context)
		if err != nil {
			return err
		}
		_, err = service.Start(gocontext.Background(), &shim.StartRequest{
			ID: context.Args().First(),
		})
		return err
	},
}

var deleteCommand = cli.Command{
	Name:  "delete",
	Usage: "delete a container with a shim",
	Action: func(context *cli.Context) error {
		service, err := getShimService(context)
		if err != nil {
			return err
		}
		r, err := service.Delete(gocontext.Background(), empty)
		if err != nil {
			return err
		}
		fmt.Printf("container deleted and returned exit status %d\n", r.ExitStatus)
		return nil
	},
}

var stateCommand = cli.Command{
	Name:  "state",
	Usage: "get the state of all the processes of the shim",
	Action: func(context *cli.Context) error {
		service, err := getShimService(context)
		if err != nil {
			return err
		}
		r, err := service.State(gocontext.Background(), &shim.StateRequest{
			ID: context.Args().First(),
		})
		if err != nil {
			return err
		}
		commands.PrintAsJSON(r)
		return nil
	},
}

var execCommand = cli.Command{
	Name:  "exec",
	Usage: "exec a new process in the shim's container",
	Flags: append(fifoFlags,
		cli.BoolFlag{
			Name:  "attach,a",
			Usage: "stay attached to the container and open the fifos",
		},
		cli.StringSliceFlag{
			Name:  "env,e",
			Usage: "add environment vars",
			Value: &cli.StringSlice{},
		},
		cli.StringFlag{
			Name:  "cwd",
			Usage: "current working directory",
		},
		cli.StringFlag{
			Name:  "spec",
			Usage: "runtime spec",
		},
	),
	Action: func(context *cli.Context) error {
		service, err := getShimService(context)
		if err != nil {
			return err
		}
		var (
			id  = context.Args().First()
			ctx = gocontext.Background()
		)

		if id == "" {
			return errors.New("exec id must be provided")
		}

		tty := context.Bool("tty")
		wg, err := prepareStdio(context.String("stdin"), context.String("stdout"), context.String("stderr"), tty)
		if err != nil {
			return err
		}

		// read spec file and extract Any object
		spec, err := ioutil.ReadFile(context.String("spec"))
		if err != nil {
			return err
		}
		url, err := typeurl.TypeURL(specs.Process{})
		if err != nil {
			return err
		}

		rq := &shim.ExecProcessRequest{
			ID: id,
			Spec: &protobuf.Any{
				TypeUrl: url,
				Value:   spec,
			},
			Stdin:    context.String("stdin"),
			Stdout:   context.String("stdout"),
			Stderr:   context.String("stderr"),
			Terminal: tty,
		}
		if _, err := service.Exec(ctx, rq); err != nil {
			return err
		}
		r, err := service.Start(ctx, &shim.StartRequest{
			ID: id,
		})
		if err != nil {
			return err
		}
		fmt.Printf("exec running with pid %d\n", r.Pid)
		if context.Bool("attach") {
			logrus.Info("attaching")
			if tty {
				current := console.Current()
				defer current.Reset()
				if err := current.SetRaw(); err != nil {
					return err
				}
				size, err := current.Size()
				if err != nil {
					return err
				}
				if _, err := service.ResizePty(ctx, &shim.ResizePtyRequest{
					ID:     id,
					Width:  uint32(size.Width),
					Height: uint32(size.Height),
				}); err != nil {
					return err
				}
			}
			wg.Wait()
		}
		return nil
	},
}

func getShimService(context *cli.Context) (shim.ShimClient, error) {
	bindSocket := context.GlobalString("socket")
	if bindSocket == "" {
		return nil, errors.New("socket path must be specified")
	}

	dialOpts := []grpc.DialOption{grpc.WithInsecure(), grpc.WithTimeout(100 * time.Second)}
	dialOpts = append(dialOpts,
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", "\x00"+bindSocket, timeout)
		},
		))
	conn, err := grpc.Dial(fmt.Sprintf("unix://%s", bindSocket), dialOpts...)
	if err != nil {
		return nil, err
	}
	return shim.NewShimClient(conn), nil
}
