package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/containers"
	"github.com/containerd/containerd/namespaces"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
	"github.com/urfave/cli"
)

const imageName = "docker.io/library/alpine:latest"

func main() {
	// morr power!
	runtime.GOMAXPROCS(runtime.NumCPU())

	app := cli.NewApp()
	app.Name = "containerd-stress"
	app.Description = "stress test a containerd daemon"
	app.Flags = []cli.Flag{
		cli.BoolFlag{
			Name:  "debug",
			Usage: "set debug output in the logs",
		},
		cli.StringFlag{
			Name:  "address,a",
			Value: "/run/containerd/containerd.sock",
			Usage: "path to the containerd socket",
		},
		cli.IntFlag{
			Name:  "concurrent,c",
			Value: 1,
			Usage: "set the concurrency of the stress test",
		},
		cli.DurationFlag{
			Name:  "duration,d",
			Value: 1 * time.Minute,
			Usage: "set the duration of the stress test",
		},
	}
	app.Before = func(context *cli.Context) error {
		if context.GlobalBool("debug") {
			logrus.SetLevel(logrus.DebugLevel)
		}
		return nil
	}
	app.Action = func(context *cli.Context) error {
		config := config{
			Address:     context.GlobalString("address"),
			Duration:    context.GlobalDuration("duration"),
			Concurrency: context.GlobalInt("concurrent"),
		}
		return test(config)
	}
	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

type config struct {
	Concurrency int
	Duration    time.Duration
	Address     string
}

func (c config) newClient() (*containerd.Client, error) {
	return containerd.New(c.Address)
}

func test(c config) error {
	var (
		wg  sync.WaitGroup
		ctx = namespaces.WithNamespace(context.Background(), "stress")
	)

	client, err := c.newClient()
	if err != nil {
		return err
	}
	defer client.Close()
	if err := cleanup(ctx, client); err != nil {
		return err
	}
	logrus.Infof("pulling %s", imageName)
	image, err := client.Pull(ctx, imageName, containerd.WithPullUnpack)
	if err != nil {
		return err
	}
	logrus.Info("generating spec from image")
	spec, err := containerd.GenerateSpec(ctx, client, &containers.Container{ID: ""}, containerd.WithImageConfig(image), containerd.WithProcessArgs("true"))
	if err != nil {
		return err
	}
	tctx, cancel := context.WithTimeout(ctx, c.Duration)
	go func() {
		s := make(chan os.Signal, 1)
		signal.Notify(s, syscall.SIGTERM, syscall.SIGINT)
		<-s
		cancel()
	}()

	var (
		workers []*worker
		start   = time.Now()
	)
	logrus.Info("starting stress test run...")
	for i := 0; i < c.Concurrency; i++ {
		wg.Add(1)
		w := &worker{
			id:     i,
			wg:     &wg,
			spec:   *spec,
			image:  image,
			client: client,
		}
		workers = append(workers, w)
		go w.run(ctx, tctx)
	}
	wg.Wait()

	var (
		total    int
		failures int
		end      = time.Now().Sub(start).Seconds()
	)
	logrus.Infof("ending test run in %0.3f seconds", end)
	for _, w := range workers {
		total += w.count
		failures += w.failures
	}
	logrus.WithField("failures", failures).Infof(
		"create/start/delete %d containers in %0.3f seconds (%0.3f c/sec) or (%0.3f sec/c)",
		total,
		end,
		float64(total)/end,
		end/float64(total),
	)
	return nil
}

type worker struct {
	id          int
	wg          *sync.WaitGroup
	count       int
	failures    int
	waitContext context.Context

	client *containerd.Client
	image  containerd.Image
	spec   specs.Spec
}

func (w *worker) run(ctx, tctx context.Context) {
	defer func() {
		w.wg.Done()
		logrus.Infof("worker %d finished", w.id)
	}()
	wctx, cancel := context.WithCancel(ctx)
	w.waitContext = wctx
	go func() {
		<-tctx.Done()
		cancel()
	}()
	for {
		select {
		case <-tctx.Done():
			return
		default:
		}

		w.count++
		id := w.getID()
		logrus.Debugf("starting container %s", id)
		if err := w.runContainer(ctx, id); err != nil {
			if err != context.DeadlineExceeded ||
				!strings.Contains(err.Error(), context.DeadlineExceeded.Error()) {
				w.failures++
				logrus.WithError(err).Errorf("running container %s", id)

			}
		}
	}
}

func (w *worker) runContainer(ctx context.Context, id string) error {
	w.spec.Linux.CgroupsPath = filepath.Join("/", fmt.Sprint(w.id), id)
	c, err := w.client.NewContainer(ctx, id,
		containerd.WithSpec(&w.spec),
		containerd.WithNewSnapshot(id, w.image),
	)
	if err != nil {
		return err
	}
	defer c.Delete(ctx, containerd.WithSnapshotCleanup)

	task, err := c.NewTask(ctx, containerd.NullIO)
	if err != nil {
		return err
	}
	defer task.Delete(ctx, containerd.WithProcessKill)

	statusC, err := task.Wait(ctx)
	if err != nil {
		return err
	}

	if err := task.Start(ctx); err != nil {
		return err
	}
	status := <-statusC
	_, _, err = status.Result()
	if err != nil {
		if err == context.DeadlineExceeded || err == context.Canceled {
			return nil
		}
		w.failures++
	}
	return nil
}

func (w *worker) getID() string {
	return fmt.Sprintf("%d-%d", w.id, w.count)
}

func (w *worker) cleanup(ctx context.Context, c containerd.Container) {
	if err := c.Delete(ctx, containerd.WithSnapshotCleanup); err != nil {
		if err == context.DeadlineExceeded {
			return
		}
		w.failures++
		logrus.WithError(err).Errorf("delete container %s", c.ID())
	}
}

// cleanup cleans up any containers in the "stress" namespace before the test run
func cleanup(ctx context.Context, client *containerd.Client) error {
	containers, err := client.Containers(ctx)
	if err != nil {
		return err
	}
	for _, c := range containers {
		task, err := c.Task(ctx, nil)
		if err == nil {
			task.Delete(ctx, containerd.WithProcessKill)
		}
		if err := c.Delete(ctx, containerd.WithSnapshotCleanup); err != nil {
			if derr := c.Delete(ctx); derr == nil {
				continue
			}
			return err
		}
	}
	return nil
}
