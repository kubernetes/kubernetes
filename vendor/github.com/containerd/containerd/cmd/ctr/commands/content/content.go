package content

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/log"
	units "github.com/docker/go-units"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
	"github.com/urfave/cli"
)

var (
	// Command is the cli command for managing content
	Command = cli.Command{
		Name:  "content",
		Usage: "manage content",
		Subcommands: cli.Commands{
			activeIngestCommand,
			deleteCommand,
			editCommand,
			fetchCommand,
			fetchObjectCommand,
			getCommand,
			ingestCommand,
			listCommand,
			pushObjectCommand,
			setLabelsCommand,
		},
	}

	getCommand = cli.Command{
		Name:        "get",
		Usage:       "get the data for an object",
		ArgsUsage:   "[<digest>, ...]",
		Description: "display the image object",
		Action: func(context *cli.Context) error {
			dgst, err := digest.Parse(context.Args().First())
			if err != nil {
				return err
			}
			client, ctx, cancel, err := commands.NewClient(context)
			if err != nil {
				return err
			}
			defer cancel()
			cs := client.ContentStore()
			ra, err := cs.ReaderAt(ctx, dgst)
			if err != nil {
				return err
			}
			defer ra.Close()

			_, err = io.Copy(os.Stdout, content.NewReader(ra))
			return err
		},
	}

	ingestCommand = cli.Command{
		Name:        "ingest",
		Usage:       "accept content into the store",
		ArgsUsage:   "[flags] <key>",
		Description: "ingest objects into the local content store",
		Flags: []cli.Flag{
			cli.Int64Flag{
				Name:  "expected-size",
				Usage: "validate against provided size",
			},
			cli.StringFlag{
				Name:  "expected-digest",
				Usage: "verify content against expected digest",
			},
		},
		Action: func(context *cli.Context) error {
			var (
				ref            = context.Args().First()
				expectedSize   = context.Int64("expected-size")
				expectedDigest = digest.Digest(context.String("expected-digest"))
			)
			if err := expectedDigest.Validate(); expectedDigest != "" && err != nil {
				return err
			}
			if ref == "" {
				return errors.New("must specify a transaction reference")
			}
			client, ctx, cancel, err := commands.NewClient(context)
			if err != nil {
				return err
			}
			defer cancel()

			cs := client.ContentStore()

			// TODO(stevvooe): Allow ingest to be reentrant. Currently, we expect
			// all data to be written in a single invocation. Allow multiple writes
			// to the same transaction key followed by a commit.
			return content.WriteBlob(ctx, cs, ref, os.Stdin, expectedSize, expectedDigest)
		},
	}

	activeIngestCommand = cli.Command{
		Name:        "active",
		Usage:       "display active transfers",
		ArgsUsage:   "[flags] [<regexp>]",
		Description: "display the ongoing transfers",
		Flags: []cli.Flag{
			cli.DurationFlag{
				Name:   "timeout, t",
				Usage:  "total timeout for fetch",
				EnvVar: "CONTAINERD_FETCH_TIMEOUT",
			},
			cli.StringFlag{
				Name:  "root",
				Usage: "path to content store root",
				Value: "/tmp/content", // TODO(stevvooe): for now, just use the PWD/.content
			},
		},
		Action: func(context *cli.Context) error {
			match := context.Args().First()
			client, ctx, cancel, err := commands.NewClient(context)
			if err != nil {
				return err
			}
			defer cancel()
			cs := client.ContentStore()
			active, err := cs.ListStatuses(ctx, match)
			if err != nil {
				return err
			}
			tw := tabwriter.NewWriter(os.Stdout, 1, 8, 1, '\t', 0)
			fmt.Fprintln(tw, "REF\tSIZE\tAGE\t")
			for _, active := range active {
				fmt.Fprintf(tw, "%s\t%s\t%s\t\n",
					active.Ref,
					units.HumanSize(float64(active.Offset)),
					units.HumanDuration(time.Since(active.StartedAt)))
			}

			return tw.Flush()
		},
	}

	listCommand = cli.Command{
		Name:        "list",
		Aliases:     []string{"ls"},
		Usage:       "list all blobs in the store",
		ArgsUsage:   "[flags]",
		Description: "list blobs in the content store",
		Flags: []cli.Flag{
			cli.BoolFlag{
				Name:  "quiet, q",
				Usage: "print only the blob digest",
			},
		},
		Action: func(context *cli.Context) error {
			var (
				quiet = context.Bool("quiet")
				args  = []string(context.Args())
			)
			client, ctx, cancel, err := commands.NewClient(context)
			if err != nil {
				return err
			}
			defer cancel()
			cs := client.ContentStore()

			var walkFn content.WalkFunc
			if quiet {
				walkFn = func(info content.Info) error {
					fmt.Println(info.Digest)
					return nil
				}
			} else {
				tw := tabwriter.NewWriter(os.Stdout, 1, 8, 1, '\t', 0)
				defer tw.Flush()

				fmt.Fprintln(tw, "DIGEST\tSIZE\tAGE\tLABELS")
				walkFn = func(info content.Info) error {
					var labelStrings []string
					for k, v := range info.Labels {
						labelStrings = append(labelStrings, strings.Join([]string{k, v}, "="))
					}
					labels := strings.Join(labelStrings, ",")
					if labels == "" {
						labels = "-"
					}

					fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n",
						info.Digest,
						units.HumanSize(float64(info.Size)),
						units.HumanDuration(time.Since(info.CreatedAt)),
						labels)
					return nil
				}

			}

			return cs.Walk(ctx, walkFn, args...)
		},
	}

	setLabelsCommand = cli.Command{
		Name:        "label",
		Usage:       "add labels to content",
		ArgsUsage:   "<digest> [<label>=<value> ...]",
		Description: "labels blobs in the content store",
		Action: func(context *cli.Context) error {
			object, labels := commands.ObjectWithLabelArgs(context)
			client, ctx, cancel, err := commands.NewClient(context)
			if err != nil {
				return err
			}
			defer cancel()

			cs := client.ContentStore()

			dgst, err := digest.Parse(object)
			if err != nil {
				return err
			}

			info := content.Info{
				Digest: dgst,
				Labels: map[string]string{},
			}

			var paths []string
			for k, v := range labels {
				paths = append(paths, fmt.Sprintf("labels.%s", k))
				if v != "" {
					info.Labels[k] = v
				}
			}

			// Nothing updated, do no clear
			if len(paths) == 0 {
				info, err = cs.Info(ctx, info.Digest)
			} else {
				info, err = cs.Update(ctx, info, paths...)
			}
			if err != nil {
				return err
			}

			var labelStrings []string
			for k, v := range info.Labels {
				labelStrings = append(labelStrings, fmt.Sprintf("%s=%s", k, v))
			}

			fmt.Println(strings.Join(labelStrings, ","))

			return nil
		},
	}

	editCommand = cli.Command{
		Name:        "edit",
		Usage:       "edit a blob and return a new digest",
		ArgsUsage:   "[flags] <digest>",
		Description: "edit a blob and return a new digest",
		Flags: []cli.Flag{
			cli.StringFlag{
				Name:  "validate",
				Usage: "validate the result against a format (json, mediatype, etc.)",
			},
		},
		Action: func(context *cli.Context) error {
			var (
				validate = context.String("validate")
				object   = context.Args().First()
			)

			if validate != "" {
				return errors.New("validating the edit result not supported")
			}

			// TODO(stevvooe): Support looking up objects by a reference through
			// the image metadata storage.

			dgst, err := digest.Parse(object)
			if err != nil {
				return err
			}
			client, ctx, cancel, err := commands.NewClient(context)
			if err != nil {
				return err
			}
			defer cancel()
			cs := client.ContentStore()
			ra, err := cs.ReaderAt(ctx, dgst)
			if err != nil {
				return err
			}
			defer ra.Close()

			nrc, err := edit(content.NewReader(ra))
			if err != nil {
				return err
			}
			defer nrc.Close()

			wr, err := cs.Writer(ctx, "edit-"+object, 0, "") // TODO(stevvooe): Choose a better key?
			if err != nil {
				return err
			}

			if _, err := io.Copy(wr, nrc); err != nil {
				return err
			}

			if err := wr.Commit(ctx, 0, wr.Digest()); err != nil {
				return err
			}

			fmt.Println(wr.Digest())
			return nil
		},
	}

	deleteCommand = cli.Command{
		Name:      "delete",
		Aliases:   []string{"del", "remove", "rm"},
		Usage:     "permanently delete one or more blobs",
		ArgsUsage: "[<digest>, ...]",
		Description: `Delete one or more blobs permanently. Successfully deleted
	blobs are printed to stdout.`,
		Action: func(context *cli.Context) error {
			var (
				args      = []string(context.Args())
				exitError error
			)
			client, ctx, cancel, err := commands.NewClient(context)
			if err != nil {
				return err
			}
			defer cancel()
			cs := client.ContentStore()

			for _, arg := range args {
				dgst, err := digest.Parse(arg)
				if err != nil {
					if exitError == nil {
						exitError = err
					}
					log.G(ctx).WithError(err).Errorf("could not delete %v", dgst)
					continue
				}

				if err := cs.Delete(ctx, dgst); err != nil {
					if !errdefs.IsNotFound(err) {
						if exitError == nil {
							exitError = err
						}
						log.G(ctx).WithError(err).Errorf("could not delete %v", dgst)
					}
					continue
				}

				fmt.Println(dgst)
			}

			return exitError
		},
	}

	// TODO(stevvooe): Create "multi-fetch" mode that just takes a remote
	// then receives object/hint lines on stdin, returning content as
	// needed.
	fetchObjectCommand = cli.Command{
		Name:        "fetch-object",
		Usage:       "retrieve objects from a remote",
		ArgsUsage:   "[flags] <remote> <object> [<hint>, ...]",
		Description: `Fetch objects by identifier from a remote.`,
		Flags:       commands.RegistryFlags,
		Action: func(context *cli.Context) error {
			var (
				ref = context.Args().First()
			)
			ctx, cancel := commands.AppContext(context)
			defer cancel()

			resolver, err := commands.GetResolver(ctx, context)
			if err != nil {
				return err
			}

			ctx = log.WithLogger(ctx, log.G(ctx).WithField("ref", ref))

			log.G(ctx).Infof("resolving")
			name, desc, err := resolver.Resolve(ctx, ref)
			if err != nil {
				return err
			}
			fetcher, err := resolver.Fetcher(ctx, name)
			if err != nil {
				return err
			}

			log.G(ctx).Infof("fetching")
			rc, err := fetcher.Fetch(ctx, desc)
			if err != nil {
				return err
			}
			defer rc.Close()

			_, err = io.Copy(os.Stdout, rc)
			return err
		},
	}

	pushObjectCommand = cli.Command{
		Name:        "push-object",
		Usage:       "push an object to a remote",
		ArgsUsage:   "[flags] <remote> <object> <type>",
		Description: `Push objects by identifier to a remote.`,
		Flags:       commands.RegistryFlags,
		Action: func(context *cli.Context) error {
			var (
				ref    = context.Args().Get(0)
				object = context.Args().Get(1)
				media  = context.Args().Get(2)
			)
			dgst, err := digest.Parse(object)
			if err != nil {
				return err
			}
			client, ctx, cancel, err := commands.NewClient(context)
			if err != nil {
				return err
			}
			defer cancel()

			resolver, err := commands.GetResolver(ctx, context)
			if err != nil {
				return err
			}

			ctx = log.WithLogger(ctx, log.G(ctx).WithField("ref", ref))

			log.G(ctx).Infof("resolving")
			pusher, err := resolver.Pusher(ctx, ref)
			if err != nil {
				return err
			}

			cs := client.ContentStore()

			info, err := cs.Info(ctx, dgst)
			if err != nil {
				return err
			}
			desc := ocispec.Descriptor{
				MediaType: media,
				Digest:    dgst,
				Size:      info.Size,
			}

			ra, err := cs.ReaderAt(ctx, dgst)
			if err != nil {
				return err
			}
			defer ra.Close()

			cw, err := pusher.Push(ctx, desc)
			if err != nil {
				return err
			}

			// TODO: Progress reader
			if err := content.Copy(ctx, cw, content.NewReader(ra), desc.Size, desc.Digest); err != nil {
				return err
			}

			fmt.Printf("Pushed %s %s\n", desc.Digest, desc.MediaType)

			return nil
		},
	}
)

func edit(rd io.Reader) (io.ReadCloser, error) {
	tmp, err := ioutil.TempFile("", "edit-")
	if err != nil {
		return nil, err
	}

	if _, err := io.Copy(tmp, rd); err != nil {
		tmp.Close()
		return nil, err
	}

	cmd := exec.Command("sh", "-c", "$EDITOR "+tmp.Name())

	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()

	if err := cmd.Run(); err != nil {
		tmp.Close()
		return nil, err
	}

	if _, err := tmp.Seek(0, io.SeekStart); err != nil {
		tmp.Close()
		return nil, err
	}

	return onCloser{ReadCloser: tmp, onClose: func() error {
		return os.RemoveAll(tmp.Name())
	}}, nil
}

type onCloser struct {
	io.ReadCloser
	onClose func() error
}

func (oc onCloser) Close() error {
	var err error
	if err1 := oc.ReadCloser.Close(); err1 != nil {
		err = err1
	}

	if oc.onClose != nil {
		err1 := oc.onClose()
		if err == nil {
			err = err1
		}
	}

	return err
}
