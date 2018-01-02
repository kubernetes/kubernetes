package images

import (
	"io"
	"os"

	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/containerd/containerd/reference"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
	"github.com/urfave/cli"
)

var exportCommand = cli.Command{
	Name:        "export",
	Usage:       "export an image",
	ArgsUsage:   "[flags] <out> <image>",
	Description: "export an image to a tar stream",
	Flags: []cli.Flag{
		cli.StringFlag{
			Name:  "oci-ref-name",
			Value: "",
			Usage: "override org.opencontainers.image.ref.name annotation",
		},
		cli.StringFlag{
			Name:  "manifest",
			Usage: "digest of manifest",
		},
		cli.StringFlag{
			Name:  "manifest-type",
			Usage: "media type of manifest digest",
			Value: ocispec.MediaTypeImageManifest,
		},
	},
	Action: func(context *cli.Context) error {
		var (
			out   = context.Args().First()
			local = context.Args().Get(1)
			desc  ocispec.Descriptor
		)
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		if manifest := context.String("manifest"); manifest != "" {
			desc.Digest, err = digest.Parse(manifest)
			if err != nil {
				return errors.Wrap(err, "invalid manifest digest")
			}
			desc.MediaType = context.String("manifest-type")
		} else {
			img, err := client.ImageService().Get(ctx, local)
			if err != nil {
				return errors.Wrap(err, "unable to resolve image to manifest")
			}
			desc = img.Target
		}

		if desc.Annotations == nil {
			desc.Annotations = make(map[string]string)
		}
		if s, ok := desc.Annotations[ocispec.AnnotationRefName]; !ok || s == "" {
			if ociRefName := determineOCIRefName(local); ociRefName != "" {
				desc.Annotations[ocispec.AnnotationRefName] = ociRefName
			}
			if ociRefName := context.String("oci-ref-name"); ociRefName != "" {
				desc.Annotations[ocispec.AnnotationRefName] = ociRefName
			}
		}
		var w io.WriteCloser
		if out == "-" {
			w = os.Stdout
		} else {
			w, err = os.Create(out)
			if err != nil {
				return nil
			}
		}
		r, err := client.Export(ctx, desc)
		if err != nil {
			return err
		}
		if _, err := io.Copy(w, r); err != nil {
			return err
		}
		if err := w.Close(); err != nil {
			return err
		}
		return r.Close()
	},
}

func determineOCIRefName(local string) string {
	refspec, err := reference.Parse(local)
	if err != nil {
		return ""
	}
	tag, _ := reference.SplitObject(refspec.Object)
	return tag
}
