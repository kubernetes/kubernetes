package commands

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/containerd/containerd"
	"github.com/urfave/cli"
)

var (
	// SnapshotterFlags are cli flags specifying snapshotter names
	SnapshotterFlags = []cli.Flag{
		cli.StringFlag{
			Name:  "snapshotter",
			Usage: "snapshotter name. Empty value stands for the daemon default value.",
			Value: containerd.DefaultSnapshotter,
		},
	}

	// LabelFlag is a cli flag specifying labels
	LabelFlag = cli.StringSliceFlag{
		Name:  "label",
		Usage: "labels to attach to the image",
	}

	// RegistryFlags are cli flags specifying registry options
	RegistryFlags = []cli.Flag{
		cli.BoolFlag{
			Name:  "skip-verify,k",
			Usage: "skip SSL certificate validation",
		},
		cli.BoolFlag{
			Name:  "plain-http",
			Usage: "allow connections using plain HTTP",
		},
		cli.StringFlag{
			Name:  "user,u",
			Usage: "user[:password] Registry user and password",
		},
		cli.StringFlag{
			Name:  "refresh",
			Usage: "refresh token for authorization server",
		},
	}
)

// ObjectWithLabelArgs returns the first arg and a LabelArgs object
func ObjectWithLabelArgs(clicontext *cli.Context) (string, map[string]string) {
	var (
		first        = clicontext.Args().First()
		labelStrings = clicontext.Args().Tail()
	)

	return first, LabelArgs(labelStrings)
}

// LabelArgs returns a map of label key,value pairs
func LabelArgs(labelStrings []string) map[string]string {
	labels := make(map[string]string, len(labelStrings))
	for _, label := range labelStrings {
		parts := strings.SplitN(label, "=", 2)
		key := parts[0]
		value := "true"
		if len(parts) > 1 {
			value = parts[1]
		}

		labels[key] = value
	}

	return labels
}

// PrintAsJSON prints input in JSON format
func PrintAsJSON(x interface{}) {
	b, err := json.MarshalIndent(x, "", "    ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "can't marshal %+v as a JSON string: %v\n", x, err)
	}
	fmt.Println(string(b))
}
