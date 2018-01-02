package events

import (
	"encoding/json"
	"fmt"

	eventsapi "github.com/containerd/containerd/api/services/events/v1"
	"github.com/containerd/containerd/cmd/ctr/commands"
	"github.com/containerd/typeurl"
	"github.com/urfave/cli"
)

// Command is the cli command for displaying containerd events
var Command = cli.Command{
	Name:  "events",
	Usage: "display containerd events",
	Action: func(context *cli.Context) error {
		client, ctx, cancel, err := commands.NewClient(context)
		if err != nil {
			return err
		}
		defer cancel()
		eventsClient := client.EventService()
		events, err := eventsClient.Subscribe(ctx, &eventsapi.SubscribeRequest{
			Filters: context.Args(),
		})
		if err != nil {
			return err
		}
		for {
			e, err := events.Recv()
			if err != nil {
				return err
			}

			var out []byte
			if e.Event != nil {
				v, err := typeurl.UnmarshalAny(e.Event)
				if err != nil {
					return err
				}
				out, err = json.Marshal(v)
				if err != nil {
					return err
				}
			}

			if _, err := fmt.Println(
				e.Timestamp,
				e.Namespace,
				e.Topic,
				string(out),
			); err != nil {
				return err
			}
		}
	},
}
