package util

import (
	"fmt"
	"time"

	"github.com/digitalocean/godo"
)

const (
	// activeFailure is the amount of times we can fail before deciding
	// the check for active is a total failure. This can help account
	// for servers randomly not answering.
	activeFailure = 3
)

// WaitForActive waits for a droplet to become active
func WaitForActive(client *godo.Client, monitorURI string) error {
	if len(monitorURI) == 0 {
		return fmt.Errorf("create had no monitor uri")
	}

	completed := false
	failCount := 0
	for !completed {
		action, _, err := client.DropletActions.GetByURI(monitorURI)

		if err != nil {
			if failCount <= activeFailure {
				failCount++
				continue
			}
			return err
		}

		switch action.Status {
		case godo.ActionInProgress:
			time.Sleep(5 * time.Second)
		case godo.ActionCompleted:
			completed = true
		default:
			return fmt.Errorf("unknown status: [%s]", action.Status)
		}
	}

	return nil
}
