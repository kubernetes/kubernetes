// Package client provides a command-line interface for Docker.
//
// Run "docker help SUBCOMMAND" or "docker SUBCOMMAND --help" to see more information on any Docker subcommand, including the full list of options supported for the subcommand.
// See https://docs.docker.com/installation/ for instructions on installing Docker.
package client

import "fmt"

// An StatusError reports an unsuccessful exit by a command.
type StatusError struct {
	Status     string
	StatusCode int
}

func (e StatusError) Error() string {
	return fmt.Sprintf("Status: %s, Code: %d", e.Status, e.StatusCode)
}
