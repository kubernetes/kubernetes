//go:build windows

package hnslib

import (
	"github.com/Microsoft/hnslib/internal/hns"
)

type EndpointNotFoundError = hns.EndpointNotFoundError
type NetworkNotFoundError = hns.NetworkNotFoundError
