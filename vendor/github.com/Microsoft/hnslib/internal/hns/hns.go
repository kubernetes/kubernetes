package hns

import "fmt"

//go:generate go run github.com/Microsoft/go-winio/tools/mkwinsyscall -output zsyscall_windows.go hns.go

//sys _hnsCall(method string, path string, object string, response **uint16) (hr error) = vmcompute.HNSCall?

type EndpointNotFoundError struct {
	EndpointName string
}

func (e EndpointNotFoundError) Error() string {
	return fmt.Sprintf("Endpoint %s not found", e.EndpointName)
}

type NetworkNotFoundError struct {
	NetworkName string
}

func (e NetworkNotFoundError) Error() string {
	return fmt.Sprintf("Network %s not found", e.NetworkName)
}
