package flexvolume

import (
	"fmt"
	"os"

	"go.pedge.io/dlog"
	"github.com/golang/protobuf/ptypes/empty"
	"golang.org/x/net/context"
)

type client struct {
	apiClient APIClient
}

const (
	volumeIDKey = "volumeID"
)

var (
	successBytes = []byte(`{"Status":"Success"}`)
)

func newClient(apiClient APIClient) *client {
	return &client{apiClient}
}

func (c *client) Init() error {
	_, err := c.apiClient.Init(
		context.Background(),
		&empty.Empty{},
	)
	return err
}

func (c *client) Attach(jsonOptions map[string]string) error {
	_, err := c.apiClient.Attach(
		context.Background(),
		&AttachRequest{
			JsonOptions: jsonOptions,
		},
	)
	if err == nil {
		writeOutput(newAttachSuccessOutput(jsonOptions[volumeIDKey]))
	} else {
		writeOutput(newFailureBytes(err))
	}
	return err
}

func (c *client) Detach(mountDevice string, unmountBeforeDetach bool) error {
	_, err := c.apiClient.Detach(
		context.Background(),
		&DetachRequest{
			MountDevice: mountDevice,
		},
	)
	writeOutput(newOutput(err))
	return err
}

func (c *client) Mount(targetMountDir string, mountDevice string, jsonOptions map[string]string) error {
	if err := os.MkdirAll(targetMountDir, os.ModeDir); err != nil {
		writeOutput(newOutput(err))
		return err
	}
	_, err := c.apiClient.Mount(
		context.Background(),
		&MountRequest{
			TargetMountDir: targetMountDir,
			MountDevice:    mountDevice,
			JsonOptions:    jsonOptions,
		},
	)
	writeOutput(newOutput(err))
	return err
}

func (c *client) Unmount(mountDir string) error {
	_, err := c.apiClient.Unmount(
		context.Background(),
		&UnmountRequest{
			MountDir: mountDir,
		},
	)
	writeOutput(newOutput(err))
	return err
}

func newFailureBytes(err error) []byte {
	return []byte(fmt.Sprintf(`{"Status":"Failure", "Message":"%s"}`, err.Error()))
}

func newOutput(err error) []byte {
	if err != nil {
		return newFailureBytes(err)
	}
	return successBytes
}

func newAttachSuccessOutput(deviceID string) []byte {
	return []byte(fmt.Sprintf(`{"Status":"Success", "Device":"%s"}`, deviceID))
}

func writeOutput(output []byte) {
	if _, err := os.Stdout.Write(output); err != nil {
		dlog.Warnf("Unable to write output to stdout : %s", err.Error())
	}
}
