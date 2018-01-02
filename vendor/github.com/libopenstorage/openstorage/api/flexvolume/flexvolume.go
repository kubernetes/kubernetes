package flexvolume

import (
	"errors"
	"fmt"
	"math"
	"net"
	
	"google.golang.org/grpc"

	"github.com/libopenstorage/openstorage/pkg/mount"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/libopenstorage/openstorage/volume/drivers"
	"github.com/libopenstorage/openstorage/pkg/flexvolume"
	
	"go.pedge.io/dlog"
)

const (
	osdDriverKey = "osdDriver"
	volumeIDKey  = "volumeID"
)

var (
	// ErrInvalidSpecDriver is returned when incorrect or no OSD driver is specified in spec file
	ErrInvalidSpecDriver = errors.New("Invalid kubernetes spec file : No OSD driver specified in flexvolume")
	// ErrInvalidSpecVolumeID is returned when incorrect or no volumeID is specified in spec file
	ErrInvalidSpecVolumeID = errors.New("Invalid kubernetes spec file : No volumeID specified in flexvolume")
	// ErrNoMountInfo is returned when flexvolume is unable to read from proc/self/mountinfo
	ErrNoMountInfo = errors.New("Could not read mountpoints from /proc/self/mountinfo. ")
	// ErrMissingMountPath is returned when no mountPath specified in mount call
	ErrMissingMountPath = errors.New("Missing target mount path")
	deviceDriverMap     = make(map[string]string)
)

type flexVolumeClient struct {
	defaultDriver string
}

func (c *flexVolumeClient) Init() error {
	return nil
}

func (c *flexVolumeClient) Attach(jsonOptions map[string]string) error {
	driverName, ok := jsonOptions[osdDriverKey]
	if !ok {
		return ErrInvalidSpecDriver
	}
	driver, err := c.getVolumeDriver(driverName)
	if err != nil {
		return err
	}
	mountDevice, ok := jsonOptions[volumeIDKey]
	if !ok {
		return ErrInvalidSpecVolumeID
	}
	if _, err := driver.Attach(mountDevice, nil); err != nil {
		return err
	}
	return nil
}

func (c *flexVolumeClient) Detach(mountDevice string, unmountBeforeDetach bool) error {
	driverName, ok := deviceDriverMap[mountDevice]
	if !ok {
		dlog.Infof("Could not find driver for (%v). Resorting to default driver ", mountDevice)
		driverName = c.defaultDriver
	}
	driver, err := c.getVolumeDriver(driverName)
	if err != nil {
		return err
	}
	if err := driver.Detach(mountDevice, unmountBeforeDetach); err != nil {
		return err
	}
	return nil
}

func (c *flexVolumeClient) Mount(targetMountDir string, mountDevice string,
	jsonOptions map[string]string) error {

	driverName, ok := jsonOptions[osdDriverKey]
	if !ok {
		return ErrInvalidSpecDriver
	}
	driver, err := c.getVolumeDriver(driverName)
	if err != nil {
		return err
	}
	if targetMountDir == "" {
		return ErrMissingMountPath
	}
	if err := driver.Mount(mountDevice, targetMountDir); err != nil {
		return err
	}
	// Update the deviceDriverMap
	mountManager, err := mount.New(mount.DeviceMount, nil, []string{""}, nil, []string{})
	if err != nil {
		dlog.Infof("Could not read mountpoints from /proc/self/mountinfo. Device - Driver mapping not saved!")
		return nil
	}
	sourcePath, err := mountManager.GetSourcePath(targetMountDir)
	if err != nil {
		return fmt.Errorf("Mount Failed. Could not find mountpoint in /proc/self/mountinfo")
	}
	deviceDriverMap[sourcePath] = driverName
	return nil
}

func (c *flexVolumeClient) Unmount(mountDir string) error {
	// Get the mountDevice from mount manager
	mountManager, err := mount.New(mount.DeviceMount, nil, []string{""}, nil, []string{})
	if err != nil {
		return ErrNoMountInfo
	}
	mountDevice, err := mountManager.GetSourcePath(mountDir)
	if err != nil {
		return fmt.Errorf("Invalid mountDir (%v). No device mounted on this path", mountDir)
	}
	driverName, ok := deviceDriverMap[mountDevice]
	if !ok {
		dlog.Infof("Could not find driver for (%v). Resorting to default driver. ", mountDevice)
		driverName = c.defaultDriver
	}
	driver, err := c.getVolumeDriver(driverName)
	if err != nil {
		return err
	}
	if err := driver.Unmount(mountDevice, mountDir); err != nil {
		return err
	}
	return nil
}

func newFlexVolumeClient(defaultDriver string) *flexVolumeClient {
	return &flexVolumeClient{
		defaultDriver: defaultDriver,
	}
}

func (c *flexVolumeClient) getVolumeDriver(driverName string) (volume.VolumeDriver, error) {
	return volumedrivers.Get(driverName)
}


// StartFlexVolumeAPI starts the flexvolume API on the given port.
func StartFlexVolumeAPI(port uint16, defaultDriver string) error {
	grpcServer := grpc.NewServer(grpc.MaxConcurrentStreams(math.MaxUint32))
	flexvolume.RegisterAPIServer(grpcServer, flexvolume.NewAPIServer(newFlexVolumeClient(defaultDriver)))
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return err
	}
	go func() {
		if err := grpcServer.Serve(listener); err != nil {
			dlog.Errorln(err.Error())
		}
	}()
	return nil
}
