package service

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"

	"k8s.io/klog"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/kubernetes-csi/csi-test/v4/mock/cache"
	"golang.org/x/net/context"
	"google.golang.org/grpc/codes"

	"github.com/golang/protobuf/ptypes"

	"github.com/robertkrimen/otto"
)

const (
	// Name is the name of the CSI plug-in.
	Name = "io.kubernetes.storage.mock"

	// VendorVersion is the version returned by GetPluginInfo.
	VendorVersion = "0.3.0"

	// TopologyKey simulates a per-node topology.
	TopologyKey = Name + "/node"

	// TopologyValue is the one, fixed node on which the driver runs.
	TopologyValue = "some-mock-node"
)

// Manifest is the SP's manifest.
var Manifest = map[string]string{
	"url": "https://github.com/kubernetes-csi/csi-test/mock",
}

// JavaScript hooks to be run to perform various tests
type Hooks struct {
	Globals                        string `yaml:"globals"` // will be executed once before all other scripts
	CreateVolumeStart              string `yaml:"createVolumeStart"`
	CreateVolumeEnd                string `yaml:"createVolumeEnd"`
	DeleteVolumeStart              string `yaml:"deleteVolumeStart"`
	DeleteVolumeEnd                string `yaml:"deleteVolumeEnd"`
	ControllerPublishVolumeStart   string `yaml:"controllerPublishVolumeStart"`
	ControllerPublishVolumeEnd     string `yaml:"controllerPublishVolumeEnd"`
	ControllerUnpublishVolumeStart string `yaml:"controllerUnpublishVolumeStart"`
	ControllerUnpublishVolumeEnd   string `yaml:"controllerUnpublishVolumeEnd"`
	ValidateVolumeCapabilities     string `yaml:"validateVolumeCapabilities"`
	ListVolumesStart               string `yaml:"listVolumesStart"`
	ListVolumesEnd                 string `yaml:"listVolumesEnd"`
	GetCapacity                    string `yaml:"getCapacity"`
	ControllerGetCapabilitiesStart string `yaml:"controllerGetCapabilitiesStart"`
	ControllerGetCapabilitiesEnd   string `yaml:"controllerGetCapabilitiesEnd"`
	CreateSnapshotStart            string `yaml:"createSnapshotStart"`
	CreateSnapshotEnd              string `yaml:"createSnapshotEnd"`
	DeleteSnapshotStart            string `yaml:"deleteSnapshotStart"`
	DeleteSnapshotEnd              string `yaml:"deleteSnapshotEnd"`
	ListSnapshots                  string `yaml:"listSnapshots"`
	ControllerExpandVolumeStart    string `yaml:"controllerExpandVolumeStart"`
	ControllerExpandVolumeEnd      string `yaml:"controllerExpandVolumeEnd"`
	NodeStageVolumeStart           string `yaml:"nodeStageVolumeStart"`
	NodeStageVolumeEnd             string `yaml:"nodeStageVolumeEnd"`
	NodeUnstageVolumeStart         string `yaml:"nodeUnstageVolumeStart"`
	NodeUnstageVolumeEnd           string `yaml:"nodeUnstageVolumeEnd"`
	NodePublishVolumeStart         string `yaml:"nodePublishVolumeStart"`
	NodePublishVolumeEnd           string `yaml:"nodePublishVolumeEnd"`
	NodeUnpublishVolumeStart       string `yaml:"nodeUnpublishVolumeStart"`
	NodeUnpublishVolumeEnd         string `yaml:"nodeUnpublishVolumeEnd"`
	NodeExpandVolumeStart          string `yaml:"nodeExpandVolumeStart"`
	NodeExpandVolumeEnd            string `yaml:"nodeExpandVolumeEnd"`
	NodeGetCapabilities            string `yaml:"nodeGetCapabilities"`
	NodeGetInfo                    string `yaml:"nodeGetInfo"`
	NodeGetVolumeStatsStart        string `yaml:"nodeGetVolumeStatsStart"`
	NodeGetVolumeStatsEnd          string `yaml:"nodeGetVolumeStatsEnd"`
}

type Config struct {
	DisableAttach              bool
	DriverName                 string
	AttachLimit                int64
	NodeExpansionRequired      bool
	DisableControllerExpansion bool
	DisableOnlineExpansion     bool
	PermissiveTargetPath       bool
	EnableTopology             bool
	ExecHooks                  *Hooks
}

// Service is the CSI Mock service provider.
type Service interface {
	csi.ControllerServer
	csi.IdentityServer
	csi.NodeServer
}

type service struct {
	sync.Mutex
	nodeID       string
	vols         []csi.Volume
	volsRWL      sync.RWMutex
	volsNID      uint64
	snapshots    cache.SnapshotCache
	snapshotsNID uint64
	config       Config
	hooksVm      *otto.Otto
}

type Volume struct {
	VolumeCSI             csi.Volume
	NodeID                string
	ISStaged              bool
	ISPublished           bool
	ISEphemeral           bool
	ISControllerPublished bool
	StageTargetPath       string
	TargetPath            string
}

var MockVolumes map[string]Volume

// New returns a new Service.
func New(config Config) Service {
	s := &service{
		nodeID: config.DriverName,
		config: config,
	}
	if config.ExecHooks != nil {
		s.hooksVm = otto.New()
		s.hooksVm.Run(grpcJSCodes) // set global variables with gRPC error codes
		_, err := s.hooksVm.Run(s.config.ExecHooks.Globals)
		if err != nil {
			klog.Exitf("Error encountered in the global exec hook: %v. Exiting\n", err)
		}
	}
	s.snapshots = cache.NewSnapshotCache()
	s.vols = []csi.Volume{
		s.newVolume("Mock Volume 1", gib100),
		s.newVolume("Mock Volume 2", gib100),
		s.newVolume("Mock Volume 3", gib100),
	}
	MockVolumes = map[string]Volume{}

	s.snapshots.Add(s.newSnapshot("Mock Snapshot 1", "1", map[string]string{"Description": "snapshot 1"}))
	s.snapshots.Add(s.newSnapshot("Mock Snapshot 2", "2", map[string]string{"Description": "snapshot 2"}))
	s.snapshots.Add(s.newSnapshot("Mock Snapshot 3", "3", map[string]string{"Description": "snapshot 3"}))

	return s
}

const (
	kib    int64 = 1024
	mib    int64 = kib * 1024
	gib    int64 = mib * 1024
	gib100 int64 = gib * 100
	tib    int64 = gib * 1024
	tib100 int64 = tib * 100
)

func (s *service) newVolume(name string, capcity int64) csi.Volume {
	vol := csi.Volume{
		VolumeId:      fmt.Sprintf("%d", atomic.AddUint64(&s.volsNID, 1)),
		VolumeContext: map[string]string{"name": name},
		CapacityBytes: capcity,
	}
	s.setTopology(&vol)
	return vol
}

func (s *service) newVolumeFromSnapshot(name string, capacity int64, snapshotID int) csi.Volume {
	vol := s.newVolume(name, capacity)
	vol.ContentSource = &csi.VolumeContentSource{
		Type: &csi.VolumeContentSource_Snapshot{
			Snapshot: &csi.VolumeContentSource_SnapshotSource{
				SnapshotId: fmt.Sprintf("%d", snapshotID),
			},
		},
	}
	s.setTopology(&vol)
	return vol
}

func (s *service) newVolumeFromVolume(name string, capacity int64, volumeID int) csi.Volume {
	vol := s.newVolume(name, capacity)
	vol.ContentSource = &csi.VolumeContentSource{
		Type: &csi.VolumeContentSource_Volume{
			Volume: &csi.VolumeContentSource_VolumeSource{
				VolumeId: fmt.Sprintf("%d", volumeID),
			},
		},
	}
	s.setTopology(&vol)
	return vol
}

func (s *service) setTopology(vol *csi.Volume) {
	if s.config.EnableTopology {
		vol.AccessibleTopology = []*csi.Topology{
			&csi.Topology{
				Segments: map[string]string{
					TopologyKey: TopologyValue,
				},
			},
		}
	}
}

func (s *service) findVol(k, v string) (volIdx int, volInfo csi.Volume) {
	s.volsRWL.RLock()
	defer s.volsRWL.RUnlock()
	return s.findVolNoLock(k, v)
}

func (s *service) findVolNoLock(k, v string) (volIdx int, volInfo csi.Volume) {
	volIdx = -1

	for i, vi := range s.vols {
		switch k {
		case "id":
			if strings.EqualFold(v, vi.GetVolumeId()) {
				return i, vi
			}
		case "name":
			if n, ok := vi.VolumeContext["name"]; ok && strings.EqualFold(v, n) {
				return i, vi
			}
		}
	}

	return
}

func (s *service) findVolByName(
	ctx context.Context, name string) (int, csi.Volume) {

	return s.findVol("name", name)
}

func (s *service) findVolByID(
	ctx context.Context, id string) (int, csi.Volume) {

	return s.findVol("id", id)
}

func (s *service) newSnapshot(name, sourceVolumeId string, parameters map[string]string) cache.Snapshot {

	ptime := ptypes.TimestampNow()
	return cache.Snapshot{
		Name:       name,
		Parameters: parameters,
		SnapshotCSI: csi.Snapshot{
			SnapshotId:     fmt.Sprintf("%d", atomic.AddUint64(&s.snapshotsNID, 1)),
			CreationTime:   ptime,
			SourceVolumeId: sourceVolumeId,
			ReadyToUse:     true,
		},
	}
}

// getAttachCount returns the number of attached volumes on the node.
func (s *service) getAttachCount(devPathKey string) int64 {
	var count int64
	for _, v := range s.vols {
		if device := v.VolumeContext[devPathKey]; device != "" {
			count++
		}
	}
	return count
}

func (s *service) execHook(hookName string) (codes.Code, string) {
	if s.hooksVm != nil {
		script := reflect.ValueOf(*s.config.ExecHooks).FieldByName(hookName).String()
		if len(script) > 0 {
			result, err := s.hooksVm.Run(script)
			if err != nil {
				klog.Exitf("Exec hook %s error: %v; exiting\n", hookName, err)
			}
			rv, err := result.ToInteger()
			if err == nil {
				// Function returned an integer, use it
				return codes.Code(rv), fmt.Sprintf("Exec hook %s returned non-OK code", hookName)
			} else {
				// Function returned non-integer data type, discard it
				return codes.OK, ""
			}
		}
	}
	return codes.OK, ""
}
