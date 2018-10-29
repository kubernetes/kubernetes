// +build linux

package blkio

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"

	dockertypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/golang/glog"
	"github.com/opencontainers/runc/libcontainer/configs"
	"k8s.io/apimachinery/pkg/api/resource"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

// var cgrouproot = "/sys/fs/zmp/test"

var ErrorNotFound error

// FakeFileStat
type FakeFileStat struct {
	name    string
	size    int64
	mode    os.FileMode
	modTime time.Time
	// sys     unix.Stat_t
	sys syscall.Stat_t
}

func (fs *FakeFileStat) Size() int64        { return fs.size }
func (fs *FakeFileStat) Mode() os.FileMode  { return fs.mode }
func (fs *FakeFileStat) ModTime() time.Time { return fs.modTime }
func (fs *FakeFileStat) Sys() interface{}   { return &fs.sys }
func (fs *FakeFileStat) Name() string       { return fs.name }
func (fs *FakeFileStat) IsDir() bool        { return false }

// FakeSubSystem
type FakeSubSystem struct {
	name         string
	expectPath   string
	expectcgroup *configs.Cgroup
}

func (f *FakeSubSystem) Name() string { return f.name }
func (f *FakeSubSystem) Set(path string, cgroup *configs.Cgroup) error {
	if f.expectPath != path {
		return fmt.Errorf("Set CGroup failed. cgroup path: %v, expect cgroup path is :%v ", path, f.expectPath)
	}
	if path != cgroup.Path {
		return fmt.Errorf("Set CGroup failed. set cgroup path: %v, configs.Cgroup.path is : %v", path, cgroup.Path)
	}

	if err := cgroupMustEqual(f.expectcgroup, cgroup); err != nil {
		return fmt.Errorf("Set CGroup failed. set cgroup expect cgroup : %+s", err.Error())
	}
	fmt.Printf("Successfull. Set CGroup: %+v cgroup:%+v Resources:%+v \n", path, cgroup, *cgroup.Resources)

	return nil
}
func weightDeviceEqual(A, B []*configs.WeightDevice) bool {
	for _, a := range A {
		if !findWeightDevice(a, B) {
			return false
		}
	}
	for _, b := range B {
		if !findWeightDevice(b, A) {
			return false
		}
	}
	return true
}
func findWeightDevice(w *configs.WeightDevice, wds []*configs.WeightDevice) bool {
	for _, wd := range wds {
		if *w == *wd {
			return true
		}
	}
	return false
}
func findThrottleDevice(t *configs.ThrottleDevice, tds []*configs.ThrottleDevice) bool {
	for _, td := range tds {
		if *t == *td {
			return true
		}
	}
	return false
}
func throttleDeviceEqual(A, B []*configs.ThrottleDevice) bool {
	for _, a := range A {
		if !findThrottleDevice(a, B) {
			return false
		}
	}
	for _, b := range B {
		if !findThrottleDevice(b, A) {
			return false
		}
	}
	return true
}
func cgroupMustEqual(A, B *configs.Cgroup) error {
	if A.Name != B.Name || A.Parent != B.Parent || A.Path != B.Path || A.ScopePrefix != B.ScopePrefix {
		return fmt.Errorf("cgroup not equal. A:%+v, B:%+v", *A, *B)
	}
	if weightDeviceEqual(A.Resources.BlkioWeightDevice, B.Resources.BlkioWeightDevice) {
		return fmt.Errorf("cgroupMustEqual BlkioWeightDevice A : %+v, B: %+v",
			A.Resources.BlkioWeightDevice, B.Resources.BlkioWeightDevice)
	}
	if throttleDeviceEqual(A.Resources.BlkioThrottleReadBpsDevice, B.Resources.BlkioThrottleReadBpsDevice) {
		return fmt.Errorf("cgroupMustEqual BlkioThrottleReadBpsDevice A : %+v, B: %+v",
			A.Resources.BlkioThrottleReadBpsDevice, B.Resources.BlkioThrottleReadBpsDevice)
	}
	if throttleDeviceEqual(A.Resources.BlkioThrottleWriteBpsDevice, B.Resources.BlkioThrottleWriteBpsDevice) {
		return fmt.Errorf("cgroupMustEqual BlkioThrottleWriteBpsDevice A : %+v, B: %+v",
			A.Resources.BlkioThrottleWriteBpsDevice, B.Resources.BlkioThrottleWriteBpsDevice)
	}
	if throttleDeviceEqual(A.Resources.BlkioThrottleReadIOPSDevice, B.Resources.BlkioThrottleReadIOPSDevice) {
		return fmt.Errorf("cgroupMustEqual BlkioThrottleReadIOPSDevice A : %+v, B: %+v",
			A.Resources.BlkioThrottleReadIOPSDevice, B.Resources.BlkioThrottleReadIOPSDevice)
	}
	if throttleDeviceEqual(A.Resources.BlkioThrottleWriteIOPSDevice, B.Resources.BlkioThrottleWriteIOPSDevice) {
		return fmt.Errorf("cgroupMustEqual BlkioThrottleWriteIOPSDevice A : %+v, B: %+v",
			A.Resources.BlkioThrottleWriteIOPSDevice, B.Resources.BlkioThrottleWriteIOPSDevice)
	}

	return nil

}

//
type BlkioMockBuilder struct {
	ContainnerID string
	Fsinfo       map[string]*FakeFileStat
	ExpectCroup  *configs.Cgroup
	CGrouproot   string
	CGroupParent string
	ContainerMap *dockertypes.ContainerJSON
	SandboxMap   *dockertypes.ContainerJSON
}

func (b *BlkioMockBuilder) setMock() (docker libdocker.Interface, err error) {
	expectCgroupPath := filepath.Join(b.CGrouproot, "blkio", b.CGroupParent, b.ContainerMap.ID)
	unixos = &containertest.FakeOS{
		StatFn: func(name string) (info os.FileInfo, err error) {
			info, ok := b.Fsinfo[name]
			if !ok {
				glog.Errorf("Notfound the file %s, Fsinfo:%+v", name, b.Fsinfo)
				return info, ErrorNotFound
			}
			return info, nil
		},
	}
	blkioSubsystem = &FakeSubSystem{
		name:         BlkioSubsystemName,
		expectcgroup: b.ExpectCroup,
		expectPath:   expectCgroupPath,
	}
	FindCgroupMountpointDir = func() (string, error) {
		return b.CGrouproot, nil
	}

	ld := libdocker.NewFakeDockerClient()
	ld.ContainerMap[b.ContainerMap.ID] = b.ContainerMap
	ld.ContainerMap[b.SandboxMap.ID] = b.SandboxMap

	return ld, nil
}

func getMockBuilder(containerID, sandboxID, roofsDevName, cgrouprootPath, cgroupParent string, blkioCfg *Blkio, ecgroup *configs.Cgroup, fstats map[string]*FakeFileStat) (data BlkioMockBuilder) {
	blkioData, _ := json.Marshal(blkioCfg)
	data = BlkioMockBuilder{
		ContainnerID: containerID,
		CGrouproot:   cgrouprootPath,
		CGroupParent: cgroupParent,
		Fsinfo:       fstats,
		ExpectCroup:  ecgroup,
		ContainerMap: &dockertypes.ContainerJSON{
			ContainerJSONBase: &dockertypes.ContainerJSONBase{
				ID: containerID,
				GraphDriver: dockertypes.GraphDriverData{
					Data: map[string]string{
						GraphDriverDeviceIdKey:   "9856789098",
						GraphDriverDeviceNameKey: roofsDevName,
						GraphDriverDeviceSizeKey: "98765445678",
					},
					Name: GraphDriverName,
				},
				HostConfig: &container.HostConfig{
					Resources: container.Resources{
						CgroupParent: cgroupParent,
					},
				},
			},
			Config: &container.Config{
				Labels: map[string]string{
					containerTypeLabelKey: containerTypeLabelContainer,
					sandboxIDLabelKey:     sandboxID,
				},
			},
		},
		SandboxMap: &dockertypes.ContainerJSON{
			ContainerJSONBase: &dockertypes.ContainerJSONBase{
				ID: sandboxID,
				HostConfig: &container.HostConfig{
					Resources: container.Resources{
						CgroupParent: cgroupParent,
					},
				},
			},
			Config: &container.Config{
				Labels: map[string]string{
					containerTypeLabelKey: containerTypeLabelSandbox,
					BlkioKey:              string(blkioData),
				},
			},
		},
	}
	return
}

func buildSubSystemData(filedatas []DeviceCgroupMeta, weight uint16, rootfsDevName, cgroupRootPath, cgroupParent, containerID string) (*configs.Cgroup, *Blkio, map[string]*FakeFileStat, error) {
	Blkiocfg := Blkio{
		Weight: uint16(87),
	}
	fileStats := make(map[string]*FakeFileStat)
	fileStats[cgroupRootPath] = &FakeFileStat{
		name: cgroupRootPath, //
		size: int64(876567),
		mode: os.FileMode(7777),
		sys: syscall.Stat_t{
			Rdev: uint64(76557734),
		},
	}

	blkioResource := configs.Resources{}
	blkioResource.BlkioWeight = weight
	for _, filedata := range filedatas {
		DevicePath := filedata.DevicePath
		if DevicePath == "rootfs" {
			DevicePath = filepath.Join("/dev/mapper/", rootfsDevName)
		}
		fileStats[DevicePath] = &FakeFileStat{
			name: DevicePath, //
			size: filedata.Size,
			mode: filedata.Mode,
			sys: syscall.Stat_t{
				Rdev: filedata.Rdev,
			},
		}
		for key, value := range filedata.CGSet {
			switch key {
			case "weight_device":
				quantity, err := resource.ParseQuantity(value)
				if err != nil {
					return nil, nil, nil, err
				}
				q := quantity.Value()
				wd := &configs.WeightDevice{}
				wd.Major = int64((filedata.Rdev >> 24) & 0xff)
				wd.Minor = int64(filedata.Rdev & 0xffffff)
				wd.Weight = uint16(q & 0x00ffff)
				blkioResource.BlkioWeightDevice = append(blkioResource.BlkioWeightDevice, wd)

				Blkiocfg.WeightDevice = append(Blkiocfg.WeightDevice,
					deviceValue{
						Device: filedata.DevicePath,
						Value:  value,
					})

			case "device_read_bps":
				quantity, err := resource.ParseQuantity(value)
				if err != nil {
					return nil, nil, nil, err
				}
				q := quantity.Value()
				td := &configs.ThrottleDevice{}
				td.Major = int64((filedata.Rdev >> 24) & 0xff)
				td.Minor = int64(filedata.Rdev & 0xffffff)
				td.Rate = uint64(q)

				blkioResource.BlkioThrottleReadBpsDevice = append(blkioResource.BlkioThrottleReadBpsDevice, td)

				Blkiocfg.DeviceReadBps = append(Blkiocfg.DeviceReadBps,
					deviceValue{
						Device: filedata.DevicePath,
						Value:  value,
					})

			case "device_write_bps":
				quantity, err := resource.ParseQuantity(value)
				if err != nil {
					return nil, nil, nil, err
				}
				q := quantity.Value()
				td := &configs.ThrottleDevice{}
				td.Major = int64((filedata.Rdev >> 24) & 0xff)
				td.Minor = int64(filedata.Rdev & 0xffffff)
				td.Rate = uint64(q)
				blkioResource.BlkioThrottleWriteBpsDevice = append(blkioResource.BlkioThrottleWriteBpsDevice, td)

				Blkiocfg.DeviceWriteBps = append(Blkiocfg.DeviceWriteBps,
					deviceValue{
						Device: filedata.DevicePath,
						Value:  value,
					})
			case "device_read_iops":
				quantity, err := resource.ParseQuantity(value)
				if err != nil {
					return nil, nil, nil, err
				}
				q := quantity.Value()
				td := &configs.ThrottleDevice{}
				td.Major = int64((filedata.Rdev >> 24) & 0xff)
				td.Minor = int64(filedata.Rdev & 0xffffff)
				td.Rate = uint64(q)
				blkioResource.BlkioThrottleReadIOPSDevice = append(blkioResource.BlkioThrottleReadIOPSDevice, td)
				Blkiocfg.DeviceReadIOps = append(Blkiocfg.DeviceReadIOps,
					deviceValue{
						Device: filedata.DevicePath,
						Value:  value,
					})
			case "device_write_iops":
				quantity, err := resource.ParseQuantity(value)
				if err != nil {
					return nil, nil, nil, err
				}
				q := quantity.Value()
				td := &configs.ThrottleDevice{}
				td.Major = int64((filedata.Rdev >> 24) & 0xff)
				td.Minor = int64(filedata.Rdev & 0xffffff)
				td.Rate = uint64(q)
				blkioResource.BlkioThrottleWriteIOPSDevice = append(blkioResource.BlkioThrottleWriteIOPSDevice, td)
				Blkiocfg.DeviceWriteIOps = append(Blkiocfg.DeviceWriteIOps,
					deviceValue{
						Device: filedata.DevicePath,
						Value:  value,
					})
			}
		}
	}

	expectcgroup := configs.Cgroup{
		Path:      filepath.Join(cgroupRootPath, "blkio", cgroupParent, containerID),
		Resources: &blkioResource,
	}
	return &expectcgroup, &Blkiocfg, fileStats, nil
}

type DeviceCgroupMeta struct {
	CGroupBlkioType string
	DevicePath      string
	CGSet           map[string]string
	Rdev            uint64
	Mode            os.FileMode
	Size            int64
}

func getMockBuilderTables() ([]BlkioMockBuilder, error) {
	ErrorNotFound = fmt.Errorf("Not found")
	MockBuilderTables := []BlkioMockBuilder{}
	cgroupRootPath := "/sys/fs/cgroup" + strconv.FormatInt(time.Now().Unix(), 10)
	for i := 0; i < 10; i++ {
		roofsDevName := "docker-253:2-6029320-525bf5183e7a4753a719b8eca03c06ed4bafbac66477bde246f6bd6f36ffb981-" + strconv.Itoa(i)
		rootrdev := uint64(0x10106655) + uint64(i)
		containerID := "JamesBryce-Avanpourm1-98766194yfkjhgfqGHGF-" + strconv.Itoa(i)
		sandboxID := "container-JamesBryce-Avanpourm-01-" + strconv.Itoa(i)
		cgroupParent := "/kubepods/podd6f49e33-d7a7-11e8-ae45-c81f66bda7a1-" + strconv.Itoa(i)
		DeviceCgroupMetas := testCase(rootrdev, i)
		weight := uint16(876 + i)
		CG, blkiocfg, Fsinfos, err := buildSubSystemData(DeviceCgroupMetas, weight, roofsDevName, cgroupRootPath, cgroupParent, containerID)
		if err != nil {
			return nil, err
		}

		MockBuilderTables = append(MockBuilderTables,
			getMockBuilder(containerID, sandboxID, roofsDevName, cgroupRootPath, cgroupParent, blkiocfg, CG, Fsinfos))
	}
	return MockBuilderTables, nil
}

func testCase(rootrdev uint64, i int) []DeviceCgroupMeta {
	DeviceCgroupMetas := []DeviceCgroupMeta{
		{
			DevicePath: "rootfs",
			CGSet: map[string]string{
				"weight_device":     "123" + strconv.Itoa(i),
				"device_read_bps":   "125" + strconv.Itoa(i) + "k",
				"device_write_bps":  "125" + strconv.Itoa(i) + "M",
				"device_read_iops":  "325" + strconv.Itoa(i) + "G",
				"device_write_iops": "525" + strconv.Itoa(i) + "k",
			},
			Rdev: rootrdev,
			Mode: os.FileMode(777),
			Size: int64(987654567) + int64(i),
		},
		{
			DevicePath: "/dev/mpa-9",
			CGSet: map[string]string{
				// "weight": "823" + strconv.Itoa(i),
				"weight_device":     "129" + strconv.Itoa(i),
				"device_read_bps":   "135" + strconv.Itoa(i) + "k",
				"device_write_bps":  "225" + strconv.Itoa(i) + "m",
				"device_read_iops":  "325" + strconv.Itoa(i) + "M",
				"device_write_iops": "325" + strconv.Itoa(i) + "G",
			},
			Rdev: uint64(0x20206657) + uint64(i),
			Mode: os.FileMode(777),
			Size: int64(187654567) + int64(i),
		},
		{
			DevicePath: "/dev/mpa-8",
			CGSet: map[string]string{
				"weight_device":     "229" + strconv.Itoa(i),
				"device_read_bps":   "335" + strconv.Itoa(i) + "m",
				"device_write_bps":  "225" + strconv.Itoa(i) + "k",
				"device_read_iops":  "325" + strconv.Itoa(i) + "M",
				"device_write_iops": "525" + strconv.Itoa(i) + "G",
			},
			Rdev: uint64(0x20226655) + uint64(i),
			Mode: os.FileMode(777),
			Size: int64(187654567) + int64(i),
		},
		{
			DevicePath: "/dev/mp8",
			CGSet: map[string]string{
				"weight_device":     "529" + strconv.Itoa(i),
				"device_read_bps":   "735" + strconv.Itoa(i) + "k",
				"device_write_bps":  "795" + strconv.Itoa(i) + "m",
				"device_read_iops":  "735" + strconv.Itoa(i) + "M",
				"device_write_iops": "355" + strconv.Itoa(i) + "G",
			},
			Rdev: uint64(0x20203655) + uint64(i),
			Mode: os.FileMode(777),
			Size: int64(187654567) + int64(i),
		},
		{
			DevicePath: "/dev/zmpbbg-Avanpourm-JamesBryce",
			CGSet: map[string]string{
				"weight_device":     "929" + strconv.Itoa(i),
				"device_read_bps":   "835" + strconv.Itoa(i) + "M",
				"device_write_bps":  "725" + strconv.Itoa(i) + "k",
				"device_read_iops":  "625" + strconv.Itoa(i) + "M",
				"device_write_iops": "525" + strconv.Itoa(i) + "G",
			},
			Rdev: uint64(0x20206755) + uint64(i),
			Mode: os.FileMode(777),
			Size: int64(187654567) + int64(i),
		},
	}

	return DeviceCgroupMetas
}

func TestUpdateBlkioLimit(t *testing.T) {
	Tables, err := getMockBuilderTables()
	if err != nil {
		t.Fatalf(err.Error())
		return
	}
	for _, builder := range Tables {
		docker, err := builder.setMock()
		if err != nil {
			t.Fatalf(err.Error())
			return
		}
		err = UpdateBlkio(builder.ContainnerID, docker)
		if err != nil {
			t.Errorf("UpdateBlkioLimit containerID:%s, err:%s ", builder.ContainnerID, err.Error())
		}

	}
}
