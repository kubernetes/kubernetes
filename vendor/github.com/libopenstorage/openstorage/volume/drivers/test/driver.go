package test

import (
	"fmt"
	"os"
	"os/exec"
	"path"
	"testing"
	"time"

	"go.pedge.io/dlog"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/portworx/kvdb"
	"github.com/portworx/kvdb/mem"
	"github.com/stretchr/testify/require"
)

func init() {
	kv, err := kvdb.New(mem.Name, "driver_test", []string{}, nil, dlog.Panicf)
	if err != nil {
		dlog.Panicf("Failed to intialize KVDB")
	}
	err = kvdb.SetInstance(kv)
	if err != nil {
		dlog.Panicf("Failed to set KVDB instance")
	}
}

// Context maintains current device state. It gets passed into tests
// so that tests can build on other tests' work
type Context struct {
	volume.VolumeDriver
	volID         string
	snapID        string
	mountPath     string
	devicePath    string
	Filesystem    api.FSType
	testPath      string
	testFile      string
	Encrypted     bool
	Passphrase    string
	AttachOptions map[string]string
}

// NewContext returns a new Context
func NewContext(d volume.VolumeDriver) *Context {
	return &Context{
		VolumeDriver: d,
		volID:        "",
		snapID:       "",
		Filesystem:   api.FSType_FS_TYPE_NONE,
		testPath:     path.Join("/tmp/openstorage/mount/", d.Name()),
		testFile:     path.Join("/tmp/", d.Name()),
	}
}

// RunShort runs the short test suite
func RunShort(t *testing.T, ctx *Context) {
	create(t, ctx)
	inspect(t, ctx)
	set(t, ctx)
	enumerate(t, ctx)
	attach(t, ctx)
	mount(t, ctx)
	io(t, ctx)
	unmount(t, ctx)
	detach(t, ctx)
	delete(t, ctx)
	runEnd(t, ctx)
}

// Run runs the complete test suite
func Run(t *testing.T, ctx *Context) {
	RunShort(t, ctx)
	RunSnap(t, ctx)
	runEnd(t, ctx)
}

func runEnd(t *testing.T, ctx *Context) {
	time.Sleep(time.Second * 2)
	os.RemoveAll(ctx.testPath)
	os.Remove(ctx.testFile)
	shutdown(t, ctx)
}

// RunSnap runs only the snap related tests
func RunSnap(t *testing.T, ctx *Context) {
	snap(t, ctx)
	snapInspect(t, ctx)
	snapEnumerate(t, ctx)
	snapDiff(t, ctx)
	snapDelete(t, ctx)
	detach(t, ctx)
	delete(t, ctx)
}

func create(t *testing.T, ctx *Context) {
	fmt.Println("create")

	volID, err := ctx.Create(
		&api.VolumeLocator{Name: "foo", VolumeLabels: map[string]string{"oh": "create"}},
		nil,
		&api.VolumeSpec{
			Size:       1 * 1024 * 1024 * 1024,
			HaLevel:    1,
			Format:     ctx.Filesystem,
			Encrypted:  ctx.Encrypted,
			Passphrase: ctx.Passphrase,
		})

	require.NoError(t, err, "Failed in Create")
	ctx.volID = volID
}

func inspect(t *testing.T, ctx *Context) {
	fmt.Println("inspect")

	vols, err := ctx.Inspect([]string{ctx.volID})
	require.NoError(t, err, "Failed in Inspect")
	require.NotNil(t, vols, "Nil vols")
	require.Equal(t, len(vols), 1, "Expect 1 volume actual %v volumes", len(vols))
	require.Equal(t, vols[0].Id, ctx.volID, "Expect volID %v actual %v", ctx.volID, vols[0].Id)

	vols, err = ctx.Inspect([]string{string("shouldNotExist")})
	require.Equal(t, 0, len(vols), "Expect 0 volume actual %v volumes", len(vols))
}

func set(t *testing.T, ctx *Context) {
	fmt.Println("update")

	vols, err := ctx.Inspect([]string{ctx.volID})
	require.NoError(t, err, "Failed in Inspect")
	require.NotNil(t, vols, "Nil vols")
	require.Equal(t, len(vols), 1, "Expect 1 volume actual %v volumes", len(vols))
	require.Equal(t, vols[0].Id, ctx.volID, "Expect volID %v actual %v", ctx.volID, vols[0].Id)

	vols[0].Locator.VolumeLabels["UpdateTest"] = "Success"
	err = ctx.Set(ctx.volID, vols[0].Locator, nil)
	if err != volume.ErrNotSupported {
		require.NoError(t, err, "Failed in Update")
		vols, err = ctx.Inspect([]string{ctx.volID})
		require.NoError(t, err, "Failed in Inspect")
		require.NotNil(t, vols, "Nil vols")
		require.Equal(t, len(vols), 1, "Expect 1 volume actual %v volumes", len(vols))
		require.Equal(t, vols[0].Locator.VolumeLabels["UpdateTest"], "Success",
			"Expect Label %v actual %v", "UpdateTest", vols[0].Locator.VolumeLabels)
	}
}

func enumerate(t *testing.T, ctx *Context) {
	fmt.Println("enumerate")

	vols, err := ctx.Enumerate(&api.VolumeLocator{}, nil)
	require.NoError(t, err, "Failed in Enumerate")
	require.NotNil(t, vols, "Nil vols")
	require.Equal(t, 1, len(vols), "Expect 1 volume actual %v volumes", len(vols))
	require.Equal(t, vols[0].Id, ctx.volID, "Expect volID %v actual %v", ctx.volID, vols[0].Id)

	vols, err = ctx.Enumerate(&api.VolumeLocator{Name: "foo"}, nil)
	require.NoError(t, err, "Failed in Enumerate")
	require.NotNil(t, vols, "Nil vols")
	require.Equal(t, len(vols), 1, "Expect 1 volume actual %v volumes", len(vols))
	require.Equal(t, vols[0].Id, ctx.volID, "Expect volID %v actual %v", ctx.volID, vols[0].Id)

	vols, err = ctx.Enumerate(&api.VolumeLocator{Name: "shouldNotExist"}, nil)
	require.Equal(t, len(vols), 0, "Expect 0 volume actual %v volumes", len(vols))
}

func waitReady(t *testing.T, ctx *Context) error {
	total := time.Minute * 5
	inc := time.Second * 2
	elapsed := time.Second * 0
	vols, err := ctx.Inspect([]string{ctx.volID})
	for err == nil && len(vols) == 1 && vols[0].Status != api.VolumeStatus_VOLUME_STATUS_UP && elapsed < total {
		time.Sleep(inc)
		elapsed += inc
		vols, err = ctx.Inspect([]string{ctx.volID})
	}
	if err != nil {
		return err
	}
	if len(vols) != 1 {
		return fmt.Errorf("Expect one volume from inspect got %v", len(vols))
	}
	if vols[0].Status != api.VolumeStatus_VOLUME_STATUS_UP {
		return fmt.Errorf("Timed out waiting for volume status %v", vols)
	}
	return err
}

func attach(t *testing.T, ctx *Context) {
	fmt.Println("attach")
	err := waitReady(t, ctx)
	require.NoError(t, err, "Volume status is not up")
	p, err := ctx.Attach(ctx.volID, ctx.AttachOptions)
	if err != nil {
		require.Equal(t, err, volume.ErrNotSupported, "Error on attach %v", err)
	}
	ctx.devicePath = p

	p, err = ctx.Attach(ctx.volID, ctx.AttachOptions)
	if err == nil {
		require.Equal(t, p, ctx.devicePath, "Multiple calls to attach if not errored should return the same path")
	}
}

func detach(t *testing.T, ctx *Context) {
	fmt.Println("detach")
	err := ctx.Detach(ctx.volID, false)
	if err != nil {
		require.Equal(t, ctx.devicePath, "", "Error on detach %s: %v", ctx.devicePath, err)
	}
	ctx.devicePath = ""
}

func mount(t *testing.T, ctx *Context) {
	fmt.Println("mount")

	err := os.MkdirAll(ctx.testPath, 0755)

	err = ctx.Mount(ctx.volID, ctx.testPath)
	require.NoError(t, err, "Failed in mount %v", ctx.testPath)

	ctx.mountPath = ctx.testPath
}

func multiMount(t *testing.T, ctx *Context) {
	ctx2 := *ctx
	create(t, ctx)
	attach(t, ctx)
	mount(t, ctx)

	create(t, &ctx2)
	attach(t, &ctx2)

	err := ctx2.Mount(ctx2.volID, ctx2.testPath)
	require.Error(t, err, "Mount of different devices to same path must fail")

	unmount(t, ctx)
	detach(t, ctx)
	delete(t, ctx)

	detach(t, &ctx2)
	delete(t, &ctx2)
}

func unmount(t *testing.T, ctx *Context) {
	fmt.Println("unmount")

	require.NotEqual(t, ctx.mountPath, "", "Device is not mounted")

	err := ctx.Unmount(ctx.volID, ctx.mountPath)
	require.NoError(t, err, "Failed in unmount %v", ctx.mountPath)

	ctx.mountPath = ""
}

func shutdown(t *testing.T, ctx *Context) {
	fmt.Println("shutdown")
	ctx.Shutdown()
}

func io(t *testing.T, ctx *Context) {
	fmt.Println("io")
	require.NotEqual(t, ctx.mountPath, "", "Device is not mounted")

	cmd := exec.Command("dd", "if=/dev/urandom", fmt.Sprintf("of=%s", ctx.testFile), "bs=1M", "count=10")
	o, err := cmd.CombinedOutput()
	require.NoError(t, err, "Failed to run dd %s", string(o))

	cmd = exec.Command("dd", fmt.Sprintf("if=%s", ctx.testFile), fmt.Sprintf("of=%s/xx", ctx.mountPath))
	o, err = cmd.CombinedOutput()
	require.NoError(t, err, "Failed to run dd on mountpoint %s/xx : %s",
		ctx.mountPath, string(o))

	cmd = exec.Command("diff", ctx.testFile, fmt.Sprintf("%s/xx", ctx.mountPath))
	o, err = cmd.CombinedOutput()
	require.NoError(t, err, "data mismatch %s", string(o))
}

func detachBad(t *testing.T, ctx *Context) {
	err := ctx.Detach(ctx.volID, false)
	require.True(t, (err == nil || err == volume.ErrNotSupported),
		"Detach on mounted device should fail")
}

func deleteBad(t *testing.T, ctx *Context) {
	fmt.Println("deleteBad")
	require.NotEqual(t, ctx.mountPath, "", "Device is not mounted")

	err := ctx.Delete(ctx.volID)
	require.Error(t, err, "Delete on mounted device must fail")
}

func delete(t *testing.T, ctx *Context) {
	fmt.Println("delete")
	err := ctx.Delete(ctx.volID)
	require.NoError(t, err, "Delete failed")
	ctx.volID = ""
}

func snap(t *testing.T, ctx *Context) {
	fmt.Println("snap")
	if ctx.volID == "" {
		create(t, ctx)
	}
	attach(t, ctx)
	labels := map[string]string{"oh": "snap"}
	require.NotEqual(t, ctx.volID, "", "invalid volume ID")
	id, err := ctx.Snapshot(ctx.volID, false,
		&api.VolumeLocator{Name: "snappy", VolumeLabels: labels})
	require.NoError(t, err, "Failed in creating a snapshot")
	ctx.snapID = id
}

func snapInspect(t *testing.T, ctx *Context) {
	fmt.Println("snapInspect")

	snaps, err := ctx.Inspect([]string{ctx.snapID})
	require.NoError(t, err, "Failed in Inspect")
	require.NotNil(t, snaps, "Nil snaps")
	require.Equal(t, len(snaps), 1, "Expect 1 snaps actual %v snaps", len(snaps))
	require.Equal(t, snaps[0].Id, ctx.snapID, "Expect snapID %v actual %v", ctx.snapID, snaps[0].Id)

	snaps, err = ctx.Inspect([]string{string("shouldNotExist")})
	require.Equal(t, 0, len(snaps), "Expect 0 snaps actual %v snaps", len(snaps))
}

func snapEnumerate(t *testing.T, ctx *Context) {
	fmt.Println("snapEnumerate")

	snaps, err := ctx.SnapEnumerate(nil, nil)
	require.NoError(t, err, "Failed in snapEnumerate")
	require.NotNil(t, snaps, "Nil snaps")
	require.Equal(t, 1, len(snaps), "Expect 1 snaps actual %v snaps", len(snaps))
	require.Equal(t, snaps[0].Id, ctx.snapID, "Expect snapID %v actual %v", ctx.snapID, snaps[0].Id)
	labels := snaps[0].Locator.VolumeLabels

	snaps, err = ctx.SnapEnumerate([]string{ctx.volID}, nil)
	require.NoError(t, err, "Failed in snapEnumerate")
	require.NotNil(t, snaps, "Nil snaps")
	require.Equal(t, len(snaps), 1, "Expect 1 snap actual %v snaps", len(snaps))
	require.Equal(t, snaps[0].Id, ctx.snapID, "Expect snapID %v actual %v", ctx.snapID, snaps[0].Id)

	snaps, err = ctx.SnapEnumerate([]string{string("shouldNotExist")}, nil)
	require.Equal(t, len(snaps), 0, "Expect 0 snap actual %v snaps", len(snaps))

	snaps, err = ctx.SnapEnumerate(nil, labels)
	require.NoError(t, err, "Failed in snapEnumerate")
	require.NotNil(t, snaps, "Nil snaps")
	require.Equal(t, len(snaps), 1, "Expect 1 snap actual %v snaps", len(snaps))
	require.Equal(t, snaps[0].Id, ctx.snapID, "Expect snapID %v actual %v", ctx.snapID, snaps[0].Id)
}

func snapDiff(t *testing.T, ctx *Context) {
	fmt.Println("snapDiff")
}

func snapDelete(t *testing.T, ctx *Context) {
	fmt.Println("snapDelete")
}
