// Copyright 2014 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//+build linux

package stage0

//
// rkt is a reference implementation of the app container specification.
//
// Execution on rkt is divided into a number of stages, and the `rkt`
// binary implements the first stage (stage 0)
//

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/apps"
	"github.com/coreos/rkt/pkg/aci"
	"github.com/coreos/rkt/pkg/fileutil"
	"github.com/coreos/rkt/pkg/label"
	"github.com/coreos/rkt/pkg/sys"
	"github.com/coreos/rkt/pkg/tpm"
	"github.com/coreos/rkt/pkg/uid"
	"github.com/coreos/rkt/store"
	"github.com/coreos/rkt/version"
	"github.com/hashicorp/errwrap"
)

const (
	// Default perm bits for the regular files
	// within the stage1 directory. (e.g. image manifest,
	// pod manifest, stage1ID, etc).
	defaultRegularFilePerm = os.FileMode(0640)

	// Default perm bits for the regular directories
	// within the stage1 directory.
	defaultRegularDirPerm = os.FileMode(0750)
)

var debugEnabled bool

// configuration parameters required by Prepare
type PrepareConfig struct {
	*CommonConfig
	Apps               *apps.Apps          // apps to prepare
	InheritEnv         bool                // inherit parent environment into apps
	ExplicitEnv        []string            // always set these environment variables for all the apps
	Ports              []types.ExposedPort // list of ports that rkt will expose on the host
	UseOverlay         bool                // prepare pod with overlay fs
	SkipTreeStoreCheck bool                // skip checking the treestore before rendering
	PodManifest        string              // use the pod manifest specified by the user, this will ignore flags such as '--volume', '--port', etc.
	PrivateUsers       *uid.UidRange       // User namespaces
}

// configuration parameters needed by Run
type RunConfig struct {
	*CommonConfig
	Net         common.NetList // pod should have its own network stack
	LockFd      int            // lock file descriptor
	Interactive bool           // whether the pod is interactive or not
	MDSRegister bool           // whether to register with metadata service or not
	Apps        schema.AppList // applications (prepare gets them via Apps)
	LocalConfig string         // Path to local configuration
	Hostname    string         // hostname of the pod
	RktGid      int            // group id of the 'rkt' group, -1 if there's no rkt group.
	DNS         []string       // DNS name servers to write in /etc/resolv.conf
	DNSSearch   []string       // DNS search domains to write in /etc/resolv.conf
	DNSOpt      []string       // DNS options to write in /etc/resolv.conf
}

// configuration shared by both Run and Prepare
type CommonConfig struct {
	Store        *store.Store // store containing all of the configured application images
	Stage1Image  types.Hash   // stage1 image containing usable /init and /enter entrypoints
	UUID         *types.UUID  // UUID of the pod
	RootHash     string       // Hash of the root filesystem
	ManifestData string       // The pod manifest data
	Debug        bool
	MountLabel   string // selinux label to use for fs
	ProcessLabel string // selinux label to use for process
}

func init() {
	// this ensures that main runs only on main thread (thread group leader).
	// since namespace ops (unshare, setns) are done for a single thread, we
	// must ensure that the goroutine does not jump from OS thread to thread
	runtime.LockOSThread()
}

func InitDebug() {
	debugEnabled = true
	log.SetDebug(true)
}

func debug(format string, i ...interface{}) {
	if debugEnabled {
		log.Printf(format, i...)
	}
}

// MergeEnvs amends appEnv setting variables in setEnv before setting anything new from os.Environ if inheritEnv = true
// setEnv is expected to be in the os.Environ() key=value format
func MergeEnvs(appEnv *types.Environment, inheritEnv bool, setEnv []string) {
	for _, ev := range setEnv {
		pair := strings.SplitN(ev, "=", 2)
		appEnv.Set(pair[0], pair[1])
	}

	if inheritEnv {
		for _, ev := range os.Environ() {
			pair := strings.SplitN(ev, "=", 2)
			if _, exists := appEnv.Get(pair[0]); !exists {
				appEnv.Set(pair[0], pair[1])
			}
		}
	}
}

func imageNameToAppName(name types.ACIdentifier) (*types.ACName, error) {
	parts := strings.Split(name.String(), "/")
	last := parts[len(parts)-1]

	sn, err := types.SanitizeACName(last)
	if err != nil {
		return nil, err
	}

	return types.MustACName(sn), nil
}

// deduplicateMPs removes Mounts with duplicated paths. If there's more than
// one Mount with the same path, it keeps the first one encountered.
func deduplicateMPs(mounts []schema.Mount) []schema.Mount {
	var res []schema.Mount
	seen := make(map[string]struct{})
	for _, m := range mounts {
		if _, ok := seen[m.Path]; !ok {
			res = append(res, m)
			seen[m.Path] = struct{}{}
		}
	}
	return res
}

// MergeMounts combines the global and per-app mount slices
func MergeMounts(mounts []schema.Mount, appMounts []schema.Mount) []schema.Mount {
	ml := append(appMounts, mounts...)
	return deduplicateMPs(ml)
}

// generatePodManifest creates the pod manifest from the command line input.
// It returns the pod manifest as []byte on success.
// This is invoked if no pod manifest is specified at the command line.
func generatePodManifest(cfg PrepareConfig, dir string) ([]byte, error) {
	pm := schema.PodManifest{
		ACKind: "PodManifest",
		Apps:   make(schema.AppList, 0),
	}

	v, err := types.NewSemVer(version.Version)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error creating version"), err)
	}
	pm.ACVersion = *v

	if err := cfg.Apps.Walk(func(app *apps.App) error {
		img := app.ImageID

		am, err := cfg.Store.GetImageManifest(img.String())
		if err != nil {
			return errwrap.Wrap(errors.New("error getting the manifest"), err)
		}
		appName, err := imageNameToAppName(am.Name)
		if err != nil {
			return errwrap.Wrap(errors.New("error converting image name to app name"), err)
		}
		if err := prepareAppImage(cfg, *appName, img, dir, cfg.UseOverlay); err != nil {
			return errwrap.Wrap(fmt.Errorf("error setting up image %s", img), err)
		}
		if pm.Apps.Get(*appName) != nil {
			return fmt.Errorf("error: multiple apps with name %s", am.Name)
		}
		if am.App == nil && app.Exec == "" {
			return fmt.Errorf("error: image %s has no app section and --exec argument is not provided", img)
		}
		ra := schema.RuntimeApp{
			// TODO(vc): leverage RuntimeApp.Name for disambiguating the apps
			Name: *appName,
			App:  am.App,
			Image: schema.RuntimeImage{
				Name:   &am.Name,
				ID:     img,
				Labels: am.Labels,
			},
			Annotations: am.Annotations,
			Mounts:      MergeMounts(cfg.Apps.Mounts, app.Mounts),
		}

		if execOverride := app.Exec; execOverride != "" {
			// Create a minimal App section if not present
			if am.App == nil {
				ra.App = &types.App{
					User:  strconv.Itoa(os.Getuid()),
					Group: strconv.Itoa(os.Getgid()),
				}
			}
			ra.App.Exec = []string{execOverride}
		}

		if execAppends := app.Args; execAppends != nil {
			ra.App.Exec = append(ra.App.Exec, execAppends...)
		}

		if memoryOverride := app.MemoryLimit; memoryOverride != nil {
			isolator := memoryOverride.AsIsolator()
			ra.App.Isolators = append(ra.App.Isolators, isolator)
		}

		if cpuOverride := app.CPULimit; cpuOverride != nil {
			isolator := cpuOverride.AsIsolator()
			ra.App.Isolators = append(ra.App.Isolators, isolator)
		}

		if cfg.InheritEnv || len(cfg.ExplicitEnv) > 0 {
			MergeEnvs(&ra.App.Environment, cfg.InheritEnv, cfg.ExplicitEnv)
		}
		pm.Apps = append(pm.Apps, ra)
		return nil
	}); err != nil {
		return nil, err
	}

	// TODO(jonboulle): check that app mountpoint expectations are
	// satisfied here, rather than waiting for stage1
	pm.Volumes = cfg.Apps.Volumes
	pm.Ports = cfg.Ports

	pmb, err := json.Marshal(pm)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error marshalling pod manifest"), err)
	}
	return pmb, nil
}

// validatePodManifest reads the user-specified pod manifest, prepares the app images
// and validates the pod manifest. If the pod manifest passes validation, it returns
// the manifest as []byte.
// TODO(yifan): More validation in the future.
func validatePodManifest(cfg PrepareConfig, dir string) ([]byte, error) {
	pmb, err := ioutil.ReadFile(cfg.PodManifest)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error reading pod manifest"), err)
	}
	var pm schema.PodManifest
	if err := json.Unmarshal(pmb, &pm); err != nil {
		return nil, errwrap.Wrap(errors.New("error unmarshaling pod manifest"), err)
	}

	appNames := make(map[types.ACName]struct{})
	for _, ra := range pm.Apps {
		img := ra.Image

		if img.ID.Empty() {
			return nil, fmt.Errorf("no image ID for app %q", ra.Name)
		}
		am, err := cfg.Store.GetImageManifest(img.ID.String())
		if err != nil {
			return nil, errwrap.Wrap(errors.New("error getting the image manifest from store"), err)
		}
		if err := prepareAppImage(cfg, ra.Name, img.ID, dir, cfg.UseOverlay); err != nil {
			return nil, errwrap.Wrap(fmt.Errorf("error setting up image %s", img), err)
		}
		if _, ok := appNames[ra.Name]; ok {
			return nil, fmt.Errorf("multiple apps with same name %s", ra.Name)
		}
		appNames[ra.Name] = struct{}{}
		if ra.App == nil && am.App == nil {
			return nil, fmt.Errorf("no app section in the pod manifest or the image manifest")
		}
	}
	return pmb, nil
}

// Prepare sets up a pod based on the given config.
func Prepare(cfg PrepareConfig, dir string, uuid *types.UUID) error {
	if err := os.MkdirAll(common.AppsInfoPath(dir), defaultRegularDirPerm); err != nil {
		return errwrap.Wrap(errors.New("error creating apps info directory"), err)
	}
	debug("Preparing stage1")
	if err := prepareStage1Image(cfg, cfg.Stage1Image, dir, cfg.UseOverlay); err != nil {
		return errwrap.Wrap(errors.New("error preparing stage1"), err)
	}

	var pmb []byte
	var err error
	if len(cfg.PodManifest) > 0 {
		pmb, err = validatePodManifest(cfg, dir)
	} else {
		pmb, err = generatePodManifest(cfg, dir)
	}
	if err != nil {
		return err
	}

	cfg.CommonConfig.ManifestData = string(pmb)

	debug("Writing pod manifest")
	fn := common.PodManifestPath(dir)
	if err := ioutil.WriteFile(fn, pmb, defaultRegularFilePerm); err != nil {
		return errwrap.Wrap(errors.New("error writing pod manifest"), err)
	}

	if cfg.UseOverlay {
		// mark the pod as prepared with overlay
		f, err := os.Create(filepath.Join(dir, common.OverlayPreparedFilename))
		if err != nil {
			return errwrap.Wrap(errors.New("error writing overlay marker file"), err)
		}
		defer f.Close()
	}

	if cfg.PrivateUsers.Shift > 0 {
		// mark the pod as prepared for user namespaces
		uidrangeBytes := cfg.PrivateUsers.Serialize()

		if err := ioutil.WriteFile(filepath.Join(dir, common.PrivateUsersPreparedFilename), uidrangeBytes, defaultRegularFilePerm); err != nil {
			return errwrap.Wrap(errors.New("error writing userns marker file"), err)
		}
	}

	return nil
}

func preparedWithOverlay(dir string) (bool, error) {
	_, err := os.Stat(filepath.Join(dir, common.OverlayPreparedFilename))
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}

	if !common.SupportsOverlay() {
		return false, fmt.Errorf("the pod was prepared with overlay but overlay is not supported")
	}

	return true, nil
}

func preparedWithPrivateUsers(dir string) (string, error) {
	bytes, err := ioutil.ReadFile(filepath.Join(dir, common.PrivateUsersPreparedFilename))
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}

	return string(bytes), nil
}

func addResolvConf(cfg RunConfig, rootfs string) {
	content := "# Generated by rkt\n\n"
	if len(cfg.DNSSearch) > 0 {
		content += fmt.Sprintf("search %s\n", strings.Join(cfg.DNSSearch, " "))
	}
	for _, server := range cfg.DNS {
		// skip empty entries
		if server == "" {
			continue
		}
		// comment invalid entries
		if net.ParseIP(server) == nil {
			content += "# "
		}
		content += "nameserver " + server + "\n"
	}
	if len(cfg.DNSOpt) > 0 {
		content += fmt.Sprintf("options %s\n", strings.Join(cfg.DNSOpt, " "))
	}
	content += "\n"

	if err := ioutil.WriteFile(filepath.Join(rootfs, "etc/rkt-resolv.conf"), []byte(content), 0644); err != nil {
		log.Fatalf("error writing /etc/rkt-resolv.conf: %v\n", err)
	}
}

// Run mounts the right overlay filesystems and actually runs the prepared
// pod by exec()ing the stage1 init inside the pod filesystem.
func Run(cfg RunConfig, dir string, dataDir string) {
	useOverlay, err := preparedWithOverlay(dir)
	if err != nil {
		log.FatalE("error preparing overlay", err)
	}

	privateUsers, err := preparedWithPrivateUsers(dir)
	if err != nil {
		log.FatalE("error preparing private users", err)
	}

	debug("Setting up stage1")
	if err := setupStage1Image(cfg, dir, useOverlay); err != nil {
		log.FatalE("error setting up stage1", err)
	}
	debug("Wrote filesystem to %s\n", dir)

	for _, app := range cfg.Apps {
		if err := setupAppImage(cfg, app.Name, app.Image.ID, dir, useOverlay); err != nil {
			log.FatalE("error setting up app image", err)
		}
	}

	destRootfs := common.Stage1RootfsPath(dir)

	if len(cfg.DNS) > 0 || len(cfg.DNSSearch) > 0 || len(cfg.DNSOpt) > 0 {
		addResolvConf(cfg, destRootfs)
	}

	if err := os.Setenv(common.EnvLockFd, fmt.Sprintf("%v", cfg.LockFd)); err != nil {
		log.FatalE("setting lock fd environment", err)
	}

	if err := os.Setenv(common.EnvSELinuxContext, fmt.Sprintf("%v", cfg.ProcessLabel)); err != nil {
		log.FatalE("setting SELinux context environment", err)
	}

	if err := os.Setenv(common.EnvSELinuxMountContext, fmt.Sprintf("%v", cfg.MountLabel)); err != nil {
		log.FatalE("setting SELinux mount context enviroment", err)
	}

	debug("Pivoting to filesystem %s", dir)
	if err := os.Chdir(dir); err != nil {
		log.FatalE("failed changing to dir", err)
	}

	ep, err := getStage1Entrypoint(dir, runEntrypoint)
	if err != nil {
		log.FatalE("error determining 'run' entrypoint", err)
	}
	args := []string{filepath.Join(destRootfs, ep)}
	debug("Execing %s", ep)

	if cfg.Debug {
		args = append(args, "--debug")
	}

	args = append(args, "--net="+cfg.Net.String())

	if cfg.Interactive {
		args = append(args, "--interactive")
	}
	if len(privateUsers) > 0 {
		args = append(args, "--private-users="+privateUsers)
	}
	if cfg.MDSRegister {
		mdsToken, err := registerPod(".", cfg.UUID, cfg.Apps)
		if err != nil {
			log.FatalE("failed to register the pod", err)
		}

		args = append(args, "--mds-token="+mdsToken)
	}

	if cfg.LocalConfig != "" {
		args = append(args, "--local-config="+cfg.LocalConfig)
	}

	s1v, err := getStage1InterfaceVersion(dir)
	if err != nil {
		log.FatalE("error determining stage1 interface version", err)
	}

	if cfg.Hostname != "" {
		if interfaceVersionSupportsHostname(s1v) {
			args = append(args, "--hostname="+cfg.Hostname)
		} else {
			log.Printf("warning: --hostname option is not supported by stage1")
		}
	}

	args = append(args, cfg.UUID.String())

	// make sure the lock fd stays open across exec
	if err := sys.CloseOnExec(cfg.LockFd, false); err != nil {
		log.Fatalf("error clearing FD_CLOEXEC on lock fd")
	}

	tpmEvent := fmt.Sprintf("rkt: Rootfs: %s Manifest: %s Stage 1 args: %s", cfg.CommonConfig.RootHash, cfg.CommonConfig.ManifestData, strings.Join(args, " "))
	// If there's no TPM available or there's a failure for some other
	// reason, ignore it and continue anyway. Long term we'll want policy
	// that enforces TPM behaviour, but we don't have any infrastructure
	// around that yet.
	_ = tpm.Extend(tpmEvent)
	if err := syscall.Exec(args[0], args, os.Environ()); err != nil {
		log.FatalE("error execing init", err)
	}
}

// prepareAppImage renders and verifies the tree cache of the app image that
// corresponds to the given app name.
// When useOverlay is false, it attempts to render and expand the app image
func prepareAppImage(cfg PrepareConfig, appName types.ACName, img types.Hash, cdir string, useOverlay bool) error {
	debug("Loading image %s", img.String())

	am, err := cfg.Store.GetImageManifest(img.String())
	if err != nil {
		return errwrap.Wrap(errors.New("error getting the manifest"), err)
	}

	if _, hasOS := am.Labels.Get("os"); !hasOS {
		return fmt.Errorf("missing os label in the image manifest")
	}
	if _, hasArch := am.Labels.Get("arch"); !hasArch {
		return fmt.Errorf("missing arch label in the image manifest")
	}

	if err := types.IsValidOSArch(am.Labels.ToMap(), ValidOSArch); err != nil {
		return err
	}

	appInfoDir := common.AppInfoPath(cdir, appName)
	if err := os.MkdirAll(appInfoDir, defaultRegularDirPerm); err != nil {
		return errwrap.Wrap(errors.New("error creating apps info directory"), err)
	}

	if useOverlay {
		if cfg.PrivateUsers.Shift > 0 {
			return fmt.Errorf("cannot use both overlay and user namespace: not implemented yet. (Try --no-overlay)")
		}
		treeStoreID, _, err := cfg.Store.RenderTreeStore(img.String(), false)
		if err != nil {
			return errwrap.Wrap(errors.New("error rendering tree image"), err)
		}

		if !cfg.SkipTreeStoreCheck {
			hash, err := cfg.Store.CheckTreeStore(treeStoreID)
			if err != nil {
				log.PrintE("warning: tree cache is in a bad state: %v. Rebuilding...", err)
				var err error
				treeStoreID, hash, err = cfg.Store.RenderTreeStore(img.String(), true)
				if err != nil {
					return errwrap.Wrap(errors.New("error rendering tree image"), err)
				}
			}
			cfg.CommonConfig.RootHash = hash
		}

		if err := ioutil.WriteFile(common.AppTreeStoreIDPath(cdir, appName), []byte(treeStoreID), defaultRegularFilePerm); err != nil {
			return errwrap.Wrap(errors.New("error writing app treeStoreID"), err)
		}
	} else {
		ad := common.AppPath(cdir, appName)
		err := os.MkdirAll(ad, defaultRegularDirPerm)
		if err != nil {
			return errwrap.Wrap(errors.New("error creating image directory"), err)
		}

		shiftedUid, shiftedGid, err := cfg.PrivateUsers.ShiftRange(uint32(os.Getuid()), uint32(os.Getgid()))
		if err != nil {
			return errwrap.Wrap(errors.New("error getting uid, gid"), err)
		}

		if err := os.Chown(ad, int(shiftedUid), int(shiftedGid)); err != nil {
			return errwrap.Wrap(fmt.Errorf("error shifting app %q's stage2 dir", appName), err)
		}

		if err := aci.RenderACIWithImageID(img, ad, cfg.Store, cfg.PrivateUsers); err != nil {
			return errwrap.Wrap(errors.New("error rendering ACI"), err)
		}
	}
	if err := writeManifest(*cfg.CommonConfig, img, appInfoDir); err != nil {
		return err
	}
	return nil
}

// setupAppImage mounts the overlay filesystem for the app image that
// corresponds to the given hash. Then, it creates the tmp directory.
// When useOverlay is false it just creates the tmp directory for this app.
func setupAppImage(cfg RunConfig, appName types.ACName, img types.Hash, cdir string, useOverlay bool) error {
	ad := common.AppPath(cdir, appName)
	if useOverlay {
		err := os.MkdirAll(ad, defaultRegularDirPerm)
		if err != nil {
			return errwrap.Wrap(errors.New("error creating image directory"), err)
		}
		treeStoreID, err := ioutil.ReadFile(common.AppTreeStoreIDPath(cdir, appName))
		if err != nil {
			return err
		}
		if err := copyAppManifest(cdir, appName, ad); err != nil {
			return err
		}
		if err := overlayRender(cfg, string(treeStoreID), cdir, ad, appName.String()); err != nil {
			return errwrap.Wrap(errors.New("error rendering overlay filesystem"), err)
		}
	}

	return nil
}

// prepareStage1Image renders and verifies tree cache of the given hash
// when using overlay.
// When useOverlay is false, it attempts to render and expand the stage1.
func prepareStage1Image(cfg PrepareConfig, img types.Hash, cdir string, useOverlay bool) error {
	s1 := common.Stage1ImagePath(cdir)
	if err := os.MkdirAll(s1, defaultRegularDirPerm); err != nil {
		return errwrap.Wrap(errors.New("error creating stage1 directory"), err)
	}

	treeStoreID, _, err := cfg.Store.RenderTreeStore(img.String(), false)
	if err != nil {
		return errwrap.Wrap(errors.New("error rendering tree image"), err)
	}

	if !cfg.SkipTreeStoreCheck {
		hash, err := cfg.Store.CheckTreeStore(treeStoreID)
		if err != nil {
			log.Printf("warning: tree cache is in a bad state: %v. Rebuilding...", err)
			var err error
			treeStoreID, hash, err = cfg.Store.RenderTreeStore(img.String(), true)
			if err != nil {
				return errwrap.Wrap(errors.New("error rendering tree image"), err)
			}
		}
		cfg.CommonConfig.RootHash = hash
	}

	if err := writeManifest(*cfg.CommonConfig, img, s1); err != nil {
		return errwrap.Wrap(errors.New("error writing manifest"), err)
	}

	if !useOverlay {
		destRootfs := filepath.Join(s1, "rootfs")
		cachedTreePath := cfg.Store.GetTreeStoreRootFS(treeStoreID)
		if err := fileutil.CopyTree(cachedTreePath, destRootfs, cfg.PrivateUsers); err != nil {
			return errwrap.Wrap(errors.New("error rendering ACI"), err)
		}
	}

	fn := path.Join(cdir, common.Stage1TreeStoreIDFilename)
	if err := ioutil.WriteFile(fn, []byte(treeStoreID), defaultRegularFilePerm); err != nil {
		return errwrap.Wrap(errors.New("error writing stage1 treeStoreID"), err)
	}
	return nil
}

// setupStage1Image mounts the overlay filesystem for stage1.
// When useOverlay is false it is a noop
func setupStage1Image(cfg RunConfig, cdir string, useOverlay bool) error {
	s1 := common.Stage1ImagePath(cdir)
	if useOverlay {
		treeStoreID, err := ioutil.ReadFile(filepath.Join(cdir, common.Stage1TreeStoreIDFilename))
		if err != nil {
			return err
		}

		// pass an empty appName: make sure it remains consistent with
		// overlayStatusDirTemplate
		if err := overlayRender(cfg, string(treeStoreID), cdir, s1, ""); err != nil {
			return errwrap.Wrap(errors.New("error rendering overlay filesystem"), err)
		}

		// we will later read the status from the upper layer of the overlay fs
		// force the status directory to be there by touching it
		statusPath := filepath.Join(s1, "rootfs", "rkt", "status")
		if err := os.Chtimes(statusPath, time.Now(), time.Now()); err != nil {
			return errwrap.Wrap(errors.New("error touching status dir"), err)
		}
	}

	return nil
}

// writeManifest takes an img ID and writes the corresponding manifest in dest
func writeManifest(cfg CommonConfig, img types.Hash, dest string) error {
	mb, err := cfg.Store.GetImageManifestJSON(img.String())
	if err != nil {
		return err
	}

	debug("Writing image manifest")
	if err := ioutil.WriteFile(filepath.Join(dest, "manifest"), mb, defaultRegularFilePerm); err != nil {
		return errwrap.Wrap(errors.New("error writing image manifest"), err)
	}

	return nil
}

// copyAppManifest copies to saved image manifest for the given appName and
// writes it in the dest directory.
func copyAppManifest(cdir string, appName types.ACName, dest string) error {
	appInfoDir := common.AppInfoPath(cdir, appName)
	sourceFn := filepath.Join(appInfoDir, "manifest")
	destFn := filepath.Join(dest, "manifest")
	if err := fileutil.CopyRegularFile(sourceFn, destFn); err != nil {
		return errwrap.Wrap(errors.New("error copying image manifest"), err)
	}
	return nil
}

// overlayRender renders the image that corresponds to the given hash using the
// overlay filesystem.
// It mounts an overlay filesystem from the cached tree of the image as rootfs.
func overlayRender(cfg RunConfig, treeStoreID string, cdir string, dest string, appName string) error {
	cachedTreePath := cfg.Store.GetTreeStoreRootFS(treeStoreID)
	fi, err := os.Stat(cachedTreePath)
	if err != nil {
		return err
	}
	imgMode := fi.Mode()

	destRootfs := path.Join(dest, "rootfs")
	if err := os.MkdirAll(destRootfs, imgMode); err != nil {
		return err
	}

	overlayDir := path.Join(cdir, "overlay")
	if err := os.MkdirAll(overlayDir, defaultRegularDirPerm); err != nil {
		return err
	}

	// Since the parent directory (rkt/pods/$STATE/$POD_UUID) has the 'S_ISGID' bit, here
	// we need to explicitly turn the bit off when creating this overlay
	// directory so that it won't inherit the bit. Otherwise the files
	// created by users within the pod will inherit the 'S_ISGID' bit
	// as well.
	if err := os.Chmod(overlayDir, defaultRegularDirPerm); err != nil {
		return err
	}

	imgDir := path.Join(overlayDir, treeStoreID)
	if err := os.MkdirAll(imgDir, defaultRegularDirPerm); err != nil {
		return err
	}

	// Also make 'rkt/pods/$STATE/$POD_UUID/overlay/$IMAGE_ID' to be readable by 'rkt' group
	// As 'rkt' status will read the 'rkt/pods/$STATE/$POD_UUID/overlay/$IMAGE_ID/upper/rkt/status/$APP'
	// to get exit status.
	if err := os.Chown(imgDir, -1, cfg.RktGid); err != nil {
		return err
	}

	upperDir := path.Join(imgDir, "upper", appName)
	if err := os.MkdirAll(upperDir, imgMode); err != nil {
		return err
	}
	if err := label.SetFileLabel(upperDir, cfg.MountLabel); err != nil {
		return err
	}

	workDir := path.Join(imgDir, "work", appName)
	if err := os.MkdirAll(workDir, defaultRegularDirPerm); err != nil {
		return err
	}
	if err := label.SetFileLabel(workDir, cfg.MountLabel); err != nil {
		return err
	}

	opts := fmt.Sprintf("lowerdir=%s,upperdir=%s,workdir=%s", cachedTreePath, upperDir, workDir)
	opts = label.FormatMountLabel(opts, cfg.MountLabel)
	if err := syscall.Mount("overlay", destRootfs, "overlay", 0, opts); err != nil {
		return errwrap.Wrap(errors.New("error mounting"), err)
	}

	return nil
}
