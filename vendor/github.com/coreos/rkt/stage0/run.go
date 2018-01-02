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
// binary implements the first stage (stage0)
//

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
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
	cnitypes "github.com/containernetworking/cni/pkg/types"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/apps"
	commonnet "github.com/coreos/rkt/common/networking"
	"github.com/coreos/rkt/common/overlay"
	"github.com/coreos/rkt/pkg/aci"
	"github.com/coreos/rkt/pkg/fileutil"
	"github.com/coreos/rkt/pkg/label"
	"github.com/coreos/rkt/pkg/sys"
	"github.com/coreos/rkt/pkg/tpm"
	"github.com/coreos/rkt/pkg/user"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/coreos/rkt/store/treestore"
	"github.com/coreos/rkt/version"
	"github.com/hashicorp/errwrap"
)

var debugEnabled bool

// PrepareConfig defines the configuration parameters required by Prepare
type PrepareConfig struct {
	*CommonConfig
	Apps               *apps.Apps            // apps to prepare
	InheritEnv         bool                  // inherit parent environment into apps
	ExplicitEnv        []string              // always set these environment variables for all the apps
	EnvFromFile        []string              // environment variables loaded from files, set for all the apps
	Ports              []types.ExposedPort   // list of ports that rkt will expose on the host
	UseOverlay         bool                  // prepare pod with overlay fs
	SkipTreeStoreCheck bool                  // skip checking the treestore before rendering
	PodManifest        string                // use the pod manifest specified by the user, this will ignore flags such as '--volume', '--port', etc.
	PrivateUsers       *user.UidRange        // user namespaces
	UserAnnotations    types.UserAnnotations // user annotations for the pod.
	UserLabels         types.UserLabels      // user labels for the pod.
}

// RunConfig defines the configuration parameters needed by Run
type RunConfig struct {
	*CommonConfig
	Net                  common.NetList // pod should have its own network stack
	LockFd               int            // lock file descriptor
	Interactive          bool           // whether the pod is interactive or not
	MDSRegister          bool           // whether to register with metadata service or not
	Apps                 schema.AppList // applications (prepare gets them via Apps)
	LocalConfig          string         // Path to local configuration
	Hostname             string         // hostname of the pod
	RktGid               int            // group id of the 'rkt' group, -1 ere's no rkt group.
	DNSConfMode          DNSConfMode    // dns configuration file mode - for stAage1
	DNSConfig            cnitypes.DNS   // the DNS configuration (nameservers, search, options)
	InsecureCapabilities bool           // Do not restrict capabilities
	InsecurePaths        bool           // Do not restrict access to files in sysfs or procfs
	InsecureSeccomp      bool           // Do not add seccomp restrictions
	UseOverlay           bool           // run pod with overlay fs
	HostsEntries         HostsEntries   // The entries in /etc/hosts
}

// CommonConfig defines the configuration shared by both Run and Prepare
type CommonConfig struct {
	DataDir      string                        // The path to the data directory, e.g. /var/lib/rkt/pods
	Store        *imagestore.Store             // store containing all of the configured application images
	TreeStore    *treestore.Store              // store containing all of the configured application images
	Stage1Image  types.Hash                    // stage1 image containing usable /init and /enter entrypoints
	UUID         *types.UUID                   // UUID of the pod
	RootHash     string                        // hash of the root filesystem
	ManifestData string                        // the pod manifest data
	Debug        bool                          // debug mode
	MountLabel   string                        // SELinux label to use for fs
	ProcessLabel string                        // SELinux label to use
	Mutable      bool                          // whether this pod is mutable
	Annotations  map[types.ACIdentifier]string // pod-level annotations, for internal/experimental usage
}

// HostsEntries encapsulates the entries in an etc-hosts file: mapping from IP
// to arbitrary list of hostnames
type HostsEntries map[string][]string

// DNSConfMode indicates what the stage1 should do with dns config files
// The values and meanings are:
// 'host': bind-mount from host
// 'stage0': the stage0 has generated it
// 'none' : do not generate it
// 'default' : do whatever was the default
type DNSConfMode struct {
	Resolv string // /etc/rkt-resolv.conf
	Hosts  string // /etc/rkt-hosts
}

func init() {
	// this ensures that main runs only on main thread (thread group leader).
	// since namespace ops (unshare, setns) are done for a single thread, we
	// must ensure that the goroutine does not jump from OS thread to thread
	runtime.LockOSThread()
}

// InitDebug enables debugging
func InitDebug() {
	debugEnabled = true
	log.SetDebug(true)
}

func debug(format string, i ...interface{}) {
	if debugEnabled {
		log.Printf(format, i...)
	}
}

// mergeEnvs merges environment variables from env into the current appEnv
// if override is set to true, then variables with the same name will be set to the value in env
// env is expected to be in the os.Environ() key=value format
func mergeEnvs(appEnv *types.Environment, env []string, override bool) {
	for _, ev := range env {
		pair := strings.SplitN(ev, "=", 2)
		if _, exists := appEnv.Get(pair[0]); override || !exists {
			appEnv.Set(pair[0], pair[1])
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

		if app.Name == "" {
			appName, err := imageNameToAppName(am.Name)
			if err != nil {
				return errwrap.Wrap(errors.New("error converting image name to app name"), err)
			}
			app.Name = appName.String()
		}

		appName, err := types.NewACName(app.Name)
		if err != nil {
			return errwrap.Wrap(errors.New("invalid app name format"), err)
		}

		if _, err := prepareAppImage(cfg, *appName, img, dir, cfg.UseOverlay); err != nil {
			return errwrap.Wrap(fmt.Errorf("error preparing image %s", img), err)
		}
		if pm.Apps.Get(*appName) != nil {
			return fmt.Errorf("error: multiple apps with name %s", app.Name)
		}
		if am.App == nil && app.Exec == "" {
			return fmt.Errorf("error: image %s has no app section and --exec argument is not provided", img)
		}

		ra, err := generateRuntimeApp(app, am, cfg.Apps.Mounts)
		if err != nil {
			return err
		}

		// loading the environment from the lowest priority to highest
		if cfg.InheritEnv {
			// Inherit environment does not override app image environment
			mergeEnvs(&ra.App.Environment, os.Environ(), false)
		}

		mergeEnvs(&ra.App.Environment, cfg.EnvFromFile, true)
		mergeEnvs(&ra.App.Environment, cfg.ExplicitEnv, true)

		pm.Apps = append(pm.Apps, ra)

		return nil
	}); err != nil {
		return nil, err
	}

	// TODO(jonboulle): check that app mountpoint expectations are
	// satisfied here, rather than waiting for stage1
	pm.Volumes = cfg.Apps.Volumes

	// Check to see if ports have any errors
	pm.Ports = cfg.Ports
	if _, err := commonnet.ForwardedPorts(&pm); err != nil {
		return nil, err
	}

	pm.Annotations = append(pm.Annotations, types.Annotation{
		Name:  "coreos.com/rkt/stage1/mutable",
		Value: strconv.FormatBool(cfg.Mutable),
	})

	pm.UserAnnotations = cfg.UserAnnotations
	pm.UserLabels = cfg.UserLabels

	// Add internal annotations for rkt experiments
	for k, v := range cfg.Annotations {
		if _, ok := pm.Annotations.Get(k.String()); ok {
			continue
		}
		pm.Annotations.Set(k, v)
	}

	pmb, err := json.Marshal(pm)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error marshalling pod manifest"), err)
	}
	return pmb, nil
}

// prepareIsolators merges the CLI app parameters with the manifest's app
func prepareIsolators(setup *apps.App, app *types.App) error {
	if memoryOverride := setup.MemoryLimit; memoryOverride != nil {
		isolator := memoryOverride.AsIsolator()
		app.Isolators = append(app.Isolators, isolator)
	}

	if cpuOverride := setup.CPULimit; cpuOverride != nil {
		isolator := cpuOverride.AsIsolator()
		app.Isolators = append(app.Isolators, isolator)
	}

	if cpuSharesOverride := setup.CPUShares; cpuSharesOverride != nil {
		isolator := cpuSharesOverride.AsIsolator()
		app.Isolators.ReplaceIsolatorsByName(isolator, []types.ACIdentifier{types.LinuxCPUSharesName})
	}

	if oomAdjOverride := setup.OOMScoreAdj; oomAdjOverride != nil {
		app.Isolators.ReplaceIsolatorsByName(oomAdjOverride.AsIsolator(), []types.ACIdentifier{types.LinuxOOMScoreAdjName})
	}

	if setup.CapsRetain != nil && setup.CapsRemove != nil {
		return fmt.Errorf("error: cannot use both --caps-retain and --caps-remove on the same image")
	}

	// Delete existing caps isolators if the user wants to override
	// them with either --caps-retain or --caps-remove
	if setup.CapsRetain != nil || setup.CapsRemove != nil {
		for i := len(app.Isolators) - 1; i >= 0; i-- {
			isolator := app.Isolators[i]
			if _, ok := isolator.Value().(types.LinuxCapabilitiesSet); ok {
				app.Isolators = append(app.Isolators[:i],
					app.Isolators[i+1:]...)
			}
		}
	}

	if capsRetain := setup.CapsRetain; capsRetain != nil {
		isolator, err := capsRetain.AsIsolator()
		if err != nil {
			return err
		}
		app.Isolators = append(app.Isolators, *isolator)
	} else if capsRemove := setup.CapsRemove; capsRemove != nil {
		isolator, err := capsRemove.AsIsolator()
		if err != nil {
			return err
		}
		app.Isolators = append(app.Isolators, *isolator)
	}

	// Override seccomp isolators via --seccomp CLI switch
	if setup.SeccompFilter != "" {
		var is *types.Isolator
		mode, errno, set, err := setup.SeccompOverride()
		if err != nil {
			return err
		}
		switch mode {
		case "retain":
			lss, err := types.NewLinuxSeccompRetainSet(errno, set...)
			if err != nil {
				return err
			}
			if is, err = lss.AsIsolator(); err != nil {
				return err
			}
		case "remove":
			lss, err := types.NewLinuxSeccompRemoveSet(errno, set...)
			if err != nil {
				return err
			}
			if is, err = lss.AsIsolator(); err != nil {
				return err
			}
		default:
			return apps.ErrInvalidSeccompMode
		}
		app.Isolators.ReplaceIsolatorsByName(*is, []types.ACIdentifier{types.LinuxSeccompRemoveSetName, types.LinuxSeccompRetainSetName})
	}
	return nil
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
		if _, err := prepareAppImage(cfg, ra.Name, img.ID, dir, cfg.UseOverlay); err != nil {
			return nil, errwrap.Wrap(fmt.Errorf("error preparing image %s", img), err)
		}
		if _, ok := appNames[ra.Name]; ok {
			return nil, fmt.Errorf("multiple apps with same name %s", ra.Name)
		}
		appNames[ra.Name] = struct{}{}
		if ra.App == nil && am.App == nil {
			return nil, fmt.Errorf("no app section in the pod manifest or the image manifest")
		}
	}

	// Validate forwarded ports
	if _, err := commonnet.ForwardedPorts(&pm); err != nil {
		return nil, err
	}
	return pmb, nil
}

// Prepare sets up a pod based on the given config.
func Prepare(cfg PrepareConfig, dir string, uuid *types.UUID) error {
	if err := os.MkdirAll(common.AppsInfoPath(dir), common.DefaultRegularDirPerm); err != nil {
		return errwrap.Wrap(errors.New("error creating apps info directory"), err)
	}
	debug("Preparing stage1")
	if err := prepareStage1Image(cfg, dir); err != nil {
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

	// create pod lock file for app add/rm operations.
	f, err := os.OpenFile(common.PodManifestLockPath(dir), os.O_CREATE|os.O_RDWR, 0600)
	if err != nil {
		return err
	}
	f.Close()

	debug("Writing pod manifest")
	fn := common.PodManifestPath(dir)
	if err := ioutil.WriteFile(fn, pmb, common.DefaultRegularFilePerm); err != nil {
		return errwrap.Wrap(errors.New("error writing pod manifest"), err)
	}

	f, err = os.OpenFile(common.PodCreatedPath(dir), os.O_CREATE|os.O_RDWR, common.DefaultRegularFilePerm)
	if err != nil {
		return err
	}
	f.Close()

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

		if err := ioutil.WriteFile(filepath.Join(dir, common.PrivateUsersPreparedFilename), uidrangeBytes, common.DefaultRegularFilePerm); err != nil {
			return errwrap.Wrap(errors.New("error writing userns marker file"), err)
		}
	}

	return nil
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

func writeDnsConfig(cfg *RunConfig, rootfs string) {
	writeResolvConf(cfg, rootfs)
	writeEtcHosts(cfg, rootfs)
}

// writeResolvConf will generate <stage1>/etc/rkt-resolv.conf if needed
func writeResolvConf(cfg *RunConfig, rootfs string) {
	if cfg.DNSConfMode.Resolv != "stage0" {
		return
	}

	if err := os.Mkdir(filepath.Join(rootfs, "etc"), common.DefaultRegularDirPerm); err != nil {
		if !os.IsExist(err) {
			log.Fatalf("error creating dir %q: %v\n", "/etc", err)
		}
	}
	resolvPath := filepath.Join(rootfs, "etc/rkt-resolv.conf")
	f, err := os.Create(resolvPath)
	if err != nil {
		log.Fatalf("error writing etc/rkt-resolv.conf: %v\n", err)
	}
	defer f.Close()

	_, err = f.WriteString(common.MakeResolvConf(cfg.DNSConfig, "Generated by rkt run"))
	if err != nil {
		log.Fatalf("error writing etc/rkt-resolv.conf: %v\n", err)
	}
}

// writeEtcHosts writes the file /etc/rkt-hosts into the stage1 rootfs.
// This will read defaults from <rootfs>/etc/hosts-fallback if it exists.
// Therefore, this should be called after the stage1 is mounted
func writeEtcHosts(cfg *RunConfig, rootfs string) {
	if cfg.DNSConfMode.Hosts != "stage0" {
		return
	}

	// Read <stage1>/rootfs/etc/hosts-fallback to get some sane defaults
	hostsTextb, err := ioutil.ReadFile(filepath.Join(rootfs, "etc/hosts-fallback"))
	if err != nil {
		// fallback-fallback :-)
		hostsTextb = []byte("#created by rkt stage0\n127.0.0.1 localhost localhost.localdomain\n")
	}
	hostsText := string(hostsTextb)

	hostsText += "\n\n# Added by rkt run --hosts-entry\n"

	for ip, hostnames := range cfg.HostsEntries {
		hostsText = fmt.Sprintf("%s%s %s\n", hostsText, ip, strings.Join(hostnames, " "))
	}

	// Create /etc if it does not exist
	etcPath := filepath.Join(rootfs, "etc")
	if _, err := os.Stat(etcPath); err != nil && os.IsNotExist(err) {
		err = os.Mkdir(etcPath, 0755)
		if err != nil {
			log.FatalE("failed to make stage1 etc directory", err)
		}
	} else if err != nil {
		log.FatalE("Failed to stat stage1 etc", err)
	}

	hostsPath := filepath.Join(etcPath, "rkt-hosts")
	err = ioutil.WriteFile(hostsPath, []byte(hostsText), 0644)
	if err != nil {
		log.FatalE("failed to write etc/rkt-hosts", err)
	}
}

// Run mounts the right overlay filesystems and actually runs the prepared
// pod by exec()ing the stage1 init inside the pod filesystem.
func Run(cfg RunConfig, dir string, dataDir string) {
	privateUsers, err := preparedWithPrivateUsers(dir)
	if err != nil {
		log.FatalE("error preparing private users", err)
	}

	debug("Setting up stage1")
	if err := setupStage1Image(cfg, dir, cfg.UseOverlay); err != nil {
		log.FatalE("error setting up stage1", err)
	}
	debug("Wrote filesystem to %s\n", dir)

	for _, app := range cfg.Apps {
		if err := setupAppImage(cfg, app.Name, app.Image.ID, dir, cfg.UseOverlay); err != nil {
			log.FatalE("error setting up app image", err)
		}
	}

	destRootfs := common.Stage1RootfsPath(dir)

	writeDnsConfig(&cfg, destRootfs)

	if err := os.Setenv(common.EnvLockFd, fmt.Sprintf("%v", cfg.LockFd)); err != nil {
		log.FatalE("setting lock fd environment", err)
	}

	if err := os.Setenv(common.EnvSELinuxContext, fmt.Sprintf("%v", cfg.ProcessLabel)); err != nil {
		log.FatalE("setting SELinux context environment", err)
	}

	if err := os.Setenv(common.EnvSELinuxMountContext, fmt.Sprintf("%v", cfg.MountLabel)); err != nil {
		log.FatalE("setting SELinux mount context environment", err)
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

	if cfg.DNSConfMode.Hosts != "default" || cfg.DNSConfMode.Resolv != "default" {
		if interfaceVersionSupportsDNSConfMode(s1v) {
			args = append(args, fmt.Sprintf("--dns-conf-mode=resolv=%s,hosts=%s", cfg.DNSConfMode.Resolv, cfg.DNSConfMode.Hosts))
		} else {
			log.Printf("warning: --dns-conf-mode option not supported by stage1")
		}
	}

	if interfaceVersionSupportsInsecureOptions(s1v) {
		if cfg.InsecureCapabilities {
			args = append(args, "--disable-capabilities-restriction")
		}
		if cfg.InsecurePaths {
			args = append(args, "--disable-paths")
		}
		if cfg.InsecureSeccomp {
			args = append(args, "--disable-seccomp")
		}
	}

	if cfg.Mutable {
		mutable, err := supportsMutableEnvironment(dir)

		switch {
		case err != nil:
			log.FatalE("error determining stage1 mutable support", err)
		case !mutable:
			log.Fatalln("stage1 does not support mutable pods")
		}

		args = append(args, "--mutable")
	}

	args = append(args, cfg.UUID.String())

	// make sure the lock fd stays open across exec
	if err := sys.CloseOnExec(cfg.LockFd, false); err != nil {
		log.Fatalf("error clearing FD_CLOEXEC on lock fd")
	}

	tpmEvent := fmt.Sprintf("rkt: Rootfs: %s Manifest: %s Stage1 args: %s", cfg.CommonConfig.RootHash, cfg.CommonConfig.ManifestData, strings.Join(args, " "))
	// If there's no TPM available or there's a failure for some other
	// reason, ignore it and continue anyway. Long term we'll want policy
	// that enforces TPM behaviour, but we don't have any infrastructure
	// around that yet.
	_ = tpm.Extend(tpmEvent)

	debug("Execing %s", args)
	if err := syscall.Exec(args[0], args, os.Environ()); err != nil {
		log.FatalE("error execing init", err)
	}
}

// prepareAppImage renders and verifies the tree cache of the app image that
// corresponds to the given app name.
// When useOverlay is false, it attempts to render and expand the app image.
// It returns the tree store ID if overlay is being used.
func prepareAppImage(cfg PrepareConfig, appName types.ACName, img types.Hash, cdir string, useOverlay bool) (string, error) {
	debug("Loading image %s", img.String())

	am, err := cfg.Store.GetImageManifest(img.String())
	if err != nil {
		return "", errwrap.Wrap(errors.New("error getting the manifest"), err)
	}

	if _, hasOS := am.Labels.Get("os"); !hasOS {
		return "", fmt.Errorf("missing os label in the image manifest")
	}

	if _, hasArch := am.Labels.Get("arch"); !hasArch {
		return "", fmt.Errorf("missing arch label in the image manifest")
	}

	if err := types.IsValidOSArch(am.Labels.ToMap(), ValidOSArch); err != nil {
		return "", err
	}

	appInfoDir := common.AppInfoPath(cdir, appName)
	if err := os.MkdirAll(appInfoDir, common.DefaultRegularDirPerm); err != nil {
		return "", errwrap.Wrap(errors.New("error creating apps info directory"), err)
	}

	var treeStoreID string
	if useOverlay {
		if cfg.PrivateUsers.Shift > 0 {
			return "", fmt.Errorf("cannot use both overlay and user namespace: not implemented yet. (Try --no-overlay)")
		}

		treeStoreID, _, err = cfg.TreeStore.Render(img.String(), false)
		if err != nil {
			return "", errwrap.Wrap(errors.New("error rendering tree image"), err)
		}

		if !cfg.SkipTreeStoreCheck {
			hash, err := cfg.TreeStore.Check(treeStoreID)
			if err != nil {
				log.PrintE("warning: tree cache is in a bad state.  Rebuilding...", err)
				var err error
				treeStoreID, hash, err = cfg.TreeStore.Render(img.String(), true)
				if err != nil {
					return "", errwrap.Wrap(errors.New("error rendering tree image"), err)
				}
			}
			cfg.CommonConfig.RootHash = hash
		}

		if err := ioutil.WriteFile(common.AppTreeStoreIDPath(cdir, appName), []byte(treeStoreID), common.DefaultRegularFilePerm); err != nil {
			return "", errwrap.Wrap(errors.New("error writing app treeStoreID"), err)
		}
	} else {
		ad := common.AppPath(cdir, appName)

		err := os.MkdirAll(ad, common.DefaultRegularDirPerm)
		if err != nil {
			return "", errwrap.Wrap(errors.New("error creating image directory"), err)
		}

		shiftedUid, shiftedGid, err := cfg.PrivateUsers.ShiftRange(uint32(os.Getuid()), uint32(os.Getgid()))
		if err != nil {
			return "", errwrap.Wrap(errors.New("error getting uid, gid"), err)
		}

		if err := os.Chown(ad, int(shiftedUid), int(shiftedGid)); err != nil {
			return "", errwrap.Wrap(fmt.Errorf("error shifting app %q's stage2 dir", appName), err)
		}

		if err := aci.RenderACIWithImageID(img, ad, cfg.Store, cfg.PrivateUsers); err != nil {
			return "", errwrap.Wrap(errors.New("error rendering ACI"), err)
		}
	}

	if err := writeManifest(*cfg.CommonConfig, img, appInfoDir); err != nil {
		return "", errwrap.Wrap(errors.New("error writing manifest"), err)
	}

	return treeStoreID, nil
}

// setupAppImage mounts the overlay filesystem for the app image that
// corresponds to the given hash if useOverlay is true.
// It also creates an mtab file in the application's rootfs if one is not
// present.
func setupAppImage(cfg RunConfig, appName types.ACName, img types.Hash, cdir string, useOverlay bool) error {
	ad := common.AppPath(cdir, appName)
	if useOverlay {
		err := os.MkdirAll(ad, common.DefaultRegularDirPerm)
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
	return ensureMtabExists(filepath.Join(ad, "rootfs"))
}

// ensureMtabExists creates a symlink from /etc/mtab -> /proc/self/mounts if
// nothing exists at /etc/mtab.
// Various tools, such as mount from util-linux 2.25, expect the mtab file to
// be populated.
func ensureMtabExists(rootfs string) error {
	stat, err := os.Stat(filepath.Join(rootfs, "etc"))
	if os.IsNotExist(err) {
		// If your image has no /etc you don't get /etc/mtab either
		return nil
	}
	if err != nil {
		return errwrap.Wrap(errors.New("error determining if /etc existed in the image"), err)
	}
	if !stat.IsDir() {
		return nil
	}
	mtabPath := filepath.Join(rootfs, "etc", "mtab")
	if _, err = os.Lstat(mtabPath); err == nil {
		// If the image already has an mtab, don't replace it
		return nil
	}
	if !os.IsNotExist(err) {
		return errwrap.Wrap(errors.New("error determining if /etc/mtab exists in the image"), err)
	}

	target := "../proc/self/mounts"
	err = os.Symlink(target, mtabPath)
	if err != nil {
		return errwrap.Wrap(errors.New("error creating mtab symlink"), err)
	}
	return nil
}

// prepareStage1Image renders and verifies tree cache of the given hash
// when using overlay.
// When useOverlay is false, it attempts to render and expand the stage1.
func prepareStage1Image(cfg PrepareConfig, cdir string) error {
	s1 := common.Stage1ImagePath(cdir)
	if err := os.MkdirAll(s1, common.DefaultRegularDirPerm); err != nil {
		return errwrap.Wrap(errors.New("error creating stage1 directory"), err)
	}

	treeStoreID, _, err := cfg.TreeStore.Render(cfg.Stage1Image.String(), false)
	if err != nil {
		return errwrap.Wrap(errors.New("error rendering tree image"), err)
	}

	if !cfg.SkipTreeStoreCheck {
		hash, err := cfg.TreeStore.Check(treeStoreID)
		if err != nil {
			log.Printf("warning: tree cache is in a bad state: %v. Rebuilding...", err)
			var err error
			treeStoreID, hash, err = cfg.TreeStore.Render(cfg.Stage1Image.String(), true)
			if err != nil {
				return errwrap.Wrap(errors.New("error rendering tree image"), err)
			}
		}
		cfg.CommonConfig.RootHash = hash
	}

	if err := writeManifest(*cfg.CommonConfig, cfg.Stage1Image, s1); err != nil {
		return errwrap.Wrap(errors.New("error writing manifest"), err)
	}

	if !cfg.UseOverlay {
		destRootfs := filepath.Join(s1, "rootfs")
		cachedTreePath := cfg.TreeStore.GetRootFS(treeStoreID)
		if err := fileutil.CopyTree(cachedTreePath, destRootfs, cfg.PrivateUsers); err != nil {
			return errwrap.Wrap(errors.New("error rendering ACI"), err)
		}
	}

	fn := path.Join(cdir, common.Stage1TreeStoreIDFilename)
	if err := ioutil.WriteFile(fn, []byte(treeStoreID), common.DefaultRegularFilePerm); err != nil {
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
	if err := ioutil.WriteFile(filepath.Join(dest, "manifest"), mb, common.DefaultRegularFilePerm); err != nil {
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
// overlay filesystem. It mounts an overlay filesystem from the cached tree of
// the image as rootfs.
func overlayRender(cfg RunConfig, treeStoreID string, cdir string, dest string, appName string) error {
	cachedTreePath := cfg.TreeStore.GetRootFS(treeStoreID)
	mc, err := prepareOverlay(cachedTreePath, treeStoreID, cdir, dest, appName, cfg.MountLabel,
		cfg.RktGid, common.DefaultRegularDirPerm)
	if err != nil {
		return errwrap.Wrap(errors.New("problem preparing overlay directories"), err)
	}
	if err = overlay.Mount(mc); err != nil {
		return errwrap.Wrap(errors.New("problem mounting overlay filesystem"), err)
	}

	return nil
}

// prepateOverlay sets up the needed directories, files and permissions for the
// overlay-rendered pods
func prepareOverlay(lower, treeStoreID, cdir, dest, appName, lbl string,
	gid int, fm os.FileMode) (*overlay.MountCfg, error) {
	fi, err := os.Stat(lower)
	if err != nil {
		return nil, err
	}
	imgMode := fi.Mode()

	dst := path.Join(dest, "rootfs")
	if err := os.MkdirAll(dst, imgMode); err != nil {
		return nil, err
	}

	overlayDir := path.Join(cdir, "overlay")
	if err := os.MkdirAll(overlayDir, fm); err != nil {
		return nil, err
	}

	// Since the parent directory (rkt/pods/$STATE/$POD_UUID) has the 'S_ISGID' bit, here
	// we need to explicitly turn the bit off when creating this overlay
	// directory so that it won't inherit the bit. Otherwise the files
	// created by users within the pod will inherit the 'S_ISGID' bit
	// as well.
	if err := os.Chmod(overlayDir, fm); err != nil {
		return nil, err
	}

	imgDir := path.Join(overlayDir, treeStoreID)
	if err := os.MkdirAll(imgDir, fm); err != nil {
		return nil, err
	}
	// Also make 'rkt/pods/$STATE/$POD_UUID/overlay/$IMAGE_ID' to be readable by 'rkt' group
	// As 'rkt' status will read the 'rkt/pods/$STATE/$POD_UUID/overlay/$IMAGE_ID/upper/rkt/status/$APP'
	// to get exgid
	if err := os.Chown(imgDir, -1, gid); err != nil {
		return nil, err
	}

	upper := path.Join(imgDir, "upper", appName)
	if err := os.MkdirAll(upper, imgMode); err != nil {
		return nil, err
	}
	if err := label.SetFileLabel(upper, lbl); err != nil {
		return nil, err
	}

	work := path.Join(imgDir, "work", appName)
	if err := os.MkdirAll(work, fm); err != nil {
		return nil, err
	}
	if err := label.SetFileLabel(work, lbl); err != nil {
		return nil, err
	}

	return &overlay.MountCfg{lower, upper, work, dst, lbl}, nil
}
