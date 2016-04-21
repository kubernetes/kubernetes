/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package rkt

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	appcschema "github.com/appc/spec/schema"
	appctypes "github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/unit"
	rktapi "github.com/coreos/rkt/api/v1alpha"
	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/errors"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/sets"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	utilwait "k8s.io/kubernetes/pkg/util/wait"
)

const (
	RktType                      = "rkt"
	DefaultRktAPIServiceEndpoint = "localhost:15441"

	minimumAppcVersion       = "0.7.4"
	minimumRktBinVersion     = "1.2.1"
	recommendedRktBinVersion = "1.2.1"
	minimumRktApiVersion     = "1.0.0-alpha"
	minimumSystemdVersion    = "219"

	systemdServiceDir = "/run/systemd/system"
	rktDataDir        = "/var/lib/rkt"
	rktLocalConfigDir = "/etc/rkt"

	kubernetesUnitPrefix  = "k8s"
	unitKubernetesSection = "X-Kubernetes"
	unitPodName           = "POD"
	unitRktID             = "RktID"
	unitRestartCount      = "RestartCount"

	k8sRktKubeletAnno                = "rkt.kubernetes.io/managed-by-kubelet"
	k8sRktKubeletAnnoValue           = "true"
	k8sRktUIDAnno                    = "rkt.kubernetes.io/uid"
	k8sRktNameAnno                   = "rkt.kubernetes.io/name"
	k8sRktNamespaceAnno              = "rkt.kubernetes.io/namespace"
	k8sRktContainerHashAnno          = "rkt.kubernetes.io/container-hash"
	k8sRktRestartCountAnno           = "rkt.kubernetes.io/restart-count"
	k8sRktTerminationMessagePathAnno = "rkt.kubernetes.io/termination-message-path"
	dockerPrefix                     = "docker://"

	authDir            = "auth.d"
	dockerAuthTemplate = `{"rktKind":"dockerAuth","rktVersion":"v1","registries":[%q],"credentials":{"user":%q,"password":%q}}`

	defaultRktAPIServiceAddr = "localhost:15441"
	defaultNetworkName       = "rkt.kubernetes.io"

	// ndots specifies the minimum number of dots that a domain name must contain for the resolver to consider it as FQDN (fully-qualified)
	// we want to able to consider SRV lookup names like _dns._udp.kube-dns.default.svc to be considered relative.
	// hence, setting ndots to be 5.
	// TODO(yifan): Move this and dockertools.ndotsDNSOption to a common package.
	defaultDNSOption = "ndots:5"

	// Annotations for the ENTRYPOINT and CMD for an ACI that's converted from Docker image.
	// TODO(yifan): Import them from docker2aci. See https://github.com/appc/docker2aci/issues/133.
	appcDockerEntrypoint = "appc.io/docker/entrypoint"
	appcDockerCmd        = "appc.io/docker/cmd"

	// TODO(yifan): Reuse this const with Docker runtime.
	minimumGracePeriodInSeconds = 2
)

// Runtime implements the Containerruntime for rkt. The implementation
// uses systemd, so in order to run this runtime, systemd must be installed
// on the machine.
type Runtime struct {
	systemd systemdInterface
	// The grpc client for rkt api-service.
	apisvcConn *grpc.ClientConn
	apisvc     rktapi.PublicAPIClient
	config     *Config
	// TODO(yifan): Refactor this to be generic keyring.
	dockerKeyring credentialprovider.DockerKeyring

	containerRefManager *kubecontainer.RefManager
	runtimeHelper       kubecontainer.RuntimeHelper
	recorder            record.EventRecorder
	livenessManager     proberesults.Manager
	volumeGetter        volumeGetter
	imagePuller         kubecontainer.ImagePuller
	runner              kubecontainer.HandlerRunner
	execer              utilexec.Interface
	os                  kubecontainer.OSInterface

	// used for a systemd Exec, which requires the full path.
	touchPath string

	versions versions
}

var _ kubecontainer.Runtime = &Runtime{}

// TODO(yifan): Remove this when volumeManager is moved to separate package.
type volumeGetter interface {
	GetVolumes(podUID types.UID) (kubecontainer.VolumeMap, bool)
}

// New creates the rkt container runtime which implements the container runtime interface.
// It will test if the rkt binary is in the $PATH, and whether we can get the
// version of it. If so, creates the rkt container runtime, otherwise returns an error.
func New(
	apiEndpoint string,
	config *Config,
	runtimeHelper kubecontainer.RuntimeHelper,
	recorder record.EventRecorder,
	containerRefManager *kubecontainer.RefManager,
	livenessManager proberesults.Manager,
	volumeGetter volumeGetter,
	httpClient kubetypes.HttpGetter,
	execer utilexec.Interface,
	os kubecontainer.OSInterface,
	imageBackOff *flowcontrol.Backoff,
	serializeImagePulls bool,
) (*Runtime, error) {
	// Create dbus connection.
	systemd, err := newSystemd()
	if err != nil {
		return nil, fmt.Errorf("rkt: cannot create systemd interface: %v", err)
	}

	// TODO(yifan): Use secure connection.
	apisvcConn, err := grpc.Dial(apiEndpoint, grpc.WithInsecure())
	if err != nil {
		return nil, fmt.Errorf("rkt: cannot connect to rkt api service: %v", err)
	}

	// TODO(yifan): Get the rkt path from API service.
	if config.Path == "" {
		// No default rkt path was set, so try to find one in $PATH.
		var err error
		config.Path, err = execer.LookPath("rkt")
		if err != nil {
			return nil, fmt.Errorf("cannot find rkt binary: %v", err)
		}
	}

	touchPath, err := execer.LookPath("touch")
	if err != nil {
		return nil, fmt.Errorf("cannot find touch binary: %v", err)
	}

	rkt := &Runtime{
		systemd:             systemd,
		apisvcConn:          apisvcConn,
		apisvc:              rktapi.NewPublicAPIClient(apisvcConn),
		config:              config,
		dockerKeyring:       credentialprovider.NewDockerKeyring(),
		containerRefManager: containerRefManager,
		runtimeHelper:       runtimeHelper,
		recorder:            recorder,
		livenessManager:     livenessManager,
		volumeGetter:        volumeGetter,
		execer:              execer,
		os:                  os,
		touchPath:           touchPath,
	}

	rkt.config, err = rkt.getConfig(rkt.config)
	if err != nil {
		return nil, fmt.Errorf("rkt: cannot get config from rkt api service: %v", err)
	}

	rkt.runner = lifecycle.NewHandlerRunner(httpClient, rkt, rkt)

	if serializeImagePulls {
		rkt.imagePuller = kubecontainer.NewSerializedImagePuller(recorder, rkt, imageBackOff)
	} else {
		rkt.imagePuller = kubecontainer.NewImagePuller(recorder, rkt, imageBackOff)
	}

	if err := rkt.getVersions(); err != nil {
		return nil, fmt.Errorf("rkt: error getting version info: %v", err)
	}

	return rkt, nil
}

func (r *Runtime) buildCommand(args ...string) *exec.Cmd {
	cmd := exec.Command(r.config.Path)
	cmd.Args = append(cmd.Args, r.config.buildGlobalOptions()...)
	cmd.Args = append(cmd.Args, args...)
	return cmd
}

// convertToACName converts a string into ACName.
func convertToACName(name string) appctypes.ACName {
	// Note that as the 'name' already matches 'DNS_LABEL'
	// defined in pkg/api/types.go, there shouldn't be error or panic.
	acname, _ := appctypes.SanitizeACName(name)
	return *appctypes.MustACName(acname)
}

// runCommand invokes rkt binary with arguments and returns the result
// from stdout in a list of strings. Each string in the list is a line.
func (r *Runtime) runCommand(args ...string) ([]string, error) {
	glog.V(4).Info("rkt: Run command:", args)

	var stdout, stderr bytes.Buffer
	cmd := r.buildCommand(args...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("failed to run %v: %v\nstdout: %v\nstderr: %v", args, err, stdout.String(), stderr.String())
	}
	return strings.Split(strings.TrimSpace(stdout.String()), "\n"), nil
}

// makePodServiceFileName constructs the unit file name for a pod using its rkt pod uuid.
func makePodServiceFileName(uuid string) string {
	// TODO(yifan): Add name for readability? We need to consider the
	// limit of the length.
	return fmt.Sprintf("%s_%s.service", kubernetesUnitPrefix, uuid)
}

// setIsolators sets the apps' isolators according to the security context and resource spec.
func setIsolators(app *appctypes.App, c *api.Container, ctx *api.SecurityContext) error {
	var isolators []appctypes.Isolator

	// Capabilities isolators.
	if ctx != nil {
		var addCaps, dropCaps []string

		if ctx.Capabilities != nil {
			addCaps, dropCaps = securitycontext.MakeCapabilities(ctx.Capabilities.Add, ctx.Capabilities.Drop)
		}
		if ctx.Privileged != nil && *ctx.Privileged {
			addCaps, dropCaps = allCapabilities(), []string{}
		}
		if len(addCaps) > 0 {
			set, err := appctypes.NewLinuxCapabilitiesRetainSet(addCaps...)
			if err != nil {
				return err
			}
			isolators = append(isolators, set.AsIsolator())
		}
		if len(dropCaps) > 0 {
			set, err := appctypes.NewLinuxCapabilitiesRevokeSet(dropCaps...)
			if err != nil {
				return err
			}
			isolators = append(isolators, set.AsIsolator())
		}
	}

	// Resources isolators.
	type resource struct {
		limit   string
		request string
	}

	// If limit is empty, populate it with request and vice versa.
	resources := make(map[api.ResourceName]*resource)
	for name, quantity := range c.Resources.Limits {
		resources[name] = &resource{limit: quantity.String(), request: quantity.String()}
	}
	for name, quantity := range c.Resources.Requests {
		r, ok := resources[name]
		if ok {
			r.request = quantity.String()
			continue
		}
		resources[name] = &resource{limit: quantity.String(), request: quantity.String()}
	}

	for name, res := range resources {
		switch name {
		case api.ResourceCPU:
			cpu, err := appctypes.NewResourceCPUIsolator(res.request, res.limit)
			if err != nil {
				return err
			}
			isolators = append(isolators, cpu.AsIsolator())
		case api.ResourceMemory:
			memory, err := appctypes.NewResourceMemoryIsolator(res.request, res.limit)
			if err != nil {
				return err
			}
			isolators = append(isolators, memory.AsIsolator())
		default:
			return fmt.Errorf("resource type not supported: %v", name)
		}
	}

	mergeIsolators(app, isolators)
	return nil
}

// mergeIsolators replaces the app.Isolators with isolators.
func mergeIsolators(app *appctypes.App, isolators []appctypes.Isolator) {
	for _, is := range isolators {
		found := false
		for j, js := range app.Isolators {
			if is.Name.Equals(js.Name) {
				switch is.Name {
				case appctypes.LinuxCapabilitiesRetainSetName:
					// TODO(yifan): More fine grain merge for capability set instead of override.
					fallthrough
				case appctypes.LinuxCapabilitiesRevokeSetName:
					fallthrough
				case appctypes.ResourceCPUName:
					fallthrough
				case appctypes.ResourceMemoryName:
					app.Isolators[j] = is
				default:
					panic(fmt.Sprintf("unexpected isolator name: %v", is.Name))
				}
				found = true
				break
			}
		}
		if !found {
			app.Isolators = append(app.Isolators, is)
		}
	}
}

// mergeEnv merges the optEnv with the image's environments.
// The environments defined in the image will be overridden by
// the ones with the same name in optEnv.
func mergeEnv(app *appctypes.App, optEnv []kubecontainer.EnvVar) {
	envMap := make(map[string]string)
	for _, e := range app.Environment {
		envMap[e.Name] = e.Value
	}
	for _, e := range optEnv {
		envMap[e.Name] = e.Value
	}
	app.Environment = nil
	for name, value := range envMap {
		app.Environment = append(app.Environment, appctypes.EnvironmentVariable{
			Name:  name,
			Value: value,
		})
	}
}

// mergeMounts merges the optMounts with the image's mount points.
// The mount points defined in the image will be overridden by the ones
// with the same name in optMounts.
func mergeMounts(app *appctypes.App, optMounts []kubecontainer.Mount) {
	mountMap := make(map[appctypes.ACName]appctypes.MountPoint)
	for _, m := range app.MountPoints {
		mountMap[m.Name] = m
	}
	for _, m := range optMounts {
		mpName := convertToACName(m.Name)
		mountMap[mpName] = appctypes.MountPoint{
			Name:     mpName,
			Path:     m.ContainerPath,
			ReadOnly: m.ReadOnly,
		}
	}
	app.MountPoints = nil
	for _, mount := range mountMap {
		app.MountPoints = append(app.MountPoints, mount)
	}
}

// mergePortMappings merges the optPortMappings with the image's port mappings.
// The port mappings defined in the image will be overridden by the ones
// with the same name in optPortMappings.
func mergePortMappings(app *appctypes.App, optPortMappings []kubecontainer.PortMapping) {
	portMap := make(map[appctypes.ACName]appctypes.Port)
	for _, p := range app.Ports {
		portMap[p.Name] = p
	}
	for _, p := range optPortMappings {
		pName := convertToACName(p.Name)
		portMap[pName] = appctypes.Port{
			Name:     pName,
			Protocol: string(p.Protocol),
			Port:     uint(p.ContainerPort),
		}
	}
	app.Ports = nil
	for _, port := range portMap {
		app.Ports = append(app.Ports, port)
	}
}

func verifyNonRoot(app *appctypes.App, ctx *api.SecurityContext) error {
	if ctx != nil && ctx.RunAsNonRoot != nil && *ctx.RunAsNonRoot {
		if ctx.RunAsUser != nil && *ctx.RunAsUser == 0 {
			return fmt.Errorf("container's runAsUser breaks non-root policy")
		}
		if ctx.RunAsUser == nil && app.User == "0" {
			return fmt.Errorf("container has no runAsUser and image will run as root")
		}
	}
	return nil
}

func setSupplementaryGIDs(app *appctypes.App, podCtx *api.PodSecurityContext) {
	if podCtx != nil {
		app.SupplementaryGIDs = app.SupplementaryGIDs[:0]
		for _, v := range podCtx.SupplementalGroups {
			app.SupplementaryGIDs = append(app.SupplementaryGIDs, int(v))
		}
		if podCtx.FSGroup != nil {
			app.SupplementaryGIDs = append(app.SupplementaryGIDs, int(*podCtx.FSGroup))
		}
	}
}

// setApp merges the container spec with the image's manifest.
func setApp(imgManifest *appcschema.ImageManifest, c *api.Container, opts *kubecontainer.RunContainerOptions, ctx *api.SecurityContext, podCtx *api.PodSecurityContext) error {
	app := imgManifest.App

	// Set up Exec.
	var command, args []string
	cmd, ok := imgManifest.Annotations.Get(appcDockerEntrypoint)
	if ok {
		err := json.Unmarshal([]byte(cmd), &command)
		if err != nil {
			return fmt.Errorf("cannot unmarshal ENTRYPOINT %q: %v", cmd, err)
		}
	}
	ag, ok := imgManifest.Annotations.Get(appcDockerCmd)
	if ok {
		err := json.Unmarshal([]byte(ag), &args)
		if err != nil {
			return fmt.Errorf("cannot unmarshal CMD %q: %v", ag, err)
		}
	}
	userCommand, userArgs := kubecontainer.ExpandContainerCommandAndArgs(c, opts.Envs)

	if len(userCommand) > 0 {
		command = userCommand
		args = nil // If 'command' is specified, then drop the default args.
	}
	if len(userArgs) > 0 {
		args = userArgs
	}

	exec := append(command, args...)
	if len(exec) > 0 {
		app.Exec = exec
	}

	// Set UID and GIDs.
	if err := verifyNonRoot(app, ctx); err != nil {
		return err
	}
	if ctx != nil && ctx.RunAsUser != nil {
		app.User = strconv.Itoa(int(*ctx.RunAsUser))
	}
	setSupplementaryGIDs(app, podCtx)

	// If 'User' or 'Group' are still empty at this point,
	// then apply the root UID and GID.
	// TODO(yifan): Instead of using root GID, we should use
	// the GID which the user is in.
	if app.User == "" {
		app.User = "0"
	}
	if app.Group == "" {
		app.Group = "0"
	}

	// Set working directory.
	if len(c.WorkingDir) > 0 {
		app.WorkingDirectory = c.WorkingDir
	}

	// Notes that we don't create Mounts section in the pod manifest here,
	// as Mounts will be automatically generated by rkt.
	mergeMounts(app, opts.Mounts)
	mergeEnv(app, opts.Envs)
	mergePortMappings(app, opts.PortMappings)

	return setIsolators(app, c, ctx)
}

// makePodManifest transforms a kubelet pod spec to the rkt pod manifest.
func (r *Runtime) makePodManifest(pod *api.Pod, pullSecrets []api.Secret) (*appcschema.PodManifest, error) {
	manifest := appcschema.BlankPodManifest()

	listResp, err := r.apisvc.ListPods(context.Background(), &rktapi.ListPodsRequest{
		Detail:  true,
		Filters: kubernetesPodFilters(pod.UID),
	})
	if err != nil {
		return nil, fmt.Errorf("couldn't list pods: %v", err)
	}

	restartCount := 0
	for _, pod := range listResp.Pods {
		manifest := &appcschema.PodManifest{}
		err = json.Unmarshal(pod.Manifest, manifest)
		if err != nil {
			glog.Warningf("rkt: error unmatshaling pod manifest: %v", err)
			continue
		}

		if countString, ok := manifest.Annotations.Get(k8sRktRestartCountAnno); ok {
			num, err := strconv.Atoi(countString)
			if err != nil {
				glog.Warningf("rkt: error reading restart count on pod: %v", err)
				continue
			}
			if num+1 > restartCount {
				restartCount = num + 1
			}
		}
	}

	manifest.Annotations.Set(*appctypes.MustACIdentifier(k8sRktKubeletAnno), k8sRktKubeletAnnoValue)
	manifest.Annotations.Set(*appctypes.MustACIdentifier(k8sRktUIDAnno), string(pod.UID))
	manifest.Annotations.Set(*appctypes.MustACIdentifier(k8sRktNameAnno), pod.Name)
	manifest.Annotations.Set(*appctypes.MustACIdentifier(k8sRktNamespaceAnno), pod.Namespace)
	manifest.Annotations.Set(*appctypes.MustACIdentifier(k8sRktRestartCountAnno), strconv.Itoa(restartCount))

	for _, c := range pod.Spec.Containers {
		err := r.newAppcRuntimeApp(pod, c, pullSecrets, manifest)
		if err != nil {
			return nil, err
		}
	}

	volumeMap, ok := r.volumeGetter.GetVolumes(pod.UID)
	if !ok {
		return nil, fmt.Errorf("cannot get the volumes for pod %q", format.Pod(pod))
	}

	// Set global volumes.
	for vname, volume := range volumeMap {
		manifest.Volumes = append(manifest.Volumes, appctypes.Volume{
			Name:   convertToACName(vname),
			Kind:   "host",
			Source: volume.Mounter.GetPath(),
		})
	}

	// TODO(yifan): Set pod-level isolators once it's supported in kubernetes.
	return manifest, nil
}

// TODO(yifan): Can make rkt handle this when '--net=host'. See https://github.com/coreos/rkt/issues/2430.
func makeHostNetworkMount(opts *kubecontainer.RunContainerOptions) (*kubecontainer.Mount, *kubecontainer.Mount) {
	hostsMount := kubecontainer.Mount{
		Name:          "kubernetes-hostnetwork-hosts-conf",
		ContainerPath: "/etc/hosts",
		HostPath:      "/etc/hosts",
		ReadOnly:      true,
	}
	resolvMount := kubecontainer.Mount{
		Name:          "kubernetes-hostnetwork-resolv-conf",
		ContainerPath: "/etc/resolv.conf",
		HostPath:      "/etc/resolv.conf",
		ReadOnly:      true,
	}
	opts.Mounts = append(opts.Mounts, hostsMount, resolvMount)
	return &hostsMount, &resolvMount
}

// podFinishedMarkerPath returns the path to a file which should be used to
// indicate the pod exiting, and the time thereof.
// If the file at the path does not exist, the pod should not be exited. If it
// does exist, then the ctime of the file should indicate the time the pod
// exited.
func podFinishedMarkerPath(podDir string, rktUID string) string {
	return filepath.Join(podDir, "finished-"+rktUID)
}

func podFinishedMarkCommand(touchPath, podDir, rktUID string) string {
	// TODO, if the path has a `'` character in it, this breaks.
	return touchPath + " " + podFinishedMarkerPath(podDir, rktUID)
}

// podFinishedAt returns the time that a pod exited, or a zero time if it has
// not.
func (r *Runtime) podFinishedAt(podUID types.UID, rktUID string) time.Time {
	markerFile := podFinishedMarkerPath(r.runtimeHelper.GetPodDir(podUID), rktUID)
	stat, err := r.os.Stat(markerFile)
	if err != nil {
		if !os.IsNotExist(err) {
			glog.Warningf("rkt: unexpected fs error checking pod finished marker: %v", err)
		}
		return time.Time{}
	}
	return stat.ModTime()
}

func makeContainerLogMount(opts *kubecontainer.RunContainerOptions, container *api.Container) (*kubecontainer.Mount, error) {
	if opts.PodContainerDir == "" || container.TerminationMessagePath == "" {
		return nil, nil
	}

	// In docker runtime, the container log path contains the container ID.
	// However, for rkt runtime, we cannot get the container ID before the
	// the container is launched, so here we generate a random uuid to enable
	// us to map a container's termination message path to an unique log file
	// on the disk.
	randomUID := util.NewUUID()
	containerLogPath := path.Join(opts.PodContainerDir, string(randomUID))
	fs, err := os.Create(containerLogPath)
	if err != nil {
		return nil, err
	}

	if err := fs.Close(); err != nil {
		return nil, err
	}

	mnt := kubecontainer.Mount{
		// Use a random name for the termination message mount, so that
		// when a container restarts, it will not overwrite the old termination
		// message.
		Name:          fmt.Sprintf("termination-message-%s", randomUID),
		ContainerPath: container.TerminationMessagePath,
		HostPath:      containerLogPath,
		ReadOnly:      false,
	}
	opts.Mounts = append(opts.Mounts, mnt)

	return &mnt, nil
}

func (r *Runtime) newAppcRuntimeApp(pod *api.Pod, c api.Container, pullSecrets []api.Secret, manifest *appcschema.PodManifest) error {
	if err, _ := r.imagePuller.PullImage(pod, &c, pullSecrets); err != nil {
		return nil
	}
	imgManifest, err := r.getImageManifest(c.Image)
	if err != nil {
		return err
	}

	if imgManifest.App == nil {
		imgManifest.App = new(appctypes.App)
	}

	imageID, err := r.getImageID(c.Image)
	if err != nil {
		return err
	}
	hash, err := appctypes.NewHash(imageID)
	if err != nil {
		return err
	}

	// TODO: determine how this should be handled for rkt
	opts, err := r.runtimeHelper.GenerateRunContainerOptions(pod, &c, "")
	if err != nil {
		return err
	}

	// create the container log file and make a mount pair.
	mnt, err := makeContainerLogMount(opts, &c)
	if err != nil {
		return err
	}

	// If run in 'hostnetwork' mode, then mount the host's /etc/resolv.conf and /etc/hosts,
	// and add volumes.
	var hostsMnt, resolvMnt *kubecontainer.Mount
	if kubecontainer.IsHostNetworkPod(pod) {
		hostsMnt, resolvMnt = makeHostNetworkMount(opts)
		manifest.Volumes = append(manifest.Volumes, appctypes.Volume{
			Name:   convertToACName(hostsMnt.Name),
			Kind:   "host",
			Source: hostsMnt.HostPath,
		})
		manifest.Volumes = append(manifest.Volumes, appctypes.Volume{
			Name:   convertToACName(resolvMnt.Name),
			Kind:   "host",
			Source: resolvMnt.HostPath,
		})
	}

	ctx := securitycontext.DetermineEffectiveSecurityContext(pod, &c)
	if err := setApp(imgManifest, &c, opts, ctx, pod.Spec.SecurityContext); err != nil {
		return err
	}

	ra := appcschema.RuntimeApp{
		Name:  convertToACName(c.Name),
		Image: appcschema.RuntimeImage{ID: *hash},
		App:   imgManifest.App,
		Annotations: []appctypes.Annotation{
			{
				Name:  *appctypes.MustACIdentifier(k8sRktContainerHashAnno),
				Value: strconv.FormatUint(kubecontainer.HashContainer(&c), 10),
			},
		},
	}

	if mnt != nil {
		ra.Annotations = append(ra.Annotations, appctypes.Annotation{
			Name:  *appctypes.MustACIdentifier(k8sRktTerminationMessagePathAnno),
			Value: mnt.HostPath,
		})

		manifest.Volumes = append(manifest.Volumes, appctypes.Volume{
			Name:   convertToACName(mnt.Name),
			Kind:   "host",
			Source: mnt.HostPath,
		})
	}

	manifest.Apps = append(manifest.Apps, ra)

	// Set global ports.
	for _, port := range opts.PortMappings {
		if port.HostPort == 0 {
			continue
		}
		manifest.Ports = append(manifest.Ports, appctypes.ExposedPort{
			Name:     convertToACName(port.Name),
			HostPort: uint(port.HostPort),
		})
	}

	return nil
}

func runningKubernetesPodFilters(uid types.UID) []*rktapi.PodFilter {
	return []*rktapi.PodFilter{
		{
			States: []rktapi.PodState{
				rktapi.PodState_POD_STATE_RUNNING,
			},
			Annotations: []*rktapi.KeyValue{
				{
					Key:   k8sRktKubeletAnno,
					Value: k8sRktKubeletAnnoValue,
				},
				{
					Key:   k8sRktUIDAnno,
					Value: string(uid),
				},
			},
		},
	}
}

func kubernetesPodFilters(uid types.UID) []*rktapi.PodFilter {
	return []*rktapi.PodFilter{
		{
			Annotations: []*rktapi.KeyValue{
				{
					Key:   k8sRktKubeletAnno,
					Value: k8sRktKubeletAnnoValue,
				},
				{
					Key:   k8sRktUIDAnno,
					Value: string(uid),
				},
			},
		},
	}
}

func newUnitOption(section, name, value string) *unit.UnitOption {
	return &unit.UnitOption{Section: section, Name: name, Value: value}
}

// apiPodToruntimePod converts an api.Pod to kubelet/container.Pod.
func apiPodToruntimePod(uuid string, pod *api.Pod) *kubecontainer.Pod {
	p := &kubecontainer.Pod{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		p.Containers = append(p.Containers, &kubecontainer.Container{
			ID:    buildContainerID(&containerID{uuid, c.Name}),
			Name:  c.Name,
			Image: c.Image,
			Hash:  kubecontainer.HashContainer(c),
		})
	}
	return p
}

// serviceFilePath returns the absolute path of the service file.
func serviceFilePath(serviceName string) string {
	return path.Join(systemdServiceDir, serviceName)
}

// generateRunCommand crafts a 'rkt run-prepared' command with necessary parameters.
func (r *Runtime) generateRunCommand(pod *api.Pod, uuid string) (string, error) {
	runPrepared := r.buildCommand("run-prepared").Args

	var hostname string
	var err error
	// Setup network configuration.
	if kubecontainer.IsHostNetworkPod(pod) {
		runPrepared = append(runPrepared, "--net=host")

		// TODO(yifan): Let runtimeHelper.GeneratePodHostNameAndDomain() to handle this.
		hostname, err = os.Hostname()
		if err != nil {
			return "", err
		}
	} else {
		runPrepared = append(runPrepared, fmt.Sprintf("--net=%s", defaultNetworkName))

		// Setup DNS.
		dnsServers, dnsSearches, err := r.runtimeHelper.GetClusterDNS(pod)
		if err != nil {
			return "", err
		}
		for _, server := range dnsServers {
			runPrepared = append(runPrepared, fmt.Sprintf("--dns=%s", server))
		}
		for _, search := range dnsSearches {
			runPrepared = append(runPrepared, fmt.Sprintf("--dns-search=%s", search))
		}
		if len(dnsServers) > 0 || len(dnsSearches) > 0 {
			runPrepared = append(runPrepared, fmt.Sprintf("--dns-opt=%s", defaultDNSOption))
		}

		// TODO(yifan): host domain is not being used.
		hostname, _, err = r.runtimeHelper.GeneratePodHostNameAndDomain(pod)
		if err != nil {
			return "", err
		}
	}
	runPrepared = append(runPrepared, fmt.Sprintf("--hostname=%s", hostname))
	runPrepared = append(runPrepared, uuid)
	return strings.Join(runPrepared, " "), nil
}

// preparePod will:
//
// 1. Invoke 'rkt prepare' to prepare the pod, and get the rkt pod uuid.
// 2. Create the unit file and save it under systemdUnitDir.
//
// On success, it will return a string that represents name of the unit file
// and the runtime pod.
func (r *Runtime) preparePod(pod *api.Pod, pullSecrets []api.Secret) (string, *kubecontainer.Pod, error) {
	// Generate the pod manifest from the pod spec.
	manifest, err := r.makePodManifest(pod, pullSecrets)
	if err != nil {
		return "", nil, err
	}
	manifestFile, err := ioutil.TempFile("", fmt.Sprintf("manifest-%s-", pod.Name))
	if err != nil {
		return "", nil, err
	}
	defer func() {
		manifestFile.Close()
		if err := os.Remove(manifestFile.Name()); err != nil {
			glog.Warningf("rkt: Cannot remove temp manifest file %q: %v", manifestFile.Name(), err)
		}
	}()

	data, err := json.Marshal(manifest)
	if err != nil {
		return "", nil, err
	}

	glog.V(4).Infof("Generating pod manifest for pod %q: %v", format.Pod(pod), string(data))
	// Since File.Write returns error if the written length is less than len(data),
	// so check error is enough for us.
	if _, err := manifestFile.Write(data); err != nil {
		return "", nil, err
	}

	// Run 'rkt prepare' to get the rkt UUID.
	cmds := []string{"prepare", "--quiet", "--pod-manifest", manifestFile.Name()}
	if r.config.Stage1Image != "" {
		cmds = append(cmds, "--stage1-path", r.config.Stage1Image)
	}
	output, err := r.runCommand(cmds...)
	if err != nil {
		return "", nil, err
	}
	if len(output) != 1 {
		return "", nil, fmt.Errorf("invalid output from 'rkt prepare': %v", output)
	}
	uuid := output[0]
	glog.V(4).Infof("'rkt prepare' returns %q", uuid)

	// Create systemd service file for the rkt pod.
	runPrepared, err := r.generateRunCommand(pod, uuid)
	if err != nil {
		return "", nil, fmt.Errorf("failed to generate 'rkt run-prepared' command: %v", err)
	}

	// TODO handle pod.Spec.HostPID
	// TODO handle pod.Spec.HostIPC

	// TODO per container finishedAt, not just per pod
	markPodFinished := podFinishedMarkCommand(r.touchPath, r.runtimeHelper.GetPodDir(pod.UID), uuid)
	units := []*unit.UnitOption{
		newUnitOption("Service", "ExecStart", runPrepared),
		newUnitOption("Service", "ExecStopPost", markPodFinished),
		// This enables graceful stop.
		newUnitOption("Service", "KillMode", "mixed"),
	}

	serviceName := makePodServiceFileName(uuid)
	glog.V(4).Infof("rkt: Creating service file %q for pod %q", serviceName, format.Pod(pod))
	serviceFile, err := os.Create(serviceFilePath(serviceName))
	if err != nil {
		return "", nil, err
	}
	if _, err := io.Copy(serviceFile, unit.Serialize(units)); err != nil {
		return "", nil, err
	}
	serviceFile.Close()

	return serviceName, apiPodToruntimePod(uuid, pod), nil
}

// generateEvents is a helper function that generates some container
// life cycle events for containers in a pod.
func (r *Runtime) generateEvents(runtimePod *kubecontainer.Pod, reason string, failure error) {
	// Set up container references.
	for _, c := range runtimePod.Containers {
		containerID := c.ID
		id, err := parseContainerID(containerID)
		if err != nil {
			glog.Warningf("Invalid container ID %q", containerID)
			continue
		}

		ref, ok := r.containerRefManager.GetRef(containerID)
		if !ok {
			glog.Warningf("No ref for container %q", containerID)
			continue
		}

		// Note that 'rkt id' is the pod id.
		uuid := utilstrings.ShortenString(id.uuid, 8)
		switch reason {
		case "Created":
			r.recorder.Eventf(ref, api.EventTypeNormal, kubecontainer.CreatedContainer, "Created with rkt id %v", uuid)
		case "Started":
			r.recorder.Eventf(ref, api.EventTypeNormal, kubecontainer.StartedContainer, "Started with rkt id %v", uuid)
		case "Failed":
			r.recorder.Eventf(ref, api.EventTypeWarning, kubecontainer.FailedToStartContainer, "Failed to start with rkt id %v with error %v", uuid, failure)
		case "Killing":
			r.recorder.Eventf(ref, api.EventTypeNormal, kubecontainer.KillingContainer, "Killing with rkt id %v", uuid)
		default:
			glog.Errorf("rkt: Unexpected event %q", reason)
		}
	}
	return
}

// RunPod first creates the unit file for a pod, and then
// starts the unit over d-bus.
func (r *Runtime) RunPod(pod *api.Pod, pullSecrets []api.Secret) error {
	glog.V(4).Infof("Rkt starts to run pod: name %q.", format.Pod(pod))

	name, runtimePod, prepareErr := r.preparePod(pod, pullSecrets)

	// Set container references and generate events.
	// If preparedPod fails, then send out 'failed' events for each container.
	// Otherwise, store the container references so we can use them later to send events.
	for i, c := range pod.Spec.Containers {
		ref, err := kubecontainer.GenerateContainerRef(pod, &c)
		if err != nil {
			glog.Errorf("Couldn't make a ref to pod %q, container %v: '%v'", format.Pod(pod), c.Name, err)
			continue
		}
		if prepareErr != nil {
			r.recorder.Eventf(ref, api.EventTypeWarning, kubecontainer.FailedToCreateContainer, "Failed to create rkt container with error: %v", prepareErr)
			continue
		}
		containerID := runtimePod.Containers[i].ID
		r.containerRefManager.SetRef(containerID, ref)
	}

	if prepareErr != nil {
		return prepareErr
	}

	r.generateEvents(runtimePod, "Created", nil)

	// RestartUnit has the same effect as StartUnit if the unit is not running, besides it can restart
	// a unit if the unit file is changed and reloaded.
	reschan := make(chan string)
	_, err := r.systemd.RestartUnit(name, "replace", reschan)
	if err != nil {
		r.generateEvents(runtimePod, "Failed", err)
		return err
	}

	res := <-reschan
	if res != "done" {
		err := fmt.Errorf("Failed to restart unit %q: %s", name, res)
		r.generateEvents(runtimePod, "Failed", err)
		return err
	}

	r.generateEvents(runtimePod, "Started", nil)

	// This is a temporary solution until we have a clean design on how
	// kubelet handles events. See https://github.com/kubernetes/kubernetes/issues/23084.
	if err := r.runLifecycleHooks(pod, runtimePod, lifecyclePostStartHook); err != nil {
		if errKill := r.KillPod(pod, *runtimePod, nil); errKill != nil {
			return errors.NewAggregate([]error{err, errKill})
		}
		return err
	}

	return nil
}

func (r *Runtime) runPreStopHook(containerID kubecontainer.ContainerID, pod *api.Pod, container *api.Container) error {
	glog.V(4).Infof("rkt: Running pre-stop hook for container %q of pod %q", container.Name, format.Pod(pod))
	return r.runner.Run(containerID, pod, container, container.Lifecycle.PreStop)
}

func (r *Runtime) runPostStartHook(containerID kubecontainer.ContainerID, pod *api.Pod, container *api.Container) error {
	glog.V(4).Infof("rkt: Running post-start hook for container %q of pod %q", container.Name, format.Pod(pod))
	cid, err := parseContainerID(containerID)
	if err != nil {
		return fmt.Errorf("cannot parse container ID %v", containerID)
	}

	isContainerRunning := func() (done bool, err error) {
		resp, err := r.apisvc.InspectPod(context.Background(), &rktapi.InspectPodRequest{Id: cid.uuid})
		if err != nil {
			return false, fmt.Errorf("failed to inspect rkt pod %q for pod %q", cid.uuid, format.Pod(pod))
		}

		for _, app := range resp.Pod.Apps {
			if app.Name == cid.appName {
				return app.State == rktapi.AppState_APP_STATE_RUNNING, nil
			}
		}
		return false, fmt.Errorf("failed to find container %q in rkt pod %q", cid.appName, cid.uuid)
	}

	// TODO(yifan): Polling the pod's state for now.
	timeout := time.Second * 5
	pollInterval := time.Millisecond * 500
	if err := utilwait.Poll(pollInterval, timeout, isContainerRunning); err != nil {
		return fmt.Errorf("rkt: Pod %q doesn't become running in %v: %v", format.Pod(pod), timeout, err)
	}

	return r.runner.Run(containerID, pod, container, container.Lifecycle.PostStart)
}

type lifecycleHookType string

const (
	lifecyclePostStartHook lifecycleHookType = "post-start"
	lifecyclePreStopHook   lifecycleHookType = "pre-stop"
)

func (r *Runtime) runLifecycleHooks(pod *api.Pod, runtimePod *kubecontainer.Pod, typ lifecycleHookType) error {
	var wg sync.WaitGroup
	var errlist []error
	errCh := make(chan error, len(pod.Spec.Containers))

	wg.Add(len(pod.Spec.Containers))

	for i, c := range pod.Spec.Containers {
		var hookFunc func(kubecontainer.ContainerID, *api.Pod, *api.Container) error

		switch typ {
		case lifecyclePostStartHook:
			if c.Lifecycle != nil && c.Lifecycle.PostStart != nil {
				hookFunc = r.runPostStartHook
			}
		case lifecyclePreStopHook:
			if c.Lifecycle != nil && c.Lifecycle.PreStop != nil {
				hookFunc = r.runPreStopHook
			}
		default:
			errCh <- fmt.Errorf("Unrecognized lifecycle hook type %q for container %q in pod %q", typ, c.Name, format.Pod(pod))
		}

		if hookFunc == nil {
			wg.Done()
			continue
		}

		container := &pod.Spec.Containers[i]
		runtimeContainer := runtimePod.FindContainerByName(container.Name)
		if runtimeContainer == nil {
			// Container already gone.
			wg.Done()
			continue
		}
		containerID := runtimeContainer.ID

		go func() {
			defer wg.Done()
			if err := hookFunc(containerID, pod, container); err != nil {
				glog.Errorf("rkt: Failed to run %s hook for container %q of pod %q: %v", typ, container.Name, format.Pod(pod), err)
				errCh <- err
			} else {
				glog.V(4).Infof("rkt: %s hook completed successfully for container %q of pod %q", typ, container.Name, format.Pod(pod))
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		errlist = append(errlist, err)
	}
	return errors.NewAggregate(errlist)
}

// convertRktPod will convert a rktapi.Pod to a kubecontainer.Pod
func (r *Runtime) convertRktPod(rktpod *rktapi.Pod) (*kubecontainer.Pod, error) {
	manifest := &appcschema.PodManifest{}
	err := json.Unmarshal(rktpod.Manifest, manifest)
	if err != nil {
		return nil, err
	}

	podUID, ok := manifest.Annotations.Get(k8sRktUIDAnno)
	if !ok {
		return nil, fmt.Errorf("pod is missing annotation %s", k8sRktUIDAnno)
	}
	podName, ok := manifest.Annotations.Get(k8sRktNameAnno)
	if !ok {
		return nil, fmt.Errorf("pod is missing annotation %s", k8sRktNameAnno)
	}
	podNamespace, ok := manifest.Annotations.Get(k8sRktNamespaceAnno)
	if !ok {
		return nil, fmt.Errorf("pod is missing annotation %s", k8sRktNamespaceAnno)
	}

	kubepod := &kubecontainer.Pod{
		ID:        types.UID(podUID),
		Name:      podName,
		Namespace: podNamespace,
	}

	for i, app := range rktpod.Apps {
		// The order of the apps is determined by the rkt pod manifest.
		// TODO(yifan): Let the server to unmarshal the annotations? https://github.com/coreos/rkt/issues/1872
		hashStr, ok := manifest.Apps[i].Annotations.Get(k8sRktContainerHashAnno)
		if !ok {
			return nil, fmt.Errorf("app %q is missing annotation %s", app.Name, k8sRktContainerHashAnno)
		}
		containerHash, err := strconv.ParseUint(hashStr, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("couldn't parse container's hash %q: %v", hashStr, err)
		}

		kubepod.Containers = append(kubepod.Containers, &kubecontainer.Container{
			ID:   buildContainerID(&containerID{rktpod.Id, app.Name}),
			Name: app.Name,
			// By default, the version returned by rkt API service will be "latest" if not specified.
			Image: fmt.Sprintf("%s:%s", app.Image.Name, app.Image.Version),
			Hash:  containerHash,
			State: appStateToContainerState(app.State),
		})
	}

	return kubepod, nil
}

// GetPods runs 'systemctl list-unit' and 'rkt list' to get the list of rkt pods.
// Then it will use the result to construct a list of container runtime pods.
// If all is false, then only running pods will be returned, otherwise all pods will be
// returned.
func (r *Runtime) GetPods(all bool) ([]*kubecontainer.Pod, error) {
	glog.V(4).Infof("Rkt getting pods")

	listReq := &rktapi.ListPodsRequest{
		Detail: true,
		Filters: []*rktapi.PodFilter{
			{
				Annotations: []*rktapi.KeyValue{
					{
						Key:   k8sRktKubeletAnno,
						Value: k8sRktKubeletAnnoValue,
					},
				},
			},
		},
	}
	if !all {
		listReq.Filters[0].States = []rktapi.PodState{rktapi.PodState_POD_STATE_RUNNING}
	}
	listResp, err := r.apisvc.ListPods(context.Background(), listReq)
	if err != nil {
		return nil, fmt.Errorf("couldn't list pods: %v", err)
	}

	pods := make(map[types.UID]*kubecontainer.Pod)
	var podIDs []types.UID
	for _, pod := range listResp.Pods {
		pod, err := r.convertRktPod(pod)
		if err != nil {
			glog.Warningf("rkt: Cannot construct pod from unit file: %v.", err)
			continue
		}

		// Group pods together.
		oldPod, found := pods[pod.ID]
		if !found {
			pods[pod.ID] = pod
			podIDs = append(podIDs, pod.ID)
			continue
		}

		oldPod.Containers = append(oldPod.Containers, pod.Containers...)
	}

	// Convert map to list, using the consistent order from the podIDs array.
	var result []*kubecontainer.Pod
	for _, id := range podIDs {
		result = append(result, pods[id])
	}

	return result, nil
}

func (r *Runtime) waitPreStopHooks(pod *api.Pod, runningPod *kubecontainer.Pod) {
	gracePeriod := int64(minimumGracePeriodInSeconds)
	switch {
	case pod.DeletionGracePeriodSeconds != nil:
		gracePeriod = *pod.DeletionGracePeriodSeconds
	case pod.Spec.TerminationGracePeriodSeconds != nil:
		gracePeriod = *pod.Spec.TerminationGracePeriodSeconds
	}

	done := make(chan struct{})
	go func() {
		if err := r.runLifecycleHooks(pod, runningPod, lifecyclePreStopHook); err != nil {
			glog.Errorf("rkt: Some pre-stop hooks failed for pod %q: %v", format.Pod(pod), err)
		}
		close(done)
	}()

	select {
	case <-time.After(time.Duration(gracePeriod) * time.Second):
		glog.V(2).Infof("rkt: Some pre-stop hooks did not complete in %d seconds for pod %q", gracePeriod, format.Pod(pod))
	case <-done:
	}
}

// KillPod invokes 'systemctl kill' to kill the unit that runs the pod.
// TODO: add support for gracePeriodOverride which is used in eviction scenarios
func (r *Runtime) KillPod(pod *api.Pod, runningPod kubecontainer.Pod, gracePeriodOverride *int64) error {
	glog.V(4).Infof("Rkt is killing pod: name %q.", runningPod.Name)

	if len(runningPod.Containers) == 0 {
		glog.V(4).Infof("rkt: Pod %q is already being killed, no action will be taken", runningPod.Name)
		return nil
	}

	if pod != nil {
		r.waitPreStopHooks(pod, &runningPod)
	}

	containerID, err := parseContainerID(runningPod.Containers[0].ID)
	if err != nil {
		glog.Errorf("rkt: Failed to get rkt uuid of the pod %q: %v", runningPod.Name, err)
		return err
	}
	serviceName := makePodServiceFileName(containerID.uuid)
	r.generateEvents(&runningPod, "Killing", nil)
	for _, c := range runningPod.Containers {
		r.containerRefManager.ClearRef(c.ID)
	}

	// Touch the systemd service file to update the mod time so it will
	// not be garbage collected too soon.
	if err := os.Chtimes(serviceFilePath(serviceName), time.Now(), time.Now()); err != nil {
		glog.Errorf("rkt: Failed to change the modification time of the service file %q: %v", serviceName, err)
		return err
	}

	// Since all service file have 'KillMode=mixed', the processes in
	// the unit's cgroup will receive a SIGKILL if the normal stop timeouts.
	reschan := make(chan string)
	if _, err = r.systemd.StopUnit(serviceName, "replace", reschan); err != nil {
		glog.Errorf("rkt: Failed to stop unit %q: %v", serviceName, err)
		return err
	}

	res := <-reschan
	if res != "done" {
		err := fmt.Errorf("invalid result: %s", res)
		glog.Errorf("rkt: Failed to stop unit %q: %v", serviceName, err)
		return err
	}

	return nil
}

func (r *Runtime) Type() string {
	return RktType
}

func (r *Runtime) Version() (kubecontainer.Version, error) {
	r.versions.RLock()
	defer r.versions.RUnlock()
	return r.versions.binVersion, nil
}

func (r *Runtime) APIVersion() (kubecontainer.Version, error) {
	r.versions.RLock()
	defer r.versions.RUnlock()
	return r.versions.apiVersion, nil
}

// Status returns error if rkt is unhealthy, nil otherwise.
func (r *Runtime) Status() error {
	return r.checkVersion(minimumRktBinVersion, recommendedRktBinVersion, minimumAppcVersion, minimumRktApiVersion, minimumSystemdVersion)
}

// SyncPod syncs the running pod to match the specified desired pod.
func (r *Runtime) SyncPod(pod *api.Pod, podStatus api.PodStatus, internalPodStatus *kubecontainer.PodStatus, pullSecrets []api.Secret, backOff *flowcontrol.Backoff) (result kubecontainer.PodSyncResult) {
	var err error
	defer func() {
		if err != nil {
			result.Fail(err)
		}
	}()
	// TODO: (random-liu) Stop using running pod in SyncPod()
	// TODO: (random-liu) Rename podStatus to apiPodStatus, rename internalPodStatus to podStatus, and use new pod status as much as possible,
	// we may stop using apiPodStatus someday.
	runningPod := kubecontainer.ConvertPodStatusToRunningPod(internalPodStatus)
	// Add references to all containers.
	unidentifiedContainers := make(map[kubecontainer.ContainerID]*kubecontainer.Container)
	for _, c := range runningPod.Containers {
		unidentifiedContainers[c.ID] = c
	}

	restartPod := false
	for _, container := range pod.Spec.Containers {
		expectedHash := kubecontainer.HashContainer(&container)

		c := runningPod.FindContainerByName(container.Name)
		if c == nil {
			if kubecontainer.ShouldContainerBeRestarted(&container, pod, internalPodStatus) {
				glog.V(3).Infof("Container %+v is dead, but RestartPolicy says that we should restart it.", container)
				// TODO(yifan): Containers in one pod are fate-sharing at this moment, see:
				// https://github.com/appc/spec/issues/276.
				restartPod = true
				break
			}
			continue
		}

		// TODO: check for non-root image directives.  See ../docker/manager.go#SyncPod

		// TODO(yifan): Take care of host network change.
		containerChanged := c.Hash != 0 && c.Hash != expectedHash
		if containerChanged {
			glog.Infof("Pod %q container %q hash changed (%d vs %d), it will be killed and re-created.", format.Pod(pod), container.Name, c.Hash, expectedHash)
			restartPod = true
			break
		}

		liveness, found := r.livenessManager.Get(c.ID)
		if found && liveness != proberesults.Success && pod.Spec.RestartPolicy != api.RestartPolicyNever {
			glog.Infof("Pod %q container %q is unhealthy, it will be killed and re-created.", format.Pod(pod), container.Name)
			restartPod = true
			break
		}

		delete(unidentifiedContainers, c.ID)
	}

	// If there is any unidentified containers, restart the pod.
	if len(unidentifiedContainers) > 0 {
		restartPod = true
	}

	if restartPod {
		// Kill the pod only if the pod is actually running.
		if len(runningPod.Containers) > 0 {
			if err = r.KillPod(pod, runningPod, nil); err != nil {
				return
			}
		}
		if err = r.RunPod(pod, pullSecrets); err != nil {
			return
		}
	}
	return
}

// GarbageCollect collects the pods/containers.
// TODO(yifan): Enforce the gc policy, also, it would be better if we can
// just GC kubernetes pods.
func (r *Runtime) GarbageCollect(gcPolicy kubecontainer.ContainerGCPolicy) error {
	if err := exec.Command("systemctl", "reset-failed").Run(); err != nil {
		glog.Errorf("rkt: Failed to reset failed systemd services: %v, continue to gc anyway...", err)
	}

	if _, err := r.runCommand("gc", "--grace-period="+gcPolicy.MinAge.String(), "--expire-prepared="+gcPolicy.MinAge.String()); err != nil {
		glog.Errorf("rkt: Failed to gc: %v", err)
	}

	// GC all inactive systemd service files.
	units, err := r.systemd.ListUnits()
	if err != nil {
		glog.Errorf("rkt: Failed to list units: %v", err)
		return err
	}
	runningKubernetesUnits := sets.NewString()
	for _, u := range units {
		if strings.HasPrefix(u.Name, kubernetesUnitPrefix) && u.SubState == "running" {
			runningKubernetesUnits.Insert(u.Name)
		}
	}

	files, err := ioutil.ReadDir(systemdServiceDir)
	if err != nil {
		glog.Errorf("rkt: Failed to read the systemd service directory: %v", err)
		return err
	}
	for _, f := range files {
		if strings.HasPrefix(f.Name(), kubernetesUnitPrefix) && !runningKubernetesUnits.Has(f.Name()) && f.ModTime().Before(time.Now().Add(-gcPolicy.MinAge)) {
			glog.V(4).Infof("rkt: Removing inactive systemd service file: %v", f.Name())
			if err := os.Remove(serviceFilePath(f.Name())); err != nil {
				glog.Warningf("rkt: Failed to remove inactive systemd service file %v: %v", f.Name(), err)
			}
		}
	}
	return nil
}

// Note: In rkt, the container ID is in the form of "UUID:appName", where
// appName is the container name.
// TODO(yifan): If the rkt is using lkvm as the stage1 image, then this function will fail.
func (r *Runtime) RunInContainer(containerID kubecontainer.ContainerID, cmd []string) ([]byte, error) {
	glog.V(4).Infof("Rkt running in container.")

	id, err := parseContainerID(containerID)
	if err != nil {
		return nil, err
	}
	args := append([]string{}, "enter", fmt.Sprintf("--app=%s", id.appName), id.uuid)
	args = append(args, cmd...)

	result, err := r.buildCommand(args...).CombinedOutput()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			err = &rktExitError{exitErr}
		}
	}
	return result, err
}

// rktExitError implemets /pkg/util/exec.ExitError interface.
type rktExitError struct{ *exec.ExitError }

var _ utilexec.ExitError = &rktExitError{}

func (r *rktExitError) ExitStatus() int {
	if status, ok := r.Sys().(syscall.WaitStatus); ok {
		return status.ExitStatus()
	}
	return 0
}

func (r *Runtime) AttachContainer(containerID kubecontainer.ContainerID, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	return fmt.Errorf("unimplemented")
}

// Note: In rkt, the container ID is in the form of "UUID:appName", where UUID is
// the rkt UUID, and appName is the container name.
// TODO(yifan): If the rkt is using lkvm as the stage1 image, then this function will fail.
func (r *Runtime) ExecInContainer(containerID kubecontainer.ContainerID, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	glog.V(4).Infof("Rkt execing in container.")

	id, err := parseContainerID(containerID)
	if err != nil {
		return err
	}
	args := append([]string{}, "enter", fmt.Sprintf("--app=%s", id.appName), id.uuid)
	args = append(args, cmd...)
	command := r.buildCommand(args...)

	if tty {
		p, err := kubecontainer.StartPty(command)
		if err != nil {
			return err
		}
		defer p.Close()

		// make sure to close the stdout stream
		defer stdout.Close()

		if stdin != nil {
			go io.Copy(p, stdin)
		}
		if stdout != nil {
			go io.Copy(stdout, p)
		}
		return command.Wait()
	}
	if stdin != nil {
		// Use an os.Pipe here as it returns true *os.File objects.
		// This way, if you run 'kubectl exec <pod> -i bash' (no tty) and type 'exit',
		// the call below to command.Run() can unblock because its Stdin is the read half
		// of the pipe.
		r, w, err := os.Pipe()
		if err != nil {
			return err
		}
		go io.Copy(w, stdin)

		command.Stdin = r
	}
	if stdout != nil {
		command.Stdout = stdout
	}
	if stderr != nil {
		command.Stderr = stderr
	}
	return command.Run()
}

// PortForward executes socat in the pod's network namespace and copies
// data between stream (representing the user's local connection on their
// computer) and the specified port in the container.
//
// TODO:
//  - match cgroups of container
//  - should we support nsenter + socat on the host? (current impl)
//  - should we support nsenter + socat in a container, running with elevated privs and --pid=host?
//
// TODO(yifan): Merge with the same function in dockertools.
// TODO(yifan): If the rkt is using lkvm as the stage1 image, then this function will fail.
func (r *Runtime) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	glog.V(4).Infof("Rkt port forwarding in container.")

	listResp, err := r.apisvc.ListPods(context.Background(), &rktapi.ListPodsRequest{
		Detail:  true,
		Filters: runningKubernetesPodFilters(pod.ID),
	})
	if err != nil {
		return fmt.Errorf("couldn't list pods: %v", err)
	}

	if len(listResp.Pods) != 1 {
		var podlist []string
		for _, p := range listResp.Pods {
			podlist = append(podlist, p.Id)
		}
		return fmt.Errorf("more than one running rkt pod for the kubernetes pod [%s]", strings.Join(podlist, ", "))
	}

	socatPath, lookupErr := exec.LookPath("socat")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: socat not found.")
	}

	args := []string{"-t", fmt.Sprintf("%d", listResp.Pods[0].Pid), "-n", socatPath, "-", fmt.Sprintf("TCP4:localhost:%d", port)}

	nsenterPath, lookupErr := exec.LookPath("nsenter")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: nsenter not found.")
	}

	command := exec.Command(nsenterPath, args...)
	command.Stdout = stream

	// If we use Stdin, command.Run() won't return until the goroutine that's copying
	// from stream finishes. Unfortunately, if you have a client like telnet connected
	// via port forwarding, as long as the user's telnet client is connected to the user's
	// local listener that port forwarding sets up, the telnet session never exits. This
	// means that even if socat has finished running, command.Run() won't ever return
	// (because the client still has the connection and stream open).
	//
	// The work around is to use StdinPipe(), as Wait() (called by Run()) closes the pipe
	// when the command (socat) exits.
	inPipe, err := command.StdinPipe()
	if err != nil {
		return fmt.Errorf("unable to do port forwarding: error creating stdin pipe: %v", err)
	}
	go func() {
		io.Copy(inPipe, stream)
		inPipe.Close()
	}()

	return command.Run()
}

// appStateToContainerState converts rktapi.AppState to kubecontainer.ContainerState.
func appStateToContainerState(state rktapi.AppState) kubecontainer.ContainerState {
	switch state {
	case rktapi.AppState_APP_STATE_RUNNING:
		return kubecontainer.ContainerStateRunning
	case rktapi.AppState_APP_STATE_EXITED:
		return kubecontainer.ContainerStateExited
	}
	return kubecontainer.ContainerStateUnknown
}

// getPodInfo returns the pod manifest, creation time and restart count of the pod.
func getPodInfo(pod *rktapi.Pod) (podManifest *appcschema.PodManifest, restartCount int, err error) {
	// TODO(yifan): The manifest is only used for getting the annotations.
	// Consider to let the server to unmarshal the annotations.
	var manifest appcschema.PodManifest
	if err = json.Unmarshal(pod.Manifest, &manifest); err != nil {
		return
	}

	if countString, ok := manifest.Annotations.Get(k8sRktRestartCountAnno); ok {
		restartCount, err = strconv.Atoi(countString)
		if err != nil {
			return
		}
	}

	return &manifest, restartCount, nil
}

// populateContainerStatus fills the container status according to the app's information.
func populateContainerStatus(pod rktapi.Pod, app rktapi.App, runtimeApp appcschema.RuntimeApp, restartCount int, finishedTime time.Time) (*kubecontainer.ContainerStatus, error) {
	hashStr, ok := runtimeApp.Annotations.Get(k8sRktContainerHashAnno)
	if !ok {
		return nil, fmt.Errorf("No container hash in pod manifest")
	}

	hashNum, err := strconv.ParseUint(hashStr, 10, 64)
	if err != nil {
		return nil, err
	}

	var reason, message string
	if app.State == rktapi.AppState_APP_STATE_EXITED {
		if app.ExitCode == 0 {
			reason = "Completed"
		} else {
			reason = "Error"
		}
	}

	terminationMessagePath, ok := runtimeApp.Annotations.Get(k8sRktTerminationMessagePathAnno)
	if ok {
		if data, err := ioutil.ReadFile(terminationMessagePath); err != nil {
			message = fmt.Sprintf("Error on reading termination-log %s: %v", terminationMessagePath, err)
		} else {
			message = string(data)
		}
	}

	createdTime := time.Unix(0, pod.CreatedAt)
	startedTime := time.Unix(0, pod.StartedAt)

	return &kubecontainer.ContainerStatus{
		ID:         buildContainerID(&containerID{uuid: pod.Id, appName: app.Name}),
		Name:       app.Name,
		State:      appStateToContainerState(app.State),
		CreatedAt:  createdTime,
		StartedAt:  startedTime,
		FinishedAt: finishedTime,
		ExitCode:   int(app.ExitCode),
		// By default, the version returned by rkt API service will be "latest" if not specified.
		Image:   fmt.Sprintf("%s:%s", app.Image.Name, app.Image.Version),
		ImageID: "rkt://" + app.Image.Id, // TODO(yifan): Add the prefix only in api.PodStatus.
		Hash:    hashNum,
		// TODO(yifan): Note that now all apps share the same restart count, this might
		// change once apps don't share the same lifecycle.
		// See https://github.com/appc/spec/pull/547.
		RestartCount: restartCount,
		Reason:       reason,
		Message:      message,
	}, nil
}

// GetPodStatus returns the status for a pod specified by a given UID, name,
// and namespace.  It will attempt to find pod's information via a request to
// the rkt api server.
// An error will be returned if the api server returns an error. If the api
// server doesn't error, but doesn't provide meaningful information about the
// pod, a status with no information (other than the passed in arguments) is
// returned anyways.
func (r *Runtime) GetPodStatus(uid types.UID, name, namespace string) (*kubecontainer.PodStatus, error) {
	podStatus := &kubecontainer.PodStatus{
		ID:        uid,
		Name:      name,
		Namespace: namespace,
	}

	listResp, err := r.apisvc.ListPods(context.Background(), &rktapi.ListPodsRequest{
		Detail:  true,
		Filters: kubernetesPodFilters(uid),
	})
	if err != nil {
		return nil, fmt.Errorf("couldn't list pods: %v", err)
	}

	var latestPod *rktapi.Pod
	var latestRestartCount int = -1

	// In this loop, we group all containers from all pods together,
	// also we try to find the latest pod, so we can fill other info of the pod below.
	for _, pod := range listResp.Pods {
		manifest, restartCount, err := getPodInfo(pod)
		if err != nil {
			glog.Warningf("rkt: Couldn't get necessary info from the rkt pod, (uuid %q): %v", pod.Id, err)
			continue
		}

		if restartCount > latestRestartCount {
			latestPod = pod
			latestRestartCount = restartCount
		}

		finishedTime := r.podFinishedAt(uid, pod.Id)
		for i, app := range pod.Apps {
			// The order of the apps is determined by the rkt pod manifest.
			cs, err := populateContainerStatus(*pod, *app, manifest.Apps[i], restartCount, finishedTime)
			if err != nil {
				glog.Warningf("rkt: Failed to populate container status(uuid %q, app %q): %v", pod.Id, app.Name, err)
				continue
			}
			podStatus.ContainerStatuses = append(podStatus.ContainerStatuses, cs)
		}
	}

	if latestPod != nil {
		// Try to fill the IP info.
		for _, n := range latestPod.Networks {
			if n.Name == defaultNetworkName {
				podStatus.IP = n.Ipv4
			}
		}
	}

	return podStatus, nil
}

// FIXME: I need to be implemented.
func (r *Runtime) ImageStats() (*kubecontainer.ImageStats, error) {
	return &kubecontainer.ImageStats{}, nil
}
