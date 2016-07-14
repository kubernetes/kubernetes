/*
Copyright 2015 The Kubernetes Authors.

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
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
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
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/leaky"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/hairpin"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/securitycontext"
	kubetypes "k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/errors"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/selinux"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/util/term"
	utilwait "k8s.io/kubernetes/pkg/util/wait"
)

const (
	RktType                      = "rkt"
	DefaultRktAPIServiceEndpoint = "localhost:15441"

	minimumRktBinVersion     = "1.9.1"
	recommendedRktBinVersion = "1.9.1"

	minimumRktApiVersion  = "1.0.0-alpha"
	minimumSystemdVersion = "219"

	systemdServiceDir = "/run/systemd/system"
	rktDataDir        = "/var/lib/rkt"
	rktLocalConfigDir = "/etc/rkt"

	kubernetesUnitPrefix  = "k8s_"
	unitKubernetesSection = "X-Kubernetes"
	unitPodUID            = "PodUID"
	unitPodName           = "PodName"
	unitPodNamespace      = "PodNamespace"
	unitPodHostNetwork    = "PodHostNetwork"

	k8sRktKubeletAnno                = "rkt.kubernetes.io/managed-by-kubelet"
	k8sRktKubeletAnnoValue           = "true"
	k8sRktContainerHashAnno          = "rkt.kubernetes.io/container-hash"
	k8sRktRestartCountAnno           = "rkt.kubernetes.io/restart-count"
	k8sRktTerminationMessagePathAnno = "rkt.kubernetes.io/termination-message-path"

	// TODO(euank): This has significant security concerns as a stage1 image is
	// effectively root.
	// Furthermore, this (using an annotation) is a hack to pass an extra
	// non-portable argument in. It should not be relied on to be stable.
	// In the future, this might be subsumed by a first-class api object, or by a
	// kitchen-sink params object (#17064).
	// See discussion in #23944
	// Also, do we want more granularity than path-at-the-kubelet-level and
	// image/name-at-the-pod-level?
	k8sRktStage1NameAnno = "rkt.alpha.kubernetes.io/stage1-name-override"
	dockerPrefix         = "docker://"

	authDir            = "auth.d"
	dockerAuthTemplate = `{"rktKind":"dockerAuth","rktVersion":"v1","registries":[%q],"credentials":{"user":%q,"password":%q}}`

	defaultRktAPIServiceAddr = "localhost:15441"

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

	// The network name of the network when no-op plugin is being used.
	// TODO(yifan): This is not ideal since today we cannot make the rkt's 'net.d' dir point to the
	// CNI directory specified by kubelet. Once that is fixed, we can just use the network config
	// under the CNI directory directly.
	// See https://github.com/coreos/rkt/pull/2312#issuecomment-200068370.
	defaultNetworkName = "rkt.kubernetes.io"

	// defaultRequestTimeout is the default timeout of rkt requests.
	defaultRequestTimeout = 2 * time.Minute
)

// Runtime implements the Containerruntime for rkt. The implementation
// uses systemd, so in order to run this runtime, systemd must be installed
// on the machine.
type Runtime struct {
	cli     cliInterface
	systemd systemdInterface
	// The grpc client for rkt api-service.
	apisvcConn *grpc.ClientConn
	apisvc     rktapi.PublicAPIClient
	config     *Config
	// TODO(yifan): Refactor this to be generic keyring.
	dockerKeyring credentialprovider.DockerKeyring

	containerRefManager *kubecontainer.RefManager
	podGetter           podGetter
	runtimeHelper       kubecontainer.RuntimeHelper
	recorder            record.EventRecorder
	livenessManager     proberesults.Manager
	imagePuller         images.ImageManager
	runner              kubecontainer.HandlerRunner
	execer              utilexec.Interface
	os                  kubecontainer.OSInterface

	// Network plugin.
	networkPlugin network.NetworkPlugin

	// If true, the "hairpin mode" flag is set on container interfaces.
	// A false value means the kubelet just backs off from setting it,
	// it might already be true.
	configureHairpinMode bool

	// used for a systemd Exec, which requires the full path.
	touchPath   string
	nsenterPath string

	versions versions

	// requestTimeout is the timeout of rkt requests.
	requestTimeout time.Duration
}

var _ kubecontainer.Runtime = &Runtime{}

// TODO(yifan): This duplicates the podGetter in dockertools.
type podGetter interface {
	GetPodByUID(kubetypes.UID) (*api.Pod, bool)
}

// cliInterface wrapps the command line calls for testing purpose.
type cliInterface interface {
	// RunCommand creates rkt commands and runs it with the given config.
	// If the config is nil, it will use the one inferred from rkt API service.
	RunCommand(config *Config, args ...string) (result []string, err error)
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
	podGetter podGetter,
	livenessManager proberesults.Manager,
	httpClient types.HttpGetter,
	networkPlugin network.NetworkPlugin,
	hairpinMode bool,
	execer utilexec.Interface,
	os kubecontainer.OSInterface,
	imageBackOff *flowcontrol.Backoff,
	serializeImagePulls bool,
	requestTimeout time.Duration,
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

	nsenterPath, err := execer.LookPath("nsenter")
	if err != nil {
		return nil, fmt.Errorf("cannot find nsenter binary: %v", err)
	}

	if requestTimeout == 0 {
		requestTimeout = defaultRequestTimeout
	}

	rkt := &Runtime{
		os:                  kubecontainer.RealOS{},
		systemd:             systemd,
		apisvcConn:          apisvcConn,
		apisvc:              rktapi.NewPublicAPIClient(apisvcConn),
		config:              config,
		dockerKeyring:       credentialprovider.NewDockerKeyring(),
		containerRefManager: containerRefManager,
		podGetter:           podGetter,
		runtimeHelper:       runtimeHelper,
		recorder:            recorder,
		livenessManager:     livenessManager,
		networkPlugin:       networkPlugin,
		execer:              execer,
		touchPath:           touchPath,
		nsenterPath:         nsenterPath,
		requestTimeout:      requestTimeout,
	}

	rkt.config, err = rkt.getConfig(rkt.config)
	if err != nil {
		return nil, fmt.Errorf("rkt: cannot get config from rkt api service: %v", err)
	}

	rkt.runner = lifecycle.NewHandlerRunner(httpClient, rkt, rkt)

	rkt.imagePuller = images.NewImageManager(recorder, rkt, imageBackOff, serializeImagePulls)

	if err := rkt.getVersions(); err != nil {
		return nil, fmt.Errorf("rkt: error getting version info: %v", err)
	}

	rkt.cli = rkt

	return rkt, nil
}

func buildCommand(config *Config, args ...string) *exec.Cmd {
	cmd := exec.Command(config.Path)
	cmd.Args = append(cmd.Args, config.buildGlobalOptions()...)
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

// RunCommand invokes rkt binary with arguments and returns the result
// from stdout in a list of strings. Each string in the list is a line.
// If config is non-nil, it will use the given config instead of the config
// inferred from rkt API service.
func (r *Runtime) RunCommand(config *Config, args ...string) ([]string, error) {
	if config == nil {
		config = r.config
	}
	glog.V(4).Infof("rkt: Run command: %q with config: %+v", args, config)

	var stdout, stderr bytes.Buffer

	cmd := buildCommand(config, args...)
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
	return fmt.Sprintf("%s%s.service", kubernetesUnitPrefix, uuid)
}

func getRktUUIDFromServiceFileName(filename string) string {
	return strings.TrimPrefix(strings.TrimSuffix(filename, path.Ext(filename)), kubernetesUnitPrefix)
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
	// TODO(yifan): If only the GID is empty, rkt should be able to determine the GID
	// using the /etc/passwd file in the image.
	// See https://github.com/appc/docker2aci/issues/175.
	// Maybe we can remove this check in the future.
	if app.User == "" {
		app.User = "0"
		app.Group = "0"
	}
	if app.Group == "" {
		return fmt.Errorf("cannot determine the GID of the app %q", imgManifest.Name)
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
func (r *Runtime) makePodManifest(pod *api.Pod, podIP string, pullSecrets []api.Secret) (*appcschema.PodManifest, error) {
	manifest := appcschema.BlankPodManifest()

	ctx, cancel := context.WithTimeout(context.Background(), r.requestTimeout)
	defer cancel()
	listResp, err := r.apisvc.ListPods(ctx, &rktapi.ListPodsRequest{
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

	requiresPrivileged := false
	manifest.Annotations.Set(*appctypes.MustACIdentifier(k8sRktKubeletAnno), k8sRktKubeletAnnoValue)
	manifest.Annotations.Set(*appctypes.MustACIdentifier(types.KubernetesPodUIDLabel), string(pod.UID))
	manifest.Annotations.Set(*appctypes.MustACIdentifier(types.KubernetesPodNameLabel), pod.Name)
	manifest.Annotations.Set(*appctypes.MustACIdentifier(types.KubernetesPodNamespaceLabel), pod.Namespace)
	manifest.Annotations.Set(*appctypes.MustACIdentifier(types.KubernetesContainerNameLabel), leaky.PodInfraContainerName)
	manifest.Annotations.Set(*appctypes.MustACIdentifier(k8sRktRestartCountAnno), strconv.Itoa(restartCount))
	if stage1Name, ok := pod.Annotations[k8sRktStage1NameAnno]; ok {
		requiresPrivileged = true
		manifest.Annotations.Set(*appctypes.MustACIdentifier(k8sRktStage1NameAnno), stage1Name)
	}

	for _, c := range pod.Spec.Containers {
		err := r.newAppcRuntimeApp(pod, podIP, c, requiresPrivileged, pullSecrets, manifest)
		if err != nil {
			return nil, err
		}
	}

	// TODO(yifan): Set pod-level isolators once it's supported in kubernetes.
	return manifest, nil
}

func copyfile(src, dst string) error {
	data, err := ioutil.ReadFile(src)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(dst, data, 0640)
}

// TODO(yifan): Can make rkt handle this when '--net=host'. See https://github.com/coreos/rkt/issues/2430.
func makeHostNetworkMount(opts *kubecontainer.RunContainerOptions) (*kubecontainer.Mount, *kubecontainer.Mount, error) {
	hostsPath := filepath.Join(opts.PodContainerDir, "etc-hosts")
	resolvPath := filepath.Join(opts.PodContainerDir, "etc-resolv-conf")

	if err := copyfile("/etc/hosts", hostsPath); err != nil {
		return nil, nil, err
	}
	if err := copyfile("/etc/resolv.conf", resolvPath); err != nil {
		return nil, nil, err
	}

	hostsMount := kubecontainer.Mount{
		Name:          "kubernetes-hostnetwork-hosts-conf",
		ContainerPath: "/etc/hosts",
		HostPath:      hostsPath,
	}
	resolvMount := kubecontainer.Mount{
		Name:          "kubernetes-hostnetwork-resolv-conf",
		ContainerPath: "/etc/resolv.conf",
		HostPath:      resolvPath,
	}
	opts.Mounts = append(opts.Mounts, hostsMount, resolvMount)
	return &hostsMount, &resolvMount, nil
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
func (r *Runtime) podFinishedAt(podUID kubetypes.UID, rktUID string) time.Time {
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

func (r *Runtime) makeContainerLogMount(opts *kubecontainer.RunContainerOptions, container *api.Container) (*kubecontainer.Mount, error) {
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
	fs, err := r.os.Create(containerLogPath)
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

func (r *Runtime) newAppcRuntimeApp(pod *api.Pod, podIP string, c api.Container, requiresPrivileged bool, pullSecrets []api.Secret, manifest *appcschema.PodManifest) error {
	if requiresPrivileged && !securitycontext.HasPrivilegedRequest(&c) {
		return fmt.Errorf("cannot make %q: running a custom stage1 requires a privileged security context", format.Pod(pod))
	}
	if err, _ := r.imagePuller.EnsureImageExists(pod, &c, pullSecrets); err != nil {
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
	opts, err := r.runtimeHelper.GenerateRunContainerOptions(pod, &c, podIP)
	if err != nil {
		return err
	}

	// create the container log file and make a mount pair.
	mnt, err := r.makeContainerLogMount(opts, &c)
	if err != nil {
		return err
	}

	// If run in 'hostnetwork' mode, then copy and mount the host's /etc/resolv.conf and /etc/hosts,
	// and add volumes.
	var hostsMnt, resolvMnt *kubecontainer.Mount
	if kubecontainer.IsHostNetworkPod(pod) {
		hostsMnt, resolvMnt, err = makeHostNetworkMount(opts)
		if err != nil {
			return err
		}
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

	for _, mnt := range opts.Mounts {
		readOnly := mnt.ReadOnly
		manifest.Volumes = append(manifest.Volumes, appctypes.Volume{
			Name:     convertToACName(mnt.Name),
			Source:   mnt.HostPath,
			Kind:     "host",
			ReadOnly: &readOnly,
		})
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
			{
				Name:  *appctypes.MustACIdentifier(types.KubernetesContainerNameLabel),
				Value: c.Name,
			},
		},
	}

	if c.SecurityContext != nil && c.SecurityContext.ReadOnlyRootFilesystem != nil {
		ra.ReadOnlyRootFS = *c.SecurityContext.ReadOnlyRootFilesystem
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

func runningKubernetesPodFilters(uid kubetypes.UID) []*rktapi.PodFilter {
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
					Key:   types.KubernetesPodUIDLabel,
					Value: string(uid),
				},
			},
		},
	}
}

func kubernetesPodFilters(uid kubetypes.UID) []*rktapi.PodFilter {
	return []*rktapi.PodFilter{
		{
			Annotations: []*rktapi.KeyValue{
				{
					Key:   k8sRktKubeletAnno,
					Value: k8sRktKubeletAnnoValue,
				},
				{
					Key:   types.KubernetesPodUIDLabel,
					Value: string(uid),
				},
			},
		},
	}
}

func kubernetesPodsFilters() []*rktapi.PodFilter {
	return []*rktapi.PodFilter{
		{
			Annotations: []*rktapi.KeyValue{
				{
					Key:   k8sRktKubeletAnno,
					Value: k8sRktKubeletAnnoValue,
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

// shouldCreateNetns returns true if:
// The pod does not run in host network. And
// The pod runs inside a netns created outside of rkt.
func (r *Runtime) shouldCreateNetns(pod *api.Pod) bool {
	return !kubecontainer.IsHostNetworkPod(pod) && r.networkPlugin.Name() != network.DefaultPluginName
}

// usesRktHostNetwork returns true if:
// The pod runs in the host network. Or
// The pod runs inside a netns created outside of rkt.
func (r *Runtime) usesRktHostNetwork(pod *api.Pod) bool {
	return kubecontainer.IsHostNetworkPod(pod) || r.shouldCreateNetns(pod)
}

// generateRunCommand crafts a 'rkt run-prepared' command with necessary parameters.
func (r *Runtime) generateRunCommand(pod *api.Pod, uuid, netnsName string) (string, error) {
	runPrepared := buildCommand(r.config, "run-prepared").Args

	var hostname string
	var err error

	osInfos, err := getOSReleaseInfo()
	if err != nil {
		glog.Errorf("rkt: Failed to read the os release info: %v", err)
	}

	// Overlay fs is not supported for SELinux yet on many distros.
	// See https://github.com/coreos/rkt/issues/1727#issuecomment-173203129.
	// For now, coreos carries a patch to support it: https://github.com/coreos/coreos-overlay/pull/1703
	if osInfos["ID"] != "coreos" && pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SELinuxOptions != nil {
		runPrepared = append(runPrepared, "--no-overlay=true")
	}

	// Apply '--net=host' to pod that is running on host network or inside a network namespace.
	if r.usesRktHostNetwork(pod) {
		runPrepared = append(runPrepared, "--net=host")
	} else {
		runPrepared = append(runPrepared, fmt.Sprintf("--net=%s", defaultNetworkName))
	}

	if kubecontainer.IsHostNetworkPod(pod) {
		// TODO(yifan): Let runtimeHelper.GeneratePodHostNameAndDomain() to handle this.
		hostname, err = r.os.Hostname()
		if err != nil {
			return "", err
		}
	} else {
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

	if r.shouldCreateNetns(pod) {
		// Drop the `rkt run-prepared` into the network namespace we
		// created.
		// TODO: switch to 'ip netns exec' once we can depend on a new
		// enough version that doesn't have bugs like
		// https://bugzilla.redhat.com/show_bug.cgi?id=882047
		nsenterExec := []string{r.nsenterPath, "--net=" + netnsPathFromName(netnsName), "--"}
		runPrepared = append(nsenterExec, runPrepared...)
	}

	return strings.Join(runPrepared, " "), nil
}

func (r *Runtime) cleanupPodNetwork(pod *api.Pod) error {
	glog.V(3).Infof("Calling network plugin %s to tear down pod for %s", r.networkPlugin.Name(), format.Pod(pod))

	// No-op if the pod is not running in a created netns.
	if !r.shouldCreateNetns(pod) {
		return nil
	}

	var teardownErr error
	containerID := kubecontainer.ContainerID{ID: string(pod.UID)}
	if err := r.networkPlugin.TearDownPod(pod.Namespace, pod.Name, containerID); err != nil {
		teardownErr = fmt.Errorf("rkt: failed to tear down network for pod %s: %v", format.Pod(pod), err)
		glog.Errorf("%v", teardownErr)
	}

	if _, err := r.execer.Command("ip", "netns", "del", makePodNetnsName(pod.UID)).Output(); err != nil {
		return fmt.Errorf("rkt: Failed to remove network namespace for pod %s: %v", format.Pod(pod), err)
	}

	return teardownErr
}

func (r *Runtime) preparePodArgs(manifest *appcschema.PodManifest, manifestFileName string) []string {
	// Order of precedence for the stage1:
	// 1) pod annotation (stage1 name)
	// 2) kubelet configured stage1 (stage1 path)
	// 3) empty; whatever rkt's compiled to default to
	stage1ImageCmd := ""
	if r.config.Stage1Image != "" {
		stage1ImageCmd = "--stage1-name=" + r.config.Stage1Image
	}
	if stage1Name, ok := manifest.Annotations.Get(k8sRktStage1NameAnno); ok {
		stage1ImageCmd = "--stage1-name=" + stage1Name
	}

	// Run 'rkt prepare' to get the rkt UUID.
	cmds := []string{"prepare", "--quiet", "--pod-manifest", manifestFileName}
	if stage1ImageCmd != "" {
		cmds = append(cmds, stage1ImageCmd)
	}
	return cmds
}

func (r *Runtime) getSelinuxContext(opt *api.SELinuxOptions) (string, error) {
	selinuxRunner := selinux.NewSelinuxContextRunner()
	str, err := selinuxRunner.Getfilecon(r.config.Dir)
	if err != nil {
		return "", err
	}

	ctx := strings.SplitN(str, ":", 4)
	if len(ctx) != 4 {
		return "", fmt.Errorf("malformated selinux context")
	}

	if opt.User != "" {
		ctx[0] = opt.User
	}
	if opt.Role != "" {
		ctx[1] = opt.Role
	}
	if opt.Type != "" {
		ctx[2] = opt.Type
	}
	if opt.Level != "" {
		ctx[3] = opt.Level
	}

	return strings.Join(ctx, ":"), nil
}

// preparePod will:
//
// 1. Invoke 'rkt prepare' to prepare the pod, and get the rkt pod uuid.
// 2. Create the unit file and save it under systemdUnitDir.
//
// On success, it will return a string that represents name of the unit file
// and the runtime pod.
func (r *Runtime) preparePod(pod *api.Pod, podIP string, pullSecrets []api.Secret, netnsName string) (string, *kubecontainer.Pod, error) {
	// Generate the appc pod manifest from the k8s pod spec.
	manifest, err := r.makePodManifest(pod, podIP, pullSecrets)
	if err != nil {
		return "", nil, err
	}
	manifestFile, err := ioutil.TempFile("", fmt.Sprintf("manifest-%s-", pod.Name))
	if err != nil {
		return "", nil, err
	}
	defer func() {
		manifestFile.Close()
		if err := r.os.Remove(manifestFile.Name()); err != nil {
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

	prepareCmd := r.preparePodArgs(manifest, manifestFile.Name())
	output, err := r.cli.RunCommand(nil, prepareCmd...)
	if err != nil {
		return "", nil, err
	}
	if len(output) != 1 {
		return "", nil, fmt.Errorf("invalid output from 'rkt prepare': %v", output)
	}
	uuid := output[0]
	glog.V(4).Infof("'rkt prepare' returns %q", uuid)

	// Create systemd service file for the rkt pod.
	runPrepared, err := r.generateRunCommand(pod, uuid, netnsName)
	if err != nil {
		return "", nil, fmt.Errorf("failed to generate 'rkt run-prepared' command: %v", err)
	}

	// TODO handle pod.Spec.HostPID
	// TODO handle pod.Spec.HostIPC

	// TODO per container finishedAt, not just per pod
	markPodFinished := podFinishedMarkCommand(r.touchPath, r.runtimeHelper.GetPodDir(pod.UID), uuid)

	hostNetwork := kubecontainer.IsHostNetworkPod(pod)
	units := []*unit.UnitOption{
		newUnitOption("Service", "ExecStart", runPrepared),
		newUnitOption("Service", "ExecStopPost", markPodFinished),
		// This enables graceful stop.
		newUnitOption("Service", "KillMode", "mixed"),
		newUnitOption("Service", "TimeoutStopSec", fmt.Sprintf("%ds", getPodTerminationGracePeriodInSecond(pod))),
		// Track pod info for garbage collection
		newUnitOption(unitKubernetesSection, unitPodUID, string(pod.UID)),
		newUnitOption(unitKubernetesSection, unitPodName, pod.Name),
		newUnitOption(unitKubernetesSection, unitPodNamespace, pod.Namespace),
		newUnitOption(unitKubernetesSection, unitPodHostNetwork, fmt.Sprintf("%v", hostNetwork)),
	}

	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SELinuxOptions != nil {
		opt := pod.Spec.SecurityContext.SELinuxOptions
		selinuxContext, err := r.getSelinuxContext(opt)
		if err != nil {
			glog.Errorf("rkt: Failed to construct selinux context with selinux option %q: %v", opt, err)
			return "", nil, err
		}
		units = append(units, newUnitOption("Service", "SELinuxContext", selinuxContext))
	}

	serviceName := makePodServiceFileName(uuid)
	glog.V(4).Infof("rkt: Creating service file %q for pod %q", serviceName, format.Pod(pod))
	serviceFile, err := r.os.Create(serviceFilePath(serviceName))
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
			r.recorder.Eventf(ref, api.EventTypeNormal, events.CreatedContainer, "Created with rkt id %v", uuid)
		case "Started":
			r.recorder.Eventf(ref, api.EventTypeNormal, events.StartedContainer, "Started with rkt id %v", uuid)
		case "Failed":
			r.recorder.Eventf(ref, api.EventTypeWarning, events.FailedToStartContainer, "Failed to start with rkt id %v with error %v", uuid, failure)
		case "Killing":
			r.recorder.Eventf(ref, api.EventTypeNormal, events.KillingContainer, "Killing with rkt id %v", uuid)
		default:
			glog.Errorf("rkt: Unexpected event %q", reason)
		}
	}
	return
}

func makePodNetnsName(podID kubetypes.UID) string {
	return fmt.Sprintf("%s%s", kubernetesUnitPrefix, string(podID))
}

func netnsPathFromName(netnsName string) string {
	return fmt.Sprintf("/var/run/netns/%s", netnsName)
}

// setupPodNetwork creates a network namespace for the given pod and calls
// configured NetworkPlugin's setup function on it.
// It returns the namespace name, configured IP (if available), and an error if
// one occurred.
//
// If the pod is running in host network or is running using the no-op plugin, then nothing will be done.
func (r *Runtime) setupPodNetwork(pod *api.Pod) (string, string, error) {
	glog.V(3).Infof("Calling network plugin %s to set up pod for %s", r.networkPlugin.Name(), format.Pod(pod))

	// No-op if the pod is not running in a created netns.
	if !r.shouldCreateNetns(pod) {
		return "", "", nil
	}

	netnsName := makePodNetnsName(pod.UID)

	// Create a new network namespace for the pod
	r.execer.Command("ip", "netns", "del", netnsName).Output()
	_, err := r.execer.Command("ip", "netns", "add", netnsName).Output()
	if err != nil {
		return "", "", fmt.Errorf("failed to create pod network namespace: %v", err)
	}

	// Set up networking with the network plugin
	glog.V(3).Infof("Calling network plugin %s to setup pod for %s", r.networkPlugin.Name(), format.Pod(pod))
	containerID := kubecontainer.ContainerID{ID: string(pod.UID)}
	err = r.networkPlugin.SetUpPod(pod.Namespace, pod.Name, containerID)
	if err != nil {
		return "", "", fmt.Errorf("failed to set up pod network: %v", err)
	}
	status, err := r.networkPlugin.GetPodNetworkStatus(pod.Namespace, pod.Name, containerID)
	if err != nil {
		return "", "", fmt.Errorf("failed to get status of pod network: %v", err)
	}

	if r.configureHairpinMode {
		if err = hairpin.SetUpContainerPath(netnsPathFromName(netnsName), network.DefaultInterfaceName); err != nil {
			glog.Warningf("Hairpin setup failed for pod %q: %v", format.Pod(pod), err)
		}
	}

	return netnsName, status.IP.String(), nil
}

// RunPod first creates the unit file for a pod, and then
// starts the unit over d-bus.
func (r *Runtime) RunPod(pod *api.Pod, pullSecrets []api.Secret) error {
	glog.V(4).Infof("Rkt starts to run pod: name %q.", format.Pod(pod))

	var err error
	var netnsName string
	var podIP string
	netnsName, podIP, err = r.setupPodNetwork(pod)
	if err != nil {
		r.cleanupPodNetwork(pod)
		return err
	}

	name, runtimePod, prepareErr := r.preparePod(pod, podIP, pullSecrets, netnsName)

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
			r.recorder.Eventf(ref, api.EventTypeWarning, events.FailedToCreateContainer, "Failed to create rkt container with error: %v", prepareErr)
			continue
		}
		containerID := runtimePod.Containers[i].ID
		r.containerRefManager.SetRef(containerID, ref)
	}

	if prepareErr != nil {
		r.cleanupPodNetwork(pod)
		return prepareErr
	}

	r.generateEvents(runtimePod, "Created", nil)

	// RestartUnit has the same effect as StartUnit if the unit is not running, besides it can restart
	// a unit if the unit file is changed and reloaded.
	reschan := make(chan string)
	_, err = r.systemd.RestartUnit(name, "replace", reschan)
	if err != nil {
		r.generateEvents(runtimePod, "Failed", err)
		r.cleanupPodNetwork(pod)
		return err
	}

	res := <-reschan
	if res != "done" {
		err := fmt.Errorf("Failed to restart unit %q: %s", name, res)
		r.generateEvents(runtimePod, "Failed", err)
		r.cleanupPodNetwork(pod)
		return err
	}

	r.generateEvents(runtimePod, "Started", nil)

	// This is a temporary solution until we have a clean design on how
	// kubelet handles events. See https://github.com/kubernetes/kubernetes/issues/23084.
	if err := r.runLifecycleHooks(pod, runtimePod, lifecyclePostStartHook); err != nil {
		if errKill := r.KillPod(pod, *runtimePod, nil); errKill != nil {
			return errors.NewAggregate([]error{err, errKill})
		}
		r.cleanupPodNetwork(pod)
		return err
	}

	return nil
}

func (r *Runtime) runPreStopHook(containerID kubecontainer.ContainerID, pod *api.Pod, container *api.Container) error {
	glog.V(4).Infof("rkt: Running pre-stop hook for container %q of pod %q", container.Name, format.Pod(pod))
	msg, err := r.runner.Run(containerID, pod, container, container.Lifecycle.PreStop)
	if err != nil {
		ref, ok := r.containerRefManager.GetRef(containerID)
		if !ok {
			glog.Warningf("No ref for container %q", containerID)
		} else {
			r.recorder.Eventf(ref, api.EventTypeWarning, events.FailedPreStopHook, msg)
		}
	}
	return err
}

func (r *Runtime) runPostStartHook(containerID kubecontainer.ContainerID, pod *api.Pod, container *api.Container) error {
	glog.V(4).Infof("rkt: Running post-start hook for container %q of pod %q", container.Name, format.Pod(pod))
	cid, err := parseContainerID(containerID)
	if err != nil {
		return fmt.Errorf("cannot parse container ID %v", containerID)
	}

	isContainerRunning := func() (done bool, err error) {
		ctx, cancel := context.WithTimeout(context.Background(), r.requestTimeout)
		defer cancel()
		resp, err := r.apisvc.InspectPod(ctx, &rktapi.InspectPodRequest{Id: cid.uuid})
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

	msg, err := r.runner.Run(containerID, pod, container, container.Lifecycle.PostStart)
	if err != nil {
		ref, ok := r.containerRefManager.GetRef(containerID)
		if !ok {
			glog.Warningf("No ref for container %q", containerID)
		} else {
			r.recorder.Eventf(ref, api.EventTypeWarning, events.FailedPostStartHook, msg)
		}
	}
	return err
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

	podUID, ok := manifest.Annotations.Get(types.KubernetesPodUIDLabel)
	if !ok {
		return nil, fmt.Errorf("pod is missing annotation %s", types.KubernetesPodUIDLabel)
	}
	podName, ok := manifest.Annotations.Get(types.KubernetesPodNameLabel)
	if !ok {
		return nil, fmt.Errorf("pod is missing annotation %s", types.KubernetesPodNameLabel)
	}
	podNamespace, ok := manifest.Annotations.Get(types.KubernetesPodNamespaceLabel)
	if !ok {
		return nil, fmt.Errorf("pod is missing annotation %s", types.KubernetesPodNamespaceLabel)
	}

	kubepod := &kubecontainer.Pod{
		ID:        kubetypes.UID(podUID),
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

// GetPods runs 'rkt list' to get the list of rkt pods.
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
	ctx, cancel := context.WithTimeout(context.Background(), r.requestTimeout)
	defer cancel()
	listResp, err := r.apisvc.ListPods(ctx, listReq)
	if err != nil {
		return nil, fmt.Errorf("couldn't list pods: %v", err)
	}

	pods := make(map[kubetypes.UID]*kubecontainer.Pod)
	var podIDs []kubetypes.UID
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

func getPodTerminationGracePeriodInSecond(pod *api.Pod) int64 {
	var gracePeriod int64
	switch {
	case pod.DeletionGracePeriodSeconds != nil:
		gracePeriod = *pod.DeletionGracePeriodSeconds
	case pod.Spec.TerminationGracePeriodSeconds != nil:
		gracePeriod = *pod.Spec.TerminationGracePeriodSeconds
	}
	if gracePeriod < minimumGracePeriodInSeconds {
		gracePeriod = minimumGracePeriodInSeconds
	}
	return gracePeriod
}

func (r *Runtime) waitPreStopHooks(pod *api.Pod, runningPod *kubecontainer.Pod) {
	gracePeriod := getPodTerminationGracePeriodInSecond(pod)

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
	serviceFile := serviceFilePath(serviceName)

	r.generateEvents(&runningPod, "Killing", nil)
	for _, c := range runningPod.Containers {
		r.containerRefManager.ClearRef(c.ID)
	}

	// Touch the systemd service file to update the mod time so it will
	// not be garbage collected too soon.
	if err := r.os.Chtimes(serviceFile, time.Now(), time.Now()); err != nil {
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

	// Clean up networking. Use the service file to get pod details since 'pod' can be nil.
	if err := r.cleanupPodNetworkFromServiceFile(serviceFile); err != nil {
		glog.Errorf("rkt: failed to tear down network for unit %q: %v", serviceName, err)
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
	return r.checkVersion(minimumRktBinVersion, recommendedRktBinVersion, minimumRktApiVersion, minimumSystemdVersion)
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

// Sort rkt pods by creation time.
type podsByCreatedAt []*rktapi.Pod

func (s podsByCreatedAt) Len() int           { return len(s) }
func (s podsByCreatedAt) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s podsByCreatedAt) Less(i, j int) bool { return s[i].CreatedAt < s[j].CreatedAt }

// getPodUID returns the pod's API UID, it returns
// empty UID if the UID cannot be determined.
func getPodUID(pod *rktapi.Pod) kubetypes.UID {
	for _, anno := range pod.Annotations {
		if anno.Key == types.KubernetesPodUIDLabel {
			return kubetypes.UID(anno.Value)
		}
	}
	return kubetypes.UID("")
}

// podIsActive returns true if the pod is embryo, preparing or running.
// If a pod is prepared, it is not guaranteed to be active (e.g. the systemd
// service might fail).
func podIsActive(pod *rktapi.Pod) bool {
	return pod.State == rktapi.PodState_POD_STATE_EMBRYO ||
		pod.State == rktapi.PodState_POD_STATE_PREPARING ||
		pod.State == rktapi.PodState_POD_STATE_RUNNING
}

// GetNetNS returns the network namespace path for the given container
func (r *Runtime) GetNetNS(containerID kubecontainer.ContainerID) (string, error) {
	// This is a slight hack, kubenet shouldn't be asking us about a container id
	// but a pod id. This is because it knows too much about the infra container.
	// We pretend the pod.UID is an infra container ID.
	// This deception is only possible because we played the same trick in
	// `networkPlugin.SetUpPod` and `networkPlugin.TearDownPod`.
	return netnsPathFromName(makePodNetnsName(kubetypes.UID(containerID.ID))), nil
}

func (r *Runtime) GetPodContainerID(pod *kubecontainer.Pod) (kubecontainer.ContainerID, error) {
	return kubecontainer.ContainerID{ID: string(pod.ID)}, nil
}

func podDetailsFromServiceFile(serviceFilePath string) (string, string, string, bool, error) {
	f, err := os.Open(serviceFilePath)
	if err != nil {
		return "", "", "", false, err
	}
	defer f.Close()

	opts, err := unit.Deserialize(f)
	if err != nil {
		return "", "", "", false, err
	}

	var id, name, namespace, hostnetwork string
	for _, o := range opts {
		if o.Section != unitKubernetesSection {
			continue
		}
		switch o.Name {
		case unitPodUID:
			id = o.Value
		case unitPodName:
			name = o.Value
		case unitPodNamespace:
			namespace = o.Value
		case unitPodHostNetwork:
			hostnetwork = o.Value
		}

		if id != "" && name != "" && namespace != "" && hostnetwork != "" {
			podHostNetwork, err := strconv.ParseBool(hostnetwork)
			if err != nil {
				return "", "", "", false, err
			}
			return id, name, namespace, podHostNetwork, nil
		}
	}

	return "", "", "", false, fmt.Errorf("failed to parse pod from file %s", serviceFilePath)
}

func (r *Runtime) DeleteContainer(containerID kubecontainer.ContainerID) error {
	return fmt.Errorf("unimplemented")
}

// GarbageCollect collects the pods/containers.
// After one GC iteration:
// - The deleted pods will be removed.
// - If the number of containers exceeds gcPolicy.MaxContainers,
//   then containers whose ages are older than gcPolicy.minAge will
//   be removed.
func (r *Runtime) GarbageCollect(gcPolicy kubecontainer.ContainerGCPolicy, allSourcesReady bool) error {
	var errlist []error
	var totalInactiveContainers int
	var inactivePods []*rktapi.Pod
	var removeCandidates []*rktapi.Pod
	var allPods = map[string]*rktapi.Pod{}

	glog.V(4).Infof("rkt: Garbage collecting triggered with policy %v", gcPolicy)

	// GC all inactive systemd service files and pods.
	files, err := r.os.ReadDir(systemdServiceDir)
	if err != nil {
		glog.Errorf("rkt: Failed to read the systemd service directory: %v", err)
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), r.requestTimeout)
	defer cancel()
	resp, err := r.apisvc.ListPods(ctx, &rktapi.ListPodsRequest{Filters: kubernetesPodsFilters()})
	if err != nil {
		glog.Errorf("rkt: Failed to list pods: %v", err)
		return err
	}

	// Mark inactive pods.
	for _, pod := range resp.Pods {
		allPods[pod.Id] = pod
		if !podIsActive(pod) {
			uid := getPodUID(pod)
			if uid == kubetypes.UID("") {
				glog.Errorf("rkt: Cannot get the UID of pod %q, pod is broken, will remove it", pod.Id)
				removeCandidates = append(removeCandidates, pod)
				continue
			}
			_, found := r.podGetter.GetPodByUID(uid)
			if !found && allSourcesReady {
				removeCandidates = append(removeCandidates, pod)
				continue
			}

			inactivePods = append(inactivePods, pod)
			totalInactiveContainers = totalInactiveContainers + len(pod.Apps)
		}
	}

	// Remove any orphan service files.
	for _, f := range files {
		serviceName := f.Name()
		if strings.HasPrefix(serviceName, kubernetesUnitPrefix) {
			rktUUID := getRktUUIDFromServiceFileName(serviceName)
			if _, ok := allPods[rktUUID]; !ok {
				glog.V(4).Infof("rkt: No rkt pod found for service file %q, will remove it", serviceName)

				if err := r.systemd.ResetFailedUnit(serviceName); err != nil {
					glog.Warningf("rkt: Failed to reset the failed systemd service %q: %v", serviceName, err)
				}
				serviceFile := serviceFilePath(serviceName)

				// Network may not be around anymore so errors are ignored
				if err := r.cleanupPodNetworkFromServiceFile(serviceFile); err != nil {
					glog.Warningf("rkt: Failed to clean up pod network from service %q: %v, the network may not be around already", serviceName, err)
				}
				if err := r.os.Remove(serviceFile); err != nil {
					errlist = append(errlist, fmt.Errorf("rkt: Failed to remove service file %q: %v", serviceFile, err))
				}
			}
		}
	}

	sort.Sort(podsByCreatedAt(inactivePods))

	// Enforce GCPolicy.MaxContainers.
	for _, pod := range inactivePods {
		if totalInactiveContainers <= gcPolicy.MaxContainers {
			break
		}
		creationTime := time.Unix(0, pod.CreatedAt)
		if creationTime.Add(gcPolicy.MinAge).Before(time.Now()) {
			// The pod is old and we are exceeding the MaxContainers limit.
			// Delete the pod.
			removeCandidates = append(removeCandidates, pod)
			totalInactiveContainers = totalInactiveContainers - len(pod.Apps)
		}
	}

	// Remove pods and their servie files.
	for _, pod := range removeCandidates {
		if err := r.removePod(pod.Id); err != nil {
			errlist = append(errlist, fmt.Errorf("rkt: Failed to clean up rkt pod %q: %v", pod.Id, err))
		}
	}

	return errors.NewAggregate(errlist)
}

// Read kubernetes pod UUID, namespace, and name from systemd service file and
// use that to clean up any pod network that may still exist.
func (r *Runtime) cleanupPodNetworkFromServiceFile(serviceFilePath string) error {
	id, name, namespace, hostnetwork, err := podDetailsFromServiceFile(serviceFilePath)
	if err != nil {
		return err
	}
	return r.cleanupPodNetwork(&api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       kubetypes.UID(id),
			Name:      name,
			Namespace: namespace,
		},
		Spec: api.PodSpec{
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: hostnetwork,
			},
		},
	})
}

// removePod calls 'rkt rm $UUID' to delete a rkt pod, it also remove the systemd service file
// related to the pod.
func (r *Runtime) removePod(uuid string) error {
	var errlist []error
	glog.V(4).Infof("rkt: GC is removing pod %q", uuid)

	serviceName := makePodServiceFileName(uuid)
	serviceFile := serviceFilePath(serviceName)

	// Network may not be around anymore so errors are ignored
	if err := r.cleanupPodNetworkFromServiceFile(serviceFile); err != nil {
		glog.Warningf("rkt: Failed to clean up pod network from service %q: %v, the network may not be around already", serviceName, err)
	}

	if _, err := r.cli.RunCommand(nil, "rm", uuid); err != nil {
		errlist = append(errlist, fmt.Errorf("rkt: Failed to remove pod %q: %v", uuid, err))
	}

	// GC systemd service files as well.
	if err := r.systemd.ResetFailedUnit(serviceName); err != nil {
		glog.Warningf("rkt: Failed to reset the failed systemd service %q: %v", serviceName, err)
	}
	if err := r.os.Remove(serviceFile); err != nil {
		errlist = append(errlist, fmt.Errorf("rkt: Failed to remove service file %q for pod %q: %v", serviceFile, uuid, err))
	}

	return errors.NewAggregate(errlist)
}

// rktExitError implements /pkg/util/exec.ExitError interface.
type rktExitError struct{ *exec.ExitError }

var _ utilexec.ExitError = &rktExitError{}

func (r *rktExitError) ExitStatus() int {
	if status, ok := r.Sys().(syscall.WaitStatus); ok {
		return status.ExitStatus()
	}
	return 0
}

func newRktExitError(e error) error {
	if exitErr, ok := e.(*exec.ExitError); ok {
		return &rktExitError{exitErr}
	}
	return e
}

func (r *Runtime) AttachContainer(containerID kubecontainer.ContainerID, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) error {
	return fmt.Errorf("unimplemented")
}

// Note: In rkt, the container ID is in the form of "UUID:appName", where UUID is
// the rkt UUID, and appName is the container name.
// TODO(yifan): If the rkt is using lkvm as the stage1 image, then this function will fail.
func (r *Runtime) ExecInContainer(containerID kubecontainer.ContainerID, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) error {
	glog.V(4).Infof("Rkt execing in container.")

	id, err := parseContainerID(containerID)
	if err != nil {
		return err
	}
	args := []string{"enter", fmt.Sprintf("--app=%s", id.appName), id.uuid}
	args = append(args, cmd...)
	command := buildCommand(r.config, args...)

	if tty {
		p, err := kubecontainer.StartPty(command)
		if err != nil {
			return err
		}
		defer p.Close()

		// make sure to close the stdout stream
		defer stdout.Close()

		kubecontainer.HandleResizing(resize, func(size term.Size) {
			term.SetSize(p.Fd(), size)
		})

		if stdin != nil {
			go io.Copy(p, stdin)
		}
		if stdout != nil {
			go io.Copy(stdout, p)
		}
		return newRktExitError(command.Wait())
	}
	if stdin != nil {
		// Use an os.Pipe here as it returns true *os.File objects.
		// This way, if you run 'kubectl exec <pod> -i bash' (no tty) and type 'exit',
		// the call below to command.Run() can unblock because its Stdin is the read half
		// of the pipe.
		r, w, err := r.os.Pipe()
		if err != nil {
			return newRktExitError(err)
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
	return newRktExitError(command.Run())
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

	ctx, cancel := context.WithTimeout(context.Background(), r.requestTimeout)
	defer cancel()
	listResp, err := r.apisvc.ListPods(ctx, &rktapi.ListPodsRequest{
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
func (r *Runtime) GetPodStatus(uid kubetypes.UID, name, namespace string) (*kubecontainer.PodStatus, error) {
	podStatus := &kubecontainer.PodStatus{
		ID:        uid,
		Name:      name,
		Namespace: namespace,
	}

	ctx, cancel := context.WithTimeout(context.Background(), r.requestTimeout)
	defer cancel()
	listResp, err := r.apisvc.ListPods(ctx, &rktapi.ListPodsRequest{
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

	// If we are running no-op network plugin, then get the pod IP from the rkt pod status.
	if r.networkPlugin.Name() == network.DefaultPluginName {
		if latestPod != nil {
			for _, n := range latestPod.Networks {
				if n.Name == defaultNetworkName {
					podStatus.IP = n.Ipv4
					break
				}
			}
		}
	} else {
		containerID := kubecontainer.ContainerID{ID: string(uid)}
		status, err := r.networkPlugin.GetPodNetworkStatus(namespace, name, containerID)
		if err != nil {
			glog.Warningf("rkt: Failed to get pod network status for pod (UID %q, name %q, namespace %q): %v", uid, name, namespace, err)
		} else if status != nil {
			// status can be nil when the pod is running on the host network, in which case the pod IP
			// will be populated by the upper layer.
			podStatus.IP = status.IP.String()
		}
	}

	return podStatus, nil
}

// getOSReleaseInfo reads /etc/os-release and returns a map
// that contains the key value pairs in that file.
func getOSReleaseInfo() (map[string]string, error) {
	result := make(map[string]string)

	path := "/etc/os-release"
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		info := strings.SplitN(line, "=", 2)
		if len(info) != 2 {
			return nil, fmt.Errorf("unexpected entry in os-release %q", line)
		}
		result[info[0]] = info[1]
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return result, nil
}
