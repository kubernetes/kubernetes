// Copyright 2015 The rkt Authors
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

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/signal"
	"path"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/go-systemd/activation"
	"github.com/coreos/rkt/api/v1alpha"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/common/cgroup/v1"
	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/coreos/rkt/pkg/set"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/coreos/rkt/version"
	"github.com/hashicorp/errwrap"
	"github.com/hashicorp/golang-lru"
	"github.com/spf13/cobra"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

const (
	// Default size for the pod LRU cache.
	defaultPodCacheSize = 512
	// Default size for the image LRU cache.
	defaultImageCacheSize = 512
)

var (
	supportedAPIVersion = "1.0.0-alpha"
	cmdAPIService       = &cobra.Command{
		Use:   fmt.Sprintf(`api-service [--listen="%s"]`, common.APIServiceListenAddr),
		Short: "Run API service (experimental)",
		Long: fmt.Sprintf(`The API service listens for gRPC requests on the address and port specified by
the --listen option, by default %s

Specify the address 0.0.0.0 to listen on all interfaces.

It will also run correctly as a systemd socket-activated service, see
systemd.socket(5).`, common.APIServiceListenAddr),
		Run: runWrapper(runAPIService),
	}

	flagAPIServiceListenAddr string
	flagSocketListen         bool
	systemdFDs               = activation.Files // for mocking
)

func init() {
	cmdRkt.AddCommand(cmdAPIService)
	cmdAPIService.Flags().StringVar(&flagAPIServiceListenAddr, "listen", "", "address to listen for client API requests")
}

type podCacheItem struct {
	pod   *v1alpha.Pod
	mtime time.Time
}

type imageCacheItem struct {
	image *v1alpha.Image
}

// copyPod copies the immutable information of the pod into the new pod.
func copyPod(pod *v1alpha.Pod) *v1alpha.Pod {
	p := &v1alpha.Pod{
		Id:          pod.Id,
		Manifest:    pod.Manifest,
		Annotations: pod.Annotations,
	}

	for _, app := range pod.Apps {
		p.Apps = append(p.Apps, &v1alpha.App{
			Name:        app.Name,
			Image:       app.Image,
			Annotations: app.Annotations,
		})
	}
	return p
}

// copyImage copies the image object to avoid modification on the original one.
func copyImage(img *v1alpha.Image) *v1alpha.Image {
	return &v1alpha.Image{
		BaseFormat:      img.BaseFormat,
		Id:              img.Id,
		Name:            img.Name,
		Version:         img.Version,
		ImportTimestamp: img.ImportTimestamp,
		Manifest:        img.Manifest,
		Size:            img.Size,
		Annotations:     img.Annotations,
		Labels:          img.Labels,
	}
}

// v1AlphaAPIServer implements v1Alpha.APIServer interface.
type v1AlphaAPIServer struct {
	store    *imagestore.Store
	podCache *lru.Cache
	imgCache *lru.Cache
}

var _ v1alpha.PublicAPIServer = &v1AlphaAPIServer{}

func newV1AlphaAPIServer() (*v1AlphaAPIServer, error) {
	s, err := imagestore.NewStore(storeDir())
	if err != nil {
		return nil, err
	}

	podCache, err := lru.New(defaultPodCacheSize)
	if err != nil {
		return nil, err
	}

	imgCache, err := lru.New(defaultImageCacheSize)
	if err != nil {
		return nil, err
	}

	return &v1AlphaAPIServer{
		store:    s,
		podCache: podCache,
		imgCache: imgCache,
	}, nil
}

// GetInfo returns the information about the rkt, appc, api server version.
func (s *v1AlphaAPIServer) GetInfo(context.Context, *v1alpha.GetInfoRequest) (*v1alpha.GetInfoResponse, error) {
	return &v1alpha.GetInfoResponse{
		Info: &v1alpha.Info{
			RktVersion:  version.Version,
			AppcVersion: schema.AppContainerVersion.String(),
			ApiVersion:  supportedAPIVersion,
			GlobalFlags: &v1alpha.GlobalFlags{
				Dir:                getDataDir(),
				SystemConfigDir:    globalFlags.SystemConfigDir,
				LocalConfigDir:     globalFlags.LocalConfigDir,
				UserConfigDir:      globalFlags.UserConfigDir,
				InsecureFlags:      globalFlags.InsecureFlags.String(),
				TrustKeysFromHttps: globalFlags.TrustKeysFromHTTPS,
			},
		},
	}, nil
}

func findInKeyValues(kvs []*v1alpha.KeyValue, key string) (string, bool) {
	for _, kv := range kvs {
		if kv.Key == key {
			return kv.Value, true
		}
	}
	return "", false
}

// containsAllKeyValues returns true if the actualKVs contains all of the key-value
// pairs listed in requiredKVs, otherwise it returns false.
func containsAllKeyValues(actualKVs []*v1alpha.KeyValue, requiredKVs []*v1alpha.KeyValue) bool {
	for _, requiredKV := range requiredKVs {
		actualValue, ok := findInKeyValues(actualKVs, requiredKV.Key)
		if !ok || actualValue != requiredKV.Value {
			return false
		}
	}
	return true
}

// isBaseNameOf returns true if 'basename' is the basename of 's'.
func isBaseNameOf(basename, s string) bool {
	return basename == path.Base(s)
}

// isPrefixOf returns true if 'prefix' is the prefix of 's'.
func isPrefixOf(prefix, s string) bool {
	return strings.HasPrefix(s, prefix)
}

// isPartOf returns true if 'keyword' is part of 's'.
func isPartOf(keyword, s string) bool {
	return strings.Contains(s, keyword)
}

// satisfiesPodFilter returns true if the pod satisfies the filter.
// The pod, filter must not be nil.
func satisfiesPodFilter(pod v1alpha.Pod, filter v1alpha.PodFilter) bool {
	// Filter according to the ID.
	if len(filter.Ids) > 0 {
		s := set.NewString(filter.Ids...)
		if !s.Has(pod.Id) {
			return false
		}
	}

	// Filter according to the state.
	if len(filter.States) > 0 {
		foundState := false
		for _, state := range filter.States {
			if pod.State == state {
				foundState = true
				break
			}
		}
		if !foundState {
			return false
		}
	}

	// Filter according to the app names.
	if len(filter.AppNames) > 0 {
		s := set.NewString()
		for _, app := range pod.Apps {
			s.Insert(app.Name)
		}
		if !s.HasAll(filter.AppNames...) {
			return false
		}
	}

	// Filter according to the image IDs.
	if len(filter.ImageIds) > 0 {
		s := set.NewString()
		for _, app := range pod.Apps {
			s.Insert(app.Image.Id)
		}
		if !s.HasAll(filter.ImageIds...) {
			return false
		}
	}

	// Filter according to the network names.
	if len(filter.NetworkNames) > 0 {
		s := set.NewString()
		for _, network := range pod.Networks {
			s.Insert(network.Name)
		}
		if !s.HasAll(filter.NetworkNames...) {
			return false
		}
	}

	// Filter according to the annotations.
	if len(filter.Annotations) > 0 {
		if !containsAllKeyValues(pod.Annotations, filter.Annotations) {
			return false
		}
	}

	// Filter according to the cgroup.
	if len(filter.Cgroups) > 0 {
		s := set.NewString(filter.Cgroups...)
		if !s.Has(pod.Cgroup) {
			return false
		}
	}

	// Filter if pod's cgroup is a prefix of the passed in cgroup
	if len(filter.PodSubCgroups) > 0 {
		matched := false
		if pod.Cgroup != "" {
			for _, cgroup := range filter.PodSubCgroups {
				if strings.HasPrefix(cgroup, pod.Cgroup) {
					matched = true
					break
				}
			}
		}

		if !matched {
			return false
		}
	}

	return true
}

// satisfiesAnyPodFilters returns true if any of the filter conditions is satisfied
// by the pod, or there's no filters.
func satisfiesAnyPodFilters(pod *v1alpha.Pod, filters []*v1alpha.PodFilter) bool {
	// No filters, return true directly.
	if len(filters) == 0 {
		return true
	}

	for _, filter := range filters {
		if satisfiesPodFilter(*pod, *filter) {
			return true
		}
	}
	return false
}

// getPodCgroup gets the cgroup path for a pod. This can be done in one of a couple ways:
//
// 1) stage1-coreos
//
// This one can be tricky because after machined registration, the cgroup might change.
// For these flavors, the cgroup it will end up in after machined moves it is
// written to a file named `subcgroup`, so we use that instead.
//
// 2) All others
//
// Assume that we can simply get the cgroup of the pod's init PID, and use that.
func getPodCgroup(p *pkgPod.Pod, pid int) (string, error) {
	// Note, this file should always exist if the pid is known since it is
	// written before the pid file
	// The contents are of the form:
	// machined-registration (whether it has happened or not, but if stage1 plans to do so):
	//   'machine.slice/machine-rkt\x2dbfb67ff1\x2dc745\x2d4aec\x2db1e1\x2d7ce85a1fd42b.scope'
	// no registration, ran as systemd-unit 'foo.service':
	//   'system.slice/foo.service'
	// stage1-fly: file does not exist
	subCgroupFile := filepath.Join(p.Path(), "subcgroup")
	data, err := ioutil.ReadFile(subCgroupFile)
	// normalize cgroup to include a leading '/'
	if err == nil {
		strCgroup := string(data)
		if strings.HasPrefix(strCgroup, "/") {
			return strCgroup, nil
		}
		return "/" + strCgroup, nil
	}

	if !os.IsNotExist(err) {
		return "", err
	}
	// if it is an "isNotExist", assume this is a stage1 flavor that won't change
	// cgroups. Just give the systemd cgroup of its pid
	// Get cgroup for the "name=systemd" controller; we assume the api-server is
	// running on a system using systemd for returning cgroups, and will just not
	// set it otherwise.
	cgroup, err := v1.GetCgroupPathByPid(pid, "name=systemd")
	if err != nil {
		return "", err
	}
	// If the stage1 systemd > v226, it will put the PID1 into "init.scope"
	// implicit scope unit in the root slice.
	// See https://github.com/coreos/rkt/pull/2331#issuecomment-203540543
	//
	// TODO(yifan): Revisit this when using unified cgroup hierarchy.
	cgroup = strings.TrimSuffix(cgroup, "/init.scope")
	return cgroup, nil
}

// getApplist returns a list of apps in the pod.
func getApplist(manifest *schema.PodManifest) []*v1alpha.App {
	var apps []*v1alpha.App
	for _, app := range manifest.Apps {
		img := &v1alpha.Image{
			BaseFormat: &v1alpha.ImageFormat{
				// Only support appc image now. If it's a docker image, then it
				// will be transformed to appc before storing in the disk store.
				Type:    v1alpha.ImageType_IMAGE_TYPE_APPC,
				Version: schema.AppContainerVersion.String(),
			},
			Id: app.Image.ID.String(),
			// Only image format and image ID are returned in 'ListPods()'.
		}

		apps = append(apps, &v1alpha.App{
			Name:        app.Name.String(),
			Image:       img,
			Annotations: convertAnnotationsToKeyValue(app.Annotations),
			State:       v1alpha.AppState_APP_STATE_UNDEFINED,
			ExitCode:    -1,
		})
	}
	return apps
}

// getNetworks returns the list of the info of the network that the pod belongs to.
func getNetworks(p *pkgPod.Pod) []*v1alpha.Network {
	var networks []*v1alpha.Network
	for _, n := range p.Nets {
		networks = append(networks, &v1alpha.Network{
			Name: n.NetName,
			// There will be IPv6 support soon so distinguish between v4 and v6
			Ipv4: n.IP.String(),
		})
	}
	return networks
}

type errs struct {
	errs []error
}

// Error is part of the error interface.
func (e errs) Error() string {
	return fmt.Sprintf("%v", e.errs)
}

// fillStaticAppInfo will modify the 'v1pod' in place with the information retrieved with 'pod'.
// Today, these information are static and will not change during the pod's lifecycle.
func fillStaticAppInfo(store *imagestore.Store, pod *pkgPod.Pod, v1pod *v1alpha.Pod) error {
	var errlist []error

	// Fill static app image info.
	for _, app := range v1pod.Apps {
		// Fill app's image info.
		app.Image = &v1alpha.Image{
			BaseFormat: &v1alpha.ImageFormat{
				// Only support appc image now. If it's a docker image, then it
				// will be transformed to appc before storing in the disk store.
				Type:    v1alpha.ImageType_IMAGE_TYPE_APPC,
				Version: schema.AppContainerVersion.String(),
			},
			Id: app.Image.Id,
			// Other information are not available because they require the image
			// info from store. Some of it is filled in below if possible.
		}

		im, err := pod.AppImageManifest(app.Name)
		if err != nil {
			stderr.PrintE(fmt.Sprintf("failed to get image manifests for app %q", app.Name), err)
			errlist = append(errlist, err)
		} else {
			app.Image.Name = im.Name.String()

			version, ok := im.Labels.Get("version")
			if !ok {
				version = "latest"
			}
			app.Image.Version = version
		}
	}

	if len(errlist) != 0 {
		return errs{errlist}
	}
	return nil
}

func (s *v1AlphaAPIServer) getBasicPodFromDisk(p *pkgPod.Pod) (*v1alpha.Pod, error) {
	pod := &v1alpha.Pod{Id: p.UUID.String()}

	data, manifest, err := p.PodManifest()
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to get the pod manifest for pod %q", p.UUID), err)
	} else {
		pod.Annotations = convertAnnotationsToKeyValue(manifest.Annotations)
		pod.Apps = getApplist(manifest)
		pod.Manifest = data
		err = fillStaticAppInfo(s.store, p, pod)
	}

	return pod, err
}

func getPodManifestModTime(p *pkgPod.Pod) (time.Time, error) {
	podFile := filepath.Join(p.Path(), "pod")
	fi, err := os.Stat(podFile)
	if err != nil {
		return time.Time{}, err
	}
	return fi.ModTime(), nil
}

// getBasicPod returns v1alpha.Pod with basic pod information.
func (s *v1AlphaAPIServer) getBasicPod(p *pkgPod.Pod) *v1alpha.Pod {
	mtime, mtimeErr := getPodManifestModTime(p)
	if mtimeErr != nil {
		stderr.PrintE(fmt.Sprintf("failed to read the pod manifest's mtime for pod %q", p.UUID), mtimeErr)
	}

	// Couldn't use pod.uuid directly as it's a pointer.
	itemValue, found := s.podCache.Get(p.UUID.String())
	if found && mtimeErr == nil {
		cacheItem := itemValue.(*podCacheItem)

		// Check the mtime to make sure we are not returning stale manifests.
		if !mtime.After(cacheItem.mtime) {
			return copyPod(cacheItem.pod)
		}
	}

	pod, err := s.getBasicPodFromDisk(p)
	if mtimeErr != nil || err != nil {
		// If any error happens or the mtime is unknown,
		// returns the raw pod directly without adding it to the cache.
		return pod
	}

	cacheItem := &podCacheItem{pod, mtime}
	s.podCache.Add(p.UUID.String(), cacheItem)

	// Return a copy of the pod, so the cached pod is not mutated later.
	return copyPod(cacheItem.pod)
}

// fillPodDetails fills the v1pod's dynamic info in place, e.g. the pod's state,
// the pod's network info, the apps' state, etc. Such information can change
// during the lifecycle of the pod, so we need to read it in every request.
func fillPodDetails(store *imagestore.Store, p *pkgPod.Pod, v1pod *v1alpha.Pod) {
	v1pod.Pid = -1

	switch p.State() {
	case pkgPod.Embryo:
		v1pod.State = v1alpha.PodState_POD_STATE_EMBRYO
		// When a pod is in embryo state, there is not much
		// information to return.
		return
	case pkgPod.Preparing:
		v1pod.State = v1alpha.PodState_POD_STATE_PREPARING
	case pkgPod.AbortedPrepare:
		v1pod.State = v1alpha.PodState_POD_STATE_ABORTED_PREPARE
	case pkgPod.Prepared:
		v1pod.State = v1alpha.PodState_POD_STATE_PREPARED
	case pkgPod.Running:
		v1pod.State = v1alpha.PodState_POD_STATE_RUNNING
		v1pod.Networks = getNetworks(p)
	case pkgPod.Deleting:
		v1pod.State = v1alpha.PodState_POD_STATE_DELETING
	case pkgPod.Exited:
		v1pod.State = v1alpha.PodState_POD_STATE_EXITED
	case pkgPod.Garbage, pkgPod.ExitedGarbage:
		v1pod.State = v1alpha.PodState_POD_STATE_GARBAGE
	default:
		v1pod.State = v1alpha.PodState_POD_STATE_UNDEFINED
		return
	}

	createdAt, err := p.CreationTime()
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to get the creation time for pod %q", p.UUID), err)
	} else if !createdAt.IsZero() {
		v1pod.CreatedAt = createdAt.UnixNano()
	}

	startedAt, err := p.StartTime()
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to get the start time for pod %q", p.UUID), err)
	} else if !startedAt.IsZero() {
		v1pod.StartedAt = startedAt.UnixNano()

	}

	gcMarkedAt, err := p.GCMarkedTime()
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to get the gc marked time for pod %q", p.UUID), err)
	} else if !gcMarkedAt.IsZero() {
		v1pod.GcMarkedAt = gcMarkedAt.UnixNano()
	}

	pid, err := p.Pid()
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to get the PID for pod %q", p.UUID), err)
	} else {
		v1pod.Pid = int32(pid)
	}

	if v1pod.State == v1alpha.PodState_POD_STATE_RUNNING {
		pid, err := p.ContainerPid1()
		if err != nil {
			stderr.PrintE(fmt.Sprintf("failed to get the container PID1 for pod %q", p.UUID), err)
		} else {
			cgroup, err := getPodCgroup(p, pid)
			if err != nil {
				stderr.PrintE(fmt.Sprintf("failed to get the cgroup path for pod %q", p.UUID), err)
			} else {
				v1pod.Cgroup = cgroup
			}
		}
	}

	for _, app := range v1pod.Apps {
		readStatus := false

		if p.State() == pkgPod.Running {
			readStatus = true
			app.State = v1alpha.AppState_APP_STATE_RUNNING
		} else if p.IsAfterRun() {
			readStatus = true
			app.State = v1alpha.AppState_APP_STATE_EXITED
		} else {
			app.State = v1alpha.AppState_APP_STATE_UNDEFINED
		}

		if readStatus {
			exitCode, err := p.AppExitCode(app.Name)
			if err != nil && !os.IsNotExist(err) {
				stderr.PrintE(fmt.Sprintf("failed to read status for app %q", app.Name), err)
			}
			app.ExitCode = int32(exitCode)
		}
	}
}

func (s *v1AlphaAPIServer) ListPods(ctx context.Context, request *v1alpha.ListPodsRequest) (*v1alpha.ListPodsResponse, error) {
	var pods []*v1alpha.Pod
	if err := pkgPod.WalkPods(getDataDir(), pkgPod.IncludeMostDirs, func(p *pkgPod.Pod) {
		pod := s.getBasicPod(p)

		fillPodDetails(s.store, p, pod)

		// Filters are combined with 'OR'.
		if !satisfiesAnyPodFilters(pod, request.Filters) {
			return
		}

		if !request.Detail {
			pod.Manifest = nil
		}

		pods = append(pods, pod)
	}); err != nil {
		stderr.PrintE("failed to list pod", err)
		return nil, err
	}
	return &v1alpha.ListPodsResponse{Pods: pods}, nil
}

func (s *v1AlphaAPIServer) InspectPod(ctx context.Context, request *v1alpha.InspectPodRequest) (*v1alpha.InspectPodResponse, error) {
	p, err := pkgPod.PodFromUUIDString(getDataDir(), request.Id)
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to get pod %q", request.Id), err)
		return nil, err
	}
	defer p.Close()

	pod := s.getBasicPod(p)

	fillPodDetails(s.store, p, pod)

	return &v1alpha.InspectPodResponse{Pod: pod}, nil
}

// aciInfoToV1AlphaAPIImage takes an aciInfo object and construct the v1alpha.Image object.
func aciInfoToV1AlphaAPIImage(store *imagestore.Store, aciInfo *imagestore.ACIInfo) (*v1alpha.Image, error) {
	manifest, err := store.GetImageManifestJSON(aciInfo.BlobKey)
	if err != nil {
		stderr.PrintE("failed to read the image manifest", err)
		return nil, err
	}

	var im schema.ImageManifest
	if err = json.Unmarshal(manifest, &im); err != nil {
		stderr.PrintE("failed to unmarshal image manifest", err)
		return nil, err
	}

	version, ok := im.Labels.Get("version")
	if !ok {
		version = "latest"
	}

	return &v1alpha.Image{
		BaseFormat: &v1alpha.ImageFormat{
			// Only support appc image now. If it's a docker image, then it
			// will be transformed to appc before storing in the disk store.
			Type:    v1alpha.ImageType_IMAGE_TYPE_APPC,
			Version: schema.AppContainerVersion.String(),
		},
		Id:              aciInfo.BlobKey,
		Name:            im.Name.String(),
		Version:         version,
		ImportTimestamp: aciInfo.ImportTime.Unix(),
		Manifest:        manifest,
		Size:            aciInfo.Size + aciInfo.TreeStoreSize,
		Annotations:     convertAnnotationsToKeyValue(im.Annotations),
		Labels:          convertLabelsToKeyValue(im.Labels),
	}, nil
}

func convertAnnotationsToKeyValue(as types.Annotations) []*v1alpha.KeyValue {
	kvs := make([]*v1alpha.KeyValue, 0, len(as))
	for _, a := range as {
		kv := &v1alpha.KeyValue{
			Key:   string(a.Name),
			Value: a.Value,
		}
		kvs = append(kvs, kv)
	}
	return kvs
}

func convertLabelsToKeyValue(ls types.Labels) []*v1alpha.KeyValue {
	kvs := make([]*v1alpha.KeyValue, 0, len(ls))
	for _, l := range ls {
		kv := &v1alpha.KeyValue{
			Key:   string(l.Name),
			Value: l.Value,
		}
		kvs = append(kvs, kv)
	}
	return kvs
}

// satisfiesImageFilter returns true if the image satisfies the filter.
// The image, filter must not be nil.
func satisfiesImageFilter(image v1alpha.Image, filter v1alpha.ImageFilter) bool {
	// Filter according to the IDs.
	if len(filter.Ids) > 0 {
		s := set.NewString(filter.Ids...)
		if !s.Has(image.Id) {
			return false
		}
	}

	// Filter according to the image full names.
	if len(filter.FullNames) > 0 {
		s := set.NewString(filter.FullNames...)
		if !s.Has(image.Name) {
			return false
		}
	}

	// Filter according to the image name prefixes.
	if len(filter.Prefixes) > 0 {
		s := set.NewString(filter.Prefixes...)
		if !s.ConditionalHas(isPrefixOf, image.Name) {
			return false
		}
	}

	// Filter according to the image base name.
	if len(filter.BaseNames) > 0 {
		s := set.NewString(filter.BaseNames...)
		if !s.ConditionalHas(isBaseNameOf, image.Name) {
			return false
		}
	}

	// Filter according to the image keywords.
	if len(filter.Keywords) > 0 {
		s := set.NewString(filter.Keywords...)
		if !s.ConditionalHas(isPartOf, image.Name) {
			return false
		}
	}

	// Filter according to the imported time.
	if filter.ImportedAfter > 0 {
		if image.ImportTimestamp <= filter.ImportedAfter {
			return false
		}
	}
	if filter.ImportedBefore > 0 {
		if image.ImportTimestamp >= filter.ImportedBefore {
			return false
		}
	}

	// Filter according to the image labels.
	if len(filter.Labels) > 0 {
		if !containsAllKeyValues(image.Labels, filter.Labels) {
			return false
		}
	}

	// Filter according to the annotations.
	if len(filter.Annotations) > 0 {
		if !containsAllKeyValues(image.Annotations, filter.Annotations) {
			return false
		}
	}

	return true
}

// satisfiesAnyImageFilters returns true if any of the filter conditions is satisfied
// by the image, or there's no filters.
func satisfiesAnyImageFilters(image *v1alpha.Image, filters []*v1alpha.ImageFilter) bool {
	// No filters, return true directly.
	if len(filters) == 0 {
		return true
	}
	for _, filter := range filters {
		if satisfiesImageFilter(*image, *filter) {
			return true
		}
	}
	return false
}

func (s *v1AlphaAPIServer) ListImages(ctx context.Context, request *v1alpha.ListImagesRequest) (*v1alpha.ListImagesResponse, error) {
	aciInfos, err := s.store.GetAllACIInfos(nil, false)
	if err != nil {
		stderr.PrintE("failed to get all ACI infos", err)
		return nil, err
	}

	var images []*v1alpha.Image
	for _, aciInfo := range aciInfos {
		image, err := s.getImageInfo(aciInfo.BlobKey)
		if err != nil {
			continue
		}
		if !satisfiesAnyImageFilters(image, request.Filters) {
			continue
		}
		if !request.Detail {
			image.Manifest = nil // Do not return image manifest in ListImages(detail=false).
		}
		images = append(images, image)
	}
	return &v1alpha.ListImagesResponse{Images: images}, nil
}

// getImageInfo returns the image info of a given image ID.
// It will first try to read the image info from cache.
// - If the image exists in cache:
//   - If the image exists on disk, then return it directly.
//   - Otherwise, return nil.
// - Else, try to return the image from disk.
func (s *v1AlphaAPIServer) getImageInfo(imageID string) (*v1alpha.Image, error) {
	item, found := s.imgCache.Get(imageID)
	if found {
		image := item.(*imageCacheItem).image

		// Check if the image has been removed.
		if found := s.store.HasFullKey(image.Id); !found {
			s.imgCache.Remove(imageID)
			return nil, fmt.Errorf("no such image with ID %q", imageID)
		}
		return copyImage(image), nil
	}

	image, err := s.getImageInfoFromDisk(s.store, imageID)
	if err != nil {
		return nil, err
	}

	s.imgCache.Add(imageID, &imageCacheItem{image})

	return copyImage(image), err
}

// getImageInfoFromDisk for a given image ID, returns the *v1alpha.Image object.
func (s *v1AlphaAPIServer) getImageInfoFromDisk(store *imagestore.Store, imageID string) (*v1alpha.Image, error) {
	aciInfo, err := store.GetACIInfoWithBlobKey(imageID)
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to get ACI info for image %q", imageID), err)
		return nil, err
	}

	image, err := aciInfoToV1AlphaAPIImage(store, aciInfo)
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to convert ACI to v1alphaAPIImage for image %q", imageID), err)
		return nil, err
	}
	return image, nil
}

func (s *v1AlphaAPIServer) InspectImage(ctx context.Context, request *v1alpha.InspectImageRequest) (*v1alpha.InspectImageResponse, error) {
	image, err := s.getImageInfo(request.Id)
	if err != nil {
		return nil, err
	}
	return &v1alpha.InspectImageResponse{Image: image}, nil
}

// LogsStreamWriter is a wrapper around a gRPC streaming server.
// Implements io.Writer interface.
type LogsStreamWriter struct {
	server v1alpha.PublicAPI_GetLogsServer
}

func (sw LogsStreamWriter) Write(b []byte) (int, error) {
	lines := common.RemoveEmptyLines(string(b))

	if err := sw.server.Send(&v1alpha.GetLogsResponse{Lines: lines}); err != nil {
		return 0, err
	}
	return len(b), nil
}

func (s *v1AlphaAPIServer) GetLogs(request *v1alpha.GetLogsRequest, server v1alpha.PublicAPI_GetLogsServer) error {
	return s.constrainedGetLogs(request, server)
}

func (s *v1AlphaAPIServer) ListenEvents(request *v1alpha.ListenEventsRequest, server v1alpha.PublicAPI_ListenEventsServer) error {
	return fmt.Errorf("not implemented yet")
}

// Open one or more listening sockets, then start the gRPC server
func runAPIService(cmd *cobra.Command, args []string) (exit int) {
	// Set up the signal handler here so we can make sure the
	// signals are caught after print the starting message.
	signal.Notify(exitCh, syscall.SIGINT, syscall.SIGTERM)

	stderr.Print("API service starting...")

	listeners, err := openAPISockets()
	if err != nil {
		stderr.PrintE("Failed to open sockets", err)
		return 254
	}
	if len(listeners) == 0 { // This is unlikely...
		stderr.Println("No sockets to listen to. Quitting.")
		return 254
	}

	publicServer := grpc.NewServer() // TODO(yifan): Add TLS credential option.

	v1AlphaAPIServer, err := newV1AlphaAPIServer()
	if err != nil {
		stderr.PrintE("failed to create API service", err)
		return 254
	}

	v1alpha.RegisterPublicAPIServer(publicServer, v1AlphaAPIServer)

	for _, l := range listeners {
		defer l.Close()
		go publicServer.Serve(l)
	}

	stderr.Printf("API service running")

	<-exitCh

	stderr.Print("API service exiting...")

	return
}

// Open API sockets based on command line parameters and
// the magic environment variable from systemd
//
// see sd_listen_fds(3)
func openAPISockets() ([]net.Listener, error) {
	listeners := []net.Listener{}

	fds := systemdFDs(true) // Try to get the socket fds from systemd
	if len(fds) > 0 {
		if flagAPIServiceListenAddr != "" {
			return nil, fmt.Errorf("started under systemd.socket(5), but --listen passed! Quitting.")
		}

		stderr.Printf("Listening on %d systemd-provided socket(s)\n", len(fds))
		for _, fd := range fds {
			l, err := net.FileListener(fd)
			if err != nil {
				return nil, errwrap.Wrap(fmt.Errorf("could not open listener"), err)
			}
			listeners = append(listeners, l)
		}
	} else {
		if flagAPIServiceListenAddr == "" {
			flagAPIServiceListenAddr = common.APIServiceListenAddr
		}
		stderr.Printf("Listening on %s\n", flagAPIServiceListenAddr)

		l, err := net.Listen("tcp", flagAPIServiceListenAddr)
		if err != nil {
			return nil, errwrap.Wrap(fmt.Errorf("could not open tcp socket"), err)
		}
		listeners = append(listeners, l)
	}

	return listeners, nil
}
