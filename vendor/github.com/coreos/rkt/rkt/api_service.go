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
	"net"
	"os"
	"os/signal"
	"path"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/api/v1alpha"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/set"
	"github.com/coreos/rkt/store"
	"github.com/coreos/rkt/version"
	"github.com/spf13/cobra"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

var (
	supportedAPIVersion = "1.0.0-alpha"
	cmdAPIService       = &cobra.Command{
		Use:   `api-service [--listen="localhost:15441"]`,
		Short: "Run API service (experimental)",
		Long: `The API service listens for gRPC requests on the address and port specified by
the --listen option.

Specify the address 0.0.0.0 to listen on all interfaces.`,
		Run: runWrapper(runAPIService),
	}

	flagAPIServiceListenAddr string
)

func init() {
	cmdRkt.AddCommand(cmdAPIService)
	cmdAPIService.Flags().StringVar(&flagAPIServiceListenAddr, "listen", common.APIServiceListenAddr, "address to listen for client API requests")
}

// v1AlphaAPIServer implements v1Alpha.APIServer interface.
type v1AlphaAPIServer struct {
	store *store.Store
}

var _ v1alpha.PublicAPIServer = &v1AlphaAPIServer{}

func newV1AlphaAPIServer() (*v1AlphaAPIServer, error) {
	s, err := store.NewStore(getDataDir())
	if err != nil {
		return nil, err
	}

	return &v1AlphaAPIServer{
		store: s,
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

type valueGetter interface {
	Get(string) (string, bool)
}

// containsAllKeyValues returns true if the actualKVs contains all of the key-value
// pairs listed in requiredKVs, otherwise it returns false.
func containsAllKeyValues(actualKVs valueGetter, requiredKVs []*v1alpha.KeyValue) bool {
	for _, requiredKV := range requiredKVs {
		actualValue, ok := actualKVs.Get(requiredKV.Key)
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
// The pod, manifest, filter must not be nil.
func satisfiesPodFilter(pod v1alpha.Pod, manifest schema.PodManifest, filter v1alpha.PodFilter) bool {
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
		if !containsAllKeyValues(manifest.Annotations, filter.Annotations) {
			return false
		}
	}

	return true
}

// satisfiesAnyPodFilters returns true if any of the filter conditions is satisfied
// by the pod, or there's no filters.
func satisfiesAnyPodFilters(pod *v1alpha.Pod, manifest *schema.PodManifest, filters []*v1alpha.PodFilter) bool {
	// No filters, return true directly.
	if len(filters) == 0 {
		return true
	}

	// No manifest, but have filters, return false.
	if manifest == nil {
		return false
	}

	for _, filter := range filters {
		if satisfiesPodFilter(*pod, *manifest, *filter) {
			return true
		}
	}
	return false
}

// getPodManifest returns the pod manifest of the pod.
// Both marshaled and unmarshaled manifest are returned.
func getPodManifest(p *pod) (*schema.PodManifest, []byte, error) {
	data, err := p.readFile("pod")
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to read pod manifest for pod %q", p.uuid), err)
		return nil, nil, err
	}

	var manifest schema.PodManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		stderr.PrintE(fmt.Sprintf("failed to unmarshal pod manifest for pod %q", p.uuid), err)
		return nil, nil, err
	}
	return &manifest, data, nil
}

// getApplist returns a list of apps in the pod.
func getApplist(p *pod) ([]*v1alpha.App, error) {
	var apps []*v1alpha.App
	applist, err := p.getApps()
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to get app list for pod %q", p.uuid), err)
		return nil, err
	}

	for _, app := range applist {
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
			// State and exit code are not returned in 'ListPods()'.
		})
	}
	return apps, nil
}

// getNetworks returns the list of the info of the network that the pod belongs to.
func getNetworks(p *pod) []*v1alpha.Network {
	var networks []*v1alpha.Network
	for _, n := range p.nets {
		networks = append(networks, &v1alpha.Network{
			Name: n.NetName,
			// There will be IPv6 support soon so distinguish between v4 and v6
			Ipv4: n.IP.String(),
		})
	}
	return networks
}

// getBasicPod returns *v1alpha.Pod with basic pod information, it also returns a *schema.PodManifest
// object.
func getBasicPod(p *pod) (*v1alpha.Pod, *schema.PodManifest, error) {
	pod := &v1alpha.Pod{Id: p.uuid.String(), Pid: -1}

	// If a pod is in Embryo, Preparing, AbortedPrepare state,
	// only id and state will be returned.
	//
	// If a pod is in other states, the pod manifest and
	// apps will be returned when 'detailed' is true in the request.
	//
	// Valid pid and networks are only returned when a pod is in Running.
	switch p.getState() {
	case Embryo:
		pod.State = v1alpha.PodState_POD_STATE_EMBRYO
		return pod, nil, nil
	case Preparing:
		pod.State = v1alpha.PodState_POD_STATE_PREPARING
		return pod, nil, nil
	case AbortedPrepare:
		pod.State = v1alpha.PodState_POD_STATE_ABORTED_PREPARE
		return pod, nil, nil
	case Prepared:
		pod.State = v1alpha.PodState_POD_STATE_PREPARED
		return pod, nil, nil
	case Running:
		pod.State = v1alpha.PodState_POD_STATE_RUNNING
		pod.Networks = getNetworks(p)
	case Deleting:
		pod.State = v1alpha.PodState_POD_STATE_DELETING
	case Exited:
		pod.State = v1alpha.PodState_POD_STATE_EXITED
	case Garbage:
		pod.State = v1alpha.PodState_POD_STATE_GARBAGE
	default:
		pod.State = v1alpha.PodState_POD_STATE_UNDEFINED
		return pod, nil, nil
	}

	manifest, data, err := getPodManifest(p)
	if err != nil {
		return nil, nil, err
	}

	apps, err := getApplist(p)
	if err != nil {
		return nil, nil, err
	}

	if pod.State == v1alpha.PodState_POD_STATE_RUNNING ||
		pod.State == v1alpha.PodState_POD_STATE_DELETING ||
		pod.State == v1alpha.PodState_POD_STATE_EXITED {
		pid, err := p.getPID()
		if err != nil {
			return nil, nil, err
		}
		pod.Pid = int32(pid)
	}

	pod.Manifest = data
	pod.Apps = apps
	pod.Annotations = convertAnnotationsToKeyValue(manifest.Annotations)

	return pod, manifest, nil
}

func (s *v1AlphaAPIServer) ListPods(ctx context.Context, request *v1alpha.ListPodsRequest) (*v1alpha.ListPodsResponse, error) {
	var pods []*v1alpha.Pod
	if err := walkPods(includeMostDirs, func(p *pod) {
		pod, manifest, err := getBasicPod(p)
		if err != nil { // Do not return partial pods.
			return
		}

		// Filters are combined with 'OR'.
		if !satisfiesAnyPodFilters(pod, manifest, request.Filters) {
			return
		}

		if request.Detail {
			if err := fillAppInfo(s.store, p, pod); err != nil { // Do not return partial pods.
				return
			}
		} else {
			pod.Manifest = nil
		}
		pods = append(pods, pod)
	}); err != nil {
		stderr.PrintE("failed to list pod", err)
		return nil, err
	}
	return &v1alpha.ListPodsResponse{Pods: pods}, nil
}

// fillAppInfo fills the apps' state and image info of the pod.
func fillAppInfo(store *store.Store, p *pod, v1pod *v1alpha.Pod) error {
	statusDir, err := p.getStatusDir()
	if err != nil {
		stderr.PrintE("failed to get pod exit status directory", err)
		return err
	}

	for _, app := range v1pod.Apps {
		// Fill app's image info (id, name, version).
		fullImageID, err := store.ResolveKey(app.Image.Id)
		if err != nil {
			stderr.PrintE(fmt.Sprintf("failed to resolve the image ID %q", app.Image.Id), err)
			return err
		}

		im, err := p.getAppImageManifest(*types.MustACName(app.Name))
		if err != nil {
			stderr.PrintE(fmt.Sprintf("failed to get image manifests for app %q", app.Name), err)
			return err
		}

		version, ok := im.Labels.Get("version")
		if !ok {
			version = "latest"
		}

		app.Image = &v1alpha.Image{
			BaseFormat: &v1alpha.ImageFormat{
				// Only support appc image now. If it's a docker image, then it
				// will be transformed to appc before storing in the disk store.
				Type:    v1alpha.ImageType_IMAGE_TYPE_APPC,
				Version: schema.AppContainerVersion.String(),
			},
			Id:      fullImageID,
			Name:    im.Name.String(),
			Version: version,
			// Other information are not available because they require the image
			// info from store.
		}

		// Fill app's state and exit code.
		value, err := p.readIntFromFile(filepath.Join(statusDir, app.Name))
		if err == nil {
			app.State = v1alpha.AppState_APP_STATE_EXITED
			app.ExitCode = int32(value)
			continue
		}

		if !os.IsNotExist(err) {
			stderr.PrintE(fmt.Sprintf("failed to read status for app %q", app.Name), err)
			return err
		}
		// If status file does not exit, that means the
		// app is either running or aborted.
		//
		// FIXME(yifan): This is not acttually true, the app can be aborted while
		// the pod is still running if the spec changes.
		switch p.getState() {
		case Running:
			app.State = v1alpha.AppState_APP_STATE_RUNNING
		default:
			app.State = v1alpha.AppState_APP_STATE_UNDEFINED
		}

	}
	return nil
}

func (s *v1AlphaAPIServer) InspectPod(ctx context.Context, request *v1alpha.InspectPodRequest) (*v1alpha.InspectPodResponse, error) {
	uuid, err := types.NewUUID(request.Id)
	if err != nil {
		stderr.PrintE(fmt.Sprintf("invalid pod id %q", request.Id), err)
		return nil, err
	}

	p, err := getPod(uuid)
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to get pod %q", request.Id), err)
		return nil, err
	}
	defer p.Close()

	pod, _, err := getBasicPod(p)
	if err != nil {
		return nil, err
	}

	// Fill the extra pod info that is not available in ListPods(detail=false).
	if err := fillAppInfo(s.store, p, pod); err != nil {
		return nil, err
	}

	return &v1alpha.InspectPodResponse{Pod: pod}, nil
}

// aciInfoToV1AlphaAPIImage takes an aciInfo object and construct the v1alpha.Image object.
// It also returns the image manifest of the image.
func aciInfoToV1AlphaAPIImage(store *store.Store, aciInfo *store.ACIInfo) (*v1alpha.Image, *schema.ImageManifest, error) {
	manifest, err := store.GetImageManifestJSON(aciInfo.BlobKey)
	if err != nil {
		stderr.PrintE("failed to read the image manifest", err)
		return nil, nil, err
	}

	var im schema.ImageManifest
	if err = json.Unmarshal(manifest, &im); err != nil {
		stderr.PrintE("failed to unmarshal image manifest", err)
		return nil, nil, err
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
		Size:            aciInfo.Size,
		Annotations:     convertAnnotationsToKeyValue(im.Annotations),
	}, &im, nil
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

// satisfiesImageFilter returns true if the image satisfies the filter.
// The image, manifest, filter must not be nil.
func satisfiesImageFilter(image v1alpha.Image, manifest schema.ImageManifest, filter v1alpha.ImageFilter) bool {
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
		if !containsAllKeyValues(manifest.Labels, filter.Labels) {
			return false
		}
	}

	// Filter according to the annotations.
	if len(filter.Annotations) > 0 {
		if !containsAllKeyValues(manifest.Annotations, filter.Annotations) {
			return false
		}
	}

	return true
}

// satisfiesAnyImageFilters returns true if any of the filter conditions is satisfied
// by the image, or there's no filters.
func satisfiesAnyImageFilters(image *v1alpha.Image, manifest *schema.ImageManifest, filters []*v1alpha.ImageFilter) bool {
	// No filters, return true directly.
	if len(filters) == 0 {
		return true
	}
	for _, filter := range filters {
		if satisfiesImageFilter(*image, *manifest, *filter) {
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
		image, manifest, err := aciInfoToV1AlphaAPIImage(s.store, aciInfo)
		if err != nil {
			continue
		}
		if !satisfiesAnyImageFilters(image, manifest, request.Filters) {
			continue
		}
		if !request.Detail {
			image.Manifest = nil // Do not return image manifest in ListImages(detail=false).
		}
		images = append(images, image)
	}
	return &v1alpha.ListImagesResponse{Images: images}, nil
}

// getImageInfo for a given image ID, returns the *v1alpha.Image object.
func getImageInfo(store *store.Store, imageID string) (*v1alpha.Image, error) {
	key, err := store.ResolveKey(imageID)
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to resolve the image ID %q", imageID), err)
		return nil, err
	}

	aciInfo, err := store.GetACIInfoWithBlobKey(key)
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to get ACI info for image %q", key), err)
		return nil, err
	}

	image, _, err := aciInfoToV1AlphaAPIImage(store, aciInfo)
	if err != nil {
		stderr.PrintE(fmt.Sprintf("failed to convert ACI to v1alphaAPIImage for image %q", key), err)
		return nil, err
	}
	return image, nil
}

func (s *v1AlphaAPIServer) InspectImage(ctx context.Context, request *v1alpha.InspectImageRequest) (*v1alpha.InspectImageResponse, error) {
	image, err := getImageInfo(s.store, request.Id)
	if err != nil {
		return nil, err
	}
	return &v1alpha.InspectImageResponse{Image: image}, nil
}

func (s *v1AlphaAPIServer) GetLogs(request *v1alpha.GetLogsRequest, server v1alpha.PublicAPI_GetLogsServer) error {
	return fmt.Errorf("not implemented yet")
}

func (s *v1AlphaAPIServer) ListenEvents(request *v1alpha.ListenEventsRequest, server v1alpha.PublicAPI_ListenEventsServer) error {
	return fmt.Errorf("not implemented yet")
}

func runAPIService(cmd *cobra.Command, args []string) (exit int) {
	// Set up the signal handler here so we can make sure the
	// signals are caught after print the starting message.
	signal.Notify(exitCh, syscall.SIGINT, syscall.SIGTERM)

	stderr.Print("API service starting...")

	tcpl, err := net.Listen("tcp", flagAPIServiceListenAddr)
	if err != nil {
		stderr.Error(err)
		return 1
	}
	defer tcpl.Close()

	publicServer := grpc.NewServer() // TODO(yifan): Add TLS credential option.

	v1AlphaAPIServer, err := newV1AlphaAPIServer()
	if err != nil {
		stderr.PrintE("failed to create API service", err)
		return 1
	}

	v1alpha.RegisterPublicAPIServer(publicServer, v1AlphaAPIServer)

	go publicServer.Serve(tcpl)

	stderr.Printf("API service running on %v...", flagAPIServiceListenAddr)

	<-exitCh

	stderr.Print("API service exiting...")

	return
}
