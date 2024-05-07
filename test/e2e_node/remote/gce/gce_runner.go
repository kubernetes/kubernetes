/*
Copyright 2023 The Kubernetes Authors.

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

package gce

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"k8s.io/kubernetes/test/e2e_node/remote"

	"github.com/google/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	"sigs.k8s.io/yaml"
)

var _ remote.Runner = (*GCERunner)(nil)

func init() {
	remote.RegisterRunner("gce", NewGCERunner)
}

// envs is the type used to collect all node envs. The key is the env name,
// and the value is the env value
type envs map[string]string

// String function of flag.Value
func (e *envs) String() string {
	return fmt.Sprint(*e)
}

// Set function of flag.Value
func (e *envs) Set(value string) error {
	if value == "" {
		return nil
	}
	kv := strings.SplitN(value, "=", 2)
	if len(kv) != 2 {
		return fmt.Errorf("invalid env string %s", value)
	}
	emap := *e
	emap[kv[0]] = kv[1]
	return nil
}

// nodeEnvs is the node envs from the flag `node-env`.
var nodeEnvs = make(envs)

var project = flag.String("project", "", "gce project the hosts live in (gce)")
var zone = flag.String("zone", "", "gce zone that the hosts live in (gce)")
var instanceMetadata = flag.String("instance-metadata", "", "key/value metadata for instances separated by '=' or '<', 'k=v' means the key is 'k' and the value is 'v'; 'k<p' means the key is 'k' and the value is extracted from the local path 'p', e.g. k1=v1,k2<p2  (gce)")
var imageProject = flag.String("image-project", "", "gce project the hosts live in  (gce)")
var instanceType = flag.String("instance-type", "e2-medium", "GCP Machine type to use for test")
var preemptibleInstances = flag.Bool("preemptible-instances", false, "If true, gce instances will be configured to be preemptible  (gce)")

func init() {
	flag.Var(&nodeEnvs, "node-env", "An environment variable passed to instance as metadata, e.g. when '--node-env=PATH=/usr/bin' is specified, there will be an extra instance metadata 'PATH=/usr/bin'.")
}

type GCERunner struct {
	cfg       remote.Config
	gceImages *internalGCEImageConfig
}

const (
	defaultGCEMachine = "e2-standard-2"
)

func NewGCERunner(cfg remote.Config) remote.Runner {
	if cfg.InstanceNamePrefix == "" {
		cfg.InstanceNamePrefix = "tmp-node-e2e-" + uuid.New().String()[:8]
	}
	return &GCERunner{cfg: cfg}
}

func (g *GCERunner) Validate() error {
	if len(g.cfg.Hosts) == 0 && g.cfg.ImageConfigFile == "" && len(g.cfg.Images) == 0 {
		klog.Fatalf("Must specify one of --image-config-file, --hosts, --images.")
	}

	_, err := runGCPCommandWithZones("compute", "instances", "list")
	if err != nil {
		klog.Fatalf("While listing GCE instances: %v", err)
	}

	if g.gceImages, err = g.prepareGceImages(); err != nil {
		klog.Fatalf("While preparing GCE images: %v", err)
	}
	return nil
}

func (g *GCERunner) StartTests(suite remote.TestSuite, archivePath string, results chan *remote.TestResult) (numTests int) {
	for shortName := range g.gceImages.images {
		imageConfig := g.gceImages.images[shortName]
		numTests++
		fmt.Printf("Initializing e2e tests using image %s/%s/%s.\n", shortName, imageConfig.project, imageConfig.image)
		go func(image *internalGCEImage, junitFileName string) {
			results <- g.testGCEImage(suite, archivePath, image, junitFileName)
		}(&imageConfig, shortName)
	}
	return
}

// Accelerator contains type and count about resource.
type Accelerator struct {
	Type  string `json:"type,omitempty"`
	Count int64  `json:"count,omitempty"`
}

// Resources contains accelerators array.
type Resources struct {
	Accelerators []Accelerator `json:"accelerators,omitempty"`
}

// internalGCEImage is an internal GCE image representation for E2E node.
type internalGCEImage struct {
	image string
	// imageDesc is the description of the image. If empty, the value in the
	// 'image' will be used.
	imageDesc       string
	kernelArguments []string
	project         string
	resources       Resources
	metadata        *gceMetadata
	machine         string
}

type internalGCEImageConfig struct {
	images map[string]internalGCEImage
}

// GCEImageConfig specifies what images should be run and how for these tests.
// It can be created via the `--images` and `--image-project` flags, or by
// specifying the `--image-config-file` flag, pointing to a json or yaml file
// of the form:
//
//	images:
//	  short-name:
//	    image: gce-image-name
//	    project: gce-image-project
//	    machine: for benchmark only, the machine type (GCE instance) to run test
//	    tests: for benchmark only, a list of ginkgo focus strings to match tests
//
// TODO(coufon): replace 'image' with 'node' in configurations
// and we plan to support testing custom machines other than GCE by specifying Host
type GCEImageConfig struct {
	Images map[string]GCEImage `json:"images"`
}

// GCEImage contains some information about GCE Image.
type GCEImage struct {
	Image      string `json:"image,omitempty"`
	ImageRegex string `json:"image_regex,omitempty"`
	// ImageFamily is the image family to use. The latest image from the image family will be used, e.g cos-81-lts.
	ImageFamily     string    `json:"image_family,omitempty"`
	ImageDesc       string    `json:"image_description,omitempty"`
	KernelArguments []string  `json:"kernel_arguments,omitempty"`
	Project         string    `json:"project"`
	Metadata        string    `json:"metadata"`
	Machine         string    `json:"machine,omitempty"`
	Resources       Resources `json:"resources,omitempty"`
}

// Returns an image name based on regex and given GCE project.
func (g *GCERunner) getGCEImage(imageRegex, imageFamily string, project string) (string, error) {
	data, err := runGCPCommandNoProject("compute", "images", "list",
		"--format=json", "--project="+project)
	if err != nil {
		return "", fmt.Errorf("failed to list images in project %q: %w", project, err)
	}
	var images []gceImage
	err = json.Unmarshal(data, &images)
	if err != nil {
		return "", fmt.Errorf("failed to parse images: %w", err)
	}

	imageObjs := []imageObj{}
	imageRe := regexp.MustCompile(imageRegex)
	for _, instance := range images {
		if imageRegex != "" && !imageRe.MatchString(instance.Name) {
			continue
		}
		if imageFamily != "" && instance.Family != imageFamily {
			continue
		}
		creationTime, err := time.Parse(time.RFC3339, instance.CreationTimestamp)
		if err != nil {
			return "", fmt.Errorf("failed to parse instance creation timestamp %q: %w", instance.CreationTimestamp, err)
		}
		io := imageObj{
			creationTime: creationTime,
			name:         instance.Name,
		}
		imageObjs = append(imageObjs, io)
	}

	// Pick the latest image after sorting.
	sort.Sort(byCreationTime(imageObjs))
	if len(imageObjs) > 0 {
		klog.V(4).Infof("found images %+v based on regex %q and family %q in project %q", imageObjs, imageRegex, imageFamily, project)
		return imageObjs[0].name, nil
	}
	return "", fmt.Errorf("found zero images based on regex %q and family %q in project %q", imageRegex, imageFamily, project)
}

func (g *GCERunner) prepareGceImages() (*internalGCEImageConfig, error) {
	gceImages := &internalGCEImageConfig{
		images: make(map[string]internalGCEImage),
	}

	// Parse images from given config file and convert them to internalGCEImage.
	if g.cfg.ImageConfigFile != "" {
		configPath := g.cfg.ImageConfigFile
		if g.cfg.ImageConfigDir != "" {
			configPath = filepath.Join(g.cfg.ImageConfigDir, g.cfg.ImageConfigFile)
		}

		imageConfigData, err := os.ReadFile(configPath)
		if err != nil {
			return nil, fmt.Errorf("Could not read image config file provided: %w", err)
		}
		// Unmarshal the given image config file. All images for this test run will be organized into a map.
		// shortName->GCEImage, e.g cos-stable->cos-stable-81-12871-103-0.
		externalImageConfig := GCEImageConfig{Images: make(map[string]GCEImage)}
		err = yaml.Unmarshal(imageConfigData, &externalImageConfig)
		if err != nil {
			return nil, fmt.Errorf("Could not parse image config file: %w", err)
		}

		for shortName, imageConfig := range externalImageConfig.Images {
			var image string
			if (imageConfig.ImageRegex != "" || imageConfig.ImageFamily != "") && imageConfig.Image == "" {
				image, err = g.getGCEImage(imageConfig.ImageRegex, imageConfig.ImageFamily, imageConfig.Project)
				if err != nil {
					return nil, fmt.Errorf("Could not retrieve a image based on image regex %q and family %q: %v",
						imageConfig.ImageRegex, imageConfig.ImageFamily, err)
				}
			} else {
				image = imageConfig.Image
			}
			// Convert the given image into an internalGCEImage.
			metadata := imageConfig.Metadata
			if len(strings.TrimSpace(*instanceMetadata)) > 0 {
				metadata += "," + *instanceMetadata
			}
			gceImage := internalGCEImage{
				image:           image,
				imageDesc:       imageConfig.ImageDesc,
				project:         imageConfig.Project,
				metadata:        g.getImageMetadata(metadata),
				kernelArguments: imageConfig.KernelArguments,
				machine:         imageConfig.Machine,
				resources:       imageConfig.Resources,
			}
			if gceImage.imageDesc == "" {
				gceImage.imageDesc = gceImage.image
			}
			gceImages.images[shortName] = gceImage
		}
	}

	// Allow users to specify additional images via cli flags for local testing
	// convenience; merge in with config file
	if len(g.cfg.Images) > 0 {
		if *imageProject == "" {
			klog.Fatal("Must specify --image-project if you specify --images")
		}
		for _, image := range g.cfg.Images {
			gceImage := internalGCEImage{
				image:    image,
				project:  *imageProject,
				metadata: g.getImageMetadata(*instanceMetadata),
			}
			gceImages.images[image] = gceImage
		}
	}

	if len(gceImages.images) != 0 && *zone == "" {
		return nil, errors.New("must specify --zone flag")
	}
	// Make sure GCP project is set. Without a project, images can't be retrieved..
	for shortName, imageConfig := range gceImages.images {
		if imageConfig.project == "" {
			return nil, fmt.Errorf("invalid config for %v; must specify a project", shortName)
		}
	}
	if len(gceImages.images) != 0 {
		if *project == "" {
			return nil, errors.New("must specify --project flag to launch images into")
		}
	}

	return gceImages, nil
}

type imageObj struct {
	creationTime time.Time
	name         string
}

type byCreationTime []imageObj

func (a byCreationTime) Len() int           { return len(a) }
func (a byCreationTime) Less(i, j int) bool { return a[i].creationTime.After(a[j].creationTime) }
func (a byCreationTime) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func (g *GCERunner) getImageMetadata(input string) *gceMetadata {
	if input == "" {
		return nil
	}
	klog.V(3).Infof("parsing instance metadata: %q", input)
	raw := g.parseInstanceMetadata(input)
	klog.V(4).Infof("parsed instance metadata: %v", raw)
	metadataItems := []gceMetadataItems{}
	for k, v := range raw {
		metadataItems = append(metadataItems, gceMetadataItems{
			Key:   k,
			Value: v,
		})
	}
	ret := gceMetadata{Items: metadataItems}
	return &ret
}

func (g *GCERunner) DeleteGCEInstance(host string) {
	klog.Infof("Deleting instance %q", host)
	_, err := runGCPCommandWithZone("compute", "instances", "delete", host)
	if err != nil {
		klog.Errorf("Error deleting instance %q: %v", host, err)
	}
}

func (g *GCERunner) parseInstanceMetadata(str string) map[string]string {
	metadata := make(map[string]string)
	ss := strings.Split(str, ",")
	for _, s := range ss {
		kv := strings.Split(s, "=")
		if len(kv) == 2 {
			metadata[kv[0]] = kv[1]
			continue
		}
		kp := strings.Split(s, "<")
		if len(kp) != 2 {
			klog.Fatalf("Invalid instance metadata: %q", s)
			continue
		}
		metaPath := kp[1]
		if g.cfg.ImageConfigDir != "" {
			metaPath = filepath.Join(g.cfg.ImageConfigDir, metaPath)
		}
		v, err := os.ReadFile(metaPath)
		if err != nil {
			klog.Fatalf("Failed to read metadata file %q: %v", metaPath, err)
			continue
		}
		metadata[kp[0]] = ignitionInjectGCEPublicKey(string(v))
	}
	for k, v := range nodeEnvs {
		metadata[k] = v
	}
	return metadata
}

// ignitionInjectGCEPublicKey tries to inject the GCE SSH public key into the
// provided ignition file path.
//
// This will only being done if the job has the
// IGNITION_INJECT_GCE_SSH_PUBLIC_KEY_FILE environment variable set, while it
// tried to replace the GCE_SSH_PUBLIC_KEY_FILE_CONTENT placeholder.
func ignitionInjectGCEPublicKey(content string) string {
	if os.Getenv("IGNITION_INJECT_GCE_SSH_PUBLIC_KEY_FILE") == "" {
		return content
	}

	klog.Infof("Injecting SSH public key into ignition")

	const publicKeyEnv = "GCE_SSH_PUBLIC_KEY_FILE"
	sshPublicKeyFile := os.Getenv(publicKeyEnv)
	if sshPublicKeyFile == "" {
		klog.Errorf("Environment variable %s is not set", publicKeyEnv)
		os.Exit(1)
	}

	sshPublicKey, err := os.ReadFile(sshPublicKeyFile)
	if err != nil {
		klog.ErrorS(err, "unable to read SSH public key file")
		os.Exit(1)
	}

	const sshPublicKeyFileContentMarker = "GCE_SSH_PUBLIC_KEY_FILE_CONTENT"
	key := base64.StdEncoding.EncodeToString(sshPublicKey)
	base64Marker := base64.StdEncoding.EncodeToString([]byte(sshPublicKeyFileContentMarker))
	replacer := strings.NewReplacer(
		sshPublicKeyFileContentMarker, key,
		base64Marker, key,
	)
	return replacer.Replace(content)
}

// Provision a gce instance using image and run the tests in archive against the instance.
// Delete the instance afterward.
func (g *GCERunner) testGCEImage(suite remote.TestSuite, archivePath string, imageConfig *internalGCEImage, junitFileName string) *remote.TestResult {
	ginkgoFlagsStr := g.cfg.GinkgoFlags

	host, err := g.createGCEInstance(imageConfig)
	if g.cfg.DeleteInstances {
		defer g.DeleteGCEInstance(host)
	}
	if err != nil {
		return &remote.TestResult{
			Err: fmt.Errorf("unable to create gce instance with running docker daemon for image %s.  %v", imageConfig.image, err),
		}
	}

	// Only delete the files if we are keeping the instance and want it cleaned up.
	// If we are going to delete the instance, don't bother with cleaning up the files
	deleteFiles := !g.cfg.DeleteInstances && g.cfg.Cleanup

	if err = g.registerGceHostIP(host); err != nil {
		return &remote.TestResult{
			Err:    err,
			Host:   host,
			ExitOK: false,
		}
	}

	output, exitOk, err := remote.RunRemote(remote.RunRemoteConfig{
		Suite:          suite,
		Archive:        archivePath,
		Host:           host,
		Cleanup:        deleteFiles,
		ImageDesc:      imageConfig.imageDesc,
		JunitFileName:  junitFileName,
		TestArgs:       g.cfg.TestArgs,
		GinkgoArgs:     ginkgoFlagsStr,
		SystemSpecName: g.cfg.SystemSpecName,
		ExtraEnvs:      g.cfg.ExtraEnvs,
		RuntimeConfig:  g.cfg.RuntimeConfig,
	})
	result := remote.TestResult{
		Output: output,
		Err:    err,
		Host:   host,
		ExitOK: exitOk,
	}

	// This is a temporary solution to collect serial node serial log. Only port 1 contains useful information.
	// TODO(random-liu): Extract out and unify log collection logic with cluste e2e.
	contents, err := g.getSerialOutput(host)
	if err != nil {
		klog.Errorf("Failed to get serial Output from node %q : %v", host, err)
	}
	logFilename := "serial-1.log"
	err = remote.WriteLog(host, logFilename, contents)
	if err != nil {
		klog.Errorf("Failed to write serial Output from node %q to %q: %v", host, logFilename, err)
	}
	return &result
}

// Provision a gce instance using image
func (g *GCERunner) createGCEInstance(imageConfig *internalGCEImage) (string, error) {
	data, err := runGCPCommand("compute", "project-info", "describe", "--format=json", "--project="+*project)
	if err != nil {
		return "", fmt.Errorf("failed to get project info for %q: %w", *project, err)
	}

	var p projectInfo
	err = json.Unmarshal(data, &p)
	if err != nil {
		return "", fmt.Errorf("failed parse project info %q: %w", *project, err)
	}
	// Use default service account
	serviceAccount := p.DefaultServiceAccount
	klog.V(1).Infof("Creating instance %+v  with service account %q", *imageConfig, serviceAccount)
	name := g.imageToInstanceName(imageConfig)

	diskArgs := []string{
		"image-project=" + imageConfig.project,
		"image=" + imageConfig.image,
		"type=pd-standard",
		"auto-delete=yes",
		"boot=yes",
		"size=20GB",
	}

	createArgs := []string{"compute", "instances", "create"}
	createArgs = append(createArgs, name)
	createArgs = append(createArgs, "--machine-type="+g.machineType(imageConfig.machine))
	createArgs = append(createArgs, "--create-disk="+strings.Join(diskArgs, ","))
	createArgs = append(createArgs, "--service-account="+serviceAccount)
	if *preemptibleInstances {
		createArgs = append(createArgs, "--preemptible")
	}
	if len(imageConfig.resources.Accelerators) > 0 {
		createArgs = append(createArgs, "--maintenance-policy=TERMINATE")
		createArgs = append(createArgs, "--restart-on-failure")
		for _, accelerator := range imageConfig.resources.Accelerators {
			createArgs = append(createArgs,
				fmt.Sprintf("--accelerator=count=%d,type=%s", accelerator.Count, accelerator.Type))
		}
	}
	if imageConfig.metadata != nil {
		var itemArgs []string
		var itemFileArgs []string
		for _, item := range imageConfig.metadata.Items {
			if strings.HasPrefix(item.Key, "user-") || strings.HasPrefix(item.Key, "startup-") ||
				strings.HasPrefix(item.Key, "containerd-") || strings.HasPrefix(item.Key, "cni-") ||
				strings.ContainsAny(item.Value, ",:") {
				dataFile, err := os.CreateTemp("", "metadata")
				if err != nil {
					return "", fmt.Errorf("unable to create temp file %v", err)
				}
				defer os.Remove(dataFile.Name()) // clean up
				if err = os.WriteFile(dataFile.Name(), []byte(item.Value), 0666); err != nil {
					return "", fmt.Errorf("could not write contents of metadata item into file %v", err)
				}
				itemFileArgs = append(itemFileArgs, item.Key+"="+dataFile.Name())
			} else {
				itemArgs = append(itemArgs, item.Key+"="+item.Value)
			}
		}
		if len(itemArgs) > 0 {
			createArgs = append(createArgs, "--metadata="+strings.Join(itemArgs, ","))
		}
		if len(itemFileArgs) > 0 {
			createArgs = append(createArgs, "--metadata-from-file="+strings.Join(itemFileArgs, ","))
		}
	}

	if _, err := getGCEInstance(name); err != nil {
		fmt.Printf("Running gcloud with parameters : %#v\n", createArgs)
		_, err := runGCPCommandWithZone(createArgs...)
		if err != nil {
			fmt.Println(err)
			return "", fmt.Errorf("failed to create instance in project %q: %w", *project, err)
		}
	}

	instanceRunning := false
	var instance *gceInstance
	for i := 0; i < 30 && !instanceRunning; i++ {
		if i > 0 {
			time.Sleep(time.Second * 20)
		}

		instance, err := getGCEInstance(name)
		if err != nil {
			continue
		}
		if strings.ToUpper(instance.Status) != "RUNNING" {
			_ = fmt.Errorf("instance %s not in state RUNNING, was %s", name, instance.Status)
			continue
		}
		externalIP := g.getExternalIP(instance)
		if len(externalIP) > 0 {
			remote.AddHostnameIP(name, externalIP)
		}

		var output string
		output, err = remote.SSH(name, "sh", "-c",
			"'systemctl list-units  --type=service  --state=running | grep -e containerd -e crio'")
		if err != nil {
			_ = fmt.Errorf("instance %s not running containerd/crio daemon - Command failed: %s", name, output)
			continue
		}
		if !strings.Contains(output, "containerd.service") &&
			!strings.Contains(output, "crio.service") {
			_ = fmt.Errorf("instance %s not running containerd/crio daemon: %s", name, output)
			continue
		}
		instanceRunning = true
	}
	// If instance didn't reach running state in time, return with error now.
	if err != nil {
		return name, err
	}
	// Instance reached running state in time, make sure that cloud-init is complete
	if g.isCloudInitUsed(imageConfig.metadata) {
		cloudInitFinished := false
		for i := 0; i < 60 && !cloudInitFinished; i++ {
			if i > 0 {
				time.Sleep(time.Second * 20)
			}
			var finished string
			finished, err = remote.SSH(name, "ls", "/var/lib/cloud/instance/boot-finished")
			if err != nil {
				err = fmt.Errorf("instance %s has not finished cloud-init script: %s", name, finished)
				continue
			}
			cloudInitFinished = true
		}
	}

	// apply additional kernel arguments to the instance
	if len(imageConfig.kernelArguments) > 0 {
		klog.Info("Update kernel arguments")
		if err := g.updateKernelArguments(instance, imageConfig.image, imageConfig.kernelArguments); err != nil {
			return name, err
		}
	}

	return name, err
}

func (g *GCERunner) isCloudInitUsed(metadata *gceMetadata) bool {
	if metadata == nil {
		return false
	}
	for _, item := range metadata.Items {
		if item.Key == "user-data" && item.Value != "" && strings.HasPrefix(item.Value, "#cloud-config") {
			return true
		}
	}
	return false
}

func (g *GCERunner) imageToInstanceName(imageConfig *internalGCEImage) string {
	if imageConfig.machine == "" {
		return g.cfg.InstanceNamePrefix + "-" + imageConfig.image
	}
	// For benchmark test, node name has the format 'machine-image-uuid' to run
	// different machine types with the same image in parallel
	name := imageConfig.machine + "-" + imageConfig.image + "-" + uuid.New().String()[:8]
	// Sometimes the image is too long, we need instance names to have a max length of 63
	if len(name) > 63 {
		return name[:63]
	}
	return name
}

func (g *GCERunner) registerGceHostIP(host string) error {
	instance, err := getGCEInstance(host)
	if err != nil {
		return err
	}
	if strings.ToUpper(instance.Status) != "RUNNING" {
		return fmt.Errorf("instance %s not in state RUNNING, was %s", host, instance.Status)
	}
	externalIP := g.getExternalIP(instance)
	if len(externalIP) > 0 {
		remote.AddHostnameIP(host, externalIP)
	}
	return nil
}

func (g *GCERunner) getExternalIP(instance *gceInstance) string {
	for i := range instance.NetworkInterfaces {
		ni := instance.NetworkInterfaces[i]
		for j := range ni.AccessConfigs {
			ac := ni.AccessConfigs[j]
			if len(ac.NatIP) > 0 {
				return ac.NatIP
			}
		}
	}
	return ""
}

func (g *GCERunner) updateKernelArguments(instance *gceInstance, image string, kernelArgs []string) error {
	kernelArgsString := strings.Join(kernelArgs, " ")

	var cmd []string
	if strings.Contains(image, "cos") {
		cmd = []string{
			"dir=$(mktemp -d)",
			"mount /dev/sda12 ${dir}",
			fmt.Sprintf("sed -i -e \"s|cros_efi|cros_efi %s|g\" ${dir}/efi/boot/grub.cfg", kernelArgsString),
			"umount ${dir}",
			"rmdir ${dir}",
		}
	}

	if strings.Contains(image, "ubuntu") {
		cmd = []string{
			fmt.Sprintf("echo \"GRUB_CMDLINE_LINUX_DEFAULT=%s ${GRUB_CMDLINE_LINUX_DEFAULT}\" > /etc/default/grub.d/99-additional-arguments.cfg", kernelArgsString),
			"/usr/sbin/update-grub",
		}
	}

	if len(cmd) == 0 {
		klog.Warningf("The image %s does not support adding an additional kernel arguments", image)
		return nil
	}

	out, err := remote.SSH(instance.Name, "sh", "-c", fmt.Sprintf("'%s'", strings.Join(cmd, "&&")))
	if err != nil {
		klog.Errorf("failed to run command %s: out: %s, Err: %v", cmd, out, err)
		return err
	}

	if err := g.rebootInstance(instance); err != nil {
		return err
	}

	return nil
}

func (g *GCERunner) machineType(machine string) string {
	var ret string
	if machine == "" && *instanceType != "" {
		ret = *instanceType
	} else if machine != "" {
		ret = machine
	} else {
		ret = defaultGCEMachine
	}
	return ret
}

func (g *GCERunner) rebootInstance(instance *gceInstance) error {
	// wait until the instance will not response to SSH
	klog.Info("Reboot the node and wait for instance not to be available via SSH")
	if waitErr := wait.PollImmediate(5*time.Second, 5*time.Minute, func() (bool, error) {
		if _, err := remote.SSH(instance.Name, "reboot"); err != nil {
			return true, nil
		}

		return false, nil
	}); waitErr != nil {
		return fmt.Errorf("the instance %s still response to SSH: %v", instance.Name, waitErr)
	}

	// wait until the instance will response again to SSH
	klog.Info("Wait for instance to be available via SSH")
	if waitErr := wait.PollImmediate(30*time.Second, 5*time.Minute, func() (bool, error) {
		if _, err := remote.SSH(instance.Name, "sh", "-c", "date"); err != nil {
			return false, nil
		}
		return true, nil
	}); waitErr != nil {
		return fmt.Errorf("the instance %s does not response to SSH: %v", instance.Name, waitErr)
	}

	return nil
}
