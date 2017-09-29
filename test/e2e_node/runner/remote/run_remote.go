/*
Copyright 2016 The Kubernetes Authors.

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

// To run the node e2e tests remotely against one or more hosts on gce:
// $ go run run_remote.go --logtostderr --v 2 --ssh-env gce --hosts <comma separated hosts>
// To run the node e2e tests remotely against one or more images on gce and provision them:
// $ go run run_remote.go --logtostderr --v 2 --project <project> --zone <zone> --ssh-env gce --images <comma separated images>
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/test/e2e_node/remote"

	"github.com/ghodss/yaml"
	"github.com/golang/glog"
	"github.com/pborman/uuid"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	compute "google.golang.org/api/compute/v0.beta"
)

var testArgs = flag.String("test_args", "", "Space-separated list of arguments to pass to Ginkgo test runner.")
var testSuite = flag.String("test-suite", "default", "Test suite the runner initializes with. Currently support default|conformance")
var instanceNamePrefix = flag.String("instance-name-prefix", "", "prefix for instance names")
var zone = flag.String("zone", "", "gce zone the hosts live in")
var project = flag.String("project", "", "gce project the hosts live in")
var imageConfigFile = flag.String("image-config-file", "", "yaml file describing images to run")
var imageProject = flag.String("image-project", "", "gce project the hosts live in")
var images = flag.String("images", "", "images to test")
var hosts = flag.String("hosts", "", "hosts to test")
var cleanup = flag.Bool("cleanup", true, "If true remove files from remote hosts and delete temporary instances")
var deleteInstances = flag.Bool("delete-instances", true, "If true, delete any instances created")
var buildOnly = flag.Bool("build-only", false, "If true, build e2e_node_test.tar.gz and exit.")
var instanceMetadata = flag.String("instance-metadata", "", "key/value metadata for instances separated by '=' or '<', 'k=v' means the key is 'k' and the value is 'v'; 'k<p' means the key is 'k' and the value is extracted from the local path 'p', e.g. k1=v1,k2<p2")
var gubernator = flag.Bool("gubernator", false, "If true, output Gubernator link to view logs")
var ginkgoFlags = flag.String("ginkgo-flags", "", "Passed to ginkgo to specify additional flags such as --skip=.")
var systemSpecName = flag.String("system-spec-name", "", "The name of the system spec used for validating the image in the node conformance test. The specs are at test/e2e_node/system/specs/. If unspecified, the default built-in spec (system.DefaultSpec) will be used.")

const (
	defaultMachine                = "n1-standard-1"
	acceleratorTypeResourceFormat = "https://www.googleapis.com/compute/beta/projects/%s/zones/%s/acceleratorTypes/%s"
)

var (
	computeService *compute.Service
	arc            Archive
	suite          remote.TestSuite
)

type Archive struct {
	sync.Once
	path string
	err  error
}

type TestResult struct {
	output string
	err    error
	host   string
	exitOk bool
}

// ImageConfig specifies what images should be run and how for these tests.
// It can be created via the `--images` and `--image-project` flags, or by
// specifying the `--image-config-file` flag, pointing to a json or yaml file
// of the form:
//
//     images:
//       short-name:
//         image: gce-image-name
//         project: gce-image-project
//         machine: for benchmark only, the machine type (GCE instance) to run test
//         tests: for benchmark only, a list of ginkgo focus strings to match tests

// TODO(coufon): replace 'image' with 'node' in configurations
// and we plan to support testing custom machines other than GCE by specifying host
type ImageConfig struct {
	Images map[string]GCEImage `json:"images"`
}

type Accelerator struct {
	Type  string `json:"type,omitempty"`
	Count int64  `json:"count, omitempty"`
}

type Resources struct {
	Accelerators []Accelerator `json:"accelerators,omitempty"`
}

type GCEImage struct {
	Image      string `json:"image, omitempty"`
	ImageDesc  string `json:"image_description, omitempty"`
	Project    string `json:"project"`
	Metadata   string `json:"metadata"`
	ImageRegex string `json:"image_regex, omitempty"`
	// Defaults to using only the latest image. Acceptible values are [0, # of images that match the regex).
	// If the number of existing previous images is lesser than what is desired, the test will use that is available.
	PreviousImages int `json:"previous_images, omitempty"`

	Machine   string    `json:"machine, omitempty"`
	Resources Resources `json:"resources, omitempty"`
	// This test is for benchmark (no limit verification, more result log, node name has format 'machine-image-uuid') if 'Tests' is non-empty.
	Tests []string `json:"tests, omitempty"`
}

type internalImageConfig struct {
	images map[string]internalGCEImage
}

type internalGCEImage struct {
	image string
	// imageDesc is the description of the image. If empty, the value in the
	// 'image' will be used.
	imageDesc string
	project   string
	resources Resources
	metadata  *compute.Metadata
	machine   string
	tests     []string
}

func main() {
	flag.Parse()
	switch *testSuite {
	case "conformance":
		suite = remote.InitConformanceRemote()
	// TODO: Add subcommand for node soaking, node conformance, cri validation.
	case "default":
		// Use node e2e suite by default if no subcommand is specified.
		suite = remote.InitNodeE2ERemote()
	default:
		glog.Fatalf("--test-suite must be one of default or conformance")
	}

	rand.Seed(time.Now().UTC().UnixNano())
	if *buildOnly {
		// Build the archive and exit
		remote.CreateTestArchive(suite, *systemSpecName)
		return
	}

	if *hosts == "" && *imageConfigFile == "" && *images == "" {
		glog.Fatalf("Must specify one of --image-config-file, --hosts, --images.")
	}
	var err error
	computeService, err = getComputeClient()
	if err != nil {
		glog.Fatalf("Unable to create gcloud compute service using defaults.  Make sure you are authenticated. %v", err)
	}

	gceImages := &internalImageConfig{
		images: make(map[string]internalGCEImage),
	}
	if *imageConfigFile != "" {
		// parse images
		imageConfigData, err := ioutil.ReadFile(*imageConfigFile)
		if err != nil {
			glog.Fatalf("Could not read image config file provided: %v", err)
		}
		externalImageConfig := ImageConfig{Images: make(map[string]GCEImage)}
		err = yaml.Unmarshal(imageConfigData, &externalImageConfig)
		if err != nil {
			glog.Fatalf("Could not parse image config file: %v", err)
		}
		for shortName, imageConfig := range externalImageConfig.Images {
			var images []string
			isRegex, name := false, shortName
			if imageConfig.ImageRegex != "" && imageConfig.Image == "" {
				isRegex = true
				images, err = getGCEImages(imageConfig.ImageRegex, imageConfig.Project, imageConfig.PreviousImages)
				if err != nil {
					glog.Fatalf("Could not retrieve list of images based on image prefix %q: %v", imageConfig.ImageRegex, err)
				}
			} else {
				images = []string{imageConfig.Image}
			}
			for _, image := range images {
				metadata := imageConfig.Metadata
				if len(strings.TrimSpace(*instanceMetadata)) > 0 {
					metadata += "," + *instanceMetadata
				}
				gceImage := internalGCEImage{
					image:     image,
					imageDesc: imageConfig.ImageDesc,
					project:   imageConfig.Project,
					metadata:  getImageMetadata(metadata),
					machine:   imageConfig.Machine,
					tests:     imageConfig.Tests,
					resources: imageConfig.Resources,
				}
				if gceImage.imageDesc == "" {
					gceImage.imageDesc = gceImage.image
				}
				if isRegex && len(images) > 1 {
					// Use image name when shortName is not unique.
					name = image
				}
				gceImages.images[name] = gceImage
			}
		}
	}

	// Allow users to specify additional images via cli flags for local testing
	// convenience; merge in with config file
	if *images != "" {
		if *imageProject == "" {
			glog.Fatal("Must specify --image-project if you specify --images")
		}
		cliImages := strings.Split(*images, ",")
		for _, img := range cliImages {
			gceImage := internalGCEImage{
				image:    img,
				project:  *imageProject,
				metadata: getImageMetadata(*instanceMetadata),
			}
			gceImages.images[img] = gceImage
		}
	}

	if len(gceImages.images) != 0 && *zone == "" {
		glog.Fatal("Must specify --zone flag")
	}
	for shortName, image := range gceImages.images {
		if image.project == "" {
			glog.Fatalf("Invalid config for %v; must specify a project", shortName)
		}
	}
	if len(gceImages.images) != 0 {
		if *project == "" {
			glog.Fatal("Must specify --project flag to launch images into")
		}
	}
	if *instanceNamePrefix == "" {
		*instanceNamePrefix = "tmp-node-e2e-" + uuid.NewUUID().String()[:8]
	}

	// Setup coloring
	stat, _ := os.Stdout.Stat()
	useColor := (stat.Mode() & os.ModeCharDevice) != 0
	blue := ""
	noColour := ""
	if useColor {
		blue = "\033[0;34m"
		noColour = "\033[0m"
	}

	go arc.getArchive()
	defer arc.deleteArchive()

	results := make(chan *TestResult)
	running := 0
	for shortName := range gceImages.images {
		imageConfig := gceImages.images[shortName]
		fmt.Printf("Initializing e2e tests using image %s.\n", shortName)
		running++
		go func(image *internalGCEImage, junitFilePrefix string) {
			results <- testImage(image, junitFilePrefix)
		}(&imageConfig, shortName)
	}
	if *hosts != "" {
		for _, host := range strings.Split(*hosts, ",") {
			fmt.Printf("Initializing e2e tests using host %s.\n", host)
			running++
			go func(host string, junitFilePrefix string) {
				results <- testHost(host, *cleanup, "", junitFilePrefix, *ginkgoFlags)
			}(host, host)
		}
	}

	// Wait for all tests to complete and emit the results
	errCount := 0
	exitOk := true
	for i := 0; i < running; i++ {
		tr := <-results
		host := tr.host
		fmt.Println() // Print an empty line
		fmt.Printf("%s>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>%s\n", blue, noColour)
		fmt.Printf("%s>                              START TEST                                >%s\n", blue, noColour)
		fmt.Printf("%s>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>%s\n", blue, noColour)
		fmt.Printf("Start Test Suite on Host %s\n", host)
		fmt.Printf("%s\n", tr.output)
		if tr.err != nil {
			errCount++
			fmt.Printf("Failure Finished Test Suite on Host %s\n%v\n", host, tr.err)
		} else {
			fmt.Printf("Success Finished Test Suite on Host %s\n", host)
		}
		exitOk = exitOk && tr.exitOk
		fmt.Printf("%s<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<%s\n", blue, noColour)
		fmt.Printf("%s<                              FINISH TEST                               <%s\n", blue, noColour)
		fmt.Printf("%s<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<%s\n", blue, noColour)
		fmt.Println() // Print an empty line
	}
	// Set the exit code if there were failures
	if !exitOk {
		fmt.Printf("Failure: %d errors encountered.\n", errCount)
		callGubernator(*gubernator)
		os.Exit(1)
	}
	callGubernator(*gubernator)
}

func callGubernator(gubernator bool) {
	if gubernator {
		fmt.Println("Running gubernator.sh")
		output, err := exec.Command("./test/e2e_node/gubernator.sh", "y").Output()

		if err != nil {
			fmt.Println("gubernator.sh Failed")
			fmt.Println(err)
			return
		}
		fmt.Printf("%s", output)
	}
	return
}

func (a *Archive) getArchive() (string, error) {
	a.Do(func() { a.path, a.err = remote.CreateTestArchive(suite, *systemSpecName) })
	return a.path, a.err
}

func (a *Archive) deleteArchive() {
	path, err := a.getArchive()
	if err != nil {
		return
	}
	os.Remove(path)
}

func getImageMetadata(input string) *compute.Metadata {
	if input == "" {
		return nil
	}
	glog.V(3).Infof("parsing instance metadata: %q", input)
	raw := parseInstanceMetadata(input)
	glog.V(4).Infof("parsed instance metadata: %v", raw)
	metadataItems := []*compute.MetadataItems{}
	for k, v := range raw {
		val := v
		metadataItems = append(metadataItems, &compute.MetadataItems{
			Key:   k,
			Value: &val,
		})
	}
	ret := compute.Metadata{Items: metadataItems}
	return &ret
}

// Run tests in archive against host
func testHost(host string, deleteFiles bool, imageDesc, junitFilePrefix, ginkgoFlagsStr string) *TestResult {
	instance, err := computeService.Instances.Get(*project, *zone, host).Do()
	if err != nil {
		return &TestResult{
			err:    err,
			host:   host,
			exitOk: false,
		}
	}
	if strings.ToUpper(instance.Status) != "RUNNING" {
		err = fmt.Errorf("instance %s not in state RUNNING, was %s.", host, instance.Status)
		return &TestResult{
			err:    err,
			host:   host,
			exitOk: false,
		}
	}
	externalIp := getExternalIp(instance)
	if len(externalIp) > 0 {
		remote.AddHostnameIp(host, externalIp)
	}

	path, err := arc.getArchive()
	if err != nil {
		// Don't log fatal because we need to do any needed cleanup contained in "defer" statements
		return &TestResult{
			err: fmt.Errorf("unable to create test archive: %v.", err),
		}
	}

	output, exitOk, err := remote.RunRemote(suite, path, host, deleteFiles, imageDesc, junitFilePrefix, *testArgs, ginkgoFlagsStr, *systemSpecName)
	return &TestResult{
		output: output,
		err:    err,
		host:   host,
		exitOk: exitOk,
	}
}

type imageObj struct {
	creationTime time.Time
	name         string
}

func (io imageObj) string() string {
	return fmt.Sprintf("%q created %q", io.name, io.creationTime.String())
}

type byCreationTime []imageObj

func (a byCreationTime) Len() int           { return len(a) }
func (a byCreationTime) Less(i, j int) bool { return a[i].creationTime.After(a[j].creationTime) }
func (a byCreationTime) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// Returns a list of image names based on regex and number of previous images requested.
func getGCEImages(imageRegex, project string, previousImages int) ([]string, error) {
	ilc, err := computeService.Images.List(project).Do()
	if err != nil {
		return nil, fmt.Errorf("Failed to list images in project %q: %v", project, err)
	}
	imageObjs := []imageObj{}
	imageRe := regexp.MustCompile(imageRegex)
	for _, instance := range ilc.Items {
		if imageRe.MatchString(instance.Name) {
			creationTime, err := time.Parse(time.RFC3339, instance.CreationTimestamp)
			if err != nil {
				return nil, fmt.Errorf("Failed to parse instance creation timestamp %q: %v", instance.CreationTimestamp, err)
			}
			io := imageObj{
				creationTime: creationTime,
				name:         instance.Name,
			}
			glog.V(4).Infof("Found image %q based on regex %q in project %q", io.string(), imageRegex, project)
			imageObjs = append(imageObjs, io)
		}
	}
	sort.Sort(byCreationTime(imageObjs))
	images := []string{}
	for _, imageObj := range imageObjs {
		images = append(images, imageObj.name)
		previousImages--
		if previousImages < 0 {
			break
		}
	}
	return images, nil
}

// Provision a gce instance using image and run the tests in archive against the instance.
// Delete the instance afterward.
func testImage(imageConfig *internalGCEImage, junitFilePrefix string) *TestResult {
	ginkgoFlagsStr := *ginkgoFlags
	// Check whether the test is for benchmark.
	if len(imageConfig.tests) > 0 {
		// Benchmark needs machine type non-empty.
		if imageConfig.machine == "" {
			imageConfig.machine = defaultMachine
		}
		// Use the Ginkgo focus in benchmark config.
		ginkgoFlagsStr += (" " + testsToGinkgoFocus(imageConfig.tests))
	}

	host, err := createInstance(imageConfig)
	if *deleteInstances {
		defer deleteInstance(host)
	}
	if err != nil {
		return &TestResult{
			err: fmt.Errorf("unable to create gce instance with running docker daemon for image %s.  %v", imageConfig.image, err),
		}
	}

	// Only delete the files if we are keeping the instance and want it cleaned up.
	// If we are going to delete the instance, don't bother with cleaning up the files
	deleteFiles := !*deleteInstances && *cleanup

	result := testHost(host, deleteFiles, imageConfig.imageDesc, junitFilePrefix, ginkgoFlagsStr)
	// This is a temporary solution to collect serial node serial log. Only port 1 contains useful information.
	// TODO(random-liu): Extract out and unify log collection logic with cluste e2e.
	serialPortOutput, err := computeService.Instances.GetSerialPortOutput(*project, *zone, host).Port(1).Do()
	if err != nil {
		glog.Errorf("Failed to collect serial output from node %q: %v", host, err)
	} else {
		logFilename := "serial-1.log"
		err := remote.WriteLog(host, logFilename, serialPortOutput.Contents)
		if err != nil {
			glog.Errorf("Failed to write serial output from node %q to %q: %v", host, logFilename, err)
		}
	}
	return result
}

// Provision a gce instance using image
func createInstance(imageConfig *internalGCEImage) (string, error) {
	glog.V(1).Infof("Creating instance %+v", *imageConfig)
	name := imageToInstanceName(imageConfig)
	i := &compute.Instance{
		Name:        name,
		MachineType: machineType(imageConfig.machine),
		NetworkInterfaces: []*compute.NetworkInterface{
			{
				AccessConfigs: []*compute.AccessConfig{
					{
						Type: "ONE_TO_ONE_NAT",
						Name: "External NAT",
					},
				}},
		},
		Disks: []*compute.AttachedDisk{
			{
				AutoDelete: true,
				Boot:       true,
				Type:       "PERSISTENT",
				InitializeParams: &compute.AttachedDiskInitializeParams{
					SourceImage: sourceImage(imageConfig.image, imageConfig.project),
					DiskSizeGb:  20,
				},
			},
		},
	}

	for _, accelerator := range imageConfig.resources.Accelerators {
		if i.GuestAccelerators == nil {
			autoRestart := true
			i.GuestAccelerators = []*compute.AcceleratorConfig{}
			i.Scheduling = &compute.Scheduling{
				OnHostMaintenance: "TERMINATE",
				AutomaticRestart:  &autoRestart,
			}
		}
		aType := fmt.Sprintf(acceleratorTypeResourceFormat, *project, *zone, accelerator.Type)
		ac := &compute.AcceleratorConfig{
			AcceleratorCount: accelerator.Count,
			AcceleratorType:  aType,
		}
		i.GuestAccelerators = append(i.GuestAccelerators, ac)
	}

	var err error
	i.Metadata = imageConfig.metadata
	if _, err := computeService.Instances.Get(*project, *zone, i.Name).Do(); err != nil {
		op, err := computeService.Instances.Insert(*project, *zone, i).Do()
		if err != nil {
			ret := fmt.Sprintf("could not create instance %s: API error: %v", name, err)
			if op != nil {
				ret = fmt.Sprintf("%s: %v", ret, op.Error)
			}
			return "", fmt.Errorf(ret)
		} else if op.Error != nil {
			return "", fmt.Errorf("could not create instance %s: %+v", name, op.Error)
		}
	}

	instanceRunning := false
	for i := 0; i < 30 && !instanceRunning; i++ {
		if i > 0 {
			time.Sleep(time.Second * 20)
		}
		var instance *compute.Instance
		instance, err = computeService.Instances.Get(*project, *zone, name).Do()
		if err != nil {
			continue
		}
		if strings.ToUpper(instance.Status) != "RUNNING" {
			err = fmt.Errorf("instance %s not in state RUNNING, was %s.", name, instance.Status)
			continue
		}
		externalIp := getExternalIp(instance)
		if len(externalIp) > 0 {
			remote.AddHostnameIp(name, externalIp)
		}
		var output string
		output, err = remote.SSH(name, "docker", "version")
		if err != nil {
			err = fmt.Errorf("instance %s not running docker daemon - Command failed: %s", name, output)
			continue
		}
		if !strings.Contains(output, "Server") {
			err = fmt.Errorf("instance %s not running docker daemon - Server not found: %s", name, output)
			continue
		}
		instanceRunning = true
	}
	// If instance didn't reach running state in time, return with error now.
	if err != nil {
		return name, err
	}
	// Instance reached running state in time, make sure that cloud-init is complete
	if isCloudInitUsed(imageConfig.metadata) {
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
	return name, err
}

func isCloudInitUsed(metadata *compute.Metadata) bool {
	if metadata == nil {
		return false
	}
	for _, item := range metadata.Items {
		if item.Key == "user-data" && item.Value != nil && strings.HasPrefix(*item.Value, "#cloud-config") {
			return true
		}
	}
	return false
}

func getExternalIp(instance *compute.Instance) string {
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

func getComputeClient() (*compute.Service, error) {
	const retries = 10
	const backoff = time.Second * 6

	// Setup the gce client for provisioning instances
	// Getting credentials on gce jenkins is flaky, so try a couple times
	var err error
	var cs *compute.Service
	for i := 0; i < retries; i++ {
		if i > 0 {
			time.Sleep(backoff)
		}

		var client *http.Client
		client, err = google.DefaultClient(oauth2.NoContext, compute.ComputeScope)
		if err != nil {
			continue
		}

		cs, err = compute.New(client)
		if err != nil {
			continue
		}
		return cs, nil
	}
	return nil, err
}

func deleteInstance(host string) {
	glog.Infof("Deleting instance %q", host)
	_, err := computeService.Instances.Delete(*project, *zone, host).Do()
	if err != nil {
		glog.Errorf("Error deleting instance %q: %v", host, err)
	}
}

func parseInstanceMetadata(str string) map[string]string {
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
			glog.Fatalf("Invalid instance metadata: %q", s)
			continue
		}
		v, err := ioutil.ReadFile(kp[1])
		if err != nil {
			glog.Fatalf("Failed to read metadata file %q: %v", kp[1], err)
			continue
		}
		metadata[kp[0]] = string(v)
	}
	return metadata
}

func imageToInstanceName(imageConfig *internalGCEImage) string {
	if imageConfig.machine == "" {
		return *instanceNamePrefix + "-" + imageConfig.image
	}
	// For benchmark test, node name has the format 'machine-image-uuid' to run
	// different machine types with the same image in parallel
	return imageConfig.machine + "-" + imageConfig.image + "-" + uuid.NewUUID().String()[:8]
}

func sourceImage(image, imageProject string) string {
	return fmt.Sprintf("projects/%s/global/images/%s", imageProject, image)
}

func machineType(machine string) string {
	if machine == "" {
		machine = defaultMachine
	}
	return fmt.Sprintf("zones/%s/machineTypes/%s", *zone, machine)
}

// testsToGinkgoFocus converts the test string list to Ginkgo focus
func testsToGinkgoFocus(tests []string) string {
	focus := "--focus=\""
	for i, test := range tests {
		if i == 0 {
			focus += test
		} else {
			focus += ("|" + test)
		}
	}
	return focus + "\""
}
