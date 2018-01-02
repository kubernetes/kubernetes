/*
Copyright 2015 The Camlistore Authors

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

// Package cloudlaunch helps binaries run themselves on The Cloud, copying
// themselves to GCE.
package cloudlaunch // import "go4.org/cloud/cloudlaunch"

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"go4.org/cloud/google/gceutil"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	storageapi "google.golang.org/api/storage/v1"
	"google.golang.org/cloud"
	"google.golang.org/cloud/storage"
)

func readFile(v string) string {
	slurp, err := ioutil.ReadFile(v)
	if err != nil {
		log.Fatalf("Error reading %s: %v", v, err)
	}
	return strings.TrimSpace(string(slurp))
}

const baseConfig = `#cloud-config
coreos:
  update:
    group: stable
    reboot-strategy: off
  units:
    - name: $NAME.service
      command: start
      content: |
        [Unit]
        Description=$NAME service
        After=network.target
        
        [Service]
        Type=simple
        ExecStartPre=/bin/sh -c 'mkdir -p /opt/bin && /usr/bin/curl --silent -f -o /opt/bin/$NAME $URL?$(date +%s) && chmod +x /opt/bin/$NAME'
        ExecStart=/opt/bin/$NAME
        RestartSec=10
        Restart=always
        StartLimitInterval=0
        
        [Install]
        WantedBy=network-online.target
`

// RestartPolicy controls whether the binary automatically restarts.
type RestartPolicy int

const (
	RestartOnUpdates RestartPolicy = iota
	RestartNever
	// TODO: more graceful restarts; make systemd own listening on network sockets,
	// don't break connections.
)

type Config struct {
	// Name is the name of a service to run.
	// This is the name of the systemd service (without .service)
	// and the name of the GCE instance.
	Name string

	// RestartPolicy controls whether the binary automatically restarts
	// on updates. The zero value means automatic.
	RestartPolicy RestartPolicy

	// BinaryBucket and BinaryObject are the GCS bucket and object
	// within that bucket containing the Linux binary to download
	// on boot and occasionally run. This binary must be public
	// (at least for now).
	BinaryBucket string
	BinaryObject string // defaults to Name

	GCEProjectID string
	Zone         string // defaults to us-central1-f
	SSD          bool

	Scopes []string // any additional scopes

	MachineType  string
	InstanceName string
}

// cloudLaunch is a launch of a Config.
type cloudLaunch struct {
	*Config
	oauthClient    *http.Client
	computeService *compute.Service
}

func (c *Config) binaryURL() string {
	return "https://storage.googleapis.com/" + c.BinaryBucket + "/" + c.binaryObject()
}

func (c *Config) instName() string     { return c.Name } // for now
func (c *Config) zone() string         { return strDefault(c.Zone, "us-central1-f") }
func (c *Config) machineType() string  { return strDefault(c.MachineType, "g1-small") }
func (c *Config) binaryObject() string { return strDefault(c.BinaryObject, c.Name) }

func (c *Config) projectAPIURL() string {
	return "https://www.googleapis.com/compute/v1/projects/" + c.GCEProjectID
}
func (c *Config) machineTypeURL() string {
	return c.projectAPIURL() + "/zones/" + c.zone() + "/machineTypes/" + c.machineType()
}

func strDefault(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

var (
	doLaunch = flag.Bool("cloudlaunch", false, "Deploy or update this binary to the cloud. Must be on Linux, for now.")
)

func (c *Config) MaybeDeploy() {
	flag.Parse()
	if !*doLaunch {
		go c.restartLoop()
		return
	}
	defer os.Exit(1) // backup, in case we return without Fatal or os.Exit later

	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		log.Fatal("Can only use --cloudlaunch on linux/amd64, for now.")
	}

	if c.GCEProjectID == "" {
		log.Fatal("cloudconfig.GCEProjectID is empty")
	}
	filename := filepath.Join(os.Getenv("HOME"), "keys", c.GCEProjectID+".key.json")
	log.Printf("Using OAuth config from JSON service file: %s", filename)
	jwtConf, err := google.JWTConfigFromJSON([]byte(readFile(filename)), append([]string{
		storageapi.DevstorageFullControlScope,
		compute.ComputeScope,
		"https://www.googleapis.com/auth/cloud-platform",
	}, c.Scopes...)...)
	if err != nil {
		log.Fatalf("ConfigFromJSON: %v", err)
	}

	cl := &cloudLaunch{
		Config:      c,
		oauthClient: jwtConf.Client(oauth2.NoContext),
	}
	cl.computeService, _ = compute.New(cl.oauthClient)

	cl.uploadBinary()
	cl.createInstance()
	os.Exit(0)
}

func (c *Config) restartLoop() {
	if c.RestartPolicy == RestartNever {
		return
	}
	url := "https://storage.googleapis.com/" + c.BinaryBucket + "/" + c.binaryObject()
	var lastEtag string
	for {
		res, err := http.Head(url + "?" + fmt.Sprint(time.Now().Unix()))
		if err != nil {
			log.Printf("Warning: %v", err)
			time.Sleep(15 * time.Second)
			continue
		}
		etag := res.Header.Get("Etag")
		if etag == "" {
			log.Printf("Warning, no ETag in response: %v", res)
			time.Sleep(15 * time.Second)
			continue
		}
		if lastEtag != "" && etag != lastEtag {
			log.Printf("Binary updated; restarting.")
			// TODO: more graceful restart, letting systemd own the network connections.
			// Then we can finish up requests here.
			os.Exit(0)
		}
		lastEtag = etag
		time.Sleep(15 * time.Second)
	}
}

// uploadBinary uploads the currently-running Linux binary.
// It crashes if it fails.
func (cl *cloudLaunch) uploadBinary() {
	ctx := context.Background()
	if cl.BinaryBucket == "" {
		log.Fatal("cloudlaunch: Config.BinaryBucket is empty")
	}
	stoClient, err := storage.NewClient(ctx, cloud.WithBaseHTTP(cl.oauthClient))
	if err != nil {
		log.Fatal(err)
	}
	w := stoClient.Bucket(cl.BinaryBucket).Object(cl.binaryObject()).NewWriter(ctx)
	if err != nil {
		log.Fatal(err)
	}
	w.ACL = []storage.ACLRule{
		// If you don't give the owners access, the web UI seems to
		// have a bug and doesn't have access to see that it's public, so
		// won't render the "Shared Publicly" link. So we do that, even
		// though it's dumb and unnecessary otherwise:
		{
			Entity: storage.ACLEntity("project-owners-" + cl.GCEProjectID),
			Role:   storage.RoleOwner,
		},
		// Public, so our systemd unit can get it easily:
		{
			Entity: storage.AllUsers,
			Role:   storage.RoleReader,
		},
	}
	w.CacheControl = "no-cache"
	selfPath := getSelfPath()
	log.Printf("Uploading %q to %v", selfPath, cl.binaryURL())
	f, err := os.Open(selfPath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	n, err := io.Copy(w, f)
	if err != nil {
		log.Fatal(err)
	}
	if err := w.Close(); err != nil {
		log.Fatal(err)
	}
	log.Printf("Uploaded %d bytes", n)
}

func getSelfPath() string {
	if runtime.GOOS != "linux" {
		panic("TODO")
	}
	v, err := os.Readlink("/proc/self/exe")
	if err != nil {
		log.Fatal(err)
	}
	return v
}

func zoneInRegion(zone, regionURL string) bool {
	if zone == "" {
		panic("empty zone")
	}
	if regionURL == "" {
		panic("empty regionURL")
	}
	// zone is like "us-central1-f"
	// regionURL is like "https://www.googleapis.com/compute/v1/projects/camlistore-website/regions/us-central1"
	region := path.Base(regionURL) // "us-central1"
	if region == "" {
		panic("empty region")
	}
	return strings.HasPrefix(zone, region)
}

// findIP finds an IP address to use, or returns the empty string if none is found.
// It tries to find a reserved one in the same region where the name of the reserved IP
// is "NAME-ip" and the IP is not in use.
func (cl *cloudLaunch) findIP() string {
	// Try to find it by name.
	aggAddrList, err := cl.computeService.Addresses.AggregatedList(cl.GCEProjectID).Do()
	if err != nil {
		log.Fatal(err)
	}
	// https://godoc.org/google.golang.org/api/compute/v1#AddressAggregatedList
	var ip string
IPLoop:
	for _, asl := range aggAddrList.Items {
		for _, addr := range asl.Addresses {
			log.Printf("  addr: %#v", addr)
			if addr.Name == cl.Name+"-ip" && addr.Status == "RESERVED" && zoneInRegion(cl.zone(), addr.Region) {
				ip = addr.Address
				break IPLoop
			}
		}
	}
	return ip
}

func (cl *cloudLaunch) createInstance() {
	inst := cl.lookupInstance()
	if inst != nil {
		log.Printf("Instance exists; not re-creating.")
		return
	}

	log.Printf("Instance doesn't exist; creating...")

	ip := cl.findIP()
	log.Printf("Found IP: %v", ip)

	cloudConfig := strings.NewReplacer(
		"$NAME", cl.Name,
		"$URL", cl.binaryURL(),
	).Replace(baseConfig)

	instance := &compute.Instance{
		Name:        cl.instName(),
		Description: cl.Name,
		MachineType: cl.machineTypeURL(),
		Disks:       []*compute.AttachedDisk{cl.instanceDisk()},
		Tags: &compute.Tags{
			Items: []string{"http-server", "https-server"},
		},
		Metadata: &compute.Metadata{
			Items: []*compute.MetadataItems{
				{
					Key:   "user-data",
					Value: googleapi.String(cloudConfig),
				},
			},
		},
		NetworkInterfaces: []*compute.NetworkInterface{
			&compute.NetworkInterface{
				AccessConfigs: []*compute.AccessConfig{
					&compute.AccessConfig{
						Type:  "ONE_TO_ONE_NAT",
						Name:  "External NAT",
						NatIP: ip,
					},
				},
				Network: cl.projectAPIURL() + "/global/networks/default",
			},
		},
		ServiceAccounts: []*compute.ServiceAccount{
			{
				Email:  "default",
				Scopes: cl.Scopes,
			},
		},
	}

	log.Printf("Creating instance...")
	op, err := cl.computeService.Instances.Insert(cl.GCEProjectID, cl.zone(), instance).Do()
	if err != nil {
		log.Fatalf("Failed to create instance: %v", err)
	}
	opName := op.Name
	log.Printf("Created. Waiting on operation %v", opName)
OpLoop:
	for {
		time.Sleep(2 * time.Second)
		op, err := cl.computeService.ZoneOperations.Get(cl.GCEProjectID, cl.zone(), opName).Do()
		if err != nil {
			log.Fatalf("Failed to get op %s: %v", opName, err)
		}
		switch op.Status {
		case "PENDING", "RUNNING":
			log.Printf("Waiting on operation %v", opName)
			continue
		case "DONE":
			if op.Error != nil {
				for _, operr := range op.Error.Errors {
					log.Printf("Error: %+v", operr)
				}
				log.Fatalf("Failed to start.")
			}
			log.Printf("Success. %+v", op)
			break OpLoop
		default:
			log.Fatalf("Unknown status %q: %+v", op.Status, op)
		}
	}

	inst, err = cl.computeService.Instances.Get(cl.GCEProjectID, cl.zone(), cl.instName()).Do()
	if err != nil {
		log.Fatalf("Error getting instance after creation: %v", err)
	}
	ij, _ := json.MarshalIndent(inst, "", "    ")
	log.Printf("%s", ij)
	log.Printf("Instance created.")
	os.Exit(0)
}

// returns nil if instance doesn't exist.
func (cl *cloudLaunch) lookupInstance() *compute.Instance {
	inst, err := cl.computeService.Instances.Get(cl.GCEProjectID, cl.zone(), cl.instName()).Do()
	if ae, ok := err.(*googleapi.Error); ok && ae.Code == 404 {
		return nil
	} else if err != nil {
		log.Fatalf("Instances.Get: %v", err)
	}
	return inst
}

func (cl *cloudLaunch) instanceDisk() *compute.AttachedDisk {
	imageURL, err := gceutil.CoreOSImageURL(cl.oauthClient)
	if err != nil {
		log.Fatalf("error looking up latest CoreOS stable image: %v", err)
	}
	diskName := cl.instName() + "-coreos-stateless-pd"
	var diskType string
	if cl.SSD {
		diskType = cl.projectAPIURL() + "/zones/" + cl.zone() + "/diskTypes/pd-ssd"
	}
	return &compute.AttachedDisk{
		AutoDelete: true,
		Boot:       true,
		Type:       "PERSISTENT",
		InitializeParams: &compute.AttachedDiskInitializeParams{
			DiskName:    diskName,
			SourceImage: imageURL,
			DiskSizeGb:  50,
			DiskType:    diskType,
		},
	}
}
