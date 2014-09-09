/*
Copyright 2014 Google Inc. All rights reserved.

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

package gce_cloud

import (
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"code.google.com/p/goauth2/compute/serviceaccount"
	"code.google.com/p/goauth2/oauth"
	compute "code.google.com/p/google-api-go-client/compute/v1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/golang/glog"
)

// GCECloud is an implementation of Interface, TCPLoadBalancer and Instances for Google Compute Engine.
type GCECloud struct {
	service    *compute.Service
	projectID  string
	zone       string
	instanceRE string
}

func init() {
	cloudprovider.RegisterCloudProvider("gce", func(config io.Reader) (cloudprovider.Interface, error) { return newGCECloud() })
}

func getProjectAndZone() (string, string, error) {
	client := http.Client{}
	url := "http://metadata/computeMetadata/v1/instance/zone"
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", "", err
	}
	req.Header.Add("X-Google-Metadata-Request", "True")
	res, err := client.Do(req)
	if err != nil {
		return "", "", err
	}
	defer res.Body.Close()
	data, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return "", "", err
	}
	parts := strings.Split(string(data), "/")
	if len(parts) != 4 {
		return "", "", fmt.Errorf("Unexpected response: %s", string(data))
	}
	return parts[1], parts[3], nil
}

// newGCECloud creates a new instance of GCECloud.
func newGCECloud() (*GCECloud, error) {
	projectID, zone, err := getProjectAndZone()
	if err != nil {
		return nil, err
	}
	client, err := serviceaccount.NewClient(&serviceaccount.Options{})
	if err != nil {
		return nil, err
	}
	svc, err := compute.New(client)
	if err != nil {
		return nil, err
	}
	return &GCECloud{
		service:   svc,
		projectID: projectID,
		zone:      zone,
	}, nil
}

// CreateGCECloud creates an instance of GCECloud for use with CLI clients.
// It uses a locally cached OAuth2 token to allow RO access to Google Storage,
// and full access to Google Compute. If the token does not exist, it launches
// the authorization request in a browser window, so a user should be present.
func CreateGCECloud(projectID, zone string) (*GCECloud, error) {
	c := CreateOAuthClient()
	svc, err := compute.New(c)
	if err != nil {
		return nil, err
	}
	return &GCECloud{
		service:   svc,
		projectID: projectID,
		zone:      zone,
	}, nil
}

var config = &oauth.Config{
	// this Id & Secret are located under the google-containers project on gce
	ClientId:     "255964991331-b0l3n9c5pqc0u0ijtniv8vls226d3d5j.apps.googleusercontent.com",
	ClientSecret: "BWm6fPAY2gS1jaRT-Xn2y-uT",
	Scope: strings.Join([]string{
		compute.DevstorageRead_onlyScope,
		compute.ComputeScope,
	}, " "),
	AuthURL:  "https://accounts.google.com/o/oauth2/auth",
	TokenURL: "https://accounts.google.com/o/oauth2/token",
}

// CreateOAuth2Client creates an Oauth client either from a cached token or a new token from the web.
func CreateOAuthClient() *http.Client {
	cacheFile := tokenCacheFile(config)
	token, err := loadToken(cacheFile)
	if err != nil {
		glog.Warningf("found a token, but couldn't open it.")
	}
	if token == nil {
		token = tokenFromWeb(config)
		saveToken(cacheFile, token)
	}
	t := &oauth.Transport{
		Token:     token,
		Config:    config,
		Transport: http.DefaultTransport,
	}
	return t.Client()
}

func saveToken(file string, token *oauth.Token) error {
	data, err := json.Marshal(token)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(file, data, 0600)
}

func tokenFromWeb(config *oauth.Config) *oauth.Token {
	ch := make(chan string)
	randState := fmt.Sprintf("st%d", time.Now().UnixNano())
	ts := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		if req.URL.Path == "/favicon.ico" {
			http.Error(rw, "", 404)
			return
		}
		if req.FormValue("state") != randState {
			glog.Infof("State doesn't match: req = %#v", req)
			http.Error(rw, "", 500)
			return
		}
		if code := req.FormValue("code"); code != "" {
			fmt.Fprintf(rw, "<h1>Success</h1>Authorized.")
			rw.(http.Flusher).Flush()
			ch <- code
			return
		}
		glog.Infof("no code")
		http.Error(rw, "", 500)
	}))
	defer ts.Close()

	config.RedirectURL = ts.URL
	authUrl := config.AuthCodeURL(randState)
	go openURL(authUrl)
	glog.Infof("Authorize this app at:\n%s", authUrl)
	code := <-ch
	glog.Infof("Got code: %s", code)

	t := &oauth.Transport{
		Config:    config,
		Transport: http.DefaultTransport,
	}
	_, err := t.Exchange(code)
	if err != nil {
		glog.Fatalf("Token exchange error: %v", err)
	}
	return t.Token
}

func tokenCacheFile(config *oauth.Config) string {
	hash := fnv.New32a()
	hash.Write([]byte(config.ClientId))
	hash.Write([]byte(config.ClientSecret))
	hash.Write([]byte(config.Scope))
	fn := fmt.Sprintf("kube-tok%v", hash.Sum32())
	return filepath.Join(osUserCacheDir(), url.QueryEscape(fn))
}

func loadToken(file string) (*oauth.Token, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var token oauth.Token
	err = json.Unmarshal(data, &token)
	return &token, err
}

func openURL(url string) {
	try := []string{"xdg-open", "google-chrome", "open"}
	for _, bin := range try {
		err := exec.Command(bin, url).Run()
		if err == nil {
			return
		}
	}
	glog.Fatalf("Error opening URL in browser.")
}

func osUserCacheDir() string {
	switch runtime.GOOS {
	case "darwin":
		return filepath.Join(os.Getenv("HOME"), "Library", "Caches")
	case "linux", "freebsd":
		return filepath.Join(os.Getenv("HOME"), ".cache")
	}
	return "."
}

// TCPLoadBalancer returns an implementation of TCPLoadBalancer for Google Compute Engine.
func (gce *GCECloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return gce, true
}

// Instances returns an implementation of Instances for Google Compute Engine.
func (gce *GCECloud) Instances() (cloudprovider.Instances, bool) {
	return gce, true
}

// Zones returns an implementation of Zones for Google Compute Engine.
func (gce *GCECloud) Zones() (cloudprovider.Zones, bool) {
	return gce, true
}

func makeHostLink(projectID, zone, host string) string {
	ix := strings.Index(host, ".")
	if ix != -1 {
		host = host[:ix]
	}
	return fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/zones/%s/instances/%s",
		projectID, zone, host)
}

func (gce *GCECloud) makeTargetPool(name, region string, hosts []string) (string, error) {
	var instances []string
	for _, host := range hosts {
		instances = append(instances, makeHostLink(gce.projectID, gce.zone, host))
	}
	pool := &compute.TargetPool{
		Name:      name,
		Instances: instances,
	}
	_, err := gce.service.TargetPools.Insert(gce.projectID, region, pool).Do()
	if err != nil {
		return "", err
	}
	link := fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/regions/%s/targetPools/%s", gce.projectID, region, name)
	return link, nil
}

func (gce *GCECloud) waitForRegionOp(op *compute.Operation, region string) error {
	pollOp := op
	for pollOp.Status != "DONE" {
		var err error
		time.Sleep(time.Second * 10)
		pollOp, err = gce.service.RegionOperations.Get(gce.projectID, region, op.Name).Do()
		if err != nil {
			return err
		}
	}
	return nil
}

type opScope int

const (
	REGION opScope = iota
	ZONE
	GLOBAL
)

// GCEOp is an abstraction of GCE's compute.Operation, providing only necessary generic information.
type GCEOp struct {
	op    *compute.Operation
	scope opScope
}

// Status provides the status of the operation, and will be either "PENDING", "RUNNING", or "DONE".
func (op *GCEOp) Status() string {
	return op.op.Status
}

// Errors provides any errors that the operation encountered.
func (op *GCEOp) Errors() []string {
	var errors []string
	if op.op.Error != nil {
		for _, err := range op.op.Error.Errors {
			errors = append(errors, err.Message)
		}
	}
	return errors
}

// OperationType provides the type of operation represented by op.
func (op *GCEOp) OperationType() string {
	return op.op.OperationType
}

// Target provides the name of the affected resource.
func (op *GCEOp) Target() string {
	target, _ := targetInfo(op.op.TargetLink)
	return target
}

// Resource provides the type of the affected resource.
func (op *GCEOp) Resource() string {
	_, resource := targetInfo(op.op.TargetLink)
	return resource
}

func targetInfo(targetLink string) (target, resourceType string) {
	i := strings.LastIndex(targetLink, "/")
	target = targetLink[i+1:]
	j := strings.LastIndex(targetLink[:i], "/")
	resourceType = targetLink[j+1 : i-1]
	return target, resourceType
}

func (gce *GCECloud) PollOp(op *GCEOp) (*GCEOp, error) {
	var err error
	switch op.scope {
	case REGION:
		region := op.op.Region[strings.LastIndex(op.op.Region, "/")+1:]
		op.op, err = gce.service.RegionOperations.Get(gce.projectID, region, op.op.Name).Do()
	case ZONE:
		zone := op.op.Zone[strings.LastIndex(op.op.Zone, "/")+1:]
		op.op, err = gce.service.ZoneOperations.Get(gce.projectID, zone, op.op.Name).Do()
	case GLOBAL:
		op.op, err = gce.service.GlobalOperations.Get(gce.projectID, op.op.Name).Do()
	default:
		err = errors.New("unknown operation scope")
	}
	return op, err
}

func (gce *GCECloud) PollGlobalOp(op *compute.Operation) (*compute.Operation, error) {
	return gce.service.GlobalOperations.Get(gce.projectID, op.Name).Do()
}

// TCPLoadBalancerExists is an implementation of TCPLoadBalancer.TCPLoadBalancerExists.
func (gce *GCECloud) TCPLoadBalancerExists(name, region string) (bool, error) {
	_, err := gce.service.ForwardingRules.Get(gce.projectID, region, name).Do()
	return false, err
}

// CreateTCPLoadBalancer is an implementation of TCPLoadBalancer.CreateTCPLoadBalancer.
func (gce *GCECloud) CreateTCPLoadBalancer(name, region string, port int, hosts []string) error {
	pool, err := gce.makeTargetPool(name, region, hosts)
	if err != nil {
		return err
	}
	req := &compute.ForwardingRule{
		Name:       name,
		IPProtocol: "TCP",
		PortRange:  strconv.Itoa(port),
		Target:     pool,
	}
	_, err = gce.service.ForwardingRules.Insert(gce.projectID, region, req).Do()
	return err
}

// UpdateTCPLoadBalancer is an implementation of TCPLoadBalancer.UpdateTCPLoadBalancer.
func (gce *GCECloud) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
	var refs []*compute.InstanceReference
	for _, host := range hosts {
		refs = append(refs, &compute.InstanceReference{host})
	}
	req := &compute.TargetPoolsAddInstanceRequest{
		Instances: refs,
	}

	_, err := gce.service.TargetPools.AddInstance(gce.projectID, region, name, req).Do()
	return err
}

// DeleteTCPLoadBalancer is an implementation of TCPLoadBalancer.DeleteTCPLoadBalancer.
func (gce *GCECloud) DeleteTCPLoadBalancer(name, region string) error {
	_, err := gce.service.ForwardingRules.Delete(gce.projectID, region, name).Do()
	if err != nil {
		return err
	}
	_, err = gce.service.TargetPools.Delete(gce.projectID, region, name).Do()
	return err
}

// IPAddress is an implementation of Instances.IPAddress.
func (gce *GCECloud) IPAddress(instance string) (net.IP, error) {
	res, err := gce.service.Instances.Get(gce.projectID, gce.zone, instance).Do()
	if err != nil {
		return nil, err
	}
	ip := net.ParseIP(res.NetworkInterfaces[0].AccessConfigs[0].NatIP)
	if ip == nil {
		return nil, fmt.Errorf("Invalid network IP: %s", res.NetworkInterfaces[0].AccessConfigs[0].NatIP)
	}
	return ip, nil
}

// ProjectName returns the Google Cloud project being used for this GCECloud
func (gce *GCECloud) ProjectID() string {
	return gce.projectID
}

// ProjectName returns the Google Cloud project being used for this GCECloud
func (gce *GCECloud) Zone() string {
	return gce.zone
}

// FullQualProj returns the full URL for the specified project.
func FullQualProj(proj string) string {
	return "https://www.googleapis.com/compute/v1/projects/" + proj
}

// CreateInstance creates the specified instance.
func (gce *GCECloud) CreateInstance(name, size, image, tag, startupScript string, ipFwd bool, scopes []string) (*GCEOp, error) {
	fqp := FullQualProj(gce.projectID)
	var serviceAccounts []*compute.ServiceAccount
	if len(scopes) > 0 {
		serviceAccounts = []*compute.ServiceAccount{
			{
				Email:  "default",
				Scopes: scopes,
			},
		}
	} else {
		serviceAccounts = nil
	}
	instance := &compute.Instance{
		Name:         name,
		MachineType:  fqp + "/zones/" + gce.zone + "/machineTypes/" + size,
		CanIpForward: ipFwd,
		Disks: []*compute.AttachedDisk{
			{
				AutoDelete: true,
				Boot:       true,
				Type:       "PERSISTENT",
				InitializeParams: &compute.AttachedDiskInitializeParams{
					DiskName:    name,
					SourceImage: image,
				},
			},
		},
		Metadata: &compute.Metadata{
			Items: []*compute.MetadataItems{
				{
					Key:   "startup-script",
					Value: startupScript,
				},
			},
		},
		NetworkInterfaces: []*compute.NetworkInterface{
			{
				AccessConfigs: []*compute.AccessConfig{
					{
						Type: "ONE_TO_ONE_NAT",
						Name: "external-nat",
					},
				},
				Network: fqp + "/global/networks/default",
			},
		},
		ServiceAccounts: serviceAccounts,
		Scheduling: &compute.Scheduling{
			AutomaticRestart: true,
		},
		Tags: &compute.Tags{
			Items: []string{
				tag,
			},
		},
	}
	computeOp, err := gce.service.Instances.Insert(gce.projectID, gce.zone, instance).Do()
	op := &GCEOp{
		op:    computeOp,
		scope: ZONE,
	}
	return op, err
}

// DeleteInstance deletes the specified instance
func (gce *GCECloud) DeleteInstance(name string) (*GCEOp, error) {
	computeOp, err := gce.service.Instances.Delete(gce.projectID, gce.zone, name).Do()
	op := &GCEOp{
		op:    computeOp,
		scope: ZONE,
	}
	return op, err
}

// fqdnSuffix is hacky function to compute the delta between hostame and hostname -f.
func fqdnSuffix() (string, error) {
	fullHostname, err := exec.Command("hostname", "-f").Output()
	if err != nil {
		return "", err
	}
	hostname, err := exec.Command("hostname").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(fullHostname)[len(string(hostname)):]), nil
}

// List is an implementation of Instances.List.
func (gce *GCECloud) List(filter string) ([]string, error) {
	// GCE gives names without their fqdn suffix, so get that here for appending.
	// This is needed because the kubelet looks for its jobs in /registry/hosts/<fqdn>/pods
	// We should really just replace this convention, with a negotiated naming protocol for kubelet's
	// to register with the master.
	suffix, err := fqdnSuffix()
	if err != nil {
		return []string{}, err
	}
	if len(suffix) > 0 {
		suffix = "." + suffix
	}
	listCall := gce.service.Instances.List(gce.projectID, gce.zone)
	if len(filter) > 0 {
		listCall = listCall.Filter("name eq " + filter)
	}
	res, err := listCall.Do()
	if err != nil {
		return nil, err
	}
	var instances []string
	for _, instance := range res.Items {
		instances = append(instances, instance.Name+suffix)
	}
	return instances, nil
}

// CreateRoute creates the specified route and adds it to the project's network configuration.
func (gce *GCECloud) CreateRoute(name, nextHop, ipRange string) (*GCEOp, error) {
	route := &compute.Route{
		DestRange:       ipRange,
		Name:            name,
		Network:         FullQualProj(gce.projectID) + "/global/networks/default",
		NextHopInstance: nextHop,
		Priority:        1000,
	}
	computeOp, err := gce.service.Routes.Insert(gce.projectID, route).Do()
	op := &GCEOp{
		op:    computeOp,
		scope: GLOBAL,
	}
	return op, err
}

func (gce *GCECloud) DeleteRoute(name string) (*GCEOp, error) {
	computeOp, err := gce.service.Routes.Delete(gce.projectID, name).Do()
	op := &GCEOp{
		op:    computeOp,
		scope: GLOBAL,
	}
	return op, err
}

// CreateFirewall creates the specified firewall rule
func (gce *GCECloud) CreateFirewall(name, sourceRange, tag, allowed string) (*GCEOp, error) {
	fwAllowed, err := parseAllowed(allowed)
	if err != nil {
		return nil, err
	}
	prefix := FullQualProj(gce.projectID)
	firewall := &compute.Firewall{
		Name:    name,
		Network: prefix + "/global/networks/default",
		Allowed: fwAllowed,
		SourceRanges: []string{
			sourceRange,
		},
		TargetTags: []string{
			tag,
		},
	}
	computeOp, err := gce.service.Firewalls.Insert(gce.projectID, firewall).Do()
	op := &GCEOp{
		op:    computeOp,
		scope: GLOBAL,
	}
	return op, err
}

// DeleteFirewall deletes the firewall specified by "name".
func (gce *GCECloud) DeleteFirewall(name string) (*GCEOp, error) {
	computeOp, err := gce.service.Firewalls.Delete(gce.projectID, name).Do()
	op := &GCEOp{
		op:    computeOp,
		scope: GLOBAL,
	}
	return op, err
}

func parseAllowed(allowedString string) ([]*compute.FirewallAllowed, error) {
	parse, err := regexp.Compile(`([a-zA-Z]+)(:([0-9]*(-[0-9]*)?))?`)
	if err != nil {
		return nil, err
	}
	allowedArr := parse.FindAllStringSubmatch(allowedString, -1)
	rules := make(map[string][]string)
	for _, match := range allowedArr {
		ipProtocol := match[1]
		ports := match[3]
		rules[ipProtocol] = append(rules[ipProtocol], ports)
	}

	var allowed []*compute.FirewallAllowed
	for ipp, prts := range rules {
		allow := &compute.FirewallAllowed{
			IPProtocol: ipp,
		}
		if len(prts) > 0 && prts[0] != "" {
			allow.Ports = prts
		}
		allowed = append(allowed, allow)
	}
	return allowed, nil
}

func (gce *GCECloud) GetZone() (cloudprovider.Zone, error) {
	region, err := getGceRegion(gce.zone)
	if err != nil {
		return cloudprovider.Zone{}, err
	}
	return cloudprovider.Zone{
		FailureDomain: gce.zone,
		Region:        region,
	}, nil
}

// getGceRegion returns region of the gce zone. Zone names
// are of the form: ${region-name}-${ix}.
// For example "us-central1-b" has a region of "us-central1".
// So we look for the last '-' and trim to just before that.
func getGceRegion(zone string) (string, error) {
	ix := strings.LastIndex(zone, "-")
	if ix == -1 {
		return "", fmt.Errorf("unexpected zone: %s", zone)
	}
	return zone[:ix], nil
}
