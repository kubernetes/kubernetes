/*
Copyright 2018 The Kubernetes Authors.

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

package ipamperf

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"regexp"
	"sync"

	"github.com/golang/glog"

	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam/cidrset"
	"k8s.io/kubernetes/test/integration/util"
)

var (
	pathInstance  = regexp.MustCompile(`/compute/(v[\d]+|beta|alpha)/projects/([\w|-]+)/zones/([\w|-]+)/instances/([\w|-]+)(\?.*)?`)
	pathUpdateNet = regexp.MustCompile(`/compute/(v[\d]+|beta|alpha)/projects/([\w|-]+)/zones/([\w|-]+)/instances/([\w|-]+)/updateNetworkInterface(\?.*)?`)
	pathOperation = regexp.MustCompile(`/compute/(v[\d]+|beta|alpha)/projects/([\w|-]+)/zones/([\w|-]+)/operations/([\w|-]+)(\?.*)?`)
)

func writeOperationStatus(w io.Writer, selfLink string, header http.Header, statusCode int, status string) error {
	return json.NewEncoder(w).Encode(&compute.Operation{
		ServerResponse: googleapi.ServerResponse{
			Header:         header,
			HTTPStatusCode: statusCode,
		},
		SelfLink: selfLink,
		Status:   status,
	})
}

type cloudServer struct {
	serveAt      int              // the qps to server at
	allocateCIDR bool             // whether cloud should allocate CIDR
	cidrSet      *cidrset.CidrSet // the cidr set if allocateCIDR is true

	lock      sync.Mutex                   // protect access to the instances map
	instances map[string]*compute.Instance // known instances
}

func newCloudServer(serveAt int) *cloudServer {
	return &cloudServer{
		serveAt:      serveAt,
		allocateCIDR: false,
		instances:    make(map[string]*compute.Instance),
	}
}

func newCloudServerWithCIDRAllocator(serveAt int, clusterCIDR *net.IPNet, subnetMaskSize int) *cloudServer {
	cidrSet, _ := cidrset.NewCIDRSet(clusterCIDR, subnetMaskSize)
	return &cloudServer{
		serveAt:      serveAt,
		allocateCIDR: true,
		cidrSet:      cidrSet,
		instances:    make(map[string]*compute.Instance),
	}
}

func (cs *cloudServer) getInstanceOrCreate(zone, name string) *compute.Instance {
	cs.lock.Lock()
	defer cs.lock.Unlock()

	if instance, found := cs.instances[name]; !found {
		instance = &compute.Instance{
			Name: name,
			Zone: zone,
			NetworkInterfaces: []*compute.NetworkInterface{
				{},
			},
		}
		if cs.allocateCIDR {
			nextRange, _ := cs.cidrSet.AllocateNext()
			instance.NetworkInterfaces[0].AliasIpRanges = []*compute.AliasIpRange{
				{
					IpCidrRange:         nextRange.String(),
					SubnetworkRangeName: util.TestSecondaryRangeName,
				},
			}
		}
		cs.instances[name] = instance
	}
	return cs.instances[name]
}

func (cs *cloudServer) assignAliasIPRange(zone, name string, netIf *compute.NetworkInterface) *compute.Instance {
	instance := cs.getInstanceOrCreate(zone, name)

	cs.lock.Lock()
	defer cs.lock.Unlock()
	instance.NetworkInterfaces[0].AliasIpRanges = netIf.AliasIpRanges
	return instance
}

func (cs *cloudServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		switch {
		case pathInstance.MatchString(r.RequestURI):
			parts := pathInstance.FindStringSubmatch(r.RequestURI)
			instance := cs.getInstanceOrCreate(parts[3], parts[4])
			err := json.NewEncoder(w).Encode(instance)
			if err != nil {
				glog.Errorf("Error encoding json result: %v", err)
				w.WriteHeader(http.StatusInternalServerError)
			}
		case pathOperation.MatchString(r.RequestURI):
			parts := pathOperation.FindStringSubmatch(r.RequestURI)
			selfLink := fmt.Sprintf("/projects/%s/zones/%s/operation/%s", parts[2], parts[3], parts[4])
			err := writeOperationStatus(w, selfLink, r.Header, http.StatusOK, "DONE")
			if err != nil {
				glog.Errorf("Error encoding json result: %v", err)
				w.WriteHeader(http.StatusInternalServerError)
			}
		}
	case http.MethodPatch:
		switch {
		case pathUpdateNet.MatchString(r.RequestURI):
			parts := pathUpdateNet.FindStringSubmatch(r.RequestURI)
			body, err := ioutil.ReadAll(r.Body)
			defer r.Body.Close()
			if err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			netInf := &compute.NetworkInterface{}
			if err := json.Unmarshal([]byte(body), netInf); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			cs.assignAliasIPRange(parts[2], parts[3], netInf)
			selfLink := fmt.Sprintf("/projects/%s/zones/%s/operation/op-%s", parts[2], parts[3], parts[4])
			writeOperationStatus(w, selfLink, r.Header, http.StatusOK, "DONE")
		}
	}
}
