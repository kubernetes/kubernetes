package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/containerinfra/v1/clustertemplates"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

const ClusterTemplateResponse = `
{
	"apiserver_port": 8081,
	"cluster_distro": "fedora-atomic",
	"coe": "kubernetes",
	"created_at": "2018-06-27T16:52:21+00:00",
	"dns_nameserver": "8.8.8.8",
	"docker_storage_driver": "devicemapper",
	"docker_volume_size": 3,
	"external_network_id": "public",
	"fixed_network": null,
	"fixed_subnet": null,
	"flavor_id": "m1.small",
	"floating_ip_enabled": true,
	"http_proxy": "http://10.164.177.169:8080",
	"https_proxy": "http://10.164.177.169:8080",
	"image_id": "Fedora-Atomic-27-20180212.2.x86_64",
	"insecure_registry": null,
	"keypair_id": "kp",
	"labels": null,
	"links": [
		{
			"href": "http://10.63.176.154:9511/v1/clustertemplates/79c0f9e5-93b8-4719-8fab-063afc67bffe",
			"rel": "self"
		},
		{
			"href": "http://10.63.176.154:9511/clustertemplates/79c0f9e5-93b8-4719-8fab-063afc67bffe",
			"rel": "bookmark"
		}
	],
	"master_flavor_id": null,
	"master_lb_enabled": true,
	"name": "kubernetes-dev",
	"network_driver": "flannel",
	"no_proxy": "10.0.0.0/8,172.0.0.0/8,192.0.0.0/8,localhost",
	"project_id": "76bd201dbc1641729904ab190d3390c6",
	"public": false,
	"registry_enabled": false,
	"server_type": "vm",
	"tls_disabled": false,
	"updated_at": null,
	"user_id": "c48d66144e9c4a54ae2b164b85cfefe3",
	"uuid": "79c0f9e5-93b8-4719-8fab-063afc67bffe",
	"volume_driver": "cinder"
}`

const ClusterTemplateResponse_EmptyTime = `
{
	"apiserver_port": null,
	"cluster_distro": "fedora-atomic",
	"coe": "kubernetes",
	"created_at": null,
	"dns_nameserver": "8.8.8.8",
	"docker_storage_driver": null,
	"docker_volume_size": 5,
	"external_network_id": "public",
	"fixed_network": null,
	"fixed_subnet": null,
	"flavor_id": "m1.small",
	"http_proxy": null,
	"https_proxy": null,
	"image_id": "fedora-atomic-latest",
	"insecure_registry": null,
	"keypair_id": "testkey",
	"labels": {},
	"links": [
		{
		  "href": "http://65.61.151.130:9511/clustertemplates/472807c2-f175-4946-9765-149701a5aba7",
		  "rel": "bookmark"
		},
		{
		  "href": "http://65.61.151.130:9511/v1/clustertemplates/472807c2-f175-4946-9765-149701a5aba7",
		  "rel": "self"
		}
	],
	"master_flavor_id": null,
	"master_lb_enabled": false,
	"name": "kubernetes-dev",
	"network_driver": "flannel",
	"no_proxy": null,
	"public": false,
	"registry_enabled": false,
	"server_type": "vm",
	"tls_disabled": false,
	"updated_at": null,
	"uuid": "472807c2-f175-4946-9765-149701a5aba7",
	"volume_driver": null
}`

const ClusterTemplateListResponse = `
{
	"clustertemplates": [
		{
			"apiserver_port": 8081,
			"cluster_distro": "fedora-atomic",
			"coe": "kubernetes",
			"created_at": "2018-06-27T16:52:21+00:00",
			"dns_nameserver": "8.8.8.8",
			"docker_storage_driver": "devicemapper",
			"docker_volume_size": 3,
			"external_network_id": "public",
			"fixed_network": null,
			"fixed_subnet": null,
			"flavor_id": "m1.small",
			"floating_ip_enabled": true,
			"http_proxy": "http://10.164.177.169:8080",
			"https_proxy": "http://10.164.177.169:8080",
			"image_id": "Fedora-Atomic-27-20180212.2.x86_64",
			"insecure_registry": null,
			"keypair_id": "kp",
			"labels": null,
			"links": [
				{
					"href": "http://10.63.176.154:9511/v1/clustertemplates/79c0f9e5-93b8-4719-8fab-063afc67bffe",
					"rel": "self"
				},
				{
					"href": "http://10.63.176.154:9511/clustertemplates/79c0f9e5-93b8-4719-8fab-063afc67bffe",
					"rel": "bookmark"
				}
			],
			"master_flavor_id": null,
			"master_lb_enabled": true,
			"name": "kubernetes-dev",
			"network_driver": "flannel",
			"no_proxy": "10.0.0.0/8,172.0.0.0/8,192.0.0.0/8,localhost",
			"project_id": "76bd201dbc1641729904ab190d3390c6",
			"public": false,
			"registry_enabled": false,
			"server_type": "vm",
			"tls_disabled": false,
			"updated_at": null,
			"user_id": "c48d66144e9c4a54ae2b164b85cfefe3",
			"uuid": "79c0f9e5-93b8-4719-8fab-063afc67bffe",
			"volume_driver": "cinder"
		},
		{
			"apiserver_port": null,
			"cluster_distro": "fedora-atomic",
			"coe": "kubernetes",
			"created_at": null,
			"dns_nameserver": "8.8.8.8",
			"docker_storage_driver": null,
			"docker_volume_size": 5,
			"external_network_id": "public",
			"fixed_network": null,
			"fixed_subnet": null,
			"flavor_id": "m1.small",
			"http_proxy": null,
			"https_proxy": null,
			"image_id": "fedora-atomic-latest",
			"insecure_registry": null,
			"keypair_id": "testkey",
			"labels": {},
			"links": [
				{
				  "href": "http://65.61.151.130:9511/clustertemplates/472807c2-f175-4946-9765-149701a5aba7",
				  "rel": "bookmark"
				},
				{
				  "href": "http://65.61.151.130:9511/v1/clustertemplates/472807c2-f175-4946-9765-149701a5aba7",
				  "rel": "self"
				}
			],
			"master_flavor_id": null,
			"master_lb_enabled": false,
			"name": "kubernetes-dev",
			"network_driver": "flannel",
			"no_proxy": null,
			"public": false,
			"registry_enabled": false,
			"server_type": "vm",
			"tls_disabled": false,
			"updated_at": null,
			"uuid": "472807c2-f175-4946-9765-149701a5aba7",
			"volume_driver": null
		}
	]
}`

var ExpectedClusterTemplate = clustertemplates.ClusterTemplate{
	APIServerPort:       8081,
	COE:                 "kubernetes",
	ClusterDistro:       "fedora-atomic",
	CreatedAt:           time.Date(2018, 6, 27, 16, 52, 21, 0, time.UTC),
	DNSNameServer:       "8.8.8.8",
	DockerStorageDriver: "devicemapper",
	DockerVolumeSize:    3,
	ExternalNetworkID:   "public",
	FixedNetwork:        "",
	FixedSubnet:         "",
	FlavorID:            "m1.small",
	FloatingIPEnabled:   true,
	HTTPProxy:           "http://10.164.177.169:8080",
	HTTPSProxy:          "http://10.164.177.169:8080",
	ImageID:             "Fedora-Atomic-27-20180212.2.x86_64",
	InsecureRegistry:    "",
	KeyPairID:           "kp",
	Labels:              map[string]string(nil),
	Links: []gophercloud.Link{
		{Href: "http://10.63.176.154:9511/v1/clustertemplates/79c0f9e5-93b8-4719-8fab-063afc67bffe", Rel: "self"},
		{Href: "http://10.63.176.154:9511/clustertemplates/79c0f9e5-93b8-4719-8fab-063afc67bffe", Rel: "bookmark"},
	},
	MasterFlavorID:  "",
	MasterLBEnabled: true,
	Name:            "kubernetes-dev",
	NetworkDriver:   "flannel",
	NoProxy:         "10.0.0.0/8,172.0.0.0/8,192.0.0.0/8,localhost",
	ProjectID:       "76bd201dbc1641729904ab190d3390c6",
	Public:          false,
	RegistryEnabled: false,
	ServerType:      "vm",
	TLSDisabled:     false,
	UUID:            "79c0f9e5-93b8-4719-8fab-063afc67bffe",
	UpdatedAt:       time.Time{},
	UserID:          "c48d66144e9c4a54ae2b164b85cfefe3",
	VolumeDriver:    "cinder",
}

var ExpectedClusterTemplate_EmptyTime = clustertemplates.ClusterTemplate{
	COE:                 "kubernetes",
	ClusterDistro:       "fedora-atomic",
	CreatedAt:           time.Time{},
	DNSNameServer:       "8.8.8.8",
	DockerStorageDriver: "",
	DockerVolumeSize:    5,
	ExternalNetworkID:   "public",
	FixedNetwork:        "",
	FixedSubnet:         "",
	FlavorID:            "m1.small",
	HTTPProxy:           "",
	HTTPSProxy:          "",
	ImageID:             "fedora-atomic-latest",
	InsecureRegistry:    "",
	KeyPairID:           "testkey",
	Labels:              map[string]string{},
	Links: []gophercloud.Link{
		{Href: "http://65.61.151.130:9511/clustertemplates/472807c2-f175-4946-9765-149701a5aba7", Rel: "bookmark"},
		{Href: "http://65.61.151.130:9511/v1/clustertemplates/472807c2-f175-4946-9765-149701a5aba7", Rel: "self"},
	},
	MasterFlavorID:  "",
	MasterLBEnabled: false,
	Name:            "kubernetes-dev",
	NetworkDriver:   "flannel",
	NoProxy:         "",
	Public:          false,
	RegistryEnabled: false,
	ServerType:      "vm",
	TLSDisabled:     false,
	UUID:            "472807c2-f175-4946-9765-149701a5aba7",
	UpdatedAt:       time.Time{},
	VolumeDriver:    "",
}

var ExpectedClusterTemplates = []clustertemplates.ClusterTemplate{ExpectedClusterTemplate, ExpectedClusterTemplate_EmptyTime}

func HandleCreateClusterTemplateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clustertemplates", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("OpenStack-API-Minimum-Version", "container-infra 1.1")
		w.Header().Add("OpenStack-API-Maximum-Version", "container-infra 1.6")
		w.Header().Add("OpenStack-API-Version", "container-infra 1.1")
		w.Header().Add("X-OpenStack-Request-Id", "req-781e9bdc-4163-46eb-91c9-786c53188bbb")
		w.WriteHeader(http.StatusCreated)

		fmt.Fprint(w, ClusterTemplateResponse)
	})
}

func HandleDeleteClusterSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clustertemplates/6dc6d336e3fc4c0a951b5698cd1236ee", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "DELETE")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("OpenStack-API-Minimum-Version", "container-infra 1.1")
		w.Header().Add("OpenStack-API-Maximum-Version", "container-infra 1.6")
		w.Header().Add("OpenStack-API-Version", "container-infra 1.1")
		w.Header().Add("X-OpenStack-Request-Id", "req-781e9bdc-4163-46eb-91c9-786c53188bbb")
		w.WriteHeader(http.StatusNoContent)
	})
}

func HandleListClusterTemplateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clustertemplates", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ClusterTemplateListResponse)
	})
}

func HandleGetClusterTemplateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clustertemplates/7d85f602-a948-4a30-afd4-e84f47471c15", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ClusterTemplateResponse)
	})
}

func HandleGetClusterTemplateEmptyTimeSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clustertemplates/7d85f602-a948-4a30-afd4-e84f47471c15", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, ClusterTemplateResponse_EmptyTime)
	})
}

const UpdateResponse = `
{
	"apiserver_port": null,
	"cluster_distro": "fedora-atomic",
	"coe": "kubernetes",
	"created_at": "2016-08-10T13:47:01+00:00",
	"dns_nameserver": "8.8.8.8",
	"docker_storage_driver": null,
	"docker_volume_size": 5,
	"external_network_id": "public",
	"fixed_network": null,
	"fixed_subnet": null,
	"flavor_id": "m1.small",
	"http_proxy": null,
	"https_proxy": null,
	"image_id": "fedora-atomic-latest",
	"insecure_registry": null,
	"keypair_id": "testkey",
	"labels": {},
	"links": [
		{
		  "href": "http://65.61.151.130:9511/v1/clustertemplates/472807c2-f175-4946-9765-149701a5aba7",
		  "rel": "self"
		},
		{
		  "href": "http://65.61.151.130:9511/clustertemplates/472807c2-f175-4946-9765-149701a5aba7",
		  "rel": "bookmark"
		}
	],
	"master_flavor_id": null,
	"master_lb_enabled": false,
	"name": "kubernetes-dev",
	"network_driver": "flannel",
	"no_proxy": null,
	"public": false,
	"registry_enabled": false,
	"server_type": "vm",
	"tls_disabled": false,
	"updated_at": null,
	"uuid": "472807c2-f175-4946-9765-149701a5aba7",
	"volume_driver": null
}`

const UpdateResponse_EmptyTime = `
{
	"apiserver_port": null,
	"cluster_distro": "fedora-atomic",
	"coe": "kubernetes",
	"created_at": null,
	"dns_nameserver": "8.8.8.8",
	"docker_storage_driver": null,
	"docker_volume_size": 5,
	"external_network_id": "public",
	"fixed_network": null,
	"fixed_subnet": null,
	"flavor_id": "m1.small",
	"http_proxy": null,
	"https_proxy": null,
	"image_id": "fedora-atomic-latest",
	"insecure_registry": null,
	"keypair_id": "testkey",
	"labels": {},
	"links": [
		{
		  "href": "http://65.61.151.130:9511/v1/clustertemplates/472807c2-f175-4946-9765-149701a5aba7",
		  "rel": "self"
		},
		{
		  "href": "http://65.61.151.130:9511/clustertemplates/472807c2-f175-4946-9765-149701a5aba7",
		  "rel": "bookmark"
		}
	],
	"master_flavor_id": null,
	"master_lb_enabled": false,
	"name": "kubernetes-dev",
	"network_driver": "flannel",
	"no_proxy": null,
	"public": false,
	"registry_enabled": false,
	"server_type": "vm",
	"tls_disabled": false,
	"updated_at": null,
	"uuid": "472807c2-f175-4946-9765-149701a5aba7",
	"volume_driver": null
}`

const UpdateResponse_InvalidUpdate = `
{
    "errors": [{\"status\": 400, \"code\": \"client\", \"links\": [], \"title\": \"'add' and 'replace' operations needs value\", \"detail\": \"'add' and 'replace' operations needs value\", \"request_id\": \"\"}]
}`

var ExpectedUpdateClusterTemplate = clustertemplates.ClusterTemplate{
	COE:                 "kubernetes",
	ClusterDistro:       "fedora-atomic",
	CreatedAt:           time.Date(2016, 8, 10, 13, 47, 01, 0, time.UTC),
	DNSNameServer:       "8.8.8.8",
	DockerStorageDriver: "",
	DockerVolumeSize:    5,
	ExternalNetworkID:   "public",
	FixedNetwork:        "",
	FixedSubnet:         "",
	FlavorID:            "m1.small",
	HTTPProxy:           "",
	HTTPSProxy:          "",
	ImageID:             "fedora-atomic-latest",
	InsecureRegistry:    "",
	KeyPairID:           "testkey",
	Labels:              map[string]string{},
	Links: []gophercloud.Link{
		{Href: "http://65.61.151.130:9511/v1/clustertemplates/472807c2-f175-4946-9765-149701a5aba7", Rel: "self"},
		{Href: "http://65.61.151.130:9511/clustertemplates/472807c2-f175-4946-9765-149701a5aba7", Rel: "bookmark"},
	},
	MasterFlavorID:  "",
	MasterLBEnabled: false,
	Name:            "kubernetes-dev",
	NetworkDriver:   "flannel",
	NoProxy:         "",
	Public:          false,
	RegistryEnabled: false,
	ServerType:      "vm",
	TLSDisabled:     false,
	UUID:            "472807c2-f175-4946-9765-149701a5aba7",
	UpdatedAt:       time.Time{},
	VolumeDriver:    "",
}

var ExpectedUpdateClusterTemplate_EmptyTime = clustertemplates.ClusterTemplate{
	COE:                 "kubernetes",
	ClusterDistro:       "fedora-atomic",
	CreatedAt:           time.Time{},
	DNSNameServer:       "8.8.8.8",
	DockerStorageDriver: "",
	DockerVolumeSize:    5,
	ExternalNetworkID:   "public",
	FixedNetwork:        "",
	FixedSubnet:         "",
	FlavorID:            "m1.small",
	HTTPProxy:           "",
	HTTPSProxy:          "",
	ImageID:             "fedora-atomic-latest",
	InsecureRegistry:    "",
	KeyPairID:           "testkey",
	Labels:              map[string]string{},
	Links: []gophercloud.Link{
		{Href: "http://65.61.151.130:9511/v1/clustertemplates/472807c2-f175-4946-9765-149701a5aba7", Rel: "self"},
		{Href: "http://65.61.151.130:9511/clustertemplates/472807c2-f175-4946-9765-149701a5aba7", Rel: "bookmark"},
	},
	MasterFlavorID:  "",
	MasterLBEnabled: false,
	Name:            "kubernetes-dev",
	NetworkDriver:   "flannel",
	NoProxy:         "",
	Public:          false,
	RegistryEnabled: false,
	ServerType:      "vm",
	TLSDisabled:     false,
	UUID:            "472807c2-f175-4946-9765-149701a5aba7",
	UpdatedAt:       time.Time{},
	VolumeDriver:    "",
}

func HandleUpdateClusterTemplateSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clustertemplates/7d85f602-a948-4a30-afd4-e84f47471c15", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, UpdateResponse)
	})
}

func HandleUpdateClusterTemplateEmptyTimeSuccessfully(t *testing.T) {
	th.Mux.HandleFunc("/v1/clustertemplates/7d85f602-a948-4a30-afd4-e84f47471c15", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprint(w, UpdateResponse_EmptyTime)
	})
}

func HandleUpdateClusterTemplateInvalidUpdate(t *testing.T) {
	th.Mux.HandleFunc("/v1/clustertemplates/7d85f602-a948-4a30-afd4-e84f47471c15", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "PATCH")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)

		fmt.Fprint(w, UpdateResponse_EmptyTime)
	})
}
