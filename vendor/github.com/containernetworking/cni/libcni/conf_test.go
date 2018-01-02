// Copyright 2016 CNI authors
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

package libcni_test

import (
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/containernetworking/cni/libcni"
	"github.com/containernetworking/cni/pkg/types"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Loading configuration from disk", func() {
	Describe("LoadConf", func() {
		var (
			configDir    string
			pluginConfig []byte
		)

		BeforeEach(func() {
			var err error
			configDir, err = ioutil.TempDir("", "plugin-conf")
			Expect(err).NotTo(HaveOccurred())

			pluginConfig = []byte(`{ "name": "some-plugin", "some-key": "some-value" }`)
			Expect(ioutil.WriteFile(filepath.Join(configDir, "50-whatever.conf"), pluginConfig, 0600)).To(Succeed())
		})

		AfterEach(func() {
			Expect(os.RemoveAll(configDir)).To(Succeed())
		})

		It("finds the network config file for the plugin of the given type", func() {
			netConfig, err := libcni.LoadConf(configDir, "some-plugin")
			Expect(err).NotTo(HaveOccurred())
			Expect(netConfig).To(Equal(&libcni.NetworkConfig{
				Network: &types.NetConf{Name: "some-plugin"},
				Bytes:   pluginConfig,
			}))
		})

		Context("when the config directory does not exist", func() {
			BeforeEach(func() {
				Expect(os.RemoveAll(configDir)).To(Succeed())
			})

			It("returns a useful error", func() {
				_, err := libcni.LoadConf(configDir, "some-plugin")
				Expect(err).To(MatchError(libcni.NoConfigsFoundError{Dir: configDir}))
			})
		})

		Context("when the config file is .json extension instead of .conf", func() {
			BeforeEach(func() {
				Expect(os.Remove(configDir + "/50-whatever.conf")).To(Succeed())
				pluginConfig = []byte(`{ "name": "some-plugin", "some-key": "some-value" }`)
				Expect(ioutil.WriteFile(filepath.Join(configDir, "50-whatever.json"), pluginConfig, 0600)).To(Succeed())
			})
			It("finds the network config file for the plugin of the given type", func() {
				netConfig, err := libcni.LoadConf(configDir, "some-plugin")
				Expect(err).NotTo(HaveOccurred())
				Expect(netConfig).To(Equal(&libcni.NetworkConfig{
					Network: &types.NetConf{Name: "some-plugin"},
					Bytes:   pluginConfig,
				}))
			})
		})

		Context("when there is no config for the desired plugin", func() {
			It("returns a useful error", func() {
				_, err := libcni.LoadConf(configDir, "some-other-plugin")
				Expect(err).To(MatchError(ContainSubstring(`no net configuration with name "some-other-plugin" in`)))
			})
		})

		Context("when a config file is malformed", func() {
			BeforeEach(func() {
				Expect(ioutil.WriteFile(filepath.Join(configDir, "00-bad.conf"), []byte(`{`), 0600)).To(Succeed())
			})

			It("returns a useful error", func() {
				_, err := libcni.LoadConf(configDir, "some-plugin")
				Expect(err).To(MatchError(`error parsing configuration: unexpected end of JSON input`))
			})
		})

		Context("when the config is in a nested subdir", func() {
			BeforeEach(func() {
				subdir := filepath.Join(configDir, "subdir1", "subdir2")
				Expect(os.MkdirAll(subdir, 0700)).To(Succeed())

				pluginConfig = []byte(`{ "name": "deep", "some-key": "some-value" }`)
				Expect(ioutil.WriteFile(filepath.Join(subdir, "90-deep.conf"), pluginConfig, 0600)).To(Succeed())
			})

			It("will not find the config", func() {
				_, err := libcni.LoadConf(configDir, "deep")
				Expect(err).To(MatchError(HavePrefix("no net configuration with name")))
			})
		})
	})

	Describe("Capabilities", func() {
		var configDir string

		BeforeEach(func() {
			var err error
			configDir, err = ioutil.TempDir("", "plugin-conf")
			Expect(err).NotTo(HaveOccurred())

			pluginConfig := []byte(`{ "name": "some-plugin", "type": "noop", "cniVersion": "0.3.1", "capabilities": { "portMappings": true, "somethingElse": true, "noCapability": false } }`)
			Expect(ioutil.WriteFile(filepath.Join(configDir, "50-whatever.conf"), pluginConfig, 0600)).To(Succeed())
		})

		AfterEach(func() {
			Expect(os.RemoveAll(configDir)).To(Succeed())
		})

		It("reads plugin capabilities from network config", func() {
			netConfig, err := libcni.LoadConf(configDir, "some-plugin")
			Expect(err).NotTo(HaveOccurred())
			Expect(netConfig.Network.Capabilities).To(Equal(map[string]bool{
				"portMappings":  true,
				"somethingElse": true,
				"noCapability":  false,
			}))
		})
	})

	Describe("ConfFromFile", func() {
		Context("when the file cannot be opened", func() {
			It("returns a useful error", func() {
				_, err := libcni.ConfFromFile("/tmp/nope/not-here")
				Expect(err).To(MatchError(HavePrefix(`error reading /tmp/nope/not-here: open /tmp/nope/not-here`)))
			})
		})
	})

	Describe("LoadConfList", func() {
		var (
			configDir  string
			configList []byte
		)

		BeforeEach(func() {
			var err error
			configDir, err = ioutil.TempDir("", "plugin-conf")
			Expect(err).NotTo(HaveOccurred())

			configList = []byte(`{
  "name": "some-list",
  "cniVersion": "0.2.0",
  "plugins": [
    {
      "type": "host-local",
      "subnet": "10.0.0.1/24"
    },
    {
      "type": "bridge",
      "mtu": 1400
    },
    {
      "type": "port-forwarding",
      "ports": {"20.0.0.1:8080": "80"}
    }
  ]
}`)
			Expect(ioutil.WriteFile(filepath.Join(configDir, "50-whatever.conflist"), configList, 0600)).To(Succeed())
		})

		AfterEach(func() {
			Expect(os.RemoveAll(configDir)).To(Succeed())
		})

		It("finds the network config file for the plugin of the given type", func() {
			netConfigList, err := libcni.LoadConfList(configDir, "some-list")
			Expect(err).NotTo(HaveOccurred())
			Expect(netConfigList).To(Equal(&libcni.NetworkConfigList{
				Name:       "some-list",
				CNIVersion: "0.2.0",
				Plugins: []*libcni.NetworkConfig{
					{
						Network: &types.NetConf{Type: "host-local"},
						Bytes:   []byte(`{"subnet":"10.0.0.1/24","type":"host-local"}`),
					},
					{
						Network: &types.NetConf{Type: "bridge"},
						Bytes:   []byte(`{"mtu":1400,"type":"bridge"}`),
					},
					{
						Network: &types.NetConf{Type: "port-forwarding"},
						Bytes:   []byte(`{"ports":{"20.0.0.1:8080":"80"},"type":"port-forwarding"}`),
					},
				},
				Bytes: configList,
			}))
		})

		Context("when there is a config file with the same name as the list", func() {
			BeforeEach(func() {
				configFile := []byte(`{
					"name": "some-list",
					"cniVersion": "0.2.0",
					"type": "bridge"
				}`)
				Expect(ioutil.WriteFile(filepath.Join(configDir, "49-whatever.conf"), configFile, 0600)).To(Succeed())
			})

			It("Loads the config list first", func() {
				netConfigList, err := libcni.LoadConfList(configDir, "some-list")
				Expect(err).NotTo(HaveOccurred())
				Expect(len(netConfigList.Plugins)).To(Equal(3))
			})

			It("falls back to the config file", func() {
				Expect(os.Remove(filepath.Join(configDir, "50-whatever.conflist"))).To(Succeed())

				netConfigList, err := libcni.LoadConfList(configDir, "some-list")
				Expect(err).NotTo(HaveOccurred())
				Expect(len(netConfigList.Plugins)).To(Equal(1))
				Expect(netConfigList.Plugins[0].Network.Type).To(Equal("bridge"))
			})
		})

		Context("when the config directory does not exist", func() {
			BeforeEach(func() {
				Expect(os.RemoveAll(configDir)).To(Succeed())
			})

			It("returns a useful error", func() {
				_, err := libcni.LoadConfList(configDir, "some-plugin")
				Expect(err).To(MatchError(libcni.NoConfigsFoundError{Dir: configDir}))
			})
		})

		Context("when there is no config for the desired plugin list", func() {
			It("returns a useful error", func() {
				_, err := libcni.LoadConfList(configDir, "some-other-plugin")
				Expect(err).To(MatchError(libcni.NotFoundError{configDir, "some-other-plugin"}))
			})
		})

		Context("when a config file is malformed", func() {
			BeforeEach(func() {
				Expect(ioutil.WriteFile(filepath.Join(configDir, "00-bad.conflist"), []byte(`{`), 0600)).To(Succeed())
			})

			It("returns a useful error", func() {
				_, err := libcni.LoadConfList(configDir, "some-plugin")
				Expect(err).To(MatchError(`error parsing configuration list: unexpected end of JSON input`))
			})
		})

		Context("when the config is in a nested subdir", func() {
			BeforeEach(func() {
				subdir := filepath.Join(configDir, "subdir1", "subdir2")
				Expect(os.MkdirAll(subdir, 0700)).To(Succeed())

				configList = []byte(`{
  "name": "deep",
  "cniVersion": "0.2.0",
  "plugins": [
    {
      "type": "host-local",
      "subnet": "10.0.0.1/24"
    },
  ]
}`)
				Expect(ioutil.WriteFile(filepath.Join(subdir, "90-deep.conflist"), configList, 0600)).To(Succeed())
			})

			It("will not find the config", func() {
				_, err := libcni.LoadConfList(configDir, "deep")
				Expect(err).To(MatchError(HavePrefix("no net configuration with name")))
			})
		})
	})

	Describe("ConfListFromFile", func() {
		Context("when the file cannot be opened", func() {
			It("returns a useful error", func() {
				_, err := libcni.ConfListFromFile("/tmp/nope/not-here")
				Expect(err).To(MatchError(HavePrefix(`error reading /tmp/nope/not-here: open /tmp/nope/not-here`)))
			})
		})
	})

	Describe("InjectConf", func() {
		var testNetConfig *libcni.NetworkConfig

		BeforeEach(func() {
			testNetConfig = &libcni.NetworkConfig{Network: &types.NetConf{Name: "some-plugin"},
				Bytes: []byte(`{ "name": "some-plugin" }`)}
		})

		Context("when function parameters are incorrect", func() {
			It("returns unmarshal error", func() {
				conf := &libcni.NetworkConfig{Network: &types.NetConf{Name: "some-plugin"},
					Bytes: []byte(`{ cc cc cc}`)}

				_, err := libcni.InjectConf(conf, map[string]interface{}{"": nil})
				Expect(err).To(MatchError(HavePrefix(`unmarshal existing network bytes`)))
			})

			It("returns key  error", func() {
				_, err := libcni.InjectConf(testNetConfig, map[string]interface{}{"": nil})
				Expect(err).To(MatchError(HavePrefix(`keys cannot be empty`)))
			})

			It("returns newValue  error", func() {
				_, err := libcni.InjectConf(testNetConfig, map[string]interface{}{"test": nil})
				Expect(err).To(MatchError(HavePrefix(`key 'test' value must not be nil`)))
			})
		})

		Context("when new string value added", func() {
			It("adds the new key & value to the config", func() {
				newPluginConfig := []byte(`{"name":"some-plugin","test":"test"}`)

				resultConfig, err := libcni.InjectConf(testNetConfig, map[string]interface{}{"test": "test"})
				Expect(err).NotTo(HaveOccurred())
				Expect(resultConfig).To(Equal(&libcni.NetworkConfig{
					Network: &types.NetConf{Name: "some-plugin"},
					Bytes:   newPluginConfig,
				}))
			})

			It("adds the new value for exiting key", func() {
				newPluginConfig := []byte(`{"name":"some-plugin","test":"changedValue"}`)

				resultConfig, err := libcni.InjectConf(testNetConfig, map[string]interface{}{"test": "test"})
				Expect(err).NotTo(HaveOccurred())

				resultConfig, err = libcni.InjectConf(resultConfig, map[string]interface{}{"test": "changedValue"})
				Expect(err).NotTo(HaveOccurred())

				Expect(resultConfig).To(Equal(&libcni.NetworkConfig{
					Network: &types.NetConf{Name: "some-plugin"},
					Bytes:   newPluginConfig,
				}))
			})

			It("adds existing key & value", func() {
				newPluginConfig := []byte(`{"name":"some-plugin","test":"test"}`)

				resultConfig, err := libcni.InjectConf(testNetConfig, map[string]interface{}{"test": "test"})
				Expect(err).NotTo(HaveOccurred())

				resultConfig, err = libcni.InjectConf(resultConfig, map[string]interface{}{"test": "test"})
				Expect(err).NotTo(HaveOccurred())

				Expect(resultConfig).To(Equal(&libcni.NetworkConfig{
					Network: &types.NetConf{Name: "some-plugin"},
					Bytes:   newPluginConfig,
				}))
			})

			It("adds sub-fields of NetworkConfig.Network to the config", func() {

				expectedPluginConfig := []byte(`{"dns":{"domain":"local","nameservers":["server1","server2"]},"name":"some-plugin","type":"bridge"}`)
				servers := []string{"server1", "server2"}
				newDNS := &types.DNS{Nameservers: servers, Domain: "local"}

				// inject DNS
				resultConfig, err := libcni.InjectConf(testNetConfig, map[string]interface{}{"dns": newDNS})
				Expect(err).NotTo(HaveOccurred())

				// inject type
				resultConfig, err = libcni.InjectConf(resultConfig, map[string]interface{}{"type": "bridge"})
				Expect(err).NotTo(HaveOccurred())

				Expect(resultConfig).To(Equal(&libcni.NetworkConfig{
					Network: &types.NetConf{Name: "some-plugin", Type: "bridge", DNS: types.DNS{Nameservers: servers, Domain: "local"}},
					Bytes:   expectedPluginConfig,
				}))
			})
		})
	})
})

var _ = Describe("ConfListFromConf", func() {
	var testNetConfig *libcni.NetworkConfig

	BeforeEach(func() {
		pb := []byte(`{"name":"some-plugin","cniVersion":"0.3.1" }`)
		tc, err := libcni.ConfFromBytes(pb)
		Expect(err).NotTo(HaveOccurred())
		testNetConfig = tc
	})

	It("correctly upconverts a NetworkConfig to a NetworkConfigList", func() {
		ncl, err := libcni.ConfListFromConf(testNetConfig)
		Expect(err).NotTo(HaveOccurred())
		bytes := ncl.Bytes

		// null out the json - we don't care about the exact marshalling
		ncl.Bytes = nil
		ncl.Plugins[0].Bytes = nil
		testNetConfig.Bytes = nil

		Expect(ncl).To(Equal(&libcni.NetworkConfigList{
			Name:       "some-plugin",
			CNIVersion: "0.3.1",
			Plugins:    []*libcni.NetworkConfig{testNetConfig},
		}))

		//Test that the json unmarshals to the same data
		ncl2, err := libcni.ConfListFromBytes(bytes)
		Expect(err).NotTo(HaveOccurred())
		ncl2.Bytes = nil
		ncl2.Plugins[0].Bytes = nil

		Expect(ncl2).To(Equal(ncl))
	})

})
