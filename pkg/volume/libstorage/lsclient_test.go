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

package libstorage

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/volume/libstorage/lstypes"
)

func tstSetupExec(t *testing.T, rootDir string) {
	tstClearExec(t, rootDir)
	cmd := "/bin/echo"
	cmdFile, err := os.Open(cmd)
	if err != nil {
		t.Skipf("Skipping, unable to access /bin/echo")
	}
	defer cmdFile.Close()
	lsxFileName := path.Join(rootDir, lsxName)
	if err := os.Mkdir(rootDir, 0755); err != nil {
		t.Fatal(err)
	}
	lsxFile, err := os.Create(lsxFileName)
	if err != nil {
		t.Fatal(err)
	}

	if err := os.Chmod(path.Join(lsxFileName), 0755); err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(lsxFile, cmdFile); err != nil {
		t.Fatal(err)
	}
	if err := lsxFile.Close(); err != nil {
		t.Fatal(err)
	}
}

func tstClearExec(t *testing.T, rootDir string) {
	if err := os.RemoveAll(rootDir); err != nil {
		if os.IsNotExist(err) {
			return
		} else {
			t.Fatal(err)
		}
	}
}

func TestHttpClientNew(t *testing.T) {
	c, err := newLsHttpClient("test", "http://local/")
	if err != nil {
		t.Fatal(err)
	}
	if c.lsUrl.String() != "http://local/" {
		t.Fail()
	}
	if c.service != "test" {
		t.Fail()
	}
}

func TestHttpClientSend(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(resp http.ResponseWriter, req *http.Request) {
			defer req.Body.Close()
			var buf bytes.Buffer
			io.Copy(&buf, req.Body)
			if buf.String() != "hello" {
				t.Errorf("Expecting hello, got %s", buf.String())
			}
			resp.WriteHeader(http.StatusOK)
			resp.Write([]byte("goodbye"))
		},
	))
	defer server.Close()
	svrUrl := server.URL

	c, err := newLsHttpClient("test", svrUrl)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest(
		"GET",
		c.lsUrl.String(),
		bytes.NewReader([]byte("hello")),
	)
	if err != nil {
		t.Error(err)
	}
	resp, err := c.send(req)
	if err != nil {
		t.Error(err)
	}
	defer resp.Body.Close()
	var buf bytes.Buffer
	io.Copy(&buf, resp.Body)
	if buf.String() != "goodbye" {
		t.Errorf("Expecting goodbye, got %s", buf.String())
	}
}

// Server serves /bin/echo binary as substitute for a real ls-executor bin.
// Issues "echo <service> instanceID" and "echo <service> localDevices 1"
// commands.  Test the stored output.  A real ls-executor would produce
// real value at standard output.
func TestHttpClientInit(t *testing.T) {
	_, err := os.Stat("/bin/echo")
	if err != nil {
		t.Skipf("skipping, unable to open /bin/echo")
	}

	server := httptest.NewServer(http.HandlerFunc(
		func(resp http.ResponseWriter, req *http.Request) {
			switch req.URL.RequestURI() {
			default:
				t.Error("Unexpected executor download path:", req.URL.Path)
				resp.WriteHeader(http.StatusBadRequest)
				return
			case "/executors":
				binEcho, err := os.Open("/bin/echo")
				if err != nil {
					resp.WriteHeader(500)
					t.Fatal(err)
				}
				defer binEcho.Close()

				md5sum, err := calcMd5(binEcho)
				if err != nil {
					resp.WriteHeader(500)
					t.Fatal(err)
				}
				execs := map[string]*lstypes.Executor{
					"lsx-linux": {
						Name:   "lsx-linux",
						MD5Sum: md5sum,
					},
				}
				if err := json.NewEncoder(resp).Encode(&execs); err != nil {
					resp.WriteHeader(500)
					t.Fatal(err)
				}
			case "/executors/lsx-linux":
				binEcho, err := os.Open("/bin/echo")
				if err != nil {
					resp.WriteHeader(500)
					t.Fatal(err)
				}
				defer binEcho.Close()

				if _, err := io.Copy(resp, binEcho); err != nil {
					resp.WriteHeader(500)
					t.Fatal(err)
				}
			case "/services/test-service":
				resp.Write([]byte(jsonService))
			case "/services/test-service?instance":
				if req.Header.Get("Libstorage-Instanceid") == "" {
					resp.WriteHeader(500)
					t.Fatal(err)
				}
				resp.Write([]byte(jsonServiceInstance))
			}
		},
	))
	defer server.Close()
	svrUrl := server.URL

	c, err := newLsHttpClient("test-service", svrUrl)
	c.setExecDir("test-exec-dir")
	defer tstClearExec(t, "test-exec-dir")
	if err := c.init(); err != nil {
		t.Fatal(err)
	}
	// ensure service info is set
	if c.service != "test-service" {
		t.Errorf("unexpected service name %s", c.service)
	}
	if c.driver != "virtualbox" {
		t.Errorf("unexpected driver name %s", c.driver)
	}
	if c.instanceID != "ca72be20-3cba-4bf9-842f-806729142d95" {
		t.Errorf("unexpected client instanceID %s", c.instanceID)
	}

	// check downloaded binary
	lsxFileName := path.Join("test-exec-dir", lsxName)
	if _, err := os.Stat(lsxFileName); err != nil {
		t.Error(err)
	}
}

// Uses /bin/echo as mock cmd
func TestHttpClientExec(t *testing.T) {
	cmd := "/bin/echo"
	_, err := os.Stat(cmd)
	if err != nil {
		t.Skipf("Command %s not found, skipping test", cmd)
	}
	c, _ := newLsHttpClient("test", "http://test/")
	expected := "hello-goodbye"
	out, err := c.exec(cmd, expected)
	if err != nil {
		t.Fatalf("Failed to exec %v", err)
	}
	if out != expected {
		t.Errorf("Expected '%s', got '%s'", expected, out)
	}
}

// Test use /bin/echo as mock cmd
func TestHttpClientIID(t *testing.T) {
	tstSetupExec(t, "test-exec-dir")
	defer tstClearExec(t, "test-exec-dir")

	c, err := newLsHttpClient("test-service", "http://test/")
	if err != nil {
		t.Fatal(err)
	}
	c.inited = true
	c.setExecDir("test-exec-dir")
	c.driver = "virtualbox"
	iid, err := c.rawIID()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(iid, "virtualbox") {
		t.Error("exec does not produce expected output:", iid)
	}
}

// Test use /bin/echo as mock cmd
func TestHttpClientLocalDevices(t *testing.T) {
	tstSetupExec(t, "test-exec-dir")
	defer tstClearExec(t, "test-exec-dir")
	c, err := newLsHttpClient("test-service", "http://test/")
	if err != nil {
		t.Fatal(err)
	}
	c.inited = true
	c.setExecDir("test-exec-dir")
	if _, err = c.LocalDevs(); err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(c.localDevices, "test-service") {
		t.Error("exec does not produce expected output:", c.instanceID)
	}
}

func TestHttpClientVolumes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(resp http.ResponseWriter, req *http.Request) {
			switch req.URL.RequestURI() {
			case "/volumes/test-service":
				resp.Write([]byte(jsonVolumesDetached))
			case "/volumes/test-service?attachments=true":
				resp.Write([]byte(jsonVolumesAttached))
			default:
				resp.WriteHeader(500)
				t.Error("unable to handle ", req.URL.RequestURI())
			}

		},
	))
	defer server.Close()
	svrUrl := server.URL

	c, err := newLsHttpClient("test-service", svrUrl)
	if err != nil {
		t.Fatal(err)
	}
	c.inited = true
	vols, err := c.Volumes(false)
	if err != nil {
		t.Fatal(err)
	}
	if len(vols) != 1 {
		t.Fatal("expected 1 volume, got ", len(vols))
	}
	if vols[0].ID != "a63c7bf6-2231-40ab-929d-490e127326d9" {
		t.Error("unxpected lsVol.id value:", vols[0].ID)
	}
	if vols[0].Size != 41 {
		t.Errorf("expected lsVol.Size = 41, got %d", vols[0].Size)
	}

	vols, err = c.Volumes(true)
	if err != nil {
		t.Fatal(err)
	}
	if len(vols) != 4 {
		t.Fatalf("expected 4 lsVols with attachments, got %d", len(vols))
	}
}

func TestHttpClientFindVolume(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(resp http.ResponseWriter, req *http.Request) {
			switch req.URL.RequestURI() {
			case "/volumes/test-service":
				resp.Write([]byte(jsonVolumesAttached))
			case "/volumes/test-service/c3932aee-ee52-44fe-84a6-fcb426724ca5?attachments=true":
				resp.Write([]byte(jsonVolume))
			default:
				resp.WriteHeader(500)
				t.Error("unable to handle ", req.URL.RequestURI())
			}

		},
	))
	defer server.Close()
	svrUrl := server.URL

	c, err := newLsHttpClient("test-service", svrUrl)
	if err != nil {
		t.Fatal(err)
	}
	c.inited = true
	vol, err := c.FindVolume("vol-0001")
	if err != nil {
		t.Fatal(err)
	}
	if vol.ID != "c3932aee-ee52-44fe-84a6-fcb426724ca5" {
		t.Error("unexpected sVol.id value:", vol.ID)
	}
}

func TestHttpClientVolume(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(resp http.ResponseWriter, req *http.Request) {
			switch req.URL.RequestURI() {
			case "/volumes/test-service/c3932aee-ee52-44fe-84a6-fcb426724ca5?attachments=true":
				resp.Write([]byte(jsonVolume))
			default:
				resp.WriteHeader(500)
				t.Error("unable to handle ", req.URL.RequestURI())
			}
		},
	))
	defer server.Close()
	svrUrl := server.URL

	c, err := newLsHttpClient("test-service", svrUrl)
	if err != nil {
		t.Fatal(err)
	}
	c.inited = true
	vol, err := c.Volume("c3932aee-ee52-44fe-84a6-fcb426724ca5")
	if err != nil {
		t.Fatal(err)
	}
	if vol.ID != "c3932aee-ee52-44fe-84a6-fcb426724ca5" {
		t.Error("unxpected lsVol.ID value:", vol.ID)
	}
	if vol.Size != 10 {
		t.Errorf("expected lsVol.Size = 10, got %d", vol.Size)
	}

	instId := vol.Attachments[0].InstanceID.ID
	if instId != "da4e52c9-79e7-423f-bc8c-509c022a98e0" {
		t.Errorf("expecting instance id da4e52c9-79e7-423f-bc8c-509c022a98e0, got %s", instId)
	}
}

func TestHttpClientCreateVolume(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(resp http.ResponseWriter, req *http.Request) {
			switch req.URL.RequestURI() {
			case "/volumes/test-service":
				if req.Method != http.MethodPost {
					resp.WriteHeader(http.StatusBadRequest)
					t.Errorf("expected POST method, got %s", req.Method)
					return
				}
				if req.ContentLength == 0 {
					resp.WriteHeader(http.StatusBadRequest)
					t.Error("request body missing")
					return
				}
				body, _ := ioutil.ReadAll(req.Body)
				if string(body) != `{"name":"vol-0001", "size":10}` {
					resp.WriteHeader(http.StatusBadRequest)
					t.Errorf("unexpected request body %s", string(body))
				}
				resp.Write([]byte(jsonVolume))
			default:
				resp.WriteHeader(500)
				t.Error("unable to handle ", req.URL.Path)
			}
		},
	))
	defer server.Close()
	svrUrl := server.URL

	c, err := newLsHttpClient("test-service", svrUrl)
	if err != nil {
		t.Fatal(err)
	}
	c.inited = true
	vol, err := c.CreateVolume("vol-0001", 10)
	if err != nil {
		t.Fatal(err)
	}
	if vol.ID != "c3932aee-ee52-44fe-84a6-fcb426724ca5" {
		t.Error("unxpected lsVol.ID value:", vol.ID)
	}
	if vol.Size != 10 {
		t.Errorf("expected lsVol.Size = 10, got %d", vol.Size)
	}

	instId := vol.Attachments[0].InstanceID.ID
	if instId != "da4e52c9-79e7-423f-bc8c-509c022a98e0" {
		t.Errorf("expecting instance id da4e52c9-79e7-423f-bc8c-509c022a98e0, got %s", instId)
	}
}

func TestHttpClientAttachVolume(t *testing.T) {
	tstSetupExec(t, "test-exec-dir")
	defer tstClearExec(t, "test-exec-dir")

	server := httptest.NewServer(http.HandlerFunc(
		func(resp http.ResponseWriter, req *http.Request) {
			switch req.URL.RequestURI() {
			case "/volumes/test-service/vol-0001?attach":
				if req.Method != http.MethodPost {
					resp.WriteHeader(http.StatusBadRequest)
					t.Errorf("expected POST method, got %s", req.Method)
					return
				}
				if req.ContentLength == 0 {
					resp.WriteHeader(http.StatusBadRequest)
					t.Error("request body missing")
					return
				}
				body, _ := ioutil.ReadAll(req.Body)
				// remember, we are useing /bin/echo as mock exec
				if string(body) != `{"nextDeviceName":"test-service nextDevice"}` {
					resp.WriteHeader(http.StatusBadRequest)
					t.Errorf("unexpected request body %s", strings.TrimSpace(string(body)))
				}
				resp.Write([]byte(jsonAttachVolume))
			default:
				resp.WriteHeader(500)
				t.Error("unable to handle ", req.URL.Path)
			}
		},
	))
	defer server.Close()
	svrUrl := server.URL

	c, err := newLsHttpClient("test-service", svrUrl)
	if err != nil {
		t.Fatal(err)
	}
	c.inited = true

	//	_, err = c.attachVolume("vol-0001")
	//	if err != nil {
	//		t.Fatal(err)
	//	}
}

func TestHttpClientDetachVolume(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(resp http.ResponseWriter, req *http.Request) {
			switch req.URL.RequestURI() {
			case "/volumes/test-service":
				resp.Write([]byte(jsonVolumesAttached))
			case "/volumes/test-service/c3932aee-ee52-44fe-84a6-fcb426724ca5?attachments=true":
				resp.Write([]byte(jsonVolume))
			case "/volumes/test-service/c3932aee-ee52-44fe-84a6-fcb426724ca5?detach":
				if req.Method != http.MethodPost {
					resp.WriteHeader(http.StatusBadRequest)
					t.Errorf("expected POST method, got %s", req.Method)
					return
				}
				resp.Write([]byte(jsonVolume))

			default:
				resp.WriteHeader(500)
				t.Error("unable to handle ", req.URL.RequestURI())
			}
		},
	))
	defer server.Close()
	svrUrl := server.URL

	c, err := newLsHttpClient("test-service", svrUrl)
	if err != nil {
		t.Fatal(err)
	}
	c.inited = true
	_, err = c.DetachVolume("c3932aee-ee52-44fe-84a6-fcb426724ca5")
	if err != nil {
		t.Fatal(err)
	}
}

func TestHttpClientDeleteVolume(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(resp http.ResponseWriter, req *http.Request) {
			switch req.URL.RequestURI() {
			case "/volumes/test-service":
				resp.Write([]byte(jsonVolumesAttached))
			case "/volumes/test-service/c3932aee-ee52-44fe-84a6-fcb426724ca5?attachments=true":
				resp.Write([]byte(jsonVolume))
			case "/volumes/test-service/c3932aee-ee52-44fe-84a6-fcb426724ca5":
				if req.Method != http.MethodDelete {
					resp.WriteHeader(http.StatusBadRequest)
					t.Errorf("expected DELETE method, got %s", req.Method)
					return
				}
			default:
				resp.WriteHeader(500)
				t.Error("unable to handle ", req.URL.Path)
			}
		},
	))
	defer server.Close()
	svrUrl := server.URL

	c, err := newLsHttpClient("test-service", svrUrl)
	if err != nil {
		t.Fatal(err)
	}
	c.inited = true
	err = c.DeleteVolume("c3932aee-ee52-44fe-84a6-fcb426724ca5")
	if err != nil {
		t.Fatal(err)
	}
}

var jsonService = `
{
  "name": "virtualbox",
  "driver": {
    "name": "virtualbox",
    "type": "block"
  }
}`

var jsonServiceInstance = `
{
  "name": "test-service",
  "instance": {
    "instanceID": {
      "id": "ca72be20-3cba-4bf9-842f-806729142d95",
      "driver": "virtualbox"
    },
    "providerName": ""
  },
  "driver": {
    "name": "virtualbox",
    "type": "block"
  }
}
`

var jsonVolume = `
{
    "attachments": [
      {
        "deviceName": "",
        "instanceID": {
          "id": "da4e52c9-79e7-423f-bc8c-509c022a98e0",
          "driver": "virtualbox"
        },
        "status": "/Users/tests/VirtualBox Volumes/vol-0001",
        "volumeID": "c3932aee-ee52-44fe-84a6-fcb426724ca5"
      }
    ],
    "name": "vol-0001",
    "size": 10,
    "status": "/Users/tests/VirtualBox Volumes/vol-0001",
    "id": "c3932aee-ee52-44fe-84a6-fcb426724ca5",
    "type": ""
}`

var jsonAttachVolume = `
{
  "attachToken":"token-000",
  "volume":{ 
     "attachments": [
      {
        "deviceName": "",
        "instanceID": {
          "id": "inst-0001",
          "driver": "virtualbox"
        },
        "status": "/Users/tests/VirtualBox/Haiku-Os/Haiku-Os.vdi",
        "volumeID": "vol-0001"
      }
    ],
    "name": "Haiku-Os.vdi",
    "size": 10,
    "status": "/Users/tests/VirtualBox VMs/Haiku-Os/Haiku-Os.vdi",
    "id": "vol-0001",
    "type": ""
  }
}`

var jsonVolumesDetached = `
{
  "a63c7bf6-2231-40ab-929d-490e127326d9": {
    "name": "test-vol-000",
    "size": 41,
    "status": "/Users/tests/Volumes/test-vol.vdi",
    "id": "a63c7bf6-2231-40ab-929d-490e127326d9",
    "type": ""
  }
}  
`

var jsonVolumesAttached = `
{
  "030806d6-67da-42c7-987b-5211627f5d63": {
    "attachments": [
      {
        "deviceName": "",
        "instanceID": {
          "id": "d171d77c-3d80-4dba-91dd-7b33245ddd44",
          "driver": "virtualbox"
        },
        "status": "/Users/tests/VirtualBox VMs/Haiku-Os/Haiku-Os.vdi",
        "volumeID": "030806d6-67da-42c7-987b-5211627f5d63"
      }
    ],
    "name": "Haiku-Os.vdi",
    "size": 10,
    "status": "/Users/tests/VirtualBox VMs/Haiku-Os/Haiku-Os.vdi",
    "id": "030806d6-67da-42c7-987b-5211627f5d63",
    "type": ""
  },
  "46ac8aee-8488-49c9-99c8-2fc72bd373b1": {
    "attachments": [
      {
        "deviceName": "",
        "instanceID": {
          "id": "98f57009-b556-44a5-93ea-3bf95d8a7b13",
          "driver": "virtualbox"
        },
        "status": "/Users/tests/VirtualBox VMs/ScaleIO_CentOS_MDM0/sds1-0.vdi",
        "volumeID": "46ac8aee-8488-49c9-99c8-2fc72bd373b1"
      }
    ],
    "name": "sds1-0.vdi",
    "size": 500,
    "status": "/Users/tests/VirtualBox VMs/ScaleIO_CentOS_MDM0/sds1-0.vdi",
    "id": "46ac8aee-8488-49c9-99c8-2fc72bd373b1",
    "type": ""
  },
  "8891a83d-eb8f-4af1-8246-796a89ce69a7": {
    "attachments": [
      {
        "deviceName": "",
        "instanceID": {
          "id": "7fcd5e1f-5b45-4eff-b4d0-a15e3799831f",
          "driver": "virtualbox"
        },
        "status": "/Users/tests/VirtualBox VMs/ScaleIO_CentOS_MDM1/sds2-0.vdi",
        "volumeID": "8891a83d-eb8f-4af1-8246-796a89ce69a7"
      }
    ],
    "name": "sds2-0.vdi",
    "size": 500,
    "status": "/Users/tests/VirtualBox VMs/ScaleIO_CentOS_MDM1/sds2-0.vdi",
    "id": "8891a83d-eb8f-4af1-8246-796a89ce69a7",
    "type": ""
  },
  "c3932aee-ee52-44fe-84a6-fcb426724ca5": {
    "attachments": [
      {
        "deviceName": "",
        "instanceID": {
          "id": "da4e52c9-79e7-423f-bc8c-509c022a98e0",
          "driver": "virtualbox"
        },
        "status": "/Users/tests/VirtualBox Volumes/vol-0001",
        "volumeID": "c3932aee-ee52-44fe-84a6-fcb426724ca5"
      }
    ],
    "name": "vol-0001",
    "size": 1,
    "status": "/Users/tests/VirtualBox Volumes/vol-0001",
    "id": "c3932aee-ee52-44fe-84a6-fcb426724ca5",
    "type": ""
  }
}
`
