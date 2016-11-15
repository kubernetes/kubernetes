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
	"crypto/md5"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"regexp"
	"strings"
	"syscall"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/volume/libstorage/lstypes"
)

var (
	lsxName            = "lsx-linux"
	lsHeaderInstanceID = "Libstorage-Instanceid"
	lsHeaderLocalDevs  = "Libstorage-Localdevices"
)

type lsHttpClient struct {
	lsUrl        *url.URL
	service      string
	driver       string
	instanceID   string
	localDevices string
	lsxDir       string
	client       *http.Client
	inited       bool
}

func newLsHttpClient(lsServ, lsUrl string) (*lsHttpClient, error) {
	u, err := url.Parse(lsUrl)
	if err != nil {
		return nil, err
	}

	return &lsHttpClient{
		service: lsServ,
		lsUrl:   u,
		client: &http.Client{
			Transport: &http.Transport{
				Dial: (&net.Dialer{
					Timeout:   10 * time.Second,
					KeepAlive: 30 * time.Second,
				}).Dial,
			},
		},
	}, nil
}

func (c *lsHttpClient) init() error {
	if c.inited {
		return nil
	}

	glog.V(4).Infof("libStorage: initializing client")

	lsxFileName := c.getLsxPath()

	// assert exec is needed, if so download it.
	_, err := os.Stat(lsxFileName)
	if err != nil {
		// attempt to get executor file
		if os.IsNotExist(err) {
			glog.V(4).Infof("libStorage: executor binary not found, downloading it.")
			if dlErr := c.dlExec(); dlErr != nil {
				glog.Error("libStorage: failed to download executor:", dlErr)
				return dlErr
			}
		} else {
			// other error, exit
			glog.V(4).Info("libStorage: executor file error:", err)
			return err
		}
	}

	//retrieve service, driver, and instanceID
	if err := c.assertService(); err != nil {
		return err
	}

	c.inited = true
	return nil
}

func (c *lsHttpClient) exec(cmdStr string, args ...string) (string, error) {
	cmd := exec.Command(cmdStr, args...)
	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			errCode := exitErr.Sys().(syscall.WaitStatus).ExitStatus()
			return "", fmt.Errorf("libStorage: executor exits with code %v", errCode)
		} else {
			return "", fmt.Errorf("libStorage: exec %v failed: %v", cmd, err)
		}
	}
	return strings.TrimSpace(string(output)), err
}

// execStat retrieves information about the executor binary
// by issuing GET /executors to retrieve all exec info
func (c *lsHttpClient) execStat() (*lstypes.Executor, error) {
	loc := fmt.Sprintf("%s/executors", c.lsUrl.String())
	req, err := http.NewRequest(http.MethodGet, loc, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.send(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var execMap map[string]*lstypes.Executor
	if err := c.decode(resp.Body, &execMap); err != nil {
		return nil, err
	}

	// look for the linux executor
	for key, exec := range execMap {
		if key == lsxName {
			return exec, nil
		}
	}
	return nil, fmt.Errorf("executor not found")

}

// calcMd5 calculates MD5 sum for provided reader
func calcMd5(source io.Reader) (string, error) {
	md := md5.New()
	if _, err := io.Copy(md, source); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", md.Sum(nil)), nil
}

func (c *lsHttpClient) setExecDir(rootDir string) {
	if rootDir == "" {
		rootDir = "/var/libstorage"
	}
	c.lsxDir = rootDir
}

func (c *lsHttpClient) getLsxPath() string {
	if c.lsxDir == "" {
		return fmt.Sprintf("/var/libstorage/%s", lsxName)
	}
	return path.Join(c.lsxDir, lsxName)
}

// delExec issues /executors/<executor> to download executor binary
func (c *lsHttpClient) dlExec() error {
	lsExec, err := c.execStat()
	if err != nil {
		return err
	}

	loc := fmt.Sprintf("%s/executors/lsx-linux", c.lsUrl.String())
	req, err := http.NewRequest(http.MethodGet, loc, nil)
	if err != nil {
		return err
	}
	glog.V(4).Infof("libStorage: downloading executor from %s", loc)
	resp, err := c.send(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	_, err = os.Stat(c.lsxDir)
	if err != nil {
		if os.IsNotExist(err) {
			os.MkdirAll(c.lsxDir, 0744)
		} else {
			return err
		}
	}

	lsxFile := c.getLsxPath()
	lsxOut, err := os.OpenFile(lsxFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0744)
	if err != nil {
		return err
	}
	_, err = io.Copy(lsxOut, resp.Body)
	if err != nil {
		return err
	}
	// close now so we can calc md5
	if err := lsxOut.Close(); err != nil {
		return err
	}

	// compare md5 val
	lsxOut, err = os.Open(lsxFile)
	if err != nil {
		return err
	}
	defer lsxOut.Close()
	md5Sum, err := calcMd5(lsxOut)
	if md5Sum != lsExec.MD5Sum {
		return fmt.Errorf("md5sum executor mismatched")
	}

	glog.V(4).Infof("libStorage: executor downloaded and saved at at %v ", lsxFile)

	return nil
}

func (c *lsHttpClient) send(req *http.Request) (*http.Response, error) {
	req.Header.Add("User-Agent", "k8s.io/libstorage")
	resp, err := c.client.Do(req)
	if err != nil {
		return resp, err
	}

	switch {
	case resp.StatusCode < 300:
		return resp, nil
	default:
		return resp, c.statCodeError(resp.StatusCode)
	}
}

func (c *lsHttpClient) statCodeError(code int) error {
	return fmt.Errorf("unexpected http status code %d", code)
}

func (c *lsHttpClient) decode(source io.Reader, sink interface{}) error {
	dec := json.NewDecoder(source)
	if err := dec.Decode(&sink); err != nil {
		return err
	}
	return nil
}

// IID gets and store the machine's instance id
func (c *lsHttpClient) IID() (string, error) {
	if err := c.init(); err != nil {
		return "", err
	}
	return c.instanceID, nil
}

// rawIID runs cmd "lsx-linux <executor-driver> instanceID"
// This must be called post c.init().
func (c *lsHttpClient) rawIID() (string, error) {
	cmd := c.getLsxPath()
	params := []string{c.driver, "instanceID"}
	out, err := c.exec(cmd, params...)
	if err != nil {
		return "", err
	}

	return out, nil
}

// localDevs gets and store local device names.
// Should be called post c.init()
func (c *lsHttpClient) LocalDevs() (string, error) {
	// build cmd string "lsx-linux <service> instanceID"
	cmd := c.getLsxPath()
	params := []string{c.service, "localDevices", "1"}
	out, err := c.exec(cmd, params...)
	if err != nil {
		return "", err
	}
	c.localDevices = out
	return c.localDevices, nil
}

// nextDev retrieve the next device name from executor
// Should be invoked post c.init()
func (c *lsHttpClient) nextDev() (string, error) {
	// build cmd string "lsx-linux <service> nextDevice"
	cmd := c.getLsxPath()
	params := []string{c.service, "nextDevice"}
	out, err := c.exec(cmd, params...)
	if err != nil {
		return "", err
	}

	return out, nil
}

// assertDriver ensures proper driver name is retrieved and stored.
// It issues a /services/<servicename> HTTP call.
func (c *lsHttpClient) assertDriver() error {
	loc := fmt.Sprintf("%s/services/%s", c.lsUrl.String(), c.service)
	req, err := http.NewRequest(http.MethodGet, loc, nil)
	if err != nil {
		return err
	}
	resp, err := c.send(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	var serv lstypes.Service
	if err := c.decode(resp.Body, &serv); err != nil {
		return err
	}
	if serv.Driver.Name != "" {
		c.driver = serv.Driver.Name
	} else {
		c.driver = serv.Name
	}

	return nil
}

// assertService ensures service instanceid is retrieved and stored.
// To do this it issues a /services/<servicename>?instance HTTP call.
// This must be called after c.assertDriver().
func (c *lsHttpClient) assertService() error {
	if err := c.assertDriver(); err != nil {
		return err
	}

	rawiid, err := c.rawIID()
	if err != nil {
		return err
	}

	loc := fmt.Sprintf("%s/services/%s?instance", c.lsUrl.String(), c.service)
	req, err := http.NewRequest(http.MethodGet, loc, nil)
	if err != nil {
		return err
	}
	req.Header.Add("Libstorage-Instanceid", rawiid)
	resp, err := c.send(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	var serv lstypes.Service
	if err := c.decode(resp.Body, &serv); err != nil {
		return err
	}
	c.service = serv.Name
	c.driver = serv.Driver.Name
	c.instanceID = serv.Instance.InstanceID.ID
	return nil
}

func (c *lsHttpClient) addHeaders(req *http.Request) {
	req.Header.Add(
		lsHeaderInstanceID,
		fmt.Sprintf("%s=%s", c.driver, c.instanceID),
	)

	devs, _ := c.LocalDevs()

	req.Header.Add(
		lsHeaderLocalDevs,
		devs,
	)
}

// volumes issues /volumes/<service> to retrieve libstorage volumes.
// If attachments is true, it issues /volues/<service>?attachements
func (c *lsHttpClient) Volumes(attachments bool) ([]*lstypes.Volume, error) {
	if err := c.init(); err != nil {
		return nil, err
	}
	loc := fmt.Sprintf("%s/volumes/%s", c.lsUrl.String(), c.service)
	if attachments {
		loc = fmt.Sprintf("%s?attachments=true", loc)
	}
	req, err := http.NewRequest(http.MethodGet, loc, nil)
	c.addHeaders(req)
	if err != nil {
		return nil, err
	}

	resp, err := c.send(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var volMap map[string]*lstypes.Volume
	if err := c.decode(resp.Body, &volMap); err != nil {
		return nil, err
	}

	var vols []*lstypes.Volume
	for _, value := range volMap {
		vols = append(vols, value)
	}
	return vols, nil
}

// FindVolume filters c.Volumes() by name
func (c *lsHttpClient) FindVolume(name string) (*lstypes.Volume, error) {
	vols, err := c.Volumes(false)
	if err != nil {
		return nil, err
	}
	for _, vol := range vols {
		if vol.Name == name {
			return c.Volume(vol.ID)
		}
	}
	return nil, errors.New("volume not found")
}

// volume issues /volumes/<service-name>/<vol-id> to retrieve
// a volume by its id.
func (c *lsHttpClient) Volume(id string) (*lstypes.Volume, error) {
	if id == "" {
		return nil, fmt.Errorf("volume not found")
	}
	if err := c.init(); err != nil {
		return nil, err
	}
	loc := fmt.Sprintf(
		"%s/volumes/%s/%s?attachments=true",
		c.lsUrl.String(), c.service, id,
	)
	req, err := http.NewRequest(http.MethodGet, loc, nil)
	if err != nil {
		return nil, err
	}

	c.addHeaders(req)
	resp, err := c.send(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var vol lstypes.Volume
	if err := c.decode(resp.Body, &vol); err != nil {
		return nil, err
	}

	return &vol, nil
}

// createVolume issues a POST /volumes/<service-name> to create a volume
// where the body {"name":"<name>", "size":<size>}
func (c *lsHttpClient) CreateVolume(name string, size int64) (*lstypes.Volume, error) {
	if name == "" {
		return nil, fmt.Errorf("volume name missing")
	}

	if err := c.init(); err != nil {
		return nil, err
	}

	loc := fmt.Sprintf("%s/volumes/%s", c.lsUrl.String(), c.service)
	body := fmt.Sprintf(`{"name":"%s", "size":%d}`, name, size)
	req, err := http.NewRequest(http.MethodPost, loc, strings.NewReader(body))
	if err != nil {
		return nil, err
	}
	c.addHeaders(req)
	resp, err := c.send(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var vol *lstypes.Volume
	if err := c.decode(resp.Body, &vol); err != nil {
		return nil, err
	}

	return vol, nil
}

// attachVolume issues a POST /volumes/<service-name>/<vol-id>?attach to
// attach a volume with body {"nextDeviceName":"<name>"}
func (c *lsHttpClient) AttachVolume(id string) (string, error) {
	if id == "" {
		return "", fmt.Errorf("volume id missing")
	}

	if err := c.init(); err != nil {
		return "", err
	}

	nextDev, err := c.nextDev()
	if err != nil {
		return "", err
	}

	loc := fmt.Sprintf(
		"%s/volumes/%s/%s?attach",
		c.lsUrl.String(),
		c.service,
		id,
	)
	body := fmt.Sprintf(`{"nextDeviceName":"%s"}`, nextDev)
	req, err := http.NewRequest(http.MethodPost, loc, strings.NewReader(body))
	c.addHeaders(req)
	if err != nil {
		return "", err
	}

	resp, err := c.send(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	type attachResponse struct {
		Volume *lstypes.Volume `json:"volume"`
		Token  string          `json:"attachToken"`
	}

	var attach attachResponse
	if err := c.decode(resp.Body, &attach); err != nil {
		return "", err
	}
	return attach.Token, nil
}

// detachVolume issues a POST /volumes/<service-name>/<vol-id>?detach to
// attach a volume with body {"force":false}
func (c *lsHttpClient) DetachVolume(id string) (*lstypes.Volume, error) {
	if id == "" {
		return nil, errors.New("volume id  missing")
	}
	if err := c.init(); err != nil {
		return nil, err
	}
	vol, err := c.Volume(id)
	if err != nil {
		return nil, err
	}
	glog.V(4).Infof("libStorage: detaching volume %s with id %v", vol.Name, vol.ID)

	loc := fmt.Sprintf(
		"%s/volumes/%s/%s?detach",
		c.lsUrl.String(),
		c.service,
		vol.ID,
	)

	body := `{"force":true}`
	req, err := http.NewRequest(http.MethodPost, loc, strings.NewReader(body))
	if err != nil {
		return nil, err
	}

	c.addHeaders(req)
	resp, err := c.send(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var detachedVol *lstypes.Volume
	if err := c.decode(resp.Body, &detachedVol); err != nil {
		return nil, err
	}

	glog.V(4).Infof("libStorage: volume %s detached successfully", vol.Name)
	return detachedVol, nil
}

// deleteVolume issues a DELETE /volumes/<service-name>/<vol-id> to
// remove a volume from the storage system.
func (c *lsHttpClient) DeleteVolume(id string) error {
	if id == "" {
		return fmt.Errorf("volume id missing")
	}
	if err := c.init(); err != nil {
		return err
	}

	glog.V(4).Infof("libStorage: deleting volume with id %s", id)

	vol, err := c.Volume(id)
	if err != nil {
		return err
	}

	loc := fmt.Sprintf(
		"%s/volumes/%s/%s",
		c.lsUrl.String(),
		c.service,
		vol.ID,
	)

	req, err := http.NewRequest(http.MethodDelete, loc, nil)
	if err != nil {
		return err
	}
	_, err = c.send(req)
	if err != nil {
		glog.Errorf("libStorage: deleting volume %s failed: %v", vol.Name, err)
		return err
	}
	glog.V(4).Infof("ligStorage: volume %s deleted successfully", vol.Name)
	return nil
}

// mapDevs converts local device strings to a map.
// According to libstorage API the devs should be presented as
// driver=<volid>::<dev-name>[,<volid>::<dev-name>,...]
func (m *lsHttpClient) mapDevs(devs string) (map[string]string, error) {
	devExp := regexp.MustCompile(`\S+::\S+`)
	parts := strings.Split(devs, "=")
	if len(parts) != 2 {
		return nil, fmt.Errorf("error parsing local device string")
	}
	devParts := strings.Split(strings.TrimSpace(parts[1]), ",")
	if len(devParts) < 1 {
		return nil, fmt.Errorf("error parsing device names")
	}

	result := make(map[string]string)
	for _, devMap := range devParts {
		if !devExp.MatchString(devMap) {
			return nil, fmt.Errorf("error paring device map")
		}
		devPair := strings.Split(devMap, "::")
		result[devPair[0]] = devPair[1]
	}
	return result, nil
}

//sets up a timer to wait for an attached device to appear in the instance's list.
func (c *lsHttpClient) WaitForAttachedDevice(token string) (string, error) {
	if token == "" {
		return "", fmt.Errorf("invalid attach token")
	}

	// wait for attach.Token to show up in local device list
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	timer := time.NewTimer(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			devs, err := c.LocalDevs()

			if err != nil {
				return "", err
			}
			if strings.Contains(devs, token) {
				devMap, err := c.mapDevs(devs)
				if err != nil {
					return "", err
				}
				return devMap[token], nil
			}
		case <-timer.C:
			return "", fmt.Errorf("volume attach timeout")
		}
	}
}

// waitForDetachedDevice waits for device to be detached
func (c *lsHttpClient) WaitForDetachedDevice(token string) error {
	if token == "" {
		return fmt.Errorf("invalid detach token")
	}

	// wait for attach.Token to show up in local device list
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	timer := time.NewTimer(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			devs, err := c.LocalDevs()
			if err != nil {
				return err
			}

			if !strings.Contains(devs, token) {
				return nil
			}
		case <-timer.C:
			return fmt.Errorf("volume detach timeout")
		}
	}
}
