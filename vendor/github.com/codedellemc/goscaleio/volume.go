package goscaleio

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	types "github.com/codedellemc/goscaleio/types/v1"
)

type SdcMappedVolume struct {
	MdmID     string
	VolumeID  string
	SdcDevice string
	// Mounted   bool
	// MountPath bool
	// Mapped    bool
}

type Volume struct {
	Volume *types.Volume
	client *Client
}

func NewVolume(client *Client) *Volume {
	return &Volume{
		Volume: new(types.Volume),
		client: client,
	}
}

func (storagePool *StoragePool) GetVolume(volumehref, volumeid, ancestorvolumeid, volumename string, getSnapshots bool) (volumes []*types.Volume, err error) {

	endpoint := storagePool.client.SIOEndpoint

	if volumename != "" {
		volumeid, err = storagePool.FindVolumeID(volumename)
		if err != nil && err.Error() == "Not found" {
			return nil, nil
		}
		if err != nil {
			return []*types.Volume{}, fmt.Errorf("Error: problem finding volume: %s", err)
		}
	}

	if volumeid != "" {
		endpoint.Path = fmt.Sprintf("/api/instances/Volume::%s", volumeid)
	} else if volumehref == "" {
		link, err := GetLink(storagePool.StoragePool.Links, "/api/StoragePool/relationship/Volume")
		if err != nil {
			return []*types.Volume{}, errors.New("Error: problem finding link")
		}
		endpoint.Path = link.HREF
	} else {
		endpoint.Path = volumehref
	}

	req := storagePool.client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", storagePool.client.Token)
	req.Header.Add("Accept", "application/json;version="+storagePool.client.configConnect.Version)

	resp, err := storagePool.client.retryCheckResp(&storagePool.client.Http, req)
	if err != nil {
		return []*types.Volume{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if volumehref == "" && volumeid == "" {
		if err = storagePool.client.decodeBody(resp, &volumes); err != nil {
			return []*types.Volume{}, fmt.Errorf("error decoding storage pool response: %s", err)
		}
		var volumesNew []*types.Volume
		for _, volume := range volumes {
			if (!getSnapshots && volume.AncestorVolumeID == ancestorvolumeid) || (getSnapshots && volume.AncestorVolumeID != "") {
				volumesNew = append(volumesNew, volume)
			}
		}
		volumes = volumesNew
	} else {
		volume := &types.Volume{}
		if err = storagePool.client.decodeBody(resp, &volume); err != nil {
			return []*types.Volume{}, fmt.Errorf("error decoding instances response: %s", err)
		}
		volumes = append(volumes, volume)
	}
	return volumes, nil
}

func (storagePool *StoragePool) FindVolumeID(volumename string) (volumeID string, err error) {

	endpoint := storagePool.client.SIOEndpoint

	volumeQeryIdByKeyParam := &types.VolumeQeryIdByKeyParam{}
	volumeQeryIdByKeyParam.Name = volumename

	jsonOutput, err := json.Marshal(&volumeQeryIdByKeyParam)
	if err != nil {
		return "", fmt.Errorf("error marshaling: %s", err)
	}
	endpoint.Path = fmt.Sprintf("/api/types/Volume/instances/action/queryIdByKey")

	req := storagePool.client.NewRequest(map[string]string{}, "POST", endpoint, bytes.NewBufferString(string(jsonOutput)))
	req.SetBasicAuth("", storagePool.client.Token)
	req.Header.Add("Accept", "application/json;version="+storagePool.client.configConnect.Version)
	req.Header.Add("Content-Type", "application/json;version="+storagePool.client.configConnect.Version)

	resp, err := storagePool.client.retryCheckResp(&storagePool.client.Http, req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	bs, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", errors.New("error reading body")
	}

	volumeID = string(bs)

	volumeID = strings.TrimRight(volumeID, `"`)
	volumeID = strings.TrimLeft(volumeID, `"`)

	return volumeID, nil
}

func GetLocalVolumeMap() (mappedVolumes []*SdcMappedVolume, err error) {

	// get sdc kernel guid
	// /bin/emc/scaleio/drv_cfg --query_guid
	// sdcKernelGuid := "271bad82-08ee-44f2-a2b1-7e2787c27be1"

	mappedVolumesMap := make(map[string]*SdcMappedVolume)

	out, err := exec.Command("/opt/emc/scaleio/sdc/bin/drv_cfg", "--query_vols").Output()
	if err != nil {
		return []*SdcMappedVolume{}, fmt.Errorf("Error querying volumes: ", err)
	}

	result := string(out)
	lines := strings.Split(result, "\n")

	for _, line := range lines {
		split := strings.Split(line, " ")
		if split[0] == "VOL-ID" {
			mappedVolume := &SdcMappedVolume{MdmID: split[3], VolumeID: split[1]}
			mdmVolumeID := fmt.Sprintf("%s-%s", mappedVolume.MdmID, mappedVolume.VolumeID)
			mappedVolumesMap[mdmVolumeID] = mappedVolume
		}
	}

	diskIDPath := "/dev/disk/by-id"
	files, _ := ioutil.ReadDir(diskIDPath)
	r, _ := regexp.Compile(`^emc-vol-\w*-\w*$`)
	for _, f := range files {
		matched := r.MatchString(f.Name())
		if matched {
			mdmVolumeID := strings.Replace(f.Name(), "emc-vol-", "", 1)
			devPath, _ := filepath.EvalSymlinks(fmt.Sprintf("%s/%s", diskIDPath, f.Name()))
			if _, ok := mappedVolumesMap[mdmVolumeID]; ok {
				mappedVolumesMap[mdmVolumeID].SdcDevice = devPath
			}
		}
	}

	keys := make([]string, 0, len(mappedVolumesMap))
	for key := range mappedVolumesMap {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		mappedVolumes = append(mappedVolumes, mappedVolumesMap[key])
	}

	return mappedVolumes, nil
}

func (storagePool *StoragePool) CreateVolume(volume *types.VolumeParam) (volumeResp *types.VolumeResp, err error) {

	endpoint := storagePool.client.SIOEndpoint

	endpoint.Path = "/api/types/Volume/instances"

	volume.StoragePoolID = storagePool.StoragePool.ID
	volume.ProtectionDomainID = storagePool.StoragePool.ProtectionDomainID

	jsonOutput, err := json.Marshal(&volume)
	if err != nil {
		return &types.VolumeResp{}, fmt.Errorf("error marshaling: %s", err)
	}

	req := storagePool.client.NewRequest(map[string]string{}, "POST", endpoint, bytes.NewBufferString(string(jsonOutput)))
	req.SetBasicAuth("", storagePool.client.Token)
	req.Header.Add("Accept", "application/json;version="+storagePool.client.configConnect.Version)
	req.Header.Add("Content-Type", "application/json;version="+storagePool.client.configConnect.Version)

	resp, err := storagePool.client.retryCheckResp(&storagePool.client.Http, req)
	if err != nil {
		return &types.VolumeResp{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if err = storagePool.client.decodeBody(resp, &volumeResp); err != nil {
		return &types.VolumeResp{}, fmt.Errorf("error decoding volume creation response: %s", err)
	}

	return volumeResp, nil
}

func (volume *Volume) GetVTree() (vtree *types.VTree, err error) {

	endpoint := volume.client.SIOEndpoint

	link, err := GetLink(volume.Volume.Links, "/api/parent/relationship/vtreeId")
	if err != nil {
		return &types.VTree{}, errors.New("Error: problem finding link")
	}
	endpoint.Path = link.HREF

	req := volume.client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", volume.client.Token)
	req.Header.Add("Accept", "application/json;version="+volume.client.configConnect.Version)

	resp, err := volume.client.retryCheckResp(&volume.client.Http, req)
	if err != nil {
		return &types.VTree{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if err = volume.client.decodeBody(resp, &vtree); err != nil {
		return &types.VTree{}, fmt.Errorf("error decoding vtree response: %s", err)
	}
	return vtree, nil
}

func (volume *Volume) RemoveVolume(removeMode string) (err error) {

	endpoint := volume.client.SIOEndpoint

	link, err := GetLink(volume.Volume.Links, "self")
	if err != nil {
		return errors.New("Error: problem finding link")
	}
	endpoint.Path = fmt.Sprintf("%v/action/removeVolume", link.HREF)

	if removeMode == "" {
		removeMode = "ONLY_ME"
	}
	removeVolumeParam := &types.RemoveVolumeParam{
		RemoveMode: removeMode,
	}

	jsonOutput, err := json.Marshal(&removeVolumeParam)
	if err != nil {
		return fmt.Errorf("error marshaling: %s", err)
	}

	req := volume.client.NewRequest(map[string]string{}, "POST", endpoint, bytes.NewBufferString(string(jsonOutput)))

	req.SetBasicAuth("", volume.client.Token)
	req.Header.Add("Accept", "application/json;version="+volume.client.configConnect.Version)
	req.Header.Add("Content-Type", "application/json;version="+volume.client.configConnect.Version)

	resp, err := volume.client.retryCheckResp(&volume.client.Http, req)
	if err != nil {
		return fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	return nil
}
