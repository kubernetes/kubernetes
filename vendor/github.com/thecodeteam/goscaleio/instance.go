package goscaleio

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"strings"

	types "github.com/thecodeteam/goscaleio/types/v1"
)

func (client *Client) GetInstance(systemhref string) (systems []*types.System, err error) {

	endpoint := client.SIOEndpoint
	if systemhref == "" {
		endpoint.Path += "/types/System/instances"
	} else {
		endpoint.Path = systemhref
	}

	req := client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", client.Token)
	req.Header.Add("Accept", "application/json;version="+client.configConnect.Version)

	resp, err := client.retryCheckResp(&client.Http, req)
	if err != nil {
		return []*types.System{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if systemhref == "" {
		if err = client.decodeBody(resp, &systems); err != nil {
			return []*types.System{}, fmt.Errorf("error decoding instances response: %s", err)
		}
	} else {
		system := &types.System{}
		if err = client.decodeBody(resp, &system); err != nil {
			return []*types.System{}, fmt.Errorf("error decoding instances response: %s", err)
		}
		systems = append(systems, system)
	}

	// bs, err := ioutil.ReadAll(resp.Body)
	// if err != nil {
	// 	return types.Systems{}, errors.New("error reading body")
	// }

	return systems, nil
}

func (client *Client) GetVolume(volumehref, volumeid, ancestorvolumeid, volumename string, getSnapshots bool) (volumes []*types.Volume, err error) {

	endpoint := client.SIOEndpoint

	if volumename != "" {
		volumeid, err = client.FindVolumeID(volumename)
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
		endpoint.Path = "/api/types/Volume/instances"
	} else {
		endpoint.Path = volumehref
	}

	req := client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", client.Token)
	req.Header.Add("Accept", "application/json;version="+client.configConnect.Version)

	resp, err := client.retryCheckResp(&client.Http, req)
	if err != nil {
		return []*types.Volume{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if volumehref == "" && volumeid == "" {
		if err = client.decodeBody(resp, &volumes); err != nil {
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
		if err = client.decodeBody(resp, &volume); err != nil {
			return []*types.Volume{}, fmt.Errorf("error decoding instances response: %s", err)
		}
		volumes = append(volumes, volume)
	}
	return volumes, nil
}

func (client *Client) FindVolumeID(volumename string) (volumeID string, err error) {

	endpoint := client.SIOEndpoint

	volumeQeryIdByKeyParam := &types.VolumeQeryIdByKeyParam{}
	volumeQeryIdByKeyParam.Name = volumename

	jsonOutput, err := json.Marshal(&volumeQeryIdByKeyParam)
	if err != nil {
		return "", fmt.Errorf("error marshaling: %s", err)
	}
	endpoint.Path = fmt.Sprintf("/api/types/Volume/instances/action/queryIdByKey")

	req := client.NewRequest(map[string]string{}, "POST", endpoint, bytes.NewBufferString(string(jsonOutput)))
	req.SetBasicAuth("", client.Token)
	req.Header.Add("Accept", "application/json;version="+client.configConnect.Version)
	req.Header.Add("Content-Type", "application/json;version="+client.configConnect.Version)

	resp, err := client.retryCheckResp(&client.Http, req)
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

func (client *Client) CreateVolume(volume *types.VolumeParam, storagePoolName string) (volumeResp *types.VolumeResp, err error) {

	endpoint := client.SIOEndpoint

	endpoint.Path = "/api/types/Volume/instances"

	storagePool, err := client.FindStoragePool("", storagePoolName, "")
	if err != nil {
		return nil, err
	}

	volume.StoragePoolID = storagePool.ID
	volume.ProtectionDomainID = storagePool.ProtectionDomainID

	jsonOutput, err := json.Marshal(&volume)
	if err != nil {
		return &types.VolumeResp{}, fmt.Errorf("error marshaling: %s", err)
	}

	req := client.NewRequest(map[string]string{}, "POST", endpoint, bytes.NewBufferString(string(jsonOutput)))
	req.SetBasicAuth("", client.Token)
	req.Header.Add("Accept", "application/json;version="+client.configConnect.Version)
	req.Header.Add("Content-Type", "application/json;version="+client.configConnect.Version)

	resp, err := client.retryCheckResp(&client.Http, req)
	if err != nil {
		return &types.VolumeResp{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if err = client.decodeBody(resp, &volumeResp); err != nil {
		return &types.VolumeResp{}, fmt.Errorf("error decoding volume creation response: %s", err)
	}

	return volumeResp, nil
}

func (client *Client) GetStoragePool(storagepoolhref string) (storagePools []*types.StoragePool, err error) {

	endpoint := client.SIOEndpoint

	if storagepoolhref == "" {
		endpoint.Path = "/api/types/StoragePool/instances"
	} else {
		endpoint.Path = storagepoolhref
	}

	req := client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", client.Token)
	req.Header.Add("Accept", "application/json;version="+client.configConnect.Version)

	resp, err := client.retryCheckResp(&client.Http, req)
	if err != nil {
		return []*types.StoragePool{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if storagepoolhref == "" {
		if err = client.decodeBody(resp, &storagePools); err != nil {
			return []*types.StoragePool{}, fmt.Errorf("error decoding storage pool response: %s", err)
		}
	} else {
		storagePool := &types.StoragePool{}
		if err = client.decodeBody(resp, &storagePool); err != nil {
			return []*types.StoragePool{}, fmt.Errorf("error decoding instances response: %s", err)
		}
		storagePools = append(storagePools, storagePool)
	}
	return storagePools, nil
}

func (client *Client) FindStoragePool(id, name, href string) (storagePool *types.StoragePool, err error) {
	storagePools, err := client.GetStoragePool(href)
	if err != nil {
		return &types.StoragePool{}, fmt.Errorf("Error getting storage pool %s", err)
	}

	for _, storagePool = range storagePools {
		if storagePool.ID == id || storagePool.Name == name || href != "" {
			return storagePool, nil
		}
	}

	return &types.StoragePool{}, errors.New("Couldn't find storage pool")

}
