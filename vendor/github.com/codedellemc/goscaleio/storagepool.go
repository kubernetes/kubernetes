package goscaleio

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"

	types "github.com/codedellemc/goscaleio/types/v1"
)

type StoragePool struct {
	StoragePool *types.StoragePool
	client      *Client
}

func NewStoragePool(client *Client) *StoragePool {
	return &StoragePool{
		StoragePool: new(types.StoragePool),
		client:      client,
	}
}

func NewStoragePoolEx(client *Client, pool *types.StoragePool) *StoragePool {
	return &StoragePool{
		StoragePool: pool,
		client:      client,
	}
}

func (protectionDomain *ProtectionDomain) CreateStoragePool(name string) (string, error) {
	endpoint := protectionDomain.client.SIOEndpoint

	storagePoolParam := &types.StoragePoolParam{}
	storagePoolParam.Name = name
	storagePoolParam.ProtectionDomainID = protectionDomain.ProtectionDomain.ID

	jsonOutput, err := json.Marshal(&storagePoolParam)
	if err != nil {
		return "", fmt.Errorf("error marshaling: %s", err)
	}
	endpoint.Path = fmt.Sprintf("/api/types/StoragePool/instances")

	req := protectionDomain.client.NewRequest(map[string]string{}, "POST", endpoint, bytes.NewBufferString(string(jsonOutput)))
	req.SetBasicAuth("", protectionDomain.client.Token)
	req.Header.Add("Accept", "application/json;version="+protectionDomain.client.configConnect.Version)
	req.Header.Add("Content-Type", "application/json;version="+protectionDomain.client.configConnect.Version)

	resp, err := protectionDomain.client.retryCheckResp(&protectionDomain.client.Http, req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	bs, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", errors.New("error reading body")
	}

	var sp types.StoragePoolResp
	err = json.Unmarshal(bs, &sp)
	if err != nil {
		return "", err
	}

	return sp.ID, nil
}

func (protectionDomain *ProtectionDomain) GetStoragePool(storagepoolhref string) (storagePools []*types.StoragePool, err error) {

	endpoint := protectionDomain.client.SIOEndpoint

	if storagepoolhref == "" {
		link, err := GetLink(protectionDomain.ProtectionDomain.Links, "/api/ProtectionDomain/relationship/StoragePool")
		if err != nil {
			return []*types.StoragePool{}, errors.New("Error: problem finding link")
		}
		endpoint.Path = link.HREF
	} else {
		endpoint.Path = storagepoolhref
	}

	req := protectionDomain.client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", protectionDomain.client.Token)
	req.Header.Add("Accept", "application/json;version="+protectionDomain.client.configConnect.Version)

	resp, err := protectionDomain.client.retryCheckResp(&protectionDomain.client.Http, req)
	if err != nil {
		return []*types.StoragePool{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if storagepoolhref == "" {
		if err = protectionDomain.client.decodeBody(resp, &storagePools); err != nil {
			return []*types.StoragePool{}, fmt.Errorf("error decoding storage pool response: %s", err)
		}
	} else {
		storagePool := &types.StoragePool{}
		if err = protectionDomain.client.decodeBody(resp, &storagePool); err != nil {
			return []*types.StoragePool{}, fmt.Errorf("error decoding instances response: %s", err)
		}
		storagePools = append(storagePools, storagePool)
	}
	return storagePools, nil
}

func (protectionDomain *ProtectionDomain) FindStoragePool(id, name, href string) (storagePool *types.StoragePool, err error) {
	storagePools, err := protectionDomain.GetStoragePool(href)
	if err != nil {
		return &types.StoragePool{}, fmt.Errorf("Error getting protection domains %s", err)
	}

	for _, storagePool = range storagePools {
		if storagePool.ID == id || storagePool.Name == name || href != "" {
			return storagePool, nil
		}
	}

	return &types.StoragePool{}, errors.New("Couldn't find protection domain")

}

func (storagePool *StoragePool) GetStatistics() (statistics *types.Statistics, err error) {
	link, err := GetLink(storagePool.StoragePool.Links, "/api/StoragePool/relationship/Statistics")
	if err != nil {
		return &types.Statistics{}, errors.New("Error: problem finding link")
	}

	endpoint := storagePool.client.SIOEndpoint
	endpoint.Path = link.HREF

	req := storagePool.client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", storagePool.client.Token)
	req.Header.Add("Accept", "application/json;version="+storagePool.client.configConnect.Version)

	resp, err := storagePool.client.retryCheckResp(&storagePool.client.Http, req)
	if err != nil {
		return &types.Statistics{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if err = storagePool.client.decodeBody(resp, &statistics); err != nil {
		return &types.Statistics{}, fmt.Errorf("error decoding instances response: %s", err)
	}

	return statistics, nil
}
