package goscaleio

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"reflect"

	types "github.com/thecodeteam/goscaleio/types/v1"
)

type Device struct {
	Device *types.Device
	client *Client
}

func NewDevice(client *Client) *Device {
	return &Device{
		Device: new(types.Device),
		client: client,
	}
}

func NewDeviceEx(client *Client, device *types.Device) *Device {
	return &Device{
		Device: device,
		client: client,
	}
}

func (storagePool *StoragePool) AttachDevice(path string, sdsID string) (string, error) {
	endpoint := storagePool.client.SIOEndpoint

	deviceParam := &types.DeviceParam{}
	deviceParam.Name = path
	deviceParam.DeviceCurrentPathname = path
	deviceParam.StoragePoolID = storagePool.StoragePool.ID
	deviceParam.SdsID = sdsID
	deviceParam.TestMode = "testAndActivate"

	jsonOutput, err := json.Marshal(&deviceParam)
	if err != nil {
		return "", fmt.Errorf("error marshaling: %s", err)
	}
	endpoint.Path = fmt.Sprintf("/api/types/Device/instances")

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

	var dev types.DeviceResp
	err = json.Unmarshal(bs, &dev)
	if err != nil {
		return "", err
	}

	return dev.ID, nil
}

func (storagePool *StoragePool) GetDevice() (devices []types.Device, err error) {
	endpoint := storagePool.client.SIOEndpoint
	endpoint.Path = fmt.Sprintf("/api/instances/StoragePool::%v/relationships/Device", storagePool.StoragePool.ID)

	req := storagePool.client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", storagePool.client.Token)
	req.Header.Add("Accept", "application/json;version="+storagePool.client.configConnect.Version)

	resp, err := storagePool.client.retryCheckResp(&storagePool.client.Http, req)
	if err != nil {
		return []types.Device{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if err = storagePool.client.decodeBody(resp, &devices); err != nil {
		return []types.Device{}, fmt.Errorf("error decoding instances response: %s", err)
	}

	return devices, nil
}

func (storagePool *StoragePool) FindDevice(field, value string) (device *types.Device, err error) {
	devices, err := storagePool.GetDevice()
	if err != nil {
		return &types.Device{}, nil
	}

	for _, device := range devices {
		valueOf := reflect.ValueOf(device)
		switch {
		case reflect.Indirect(valueOf).FieldByName(field).String() == value:
			return &device, nil
		}
	}

	return &types.Device{}, errors.New("Couldn't find DEV")
}
