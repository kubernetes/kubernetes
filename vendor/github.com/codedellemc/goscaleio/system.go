package goscaleio

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	types "github.com/codedellemc/goscaleio/types/v1"
)

type System struct {
	System *types.System
	client *Client
}

func NewSystem(client *Client) *System {
	return &System{
		System: new(types.System),
		client: client,
	}
}

func (client *Client) FindSystem(instanceID, name, href string) (*System, error) {
	systems, err := client.GetInstance(href)
	if err != nil {
		return &System{}, fmt.Errorf("err: problem getting instances: %s", err)
	}

	for _, system := range systems {
		if system.ID == instanceID || system.Name == name || href != "" {
			outSystem := NewSystem(client)
			outSystem.System = system
			return outSystem, nil
		}
	}
	return &System{}, fmt.Errorf("err: systemid or systemname not found")
}

func (system *System) GetStatistics() (statistics *types.Statistics, err error) {
	endpoint := system.client.SIOEndpoint
	// endpoint.Path = fmt.Sprintf("/api/instances/System::%v/relationships/Statistics", system.System.ID)

	link, err := GetLink(system.System.Links, "/api/System/relationship/Statistics")
	if err != nil {
		return &types.Statistics{}, errors.New("Error: problem finding link")
	}

	endpoint.Path = link.HREF

	req := system.client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", system.client.Token)
	req.Header.Add("Accept", "application/json;version="+system.client.configConnect.Version)

	resp, err := system.client.retryCheckResp(&system.client.Http, req)
	if err != nil {
		return &types.Statistics{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if err = system.client.decodeBody(resp, &statistics); err != nil {
		return &types.Statistics{}, fmt.Errorf("error decoding instances response: %s", err)
	}

	// bs, err := ioutil.ReadAll(resp.Body)
	// if err != nil {
	// 	return errors.New("error reading body")
	// }
	//
	// fmt.Println(string(bs))
	return statistics, nil
}

func (system *System) CreateSnapshotConsistencyGroup(snapshotVolumesParam *types.SnapshotVolumesParam) (snapshotVolumesResp *types.SnapshotVolumesResp, err error) {
	endpoint := system.client.SIOEndpoint

	link, err := GetLink(system.System.Links, "self")
	if err != nil {
		return &types.SnapshotVolumesResp{}, errors.New("Error: problem finding link")
	}

	endpoint.Path = fmt.Sprintf("%v/action/snapshotVolumes", link.HREF)

	jsonOutput, err := json.Marshal(&snapshotVolumesParam)
	if err != nil {
		return &types.SnapshotVolumesResp{}, fmt.Errorf("error marshaling: %s", err)
	}

	req := system.client.NewRequest(map[string]string{}, "POST", endpoint, bytes.NewBufferString(string(jsonOutput)))
	req.SetBasicAuth("", system.client.Token)
	req.Header.Add("Accept", "application/json;version="+system.client.configConnect.Version)
	req.Header.Add("Content-Type", "application/json;version="+system.client.configConnect.Version)

	resp, err := system.client.retryCheckResp(&system.client.Http, req)
	if err != nil {
		return &types.SnapshotVolumesResp{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if err = system.client.decodeBody(resp, &snapshotVolumesResp); err != nil {
		return &types.SnapshotVolumesResp{}, fmt.Errorf("error decoding snapshotvolumes response: %s", err)
	}

	return snapshotVolumesResp, nil

}
