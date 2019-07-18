package goscaleio

import (
	"fmt"

	types "github.com/thecodeteam/goscaleio/types/v1"
)

func (system *System) GetScsiInitiator() (scsiInitiators []types.ScsiInitiator, err error) {
	endpoint := system.client.SIOEndpoint
	endpoint.Path = fmt.Sprintf("/api/instances/System::%v/relationships/ScsiInitiator", system.System.ID)

	req := system.client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", system.client.Token)
	req.Header.Add("Accept", "application/json;version="+system.client.configConnect.Version)

	resp, err := system.client.retryCheckResp(&system.client.Http, req)
	if err != nil {
		return []types.ScsiInitiator{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if err = system.client.decodeBody(resp, &scsiInitiators); err != nil {
		return []types.ScsiInitiator{}, fmt.Errorf("error decoding instances response: %s", err)
	}

	// bs, err := ioutil.ReadAll(resp.Body)
	// if err != nil {
	// 	return types.ScsiInitiator{}, errors.New("error reading body")
	// }
	//
	// log.Fatalf("here")
	// return types.ScsiInitiator{}, nil
	return scsiInitiators, nil
}
