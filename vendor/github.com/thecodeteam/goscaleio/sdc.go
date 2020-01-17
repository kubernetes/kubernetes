package goscaleio

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os/exec"
	"reflect"
	"strings"

	types "github.com/thecodeteam/goscaleio/types/v1"
)

type Sdc struct {
	Sdc    *types.Sdc
	client *Client
}

func NewSdc(client *Client, sdc *types.Sdc) *Sdc {
	return &Sdc{
		Sdc:    sdc,
		client: client,
	}
}

func (system *System) GetSdc() (sdcs []types.Sdc, err error) {
	endpoint := system.client.SIOEndpoint
	endpoint.Path = fmt.Sprintf("/api/instances/System::%v/relationships/Sdc", system.System.ID)

	req := system.client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", system.client.Token)
	req.Header.Add("Accept", "application/json;version="+system.client.configConnect.Version)

	resp, err := system.client.retryCheckResp(&system.client.Http, req)
	if err != nil {
		return []types.Sdc{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if err = system.client.decodeBody(resp, &sdcs); err != nil {
		return []types.Sdc{}, fmt.Errorf("error decoding instances response: %s", err)
	}

	// bs, err := ioutil.ReadAll(resp.Body)
	// if err != nil {
	// 	return []types.Sdc{}, errors.New("error reading body")
	// }
	//
	// fmt.Println(string(bs))
	// log.Fatalf("here")
	// return []types.Sdc{}, nil
	return sdcs, nil
}

func (system *System) FindSdc(field, value string) (sdc *Sdc, err error) {
	sdcs, err := system.GetSdc()
	if err != nil {
		return &Sdc{}, nil
	}

	for _, sdc := range sdcs {
		valueOf := reflect.ValueOf(sdc)
		switch {
		case reflect.Indirect(valueOf).FieldByName(field).String() == value:
			return NewSdc(system.client, &sdc), nil
		}
	}

	return &Sdc{}, errors.New("Couldn't find SDC")
}

func (sdc *Sdc) GetStatistics() (statistics *types.Statistics, err error) {
	endpoint := sdc.client.SIOEndpoint

	link, err := GetLink(sdc.Sdc.Links, "/api/Sdc/relationship/Statistics")
	if err != nil {
		return &types.Statistics{}, errors.New("Error: problem finding link")
	}
	endpoint.Path = link.HREF

	req := sdc.client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", sdc.client.Token)
	req.Header.Add("Accept", "application/json;version="+sdc.client.configConnect.Version)

	resp, err := sdc.client.retryCheckResp(&sdc.client.Http, req)
	if err != nil {
		return &types.Statistics{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if err = sdc.client.decodeBody(resp, &statistics); err != nil {
		return &types.Statistics{}, fmt.Errorf("error decoding instances response: %s", err)
	}

	return statistics, nil
}

func (sdc *Sdc) GetVolume() (volumes []*types.Volume, err error) {
	endpoint := sdc.client.SIOEndpoint

	link, err := GetLink(sdc.Sdc.Links, "/api/Sdc/relationship/Volume")
	if err != nil {
		return []*types.Volume{}, errors.New("Error: problem finding link")
	}
	endpoint.Path = link.HREF

	req := sdc.client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", sdc.client.Token)
	req.Header.Add("Accept", "application/json;version="+sdc.client.configConnect.Version)

	resp, err := sdc.client.retryCheckResp(&sdc.client.Http, req)
	if err != nil {
		return []*types.Volume{}, fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	if err = sdc.client.decodeBody(resp, &volumes); err != nil {
		return []*types.Volume{}, fmt.Errorf("error decoding instances response: %s", err)
	}

	return volumes, nil
}

func GetSdcLocalGUID() (sdcGUID string, err error) {

	// get sdc kernel guid
	// /bin/emc/scaleio/drv_cfg --query_guid
	// sdcKernelGuid := "271bad82-08ee-44f2-a2b1-7e2787c27be1"

	out, err := exec.Command("/opt/emc/scaleio/sdc/bin/drv_cfg", "--query_guid").Output()
	if err != nil {
		return "", fmt.Errorf("GetSdcLocalGUID: query vols failed: %v", err)
	}

	sdcGUID = strings.Replace(string(out), "\n", "", -1)

	return sdcGUID, nil
}

func (volume *Volume) MapVolumeSdc(mapVolumeSdcParam *types.MapVolumeSdcParam) (err error) {
	endpoint := volume.client.SIOEndpoint

	endpoint.Path = fmt.Sprintf("/api/instances/Volume::%s/action/addMappedSdc", volume.Volume.ID)

	jsonOutput, err := json.Marshal(&mapVolumeSdcParam)
	if err != nil {
		log.Fatalf("error marshaling: %s", err)
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

func (volume *Volume) UnmapVolumeSdc(unmapVolumeSdcParam *types.UnmapVolumeSdcParam) (err error) {
	endpoint := volume.client.SIOEndpoint

	endpoint.Path = fmt.Sprintf("/api/instances/Volume::%s/action/removeMappedSdc", volume.Volume.ID)

	jsonOutput, err := json.Marshal(&unmapVolumeSdcParam)
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
