package kubelet

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sync"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type fakeKubelet struct {
	infoFunc func(name string) (string, error)
	idFunc   func(name string) (string, error)
}

func (fk *fakeKubelet) GetContainerInfo(name string) (string, error) {
	return fk.infoFunc(name)
}

func (fk *fakeKubelet) GetContainerID(name string) (string, error) {
	return fk.idFunc(name)
}

// If we made everything distribute a list of ContainerManifests, we could just use
// channelReader.
type channelReaderSingle struct {
	list []api.ContainerManifest
	wg   sync.WaitGroup
}

func startReadingSingle(channel <-chan api.ContainerManifest) *channelReaderSingle {
	cr := &channelReaderSingle{}
	cr.wg.Add(1)
	go func() {
		for {
			manifest, ok := <-channel
			if !ok {
				break
			}
			cr.list = append(cr.list, manifest)
		}
		cr.wg.Done()
	}()
	return cr
}

func (cr *channelReaderSingle) GetList() []api.ContainerManifest {
	cr.wg.Wait()
	return cr.list
}

type serverTestFramework struct {
	updateChan      chan api.ContainerManifest
	updateReader    *channelReaderSingle
	serverUnderTest *KubeletServer
	fakeKubelet     *fakeKubelet
	testHttpServer  *httptest.Server
}

func makeServerTest() *serverTestFramework {
	fw := &serverTestFramework{
		updateChan: make(chan api.ContainerManifest),
	}
	fw.updateReader = startReadingSingle(fw.updateChan)
	fw.fakeKubelet = &fakeKubelet{}
	fw.serverUnderTest = &KubeletServer{
		Kubelet:       fw.fakeKubelet,
		UpdateChannel: fw.updateChan,
	}
	fw.testHttpServer = httptest.NewServer(fw.serverUnderTest)
	return fw
}

func readResp(resp *http.Response) (string, error) {
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	return string(body), err
}

func TestContainer(t *testing.T) {
	fw := makeServerTest()
	expected := api.ContainerManifest{Id: "test_manifest"}
	body := bytes.NewBuffer([]byte(util.MakeJSONString(expected)))
	resp, err := http.Post(fw.testHttpServer.URL+"/container", "application/json", body)
	if err != nil {
		t.Errorf("Post returned: %v", err)
	}
	resp.Body.Close()
	close(fw.updateChan)
	received := fw.updateReader.GetList()
	if len(received) != 1 {
		t.Errorf("Expected 1 manifest, but got %v", len(received))
	}
	if !reflect.DeepEqual(expected, received[0]) {
		t.Errorf("Expected %#v, but got %#v", expected, received[0])
	}
}

func TestContainerInfo(t *testing.T) {
	fw := makeServerTest()
	expected := "good container info string"
	fw.fakeKubelet.idFunc = func(name string) (string, error) {
		if name == "goodcontainer" {
			return name, nil
		}
		return "", fmt.Errorf("bad container")
	}
	fw.fakeKubelet.infoFunc = func(name string) (string, error) {
		if name == "goodcontainer" {
			return expected, nil
		}
		return "", fmt.Errorf("bad container")
	}
	resp, err := http.Get(fw.testHttpServer.URL + "/containerInfo?container=goodcontainer")
	if err != nil {
		t.Errorf("Got error GETing: %v", err)
	}
	got, err := readResp(resp)
	if err != nil {
		t.Errorf("Error reading body: %v", err)
	}
	if got != expected {
		t.Errorf("Expected: '%v', got: '%v'", expected, got)
	}
}
