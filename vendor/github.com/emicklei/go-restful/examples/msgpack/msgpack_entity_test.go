package restPack

import (
	"bytes"
	"errors"
	"log"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"io/ioutil"

	restful "github.com/emicklei/go-restful"
)

func TestMsgPack(t *testing.T) {

	// register msg pack entity
	restful.RegisterEntityAccessor(MIME_MSGPACK, NewEntityAccessorMsgPack())
	type Tool struct {
		Name   string
		Vendor string
	}

	// Write
	httpWriter := httptest.NewRecorder()
	mpack := &Tool{Name: "json", Vendor: "apple"}
	resp := restful.NewResponse(httpWriter)
	resp.SetRequestAccepts("application/x-msgpack,*/*;q=0.8")

	err := resp.WriteEntity(mpack)
	if err != nil {
		t.Errorf("err %v", err)
	}

	// Read
	bodyReader := bytes.NewReader(httpWriter.Body.Bytes())
	httpRequest, _ := http.NewRequest("GET", "/test", bodyReader)
	httpRequest.Header.Set("Content-Type", MIME_MSGPACK)
	request := restful.NewRequest(httpRequest)
	readMsgPack := new(Tool)
	err = request.ReadEntity(&readMsgPack)
	if err != nil {
		t.Errorf("err %v", err)
	}
	if equal := reflect.DeepEqual(mpack, readMsgPack); !equal {
		t.Fatalf("should not be error")
	}
}

func TestWithWebService(t *testing.T) {
	serverURL := "http://127.0.0.1:8090"
	go func() {
		runRestfulMsgPackRouterServer()
	}()
	if err := waitForServerUp(serverURL); err != nil {
		t.Errorf("%v", err)
	}

	// send a post request
	userData := user{Id: "0001", Name: "Tony"}
	msgPackData, err := msgpack.Marshal(userData)
	req, err := http.NewRequest("POST", serverURL+"/test/msgpack", bytes.NewBuffer(msgPackData))
	req.Header.Set("Content-Type", MIME_MSGPACK)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		t.Errorf("unexpected error in sending req: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("unexpected response: %v, expected: %v", resp.StatusCode, http.StatusOK)
	}

	ur := &userResponse{}
	expectMsgPackDocument(t, resp, ur)
	if ur.Status != statusActive {
		t.Fatalf("should not error")
	}
	log.Printf("user response:%v", ur)
}

func expectMsgPackDocument(t *testing.T, r *http.Response, doc interface{}) {
	data, err := ioutil.ReadAll(r.Body)
	defer r.Body.Close()
	if err != nil {
		t.Errorf("ExpectMsgPackDocument: unable to read response body :%v", err)
		return
	}
	// put the body back for re-reads
	r.Body = ioutil.NopCloser(bytes.NewReader(data))

	err = msgpack.Unmarshal(data, doc)
	if err != nil {
		t.Errorf("ExpectMsgPackDocument: unable to unmarshal MsgPack:%v", err)
	}
}

func runRestfulMsgPackRouterServer() {

	container := restful.NewContainer()
	register(container)

	log.Printf("start listening on localhost:8090")
	server := &http.Server{Addr: ":8090", Handler: container}
	log.Fatal(server.ListenAndServe())
}

func waitForServerUp(serverURL string) error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		_, err := http.Get(serverURL + "/")
		if err == nil {
			return nil
		}
	}
	return errors.New("waiting for server timed out")
}

var (
	statusActive = "active"
)

type user struct {
	Id, Name string
}

type userResponse struct {
	Status string
}

func register(container *restful.Container) {
	restful.RegisterEntityAccessor(MIME_MSGPACK, NewEntityAccessorMsgPack())
	ws := new(restful.WebService)
	ws.
		Path("/test").
		Consumes(restful.MIME_JSON, MIME_MSGPACK).
		Produces(restful.MIME_JSON, MIME_MSGPACK)
	// route user api
	ws.Route(ws.POST("/msgpack").
		To(do).
		Reads(user{}).
		Writes(userResponse{}))
	container.Add(ws)
}

func do(request *restful.Request, response *restful.Response) {
	u := &user{}
	err := request.ReadEntity(u)
	if err != nil {
		log.Printf("should be no error, got:%v", err)
	}
	log.Printf("got:%v", u)

	ur := &userResponse{Status: statusActive}

	response.SetRequestAccepts(MIME_MSGPACK)
	response.WriteEntity(ur)
}
