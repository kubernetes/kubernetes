// +build !solaris

package registry

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/docker/distribution/reference"
	registrytypes "github.com/docker/docker/api/types/registry"
	"github.com/gorilla/mux"

	"github.com/sirupsen/logrus"
)

var (
	testHTTPServer  *httptest.Server
	testHTTPSServer *httptest.Server
	testLayers      = map[string]map[string]string{
		"77dbf71da1d00e3fbddc480176eac8994025630c6590d11cfc8fe1209c2a1d20": {
			"json": `{"id":"77dbf71da1d00e3fbddc480176eac8994025630c6590d11cfc8fe1209c2a1d20",
				"comment":"test base image","created":"2013-03-23T12:53:11.10432-07:00",
				"container_config":{"Hostname":"","User":"","Memory":0,"MemorySwap":0,
				"CpuShares":0,"AttachStdin":false,"AttachStdout":false,"AttachStderr":false,
				"Tty":false,"OpenStdin":false,"StdinOnce":false,
				"Env":null,"Cmd":null,"Dns":null,"Image":"","Volumes":null,
				"VolumesFrom":"","Entrypoint":null},"Size":424242}`,
			"checksum_simple": "sha256:1ac330d56e05eef6d438586545ceff7550d3bdcb6b19961f12c5ba714ee1bb37",
			"checksum_tarsum": "tarsum+sha256:4409a0685741ca86d38df878ed6f8cbba4c99de5dc73cd71aef04be3bb70be7c",
			"ancestry":        `["77dbf71da1d00e3fbddc480176eac8994025630c6590d11cfc8fe1209c2a1d20"]`,
			"layer": string([]byte{
				0x1f, 0x8b, 0x08, 0x08, 0x0e, 0xb0, 0xee, 0x51, 0x02, 0x03, 0x6c, 0x61, 0x79, 0x65,
				0x72, 0x2e, 0x74, 0x61, 0x72, 0x00, 0xed, 0xd2, 0x31, 0x0e, 0xc2, 0x30, 0x0c, 0x05,
				0x50, 0xcf, 0x9c, 0xc2, 0x27, 0x48, 0xed, 0x38, 0x4e, 0xce, 0x13, 0x44, 0x2b, 0x66,
				0x62, 0x24, 0x8e, 0x4f, 0xa0, 0x15, 0x63, 0xb6, 0x20, 0x21, 0xfc, 0x96, 0xbf, 0x78,
				0xb0, 0xf5, 0x1d, 0x16, 0x98, 0x8e, 0x88, 0x8a, 0x2a, 0xbe, 0x33, 0xef, 0x49, 0x31,
				0xed, 0x79, 0x40, 0x8e, 0x5c, 0x44, 0x85, 0x88, 0x33, 0x12, 0x73, 0x2c, 0x02, 0xa8,
				0xf0, 0x05, 0xf7, 0x66, 0xf5, 0xd6, 0x57, 0x69, 0xd7, 0x7a, 0x19, 0xcd, 0xf5, 0xb1,
				0x6d, 0x1b, 0x1f, 0xf9, 0xba, 0xe3, 0x93, 0x3f, 0x22, 0x2c, 0xb6, 0x36, 0x0b, 0xf6,
				0xb0, 0xa9, 0xfd, 0xe7, 0x94, 0x46, 0xfd, 0xeb, 0xd1, 0x7f, 0x2c, 0xc4, 0xd2, 0xfb,
				0x97, 0xfe, 0x02, 0x80, 0xe4, 0xfd, 0x4f, 0x77, 0xae, 0x6d, 0x3d, 0x81, 0x73, 0xce,
				0xb9, 0x7f, 0xf3, 0x04, 0x41, 0xc1, 0xab, 0xc6, 0x00, 0x0a, 0x00, 0x00,
			}),
		},
		"42d718c941f5c532ac049bf0b0ab53f0062f09a03afd4aa4a02c098e46032b9d": {
			"json": `{"id":"42d718c941f5c532ac049bf0b0ab53f0062f09a03afd4aa4a02c098e46032b9d",
				"parent":"77dbf71da1d00e3fbddc480176eac8994025630c6590d11cfc8fe1209c2a1d20",
				"comment":"test base image","created":"2013-03-23T12:55:11.10432-07:00",
				"container_config":{"Hostname":"","User":"","Memory":0,"MemorySwap":0,
				"CpuShares":0,"AttachStdin":false,"AttachStdout":false,"AttachStderr":false,
				"Tty":false,"OpenStdin":false,"StdinOnce":false,
				"Env":null,"Cmd":null,"Dns":null,"Image":"","Volumes":null,
				"VolumesFrom":"","Entrypoint":null},"Size":424242}`,
			"checksum_simple": "sha256:bea7bf2e4bacd479344b737328db47b18880d09096e6674165533aa994f5e9f2",
			"checksum_tarsum": "tarsum+sha256:68fdb56fb364f074eec2c9b3f85ca175329c4dcabc4a6a452b7272aa613a07a2",
			"ancestry": `["42d718c941f5c532ac049bf0b0ab53f0062f09a03afd4aa4a02c098e46032b9d",
				"77dbf71da1d00e3fbddc480176eac8994025630c6590d11cfc8fe1209c2a1d20"]`,
			"layer": string([]byte{
				0x1f, 0x8b, 0x08, 0x08, 0xbd, 0xb3, 0xee, 0x51, 0x02, 0x03, 0x6c, 0x61, 0x79, 0x65,
				0x72, 0x2e, 0x74, 0x61, 0x72, 0x00, 0xed, 0xd1, 0x31, 0x0e, 0xc2, 0x30, 0x0c, 0x05,
				0x50, 0xcf, 0x9c, 0xc2, 0x27, 0x48, 0x9d, 0x38, 0x8e, 0xcf, 0x53, 0x51, 0xaa, 0x56,
				0xea, 0x44, 0x82, 0xc4, 0xf1, 0x09, 0xb4, 0xea, 0x98, 0x2d, 0x48, 0x08, 0xbf, 0xe5,
				0x2f, 0x1e, 0xfc, 0xf5, 0xdd, 0x00, 0xdd, 0x11, 0x91, 0x8a, 0xe0, 0x27, 0xd3, 0x9e,
				0x14, 0xe2, 0x9e, 0x07, 0xf4, 0xc1, 0x2b, 0x0b, 0xfb, 0xa4, 0x82, 0xe4, 0x3d, 0x93,
				0x02, 0x0a, 0x7c, 0xc1, 0x23, 0x97, 0xf1, 0x5e, 0x5f, 0xc9, 0xcb, 0x38, 0xb5, 0xee,
				0xea, 0xd9, 0x3c, 0xb7, 0x4b, 0xbe, 0x7b, 0x9c, 0xf9, 0x23, 0xdc, 0x50, 0x6e, 0xb9,
				0xb8, 0xf2, 0x2c, 0x5d, 0xf7, 0x4f, 0x31, 0xb6, 0xf6, 0x4f, 0xc7, 0xfe, 0x41, 0x55,
				0x63, 0xdd, 0x9f, 0x89, 0x09, 0x90, 0x6c, 0xff, 0xee, 0xae, 0xcb, 0xba, 0x4d, 0x17,
				0x30, 0xc6, 0x18, 0xf3, 0x67, 0x5e, 0xc1, 0xed, 0x21, 0x5d, 0x00, 0x0a, 0x00, 0x00,
			}),
		},
	}
	testRepositories = map[string]map[string]string{
		"foo42/bar": {
			"latest": "42d718c941f5c532ac049bf0b0ab53f0062f09a03afd4aa4a02c098e46032b9d",
			"test":   "42d718c941f5c532ac049bf0b0ab53f0062f09a03afd4aa4a02c098e46032b9d",
		},
	}
	mockHosts = map[string][]net.IP{
		"":            {net.ParseIP("0.0.0.0")},
		"localhost":   {net.ParseIP("127.0.0.1"), net.ParseIP("::1")},
		"example.com": {net.ParseIP("42.42.42.42")},
		"other.com":   {net.ParseIP("43.43.43.43")},
	}
)

func init() {
	r := mux.NewRouter()

	// /v1/
	r.HandleFunc("/v1/_ping", handlerGetPing).Methods("GET")
	r.HandleFunc("/v1/images/{image_id:[^/]+}/{action:json|layer|ancestry}", handlerGetImage).Methods("GET")
	r.HandleFunc("/v1/images/{image_id:[^/]+}/{action:json|layer|checksum}", handlerPutImage).Methods("PUT")
	r.HandleFunc("/v1/repositories/{repository:.+}/tags", handlerGetDeleteTags).Methods("GET", "DELETE")
	r.HandleFunc("/v1/repositories/{repository:.+}/tags/{tag:.+}", handlerGetTag).Methods("GET")
	r.HandleFunc("/v1/repositories/{repository:.+}/tags/{tag:.+}", handlerPutTag).Methods("PUT")
	r.HandleFunc("/v1/users{null:.*}", handlerUsers).Methods("GET", "POST", "PUT")
	r.HandleFunc("/v1/repositories/{repository:.+}{action:/images|/}", handlerImages).Methods("GET", "PUT", "DELETE")
	r.HandleFunc("/v1/repositories/{repository:.+}/auth", handlerAuth).Methods("PUT")
	r.HandleFunc("/v1/search", handlerSearch).Methods("GET")

	// /v2/
	r.HandleFunc("/v2/version", handlerGetPing).Methods("GET")

	testHTTPServer = httptest.NewServer(handlerAccessLog(r))
	testHTTPSServer = httptest.NewTLSServer(handlerAccessLog(r))

	// override net.LookupIP
	lookupIP = func(host string) ([]net.IP, error) {
		if host == "127.0.0.1" {
			// I believe in future Go versions this will fail, so let's fix it later
			return net.LookupIP(host)
		}
		for h, addrs := range mockHosts {
			if host == h {
				return addrs, nil
			}
			for _, addr := range addrs {
				if addr.String() == host {
					return []net.IP{addr}, nil
				}
			}
		}
		return nil, errors.New("lookup: no such host")
	}
}

func handlerAccessLog(handler http.Handler) http.Handler {
	logHandler := func(w http.ResponseWriter, r *http.Request) {
		logrus.Debugf("%s \"%s %s\"", r.RemoteAddr, r.Method, r.URL)
		handler.ServeHTTP(w, r)
	}
	return http.HandlerFunc(logHandler)
}

func makeURL(req string) string {
	return testHTTPServer.URL + req
}

func makeHTTPSURL(req string) string {
	return testHTTPSServer.URL + req
}

func makeIndex(req string) *registrytypes.IndexInfo {
	index := &registrytypes.IndexInfo{
		Name: makeURL(req),
	}
	return index
}

func makeHTTPSIndex(req string) *registrytypes.IndexInfo {
	index := &registrytypes.IndexInfo{
		Name: makeHTTPSURL(req),
	}
	return index
}

func makePublicIndex() *registrytypes.IndexInfo {
	index := &registrytypes.IndexInfo{
		Name:     IndexServer,
		Secure:   true,
		Official: true,
	}
	return index
}

func makeServiceConfig(mirrors []string, insecureRegistries []string) *serviceConfig {
	options := ServiceOptions{
		Mirrors:            mirrors,
		InsecureRegistries: insecureRegistries,
	}

	return newServiceConfig(options)
}

func writeHeaders(w http.ResponseWriter) {
	h := w.Header()
	h.Add("Server", "docker-tests/mock")
	h.Add("Expires", "-1")
	h.Add("Content-Type", "application/json")
	h.Add("Pragma", "no-cache")
	h.Add("Cache-Control", "no-cache")
	h.Add("X-Docker-Registry-Version", "0.0.0")
	h.Add("X-Docker-Registry-Config", "mock")
}

func writeResponse(w http.ResponseWriter, message interface{}, code int) {
	writeHeaders(w)
	w.WriteHeader(code)
	body, err := json.Marshal(message)
	if err != nil {
		io.WriteString(w, err.Error())
		return
	}
	w.Write(body)
}

func readJSON(r *http.Request, dest interface{}) error {
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return err
	}
	return json.Unmarshal(body, dest)
}

func apiError(w http.ResponseWriter, message string, code int) {
	body := map[string]string{
		"error": message,
	}
	writeResponse(w, body, code)
}

func assertEqual(t *testing.T, a interface{}, b interface{}, message string) {
	if a == b {
		return
	}
	if len(message) == 0 {
		message = fmt.Sprintf("%v != %v", a, b)
	}
	t.Fatal(message)
}

func assertNotEqual(t *testing.T, a interface{}, b interface{}, message string) {
	if a != b {
		return
	}
	if len(message) == 0 {
		message = fmt.Sprintf("%v == %v", a, b)
	}
	t.Fatal(message)
}

// Similar to assertEqual, but does not stop test
func checkEqual(t *testing.T, a interface{}, b interface{}, messagePrefix string) {
	if a == b {
		return
	}
	message := fmt.Sprintf("%v != %v", a, b)
	if len(messagePrefix) != 0 {
		message = messagePrefix + ": " + message
	}
	t.Error(message)
}

// Similar to assertNotEqual, but does not stop test
func checkNotEqual(t *testing.T, a interface{}, b interface{}, messagePrefix string) {
	if a != b {
		return
	}
	message := fmt.Sprintf("%v == %v", a, b)
	if len(messagePrefix) != 0 {
		message = messagePrefix + ": " + message
	}
	t.Error(message)
}

func requiresAuth(w http.ResponseWriter, r *http.Request) bool {
	writeCookie := func() {
		value := fmt.Sprintf("FAKE-SESSION-%d", time.Now().UnixNano())
		cookie := &http.Cookie{Name: "session", Value: value, MaxAge: 3600}
		http.SetCookie(w, cookie)
		//FIXME(sam): this should be sent only on Index routes
		value = fmt.Sprintf("FAKE-TOKEN-%d", time.Now().UnixNano())
		w.Header().Add("X-Docker-Token", value)
	}
	if len(r.Cookies()) > 0 {
		writeCookie()
		return true
	}
	if len(r.Header.Get("Authorization")) > 0 {
		writeCookie()
		return true
	}
	w.Header().Add("WWW-Authenticate", "token")
	apiError(w, "Wrong auth", 401)
	return false
}

func handlerGetPing(w http.ResponseWriter, r *http.Request) {
	writeResponse(w, true, 200)
}

func handlerGetImage(w http.ResponseWriter, r *http.Request) {
	if !requiresAuth(w, r) {
		return
	}
	vars := mux.Vars(r)
	layer, exists := testLayers[vars["image_id"]]
	if !exists {
		http.NotFound(w, r)
		return
	}
	writeHeaders(w)
	layerSize := len(layer["layer"])
	w.Header().Add("X-Docker-Size", strconv.Itoa(layerSize))
	io.WriteString(w, layer[vars["action"]])
}

func handlerPutImage(w http.ResponseWriter, r *http.Request) {
	if !requiresAuth(w, r) {
		return
	}
	vars := mux.Vars(r)
	imageID := vars["image_id"]
	action := vars["action"]
	layer, exists := testLayers[imageID]
	if !exists {
		if action != "json" {
			http.NotFound(w, r)
			return
		}
		layer = make(map[string]string)
		testLayers[imageID] = layer
	}
	if checksum := r.Header.Get("X-Docker-Checksum"); checksum != "" {
		if checksum != layer["checksum_simple"] && checksum != layer["checksum_tarsum"] {
			apiError(w, "Wrong checksum", 400)
			return
		}
	}
	body, err := ioutil.ReadAll(r.Body)
	if err != nil {
		apiError(w, fmt.Sprintf("Error: %s", err), 500)
		return
	}
	layer[action] = string(body)
	writeResponse(w, true, 200)
}

func handlerGetDeleteTags(w http.ResponseWriter, r *http.Request) {
	if !requiresAuth(w, r) {
		return
	}
	repositoryName, err := reference.WithName(mux.Vars(r)["repository"])
	if err != nil {
		apiError(w, "Could not parse repository", 400)
		return
	}
	tags, exists := testRepositories[repositoryName.String()]
	if !exists {
		apiError(w, "Repository not found", 404)
		return
	}
	if r.Method == "DELETE" {
		delete(testRepositories, repositoryName.String())
		writeResponse(w, true, 200)
		return
	}
	writeResponse(w, tags, 200)
}

func handlerGetTag(w http.ResponseWriter, r *http.Request) {
	if !requiresAuth(w, r) {
		return
	}
	vars := mux.Vars(r)
	repositoryName, err := reference.WithName(vars["repository"])
	if err != nil {
		apiError(w, "Could not parse repository", 400)
		return
	}
	tagName := vars["tag"]
	tags, exists := testRepositories[repositoryName.String()]
	if !exists {
		apiError(w, "Repository not found", 404)
		return
	}
	tag, exists := tags[tagName]
	if !exists {
		apiError(w, "Tag not found", 404)
		return
	}
	writeResponse(w, tag, 200)
}

func handlerPutTag(w http.ResponseWriter, r *http.Request) {
	if !requiresAuth(w, r) {
		return
	}
	vars := mux.Vars(r)
	repositoryName, err := reference.WithName(vars["repository"])
	if err != nil {
		apiError(w, "Could not parse repository", 400)
		return
	}
	tagName := vars["tag"]
	tags, exists := testRepositories[repositoryName.String()]
	if !exists {
		tags = make(map[string]string)
		testRepositories[repositoryName.String()] = tags
	}
	tagValue := ""
	readJSON(r, tagValue)
	tags[tagName] = tagValue
	writeResponse(w, true, 200)
}

func handlerUsers(w http.ResponseWriter, r *http.Request) {
	code := 200
	if r.Method == "POST" {
		code = 201
	} else if r.Method == "PUT" {
		code = 204
	}
	writeResponse(w, "", code)
}

func handlerImages(w http.ResponseWriter, r *http.Request) {
	u, _ := url.Parse(testHTTPServer.URL)
	w.Header().Add("X-Docker-Endpoints", fmt.Sprintf("%s 	,  %s ", u.Host, "test.example.com"))
	w.Header().Add("X-Docker-Token", fmt.Sprintf("FAKE-SESSION-%d", time.Now().UnixNano()))
	if r.Method == "PUT" {
		if strings.HasSuffix(r.URL.Path, "images") {
			writeResponse(w, "", 204)
			return
		}
		writeResponse(w, "", 200)
		return
	}
	if r.Method == "DELETE" {
		writeResponse(w, "", 204)
		return
	}
	images := []map[string]string{}
	for imageID, layer := range testLayers {
		image := make(map[string]string)
		image["id"] = imageID
		image["checksum"] = layer["checksum_tarsum"]
		image["Tag"] = "latest"
		images = append(images, image)
	}
	writeResponse(w, images, 200)
}

func handlerAuth(w http.ResponseWriter, r *http.Request) {
	writeResponse(w, "OK", 200)
}

func handlerSearch(w http.ResponseWriter, r *http.Request) {
	result := &registrytypes.SearchResults{
		Query:      "fakequery",
		NumResults: 1,
		Results:    []registrytypes.SearchResult{{Name: "fakeimage", StarCount: 42}},
	}
	writeResponse(w, result, 200)
}

func TestPing(t *testing.T) {
	res, err := http.Get(makeURL("/v1/_ping"))
	if err != nil {
		t.Fatal(err)
	}
	assertEqual(t, res.StatusCode, 200, "")
	assertEqual(t, res.Header.Get("X-Docker-Registry-Config"), "mock",
		"This is not a Mocked Registry")
}

/* Uncomment this to test Mocked Registry locally with curl
 * WARNING: Don't push on the repos uncommented, it'll block the tests
 *
func TestWait(t *testing.T) {
	logrus.Println("Test HTTP server ready and waiting:", testHTTPServer.URL)
	c := make(chan int)
	<-c
}

//*/
