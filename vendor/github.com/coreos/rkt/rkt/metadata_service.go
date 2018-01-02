// Copyright 2014 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha512"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"sync"
	"syscall"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	"github.com/gorilla/mux"
	"github.com/hashicorp/errwrap"
	"github.com/spf13/cobra"
)

var (
	cmdMetadataService = &cobra.Command{
		Use:   "metadata-service [--listen-port=PORT]",
		Short: "Run metadata service",
		Long: `Provides a means for running apps to introspect their execution
environment and assert their identity.`,
		Run: runWrapper(runMetadataService),
	}
)

var (
	hmacKey        [sha512.Size]byte
	pods           = newPodStore()
	errPodNotFound = errors.New("pod not found")
	errAppNotFound = errors.New("app not found")

	flagListenPort int

	exitCh = make(chan os.Signal, 1)
)

const (
	listenFdsStart = 3
)

func init() {
	cmdRkt.AddCommand(cmdMetadataService)
	cmdMetadataService.Flags().IntVar(&flagListenPort, "listen-port", common.MetadataServicePort, "listen port")
}

type mdsPod struct {
	uuid     types.UUID
	token    string
	manifest *schema.PodManifest
	apps     map[string]*schema.ImageManifest
}

type podStore struct {
	byToken map[string]*mdsPod
	byUUID  map[types.UUID]*mdsPod
	mutex   sync.Mutex
}

func newPodStore() *podStore {
	return &podStore{
		byToken: make(map[string]*mdsPod),
		byUUID:  make(map[types.UUID]*mdsPod),
	}
}

func (ps *podStore) addPod(u *types.UUID, token string, manifest *schema.PodManifest) {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	p := &mdsPod{
		uuid:     *u,
		token:    token,
		manifest: manifest,
		apps:     make(map[string]*schema.ImageManifest),
	}

	ps.byUUID[*u] = p
	ps.byToken[token] = p
}

func (ps *podStore) addApp(u *types.UUID, app string, manifest *schema.ImageManifest) error {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	p, ok := ps.byUUID[*u]
	if !ok {
		return errPodNotFound
	}

	p.apps[app] = manifest

	return nil
}

func (ps *podStore) remove(u *types.UUID) error {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	p, ok := ps.byUUID[*u]
	if !ok {
		return errPodNotFound
	}

	delete(ps.byUUID, *u)
	delete(ps.byToken, p.token)

	return nil
}

func (ps *podStore) getUUID(token string) (*types.UUID, error) {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	p, ok := ps.byToken[token]
	if !ok {
		return nil, errPodNotFound
	}
	return &p.uuid, nil
}

func (ps *podStore) getPodManifest(token string) (*schema.PodManifest, error) {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	p, ok := ps.byToken[token]
	if !ok {
		return nil, errPodNotFound
	}
	return p.manifest, nil
}

func (ps *podStore) getManifests(token, an string) (*schema.PodManifest, *schema.ImageManifest, error) {
	ps.mutex.Lock()
	defer ps.mutex.Unlock()

	p, ok := ps.byToken[token]
	if !ok {
		return nil, nil, errPodNotFound
	}

	im, ok := p.apps[an]
	if !ok {
		return nil, nil, errAppNotFound
	}

	return p.manifest, im, nil
}

func queryValue(u *url.URL, key string) string {
	vals, ok := u.Query()[key]
	if !ok || len(vals) != 1 {
		return ""
	}
	return vals[0]
}

func handleRegisterPod(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	uuid, err := types.NewUUID(mux.Vars(r)["uuid"])
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "UUID is missing or malformed: %v", err)
		return
	}

	token := queryValue(r.URL, "token")
	if token == "" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, "token missing")
		return
	}

	pm := &schema.PodManifest{}

	if err := json.NewDecoder(r.Body).Decode(pm); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "JSON-decoding failed: %v", err)
		return
	}

	pods.addPod(uuid, token, pm)

	w.WriteHeader(http.StatusOK)
}

func handleUnregisterPod(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	uuid, err := types.NewUUID(mux.Vars(r)["uuid"])
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "UUID is missing or malformed: %v", err)
		return
	}

	if err := pods.remove(uuid); err != nil {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, err)
		return
	}

	w.WriteHeader(http.StatusOK)
}

func handleRegisterApp(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	uuid, err := types.NewUUID(mux.Vars(r)["uuid"])
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "UUID is missing or malformed: %v", err)
		return
	}

	an := mux.Vars(r)["app"]
	if an == "" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, "app missing")
		return
	}

	im := &schema.ImageManifest{}
	if err := json.NewDecoder(r.Body).Decode(im); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "JSON-decoding failed: %v", err)
		return
	}

	err = pods.addApp(uuid, an, im)
	if err != nil {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, "Pod with given UUID not found")
		return
	}

	w.WriteHeader(http.StatusOK)
}

func podGet(h func(http.ResponseWriter, *http.Request, *schema.PodManifest)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		token := mux.Vars(r)["token"]

		pm, err := pods.getPodManifest(token)
		if err != nil {
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprintln(w, err)
			return
		}

		h(w, r, pm)
	}
}

func appGet(h func(http.ResponseWriter, *http.Request, *schema.PodManifest, *schema.ImageManifest)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		token := mux.Vars(r)["token"]

		an := mux.Vars(r)["app"]
		if an == "" {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, "app missing")
			return
		}

		pm, im, err := pods.getManifests(token, an)
		switch {
		case err == nil:
			h(w, r, pm, im)

		case err == errPodNotFound:
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprintln(w, err)

		default:
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintln(w, err)
		}
	}
}

func handlePodAnnotations(w http.ResponseWriter, r *http.Request, pm *schema.PodManifest) {
	defer r.Body.Close()

	out, err := pm.Annotations.MarshalJSON()
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, "invalid annotations: %v", err)
		return
	}

	w.Header().Add("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(out)

}

func handlePodAnnotation(w http.ResponseWriter, r *http.Request, pm *schema.PodManifest) {
	defer r.Body.Close()

	n := mux.Vars(r)["name"]
	k, err := types.NewACIdentifier(n)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "Pod annotation %q is not a valid AC Identifier", n)
		return
	}

	v, ok := pm.Annotations.Get(k.String())
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintf(w, "Pod annotation %q not found", k)
		return
	}

	w.Header().Add("Content-Type", "text/plain; charset=us-ascii")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(v))
}

func handlePodManifest(w http.ResponseWriter, r *http.Request, pm *schema.PodManifest) {
	defer r.Body.Close()

	out, err := json.Marshal(pm)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprintf(w, "JSON encoding error: %v", err)
		return
	}

	w.Header().Add("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	w.Write(out)
}

func handlePodUUID(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	token := mux.Vars(r)["token"]

	uuid, err := pods.getUUID(token)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprintln(w, err)
		return
	}

	w.Header().Add("Content-Type", "text/plain; charset=us-ascii")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(uuid.String()))
}

func mergeAppAnnotations(im *schema.ImageManifest, pm *schema.PodManifest, appName *types.ACName) types.Annotations {
	merged := types.Annotations{}

	for _, annot := range im.Annotations {
		merged.Set(annot.Name, annot.Value)
	}

	if app := pm.Apps.Get(*appName); app != nil {
		for _, annot := range app.Annotations {
			merged.Set(annot.Name, annot.Value)
		}
	}

	return merged
}

func handleAppAnnotations(w http.ResponseWriter, r *http.Request, pm *schema.PodManifest, im *schema.ImageManifest) {
	defer r.Body.Close()

	n := mux.Vars(r)["app"]
	an, err := types.NewACName(n)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "App name %q is not a valid AC Name", n)
		return
	}

	anno := mergeAppAnnotations(im, pm, an)
	out, err := anno.MarshalJSON()

	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, err.Error())
		return
	}

	w.Header().Add("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(out)

}

func handleAppAnnotation(w http.ResponseWriter, r *http.Request, pm *schema.PodManifest, im *schema.ImageManifest) {
	defer r.Body.Close()

	n := mux.Vars(r)["name"]
	k, err := types.NewACIdentifier(n)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "App annotation name %q is not a valid AC Identifier", n)
		return
	}

	n = mux.Vars(r)["app"]
	an, err := types.NewACName(n)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "App name %q is not a valid AC Name", n)
		return
	}

	merged := mergeAppAnnotations(im, pm, an)

	v, ok := merged.Get(k.String())
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprintf(w, "App annotation %q not found", k)
		return
	}

	w.Header().Add("Content-Type", "text/plain; charset=us-ascii")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(v))
}

func handleImageManifest(w http.ResponseWriter, r *http.Request, _ *schema.PodManifest, im *schema.ImageManifest) {
	defer r.Body.Close()

	w.Header().Add("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	out, err := json.Marshal(im)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, err.Error())
		return
	}
	w.Write(out)

}

func handleAppID(w http.ResponseWriter, r *http.Request, pm *schema.PodManifest, im *schema.ImageManifest) {
	defer r.Body.Close()

	n := mux.Vars(r)["app"]
	an, err := types.NewACName(n)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "App name %q is not a valid AC Name", n)
		return
	}

	w.Header().Add("Content-Type", "text/plain; charset=us-ascii")
	w.WriteHeader(http.StatusOK)
	app := pm.Apps.Get(*an)
	if app == nil {
		// This is impossible as we have already checked that
		// the image manifest is not nil in the parent function.
		panic("could not find app in manifest!")
	}
	w.Write([]byte(app.Image.ID.String()))
}

func initCrypto() error {
	if n, err := rand.Reader.Read(hmacKey[:]); err != nil || n != len(hmacKey) {
		return fmt.Errorf("failed to generate HMAC Key")
	}
	return nil
}

func handlePodSign(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	token := mux.Vars(r)["token"]

	uuid, err := pods.getUUID(token)
	if err != nil {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprintln(w, err)
		return
	}

	content := r.FormValue("content")
	if content == "" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "content form value not found")
		return
	}

	// HMAC(UID:content)
	h := hmac.New(sha512.New, hmacKey[:])
	h.Write((*uuid)[:])
	h.Write([]byte(content))

	// Send back HMAC as the signature
	w.Header().Add("Content-Type", "text/plain; charset=us-ascii")
	w.WriteHeader(http.StatusOK)
	enc := base64.NewEncoder(base64.StdEncoding, w)
	enc.Write(h.Sum(nil))
	enc.Close()
}

func handlePodVerify(w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	w.Header().Add("Content-Type", "text/plain; charset=us-ascii")
	uuid, err := types.NewUUID(r.FormValue("uuid"))
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "uuid field missing or malformed: %v", err)
		return
	}

	content := r.FormValue("content")
	if content == "" {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "content field missing")
		return
	}

	sig, err := base64.StdEncoding.DecodeString(r.FormValue("signature"))
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, "signature field missing or corrupt: %v", err)
		return
	}

	h := hmac.New(sha512.New, hmacKey[:])
	h.Write((*uuid)[:])
	h.Write([]byte(content))

	if hmac.Equal(sig, h.Sum(nil)) {
		w.WriteHeader(http.StatusOK)
	} else {
		w.WriteHeader(http.StatusForbidden)
	}
}

type httpResp struct {
	writer http.ResponseWriter
	status int
}

func (r *httpResp) Header() http.Header {
	return r.writer.Header()
}

func (r *httpResp) Write(d []byte) (int, error) {
	return r.writer.Write(d)
}

func (r *httpResp) WriteHeader(status int) {
	r.status = status
	r.writer.WriteHeader(status)
}

func logReq(h func(w http.ResponseWriter, r *http.Request)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		resp := &httpResp{w, 0}
		h(resp, r)
		stderr.Printf("%v %v - %v", r.Method, r.RequestURI, resp.status)
	}
}

// unixListener returns the listener used for registrations (over unix sock)
func unixListener() (net.Listener, error) {
	s := os.Getenv("LISTEN_FDS")
	if s != "" {
		// socket activated
		lfds, err := strconv.ParseInt(s, 10, 16)
		if err != nil {
			return nil, errwrap.Wrap(errors.New("error parsing LISTEN_FDS env var"), err)
		}
		if lfds < 1 {
			return nil, fmt.Errorf("LISTEN_FDS < 1")
		}

		return net.FileListener(os.NewFile(uintptr(listenFdsStart), "listen"))
	} else {
		dir := filepath.Dir(common.MetadataServiceRegSock)
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return nil, errwrap.Wrap(fmt.Errorf("failed to create %v", dir), err)
		}

		return net.ListenUnix("unix", &net.UnixAddr{
			Net:  "unix",
			Name: common.MetadataServiceRegSock,
		})
	}
}

func runRegistrationServer(l net.Listener) {
	r := mux.NewRouter()
	r.HandleFunc("/pods/{uuid}", logReq(handleRegisterPod)).Methods("PUT")
	r.HandleFunc("/pods/{uuid}", logReq(handleUnregisterPod)).Methods("DELETE")
	r.HandleFunc("/pods/{uuid}/{app:.*}", logReq(handleRegisterApp)).Methods("PUT")

	if err := http.Serve(l, r); err != nil {
		stderr.PrintE("error serving registration HTTP", err)
	}
	close(exitCh)
}

func runPublicServer(l net.Listener) {
	r := mux.NewRouter().PathPrefix("/{token}/acMetadata/v1").Subrouter()

	mr := r.Methods("GET").Subrouter()

	mr.HandleFunc("/pod/annotations", logReq(podGet(handlePodAnnotations)))
	mr.HandleFunc("/pod/annotations/{name:.*}", logReq(podGet(handlePodAnnotation)))
	mr.HandleFunc("/pod/manifest", logReq(podGet(handlePodManifest)))
	mr.HandleFunc("/pod/uuid", logReq(handlePodUUID))

	mr.HandleFunc("/apps/{app:.*}/annotations", logReq(appGet(handleAppAnnotations)))
	mr.HandleFunc("/apps/{app:.*}/annotations/{name:.*}", logReq(appGet(handleAppAnnotation)))
	mr.HandleFunc("/apps/{app:.*}/image/manifest", logReq(appGet(handleImageManifest)))
	mr.HandleFunc("/apps/{app:.*}/image/id", logReq(appGet(handleAppID)))

	r.HandleFunc("/pod/hmac/sign", logReq(handlePodSign)).Methods("POST")
	r.HandleFunc("/pod/hmac/verify", logReq(handlePodVerify)).Methods("POST")

	if err := http.Serve(l, r); err != nil {
		stderr.PrintE("error serving pod HTTP", err)
	}
	close(exitCh)
}

func runMetadataService(cmd *cobra.Command, args []string) (exit int) {
	signal.Notify(exitCh, syscall.SIGINT, syscall.SIGTERM)
	stderr.Print("metadata service starting...")

	unixl, err := unixListener()
	if err != nil {
		stderr.Error(err)
		return 254
	}
	defer unixl.Close()

	tcpl, err := net.ListenTCP("tcp4", &net.TCPAddr{Port: flagListenPort})
	if err != nil {
		stderr.PrintE(fmt.Sprintf("error listening on port %v", flagListenPort), err)
		return 254
	}
	defer tcpl.Close()

	if err := initCrypto(); err != nil {
		stderr.Error(err)
		return 254
	}

	go runRegistrationServer(unixl)
	go runPublicServer(tcpl)

	stderr.Print("metadata service running...")

	<-exitCh

	stderr.Print("metadata service exiting...")

	return
}
