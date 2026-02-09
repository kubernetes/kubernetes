/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package auditproxy

import (
	"io"
	"log"
	"net/http"
	"os"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	auditinstall "k8s.io/apiserver/pkg/apis/audit/install"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	"k8s.io/apiserver/pkg/audit"
)

// CmdAuditProxy is used by agnhost Cobra.
var CmdAuditProxy = &cobra.Command{
	Use:   "audit-proxy",
	Short: "Listens on port 8080 for incoming audit events",
	Long:  "Used to test dynamic auditing. It listens on port 8080 for incoming audit events and writes them in a uniform manner to stdout.",
	Args:  cobra.MaximumNArgs(0),
	Run:   main,
}

var (
	encoder runtime.Encoder
	decoder runtime.Decoder
)

func main(cmd *cobra.Command, args []string) {
	scheme := runtime.NewScheme()
	auditinstall.Install(scheme)
	serializer := json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, json.SerializerOptions{Pretty: false})
	encoder = audit.Codecs.EncoderForVersion(serializer, auditv1.SchemeGroupVersion)
	decoder = audit.Codecs.UniversalDecoder(auditv1.SchemeGroupVersion)

	http.HandleFunc("/", handler)
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func handler(w http.ResponseWriter, req *http.Request) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		log.Printf("could not read request body: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	el := &auditv1.EventList{}

	if err := runtime.DecodeInto(decoder, body, el); err != nil {
		log.Printf("failed decoding buf: %b, apiVersion: %s", body, auditv1.SchemeGroupVersion)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	defer req.Body.Close()

	// write events to stdout
	for _, event := range el.Items {
		err := encoder.Encode(&event, os.Stdout)
		if err != nil {
			log.Printf("could not encode audit event: %v", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
	}
	w.WriteHeader(http.StatusOK)
}
