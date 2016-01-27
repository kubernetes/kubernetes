package main

import (
	"log"
	"os"
	"strings"

	"github.com/emicklei/go-restful"
	"github.com/go-swagger/go-swagger/scan"
	"github.com/go-swagger/go-swagger/spec"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/genericapiserver"
	kubeletclient "k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/storage/etcd"
)

func main() {
	s := &spec.Swagger{}
	if s.Paths == nil {
		s.Paths = &spec.Paths{}
		s.Paths.Paths = make(map[string]spec.PathItem)
	}

	storage, _ := (&etcd.EtcdConfig{}).NewStorage()
	sd := genericapiserver.NewStorageDestinations()
	sd.AddAPIGroup("", storage)
	m := master.New(&master.Config{
		Config: &genericapiserver.Config{
			Serializer:               api.Codecs,
			StorageDestinations:      sd,
			APIGroupVersionOverrides: map[string]genericapiserver.APIGroupVersionOverride{"extensions/v1beta1": {Disable: true}},
		},
		KubeletClient: &kubeletclient.FakeKubeletClient{},
	})
	for _, ws := range m.HandlerContainer.RegisteredWebServices() {
		for _, r := range ws.Routes() {
			path := r.Path
			if !strings.HasPrefix(path, "/") {
				path = "/" + path
			}
			log.Printf("ws %s: route %s %s", ws.RootPath(), path, r.Method)
			item := s.Paths.Paths[path]
			switch r.Method {
			case "GET":
				op := (&spec.Operation{}).WithDescription(r.Doc).WithConsumes(r.Consumes...).WithProduces(r.Produces...)
				item.Get = op
				for _, p := range r.ParameterDocs {
					switch p.Kind() {
					case restful.PathParameterKind:
						param := (&spec.Parameter{}).
							WithLocation("path").
							Named(p.Data().Name).
							WithDescription(p.Data().Description)
						if p.Data().Required {
							param.AsRequired()
						}
						item.Get.AddParam(param)
					}
				}
			}
			s.Paths.Paths[path] = item
		}
	}
	out, err := scan.Application(os.Args[1], s, nil, nil)
	if err != nil {
		log.Fatal(err)
	}
	data, err := out.MarshalJSON()
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("%s", data)
}
