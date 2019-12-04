// This is a generated file. Do not edit directly.

module k8s.io/apiserver

go 1.12

require (
	github.com/Azure/go-ansiterm v0.0.0-20170929234023-d6e3b3328b78 // indirect
	github.com/coreos/go-oidc v2.1.0+incompatible
	github.com/coreos/go-semver v0.3.0 // indirect
	github.com/coreos/go-systemd v0.0.0-20190321100706-95778dfbb74e
	github.com/coreos/pkg v0.0.0-20180108230652-97fdf19511ea
	github.com/davecgh/go-spew v1.1.1
	github.com/docker/docker v0.7.3-0.20190327010347-be7ac8be2ae0
	github.com/dustin/go-humanize v1.0.0 // indirect
	github.com/emicklei/go-restful v2.9.5+incompatible
	github.com/evanphx/json-patch v4.2.0+incompatible
	github.com/go-openapi/jsonreference v0.19.3 // indirect
	github.com/go-openapi/spec v0.19.3
	github.com/gogo/protobuf v1.2.2-0.20190723190241-65acae22fc9d
	github.com/google/go-cmp v0.3.0
	github.com/google/gofuzz v1.0.0
	github.com/google/uuid v1.1.1
	github.com/googleapis/gnostic v0.0.0-20170729233727-0c5108395e2d
	github.com/gorilla/websocket v1.4.0 // indirect
	github.com/grpc-ecosystem/go-grpc-prometheus v1.2.0
	github.com/hashicorp/golang-lru v0.5.1
	github.com/mailru/easyjson v0.7.0 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822
	github.com/pkg/errors v0.8.1
	github.com/pquerna/cachecontrol v0.0.0-20171018203845-0dec1b30a021 // indirect
	github.com/prometheus/client_golang v1.0.0
	github.com/prometheus/client_model v0.0.0-20190129233127-fd36f4220a90
	github.com/sirupsen/logrus v1.4.2 // indirect
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.4.0
	go.etcd.io/etcd v0.0.0-20191023171146-3cf2f69b5738
	go.uber.org/zap v1.10.0
	golang.org/x/crypto v0.0.0-20190820162420-60c769a6c586
	golang.org/x/net v0.0.0-20191004110552-13f9640d40b9
	google.golang.org/genproto v0.0.0-20190502173448-54afdca5d873 // indirect
	google.golang.org/grpc v1.23.1
	gopkg.in/natefinch/lumberjack.v2 v2.0.0
	gopkg.in/square/go-jose.v2 v2.2.2
	gopkg.in/yaml.v2 v2.2.4
	gotest.tools v2.2.0+incompatible // indirect
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/component-base v0.0.0
	k8s.io/klog v1.0.0
	k8s.io/kube-openapi v0.0.0-20191107075043-30be4d16710a
	k8s.io/utils v0.0.0-20191114184206-e782cd3c129f
	sigs.k8s.io/structured-merge-diff v1.0.1-0.20191108220359-b1b620dd3f06
	sigs.k8s.io/yaml v1.1.0
)

replace (
	golang.org/x/sys => golang.org/x/sys v0.0.0-20190813064441-fde4db37ae7a // pinned to release-branch.go1.13
	golang.org/x/tools => golang.org/x/tools v0.0.0-20190821162956-65e3620a7ae7 // pinned to release-branch.go1.13
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/apiserver => ../apiserver
	k8s.io/client-go => ../client-go
	k8s.io/component-base => ../component-base
)
