// This is a generated file. Do not edit directly.
// Ensure you've carefully read
// https://git.k8s.io/community/contributors/devel/sig-architecture/vendor.md
// Run hack/pin-dependency.sh to change pinned dependency versions.
// Run hack/update-vendor.sh to update go.mod files and the vendor directory.

module k8s.io/apimachinery

go 1.20

require (
	github.com/armon/go-socks5 v0.0.0-20160902184237-e75332964ef5
	github.com/davecgh/go-spew v1.1.1
	github.com/evanphx/json-patch v4.12.0+incompatible
	github.com/gogo/protobuf v1.3.2
	github.com/golang/protobuf v1.5.3
	github.com/google/gnostic v0.5.7-v3refs
	github.com/google/go-cmp v0.5.9
	github.com/google/gofuzz v1.1.0
	github.com/google/uuid v1.3.0
	github.com/moby/spdystream v0.2.0
	github.com/mxk/go-flowrate v0.0.0-20140419014527-cca7078d478f
	github.com/onsi/ginkgo/v2 v2.9.1
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.8.1
	golang.org/x/net v0.8.0
	golang.org/x/time v0.0.0-20220210224613-90d013bbcef8
	gopkg.in/inf.v0 v0.9.1
	k8s.io/klog/v2 v2.90.1
	k8s.io/kube-openapi v0.0.0-20230308215209-15aac26d736a
	k8s.io/utils v0.0.0-20230209194617-a36077c30491
	sigs.k8s.io/json v0.0.0-20221116044647-bc3834ca7abd
	sigs.k8s.io/structured-merge-diff/v4 v4.2.3
	sigs.k8s.io/yaml v1.3.0
)

require (
	bitbucket.org/bertimus9/systemstat v0.5.0
	cloud.google.com/go v0.97.0
	github.com/Azure/azure-sdk-for-go v55.0.0+incompatible
	github.com/Azure/go-ansiterm v0.0.0-20210617225240-d185dfc1b5a1
	github.com/Azure/go-autorest v14.2.0+incompatible
	github.com/Azure/go-autorest/autorest v0.11.27
	github.com/Azure/go-autorest/autorest/adal v0.9.20
	github.com/Azure/go-autorest/autorest/date v0.3.0
	github.com/Azure/go-autorest/autorest/mocks v0.4.2
	github.com/Azure/go-autorest/autorest/to v0.4.0
	github.com/Azure/go-autorest/autorest/validation v0.1.0
	github.com/Azure/go-autorest/logger v0.2.1
	github.com/Azure/go-autorest/tracing v0.6.0
	github.com/GoogleCloudPlatform/k8s-cloud-provider v1.18.1-0.20220218231025-f11817397a1b
	github.com/JeffAshton/win_pdh v0.0.0-20161109143554-76bb4ee9f0ab
	github.com/MakeNowJust/heredoc v1.0.0
	github.com/Microsoft/go-winio v0.4.17
	github.com/Microsoft/hcsshim v0.8.25
	github.com/NYTimes/gziphandler v1.1.1
	github.com/antlr/antlr4/runtime/Go/antlr v1.4.10
	github.com/armon/circbuf v0.0.0-20150827004946-bbbad097214e
	github.com/asaskevich/govalidator v0.0.0-20190424111038-f61b66f89f4a
	github.com/beorn7/perks v1.0.1
	github.com/blang/semver/v4 v4.0.0
	github.com/cenkalti/backoff/v4 v4.1.3
	github.com/cespare/xxhash/v2 v2.1.2
	github.com/chai2010/gettext-go v1.0.2
	github.com/checkpoint-restore/go-criu/v5 v5.3.0
	github.com/cilium/ebpf v0.7.0
	github.com/container-storage-interface/spec v1.7.0
	github.com/containerd/cgroups v1.0.1
	github.com/containerd/console v1.0.3
	github.com/containerd/ttrpc v1.1.0
	github.com/coredns/caddy v1.1.0
	github.com/coredns/corefile-migration v1.0.20
	github.com/coreos/go-oidc v2.1.0+incompatible
	github.com/coreos/go-semver v0.3.0
	github.com/coreos/go-systemd/v22 v22.4.0
	github.com/cpuguy83/go-md2man/v2 v2.0.2
	github.com/cyphar/filepath-securejoin v0.2.3
	github.com/daviddengcn/go-colortext v1.0.0
	github.com/docker/distribution v2.8.1+incompatible
	github.com/docker/go-units v0.5.0
	github.com/dustin/go-humanize v1.0.0
	github.com/emicklei/go-restful/v3 v3.9.0
	github.com/euank/go-kmsg-parser v2.0.0+incompatible
	github.com/exponent-io/jsonpath v0.0.0-20151013193312-d6023ce2651d
	github.com/fatih/camelcase v1.0.0
	github.com/felixge/httpsnoop v1.0.3
	github.com/fsnotify/fsnotify v1.6.0
	github.com/fvbommel/sortorder v1.0.1
	github.com/go-errors/errors v1.4.2
	github.com/go-logr/logr v1.2.3 // indirect
	github.com/go-logr/stdr v1.2.2
	github.com/go-logr/zapr v1.2.3
	github.com/go-openapi/jsonpointer v0.19.6 // indirect
	github.com/go-openapi/jsonreference v0.20.1 // indirect
	github.com/go-openapi/swag v0.22.3 // indirect
	github.com/go-task/slim-sprig v0.0.0-20210107165309-348f09dbbbc0 // indirect
	github.com/godbus/dbus/v5 v5.0.6
	github.com/gofrs/uuid v4.0.0+incompatible
	github.com/golang-jwt/jwt/v4 v4.4.2
	github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da
	github.com/golang/mock v1.6.0
	github.com/google/btree v1.0.1
	github.com/google/cadvisor v0.47.1
	github.com/google/cel-go v0.12.6
	github.com/google/pprof v0.0.0-20210720184732-4bb14d4b1be1 // indirect
	github.com/google/shlex v0.0.0-20191202100458-e7afc7fbc510
	github.com/googleapis/gax-go/v2 v2.1.1
	github.com/gorilla/websocket v1.4.2
	github.com/gregjones/httpcache v0.0.0-20180305231024-9cad4c3443a7
	github.com/grpc-ecosystem/go-grpc-middleware v1.3.0
	github.com/grpc-ecosystem/go-grpc-prometheus v1.2.0
	github.com/grpc-ecosystem/grpc-gateway v1.16.0
	github.com/grpc-ecosystem/grpc-gateway/v2 v2.7.0
	github.com/imdario/mergo v0.3.6
	github.com/inconshreveable/mousetrap v1.0.1
	github.com/ishidawataru/sctp v0.0.0-20190723014705-7c296d48a2b5
	github.com/jonboulle/clockwork v0.2.2
	github.com/josharian/intern v1.0.0 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/karrick/godirwalk v1.17.0
	github.com/kr/pretty v0.3.0 // indirect
	github.com/libopenstorage/openstorage v1.0.0
	github.com/liggitt/tabwriter v0.0.0-20181228230101-89fcab3d43de
	github.com/lithammer/dedent v1.1.0
	github.com/mailru/easyjson v0.7.7 // indirect
	github.com/matttproud/golang_protobuf_extensions v1.0.2
	github.com/mistifyio/go-zfs v2.1.2-0.20190413222219-f784269be439+incompatible
	github.com/mitchellh/go-wordwrap v1.0.0
	github.com/mitchellh/mapstructure v1.4.1
	github.com/moby/ipvs v1.1.0
	github.com/moby/sys/mountinfo v0.6.2
	github.com/moby/term v0.0.0-20221205130635-1aeaba878587
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/mohae/deepcopy v0.0.0-20170603005431-491d3605edfb
	github.com/monochromegane/go-gitignore v0.0.0-20200626010858-205db1a8cc00
	github.com/mrunalp/fileutils v0.5.0
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822
	github.com/onsi/gomega v1.27.4 // indirect
	github.com/opencontainers/go-digest v1.0.0
	github.com/opencontainers/runc v1.1.4
	github.com/opencontainers/runtime-spec v1.0.3-0.20220909204839-494a5a6aca78
	github.com/opencontainers/selinux v1.10.0
	github.com/peterbourgon/diskv v2.0.1+incompatible
	github.com/pkg/errors v0.9.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/pquerna/cachecontrol v0.1.0
	github.com/prometheus/client_golang v1.14.0
	github.com/prometheus/client_model v0.3.0
	github.com/prometheus/common v0.37.0
	github.com/prometheus/procfs v0.8.0
	github.com/robfig/cron/v3 v3.0.1
	github.com/rubiojr/go-vhd v0.0.0-20200706105327-02e210299021
	github.com/russross/blackfriday/v2 v2.1.0
	github.com/seccomp/libseccomp-golang v0.9.2-0.20220502022130-f33da4d89646
	github.com/sirupsen/logrus v1.9.0
	github.com/soheilhy/cmux v0.1.5
	github.com/spf13/cobra v1.6.0
	github.com/stoewer/go-strcase v1.2.0
	github.com/syndtr/gocapability v0.0.0-20200815063812-42c35b437635
	github.com/tmc/grpc-websocket-proxy v0.0.0-20220101234140-673ab2c3ae75
	github.com/vishvananda/netlink v1.1.0
	github.com/vishvananda/netns v0.0.2
	github.com/vmware/govmomi v0.30.0
	github.com/xiang90/probing v0.0.0-20190116061207-43a291ad63a2
	github.com/xlab/treeprint v1.1.0
	go.etcd.io/bbolt v1.3.6
	go.etcd.io/etcd/api/v3 v3.5.7
	go.etcd.io/etcd/client/pkg/v3 v3.5.7
	go.etcd.io/etcd/client/v2 v2.305.7
	go.etcd.io/etcd/client/v3 v3.5.7
	go.etcd.io/etcd/pkg/v3 v3.5.7
	go.etcd.io/etcd/raft/v3 v3.5.7
	go.etcd.io/etcd/server/v3 v3.5.7
	go.opencensus.io v0.23.0
	go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful v0.35.0
	go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc v0.35.0
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.35.1
	go.opentelemetry.io/otel v1.10.0
	go.opentelemetry.io/otel/exporters/otlp/internal/retry v1.10.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace v1.10.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc v1.10.0
	go.opentelemetry.io/otel/metric v0.31.0
	go.opentelemetry.io/otel/sdk v1.10.0
	go.opentelemetry.io/otel/trace v1.10.0
	go.opentelemetry.io/proto/otlp v0.19.0
	go.starlark.net v0.0.0-20200306205701-8dd3e2ee1dd5
	go.uber.org/atomic v1.7.0
	go.uber.org/goleak v1.2.1
	go.uber.org/multierr v1.6.0
	go.uber.org/zap v1.19.0
	golang.org/x/crypto v0.1.0
	golang.org/x/mod v0.9.0
	golang.org/x/oauth2 v0.0.0-20220223155221-ee480838109b
	golang.org/x/sync v0.1.0
	golang.org/x/sys v0.6.0 // indirect
	golang.org/x/term v0.6.0
	golang.org/x/text v0.8.0 // indirect
	golang.org/x/tools v0.7.0 // indirect
	google.golang.org/api v0.60.0
	google.golang.org/appengine v1.6.7
	google.golang.org/genproto v0.0.0-20220502173005-c8bf987b8c21
	google.golang.org/grpc v1.51.0
	google.golang.org/protobuf v1.28.1 // indirect
	gopkg.in/gcfg.v1 v1.2.0
	gopkg.in/natefinch/lumberjack.v2 v2.0.0
	gopkg.in/square/go-jose.v2 v2.6.0
	gopkg.in/warnings.v0 v0.1.1
	gopkg.in/yaml.v2 v2.4.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	k8s.io/gengo v0.0.0-20220902162205-c0856e24416d
	k8s.io/system-validators v1.8.0
	sigs.k8s.io/apiserver-network-proxy/konnectivity-client v0.1.1
	sigs.k8s.io/kustomize/api v0.13.2
	sigs.k8s.io/kustomize/kustomize/v5 v5.0.1
	sigs.k8s.io/kustomize/kyaml v0.14.1
)

replace k8s.io/apimachinery => ../apimachinery
