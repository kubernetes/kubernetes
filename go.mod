// This is a generated file. Do not edit directly.
// Ensure you've carefully read
// https://git.k8s.io/community/contributors/devel/sig-architecture/vendor.md
// Run hack/pin-dependency.sh to change pinned dependency versions.
// Run hack/update-vendor.sh to update go.mod files and the vendor directory.

module k8s.io/kubernetes

go 1.19

require (
	bitbucket.org/bertimus9/systemstat v0.5.0
	github.com/Azure/azure-sdk-for-go v55.0.0+incompatible
	github.com/Azure/go-autorest/autorest v0.11.27
	github.com/Azure/go-autorest/autorest/adal v0.9.20
	github.com/Azure/go-autorest/autorest/to v0.4.0
	github.com/GoogleCloudPlatform/k8s-cloud-provider v1.18.1-0.20220218231025-f11817397a1b
	github.com/JeffAshton/win_pdh v0.0.0-20161109143554-76bb4ee9f0ab
	github.com/Microsoft/go-winio v0.4.17
	github.com/Microsoft/hcsshim v0.8.22
	github.com/armon/circbuf v0.0.0-20150827004946-bbbad097214e
	github.com/armon/go-socks5 v0.0.0-20160902184237-e75332964ef5
	github.com/aws/aws-sdk-go v1.38.49
	github.com/blang/semver/v4 v4.0.0
	github.com/container-storage-interface/spec v1.6.0
	github.com/coredns/corefile-migration v1.0.17
	github.com/coreos/go-oidc v2.1.0+incompatible
	github.com/coreos/go-systemd/v22 v22.3.2
	github.com/cpuguy83/go-md2man/v2 v2.0.1
	github.com/davecgh/go-spew v1.1.1
	github.com/docker/distribution v2.8.1+incompatible
	github.com/docker/go-units v0.4.0
	github.com/elazarl/goproxy v0.0.0-20180725130230-947c36da3153
	github.com/emicklei/go-restful/v3 v3.8.0
	github.com/evanphx/json-patch v4.12.0+incompatible
	github.com/fsnotify/fsnotify v1.4.9
	github.com/go-logr/logr v1.2.3
	github.com/godbus/dbus/v5 v5.0.6
	github.com/gogo/protobuf v1.3.2
	github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da
	github.com/golang/mock v1.6.0
	github.com/golang/protobuf v1.5.2
	github.com/google/cadvisor v0.45.0
	github.com/google/gnostic v0.5.7-v3refs
	github.com/google/go-cmp v0.5.6
	github.com/google/gofuzz v1.1.0
	github.com/google/uuid v1.1.2
	github.com/heketi/heketi v10.3.0+incompatible
	github.com/ishidawataru/sctp v0.0.0-20190723014705-7c296d48a2b5
	github.com/libopenstorage/openstorage v1.0.0
	github.com/lithammer/dedent v1.1.0
	github.com/moby/ipvs v1.0.1
	github.com/mrunalp/fileutils v0.5.0
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822
	github.com/mvdan/xurls v1.1.0
	github.com/onsi/ginkgo/v2 v2.1.4
	github.com/onsi/gomega v1.19.0
	github.com/opencontainers/runc v1.1.3
	github.com/opencontainers/selinux v1.10.0
	github.com/pkg/errors v0.9.1
	github.com/pmezard/go-difflib v1.0.0
	github.com/prometheus/client_golang v1.12.1
	github.com/prometheus/client_model v0.2.0
	github.com/prometheus/common v0.32.1
	github.com/robfig/cron/v3 v3.0.1
	github.com/spf13/cobra v1.4.0
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.7.0
	github.com/vishvananda/netlink v1.1.0
	github.com/vmware/govmomi v0.20.3
	go.etcd.io/etcd/api/v3 v3.5.4
	go.etcd.io/etcd/client/pkg/v3 v3.5.4
	go.etcd.io/etcd/client/v3 v3.5.4
	go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful v0.20.0
	go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc v0.20.0
	go.opentelemetry.io/otel v0.20.0
	go.opentelemetry.io/otel/exporters/otlp v0.20.0
	go.opentelemetry.io/otel/sdk v0.20.0
	go.opentelemetry.io/otel/trace v0.20.0
	go.opentelemetry.io/proto/otlp v0.7.0
	go.uber.org/zap v1.19.0
	golang.org/x/crypto v0.0.0-20220315160706-3147a52a75dd
	golang.org/x/net v0.0.0-20220722155237-a158d28d115b
	golang.org/x/oauth2 v0.0.0-20211104180415-d3ed0bb246c8
	golang.org/x/sync v0.0.0-20220722155255-886fb9371eb4
	golang.org/x/sys v0.0.0-20220722155257-8c9f86f7a55f
	golang.org/x/term v0.0.0-20210927222741-03fcf44c2211
	golang.org/x/time v0.0.0-20220210224613-90d013bbcef8
	golang.org/x/tools v0.1.12
	gonum.org/v1/gonum v0.6.2
	google.golang.org/api v0.60.0
	google.golang.org/genproto v0.0.0-20220502173005-c8bf987b8c21
	google.golang.org/grpc v1.47.0
	google.golang.org/protobuf v1.28.0
	gopkg.in/gcfg.v1 v1.2.0
	gopkg.in/square/go-jose.v2 v2.2.2
	gopkg.in/yaml.v2 v2.4.0
	gopkg.in/yaml.v3 v3.0.1
	k8s.io/api v0.0.0
	k8s.io/apiextensions-apiserver v0.0.0
	k8s.io/apimachinery v0.0.0
	k8s.io/apiserver v0.0.0
	k8s.io/cli-runtime v0.0.0
	k8s.io/client-go v0.0.0
	k8s.io/cloud-provider v0.0.0
	k8s.io/cluster-bootstrap v0.0.0
	k8s.io/code-generator v0.0.0
	k8s.io/component-base v0.0.0
	k8s.io/component-helpers v0.0.0
	k8s.io/controller-manager v0.0.0
	k8s.io/cri-api v0.0.0
	k8s.io/csi-translation-lib v0.0.0
	k8s.io/gengo v0.0.0-20211129171323-c02415ce4185
	k8s.io/klog/v2 v2.70.1
	k8s.io/kube-aggregator v0.0.0
	k8s.io/kube-controller-manager v0.0.0
	k8s.io/kube-openapi v0.0.0-20220803162953-67bda5d908f1
	k8s.io/kube-proxy v0.0.0
	k8s.io/kube-scheduler v0.0.0
	k8s.io/kubectl v0.0.0
	k8s.io/kubelet v0.0.0
	k8s.io/legacy-cloud-providers v0.0.0
	k8s.io/metrics v0.0.0
	k8s.io/mount-utils v0.0.0
	k8s.io/pod-security-admission v0.0.0
	k8s.io/sample-apiserver v0.0.0
	k8s.io/system-validators v1.7.0
	k8s.io/utils v0.0.0-20220728103510-ee6ede2d64ed
	sigs.k8s.io/structured-merge-diff/v4 v4.2.3
	sigs.k8s.io/yaml v1.2.0
)

require (
	cloud.google.com/go v0.97.0 // indirect
	github.com/Azure/go-ansiterm v0.0.0-20210617225240-d185dfc1b5a1 // indirect
	github.com/Azure/go-autorest v14.2.0+incompatible // indirect
	github.com/Azure/go-autorest/autorest/date v0.3.0 // indirect
	github.com/Azure/go-autorest/autorest/mocks v0.4.2 // indirect
	github.com/Azure/go-autorest/autorest/validation v0.1.0 // indirect
	github.com/Azure/go-autorest/logger v0.2.1 // indirect
	github.com/Azure/go-autorest/tracing v0.6.0 // indirect
	github.com/MakeNowJust/heredoc v1.0.0 // indirect
	github.com/NYTimes/gziphandler v1.1.1 // indirect
	github.com/PuerkitoBio/purell v1.1.1 // indirect
	github.com/PuerkitoBio/urlesc v0.0.0-20170810143723-de5bf2ad4578 // indirect
	github.com/antlr/antlr4/runtime/Go/antlr v0.0.0-20220418222510-f25a4f6275ed // indirect
	github.com/asaskevich/govalidator v0.0.0-20190424111038-f61b66f89f4a // indirect
	github.com/auth0/go-jwt-middleware v1.0.1 // indirect
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/boltdb/bolt v1.3.1 // indirect
	github.com/cespare/xxhash/v2 v2.1.2 // indirect
	github.com/chai2010/gettext-go v1.0.2 // indirect
	github.com/checkpoint-restore/go-criu/v5 v5.3.0 // indirect
	github.com/cilium/ebpf v0.7.0 // indirect
	github.com/containerd/cgroups v1.0.1 // indirect
	github.com/containerd/console v1.0.3 // indirect
	github.com/containerd/ttrpc v1.0.2 // indirect
	github.com/coredns/caddy v1.1.0 // indirect
	github.com/coreos/go-semver v0.3.0 // indirect
	github.com/cyphar/filepath-securejoin v0.2.3 // indirect
	github.com/daviddengcn/go-colortext v1.0.0 // indirect
	github.com/dustin/go-humanize v1.0.0 // indirect
	github.com/euank/go-kmsg-parser v2.0.0+incompatible // indirect
	github.com/exponent-io/jsonpath v0.0.0-20151013193312-d6023ce2651d // indirect
	github.com/fatih/camelcase v1.0.0 // indirect
	github.com/felixge/httpsnoop v1.0.1 // indirect
	github.com/form3tech-oss/jwt-go v3.2.3+incompatible // indirect
	github.com/fvbommel/sortorder v1.0.1 // indirect
	github.com/go-errors/errors v1.0.1 // indirect
	github.com/go-logr/zapr v1.2.3 // indirect
	github.com/go-openapi/jsonpointer v0.19.5 // indirect
	github.com/go-openapi/jsonreference v0.19.5 // indirect
	github.com/go-openapi/swag v0.19.14 // indirect
	github.com/go-ozzo/ozzo-validation v3.5.0+incompatible // indirect
	github.com/go-task/slim-sprig v0.0.0-20210107165309-348f09dbbbc0 // indirect
	github.com/gofrs/uuid v4.0.0+incompatible // indirect
	github.com/golang-jwt/jwt/v4 v4.2.0 // indirect
	github.com/google/btree v1.0.1 // indirect
	github.com/google/cel-go v0.12.4 // indirect
	github.com/google/pprof v0.0.0-20210407192527-94a9f03dee38 // indirect
	github.com/google/shlex v0.0.0-20191202100458-e7afc7fbc510 // indirect
	github.com/googleapis/gax-go/v2 v2.1.1 // indirect
	github.com/gorilla/mux v1.8.0 // indirect
	github.com/gorilla/websocket v1.4.2 // indirect
	github.com/gregjones/httpcache v0.0.0-20180305231024-9cad4c3443a7 // indirect
	github.com/grpc-ecosystem/go-grpc-middleware v1.3.0 // indirect
	github.com/grpc-ecosystem/go-grpc-prometheus v1.2.0 // indirect
	github.com/grpc-ecosystem/grpc-gateway v1.16.0 // indirect
	github.com/heketi/tests v0.0.0-20151005000721-f3775cbcefd6 // indirect
	github.com/imdario/mergo v0.3.6 // indirect
	github.com/inconshreveable/mousetrap v1.0.0 // indirect
	github.com/jmespath/go-jmespath v0.4.0 // indirect
	github.com/jonboulle/clockwork v0.2.2 // indirect
	github.com/josharian/intern v1.0.0 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/karrick/godirwalk v1.16.1 // indirect
	github.com/liggitt/tabwriter v0.0.0-20181228230101-89fcab3d43de // indirect
	github.com/lpabon/godbc v0.1.1 // indirect
	github.com/mailru/easyjson v0.7.6 // indirect
	github.com/matttproud/golang_protobuf_extensions v1.0.1 // indirect
	github.com/mindprince/gonvml v0.0.0-20190828220739-9ebdce4bb989 // indirect
	github.com/mistifyio/go-zfs v2.1.2-0.20190413222219-f784269be439+incompatible // indirect
	github.com/mitchellh/go-wordwrap v1.0.0 // indirect
	github.com/mitchellh/mapstructure v1.4.1 // indirect
	github.com/moby/spdystream v0.2.0 // indirect
	github.com/moby/sys/mountinfo v0.6.0 // indirect
	github.com/moby/term v0.0.0-20210619224110-3f7ff695adc6 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/mohae/deepcopy v0.0.0-20170603005431-491d3605edfb // indirect
	github.com/monochromegane/go-gitignore v0.0.0-20200626010858-205db1a8cc00 // indirect
	github.com/mxk/go-flowrate v0.0.0-20140419014527-cca7078d478f // indirect
	github.com/opencontainers/go-digest v1.0.0 // indirect
	github.com/opencontainers/runtime-spec v1.0.3-0.20210326190908-1c3f411f0417 // indirect
	github.com/peterbourgon/diskv v2.0.1+incompatible // indirect
	github.com/pquerna/cachecontrol v0.1.0 // indirect
	github.com/prometheus/procfs v0.7.3 // indirect
	github.com/rubiojr/go-vhd v0.0.0-20200706105327-02e210299021 // indirect
	github.com/russross/blackfriday v1.5.2 // indirect
	github.com/russross/blackfriday/v2 v2.1.0 // indirect
	github.com/seccomp/libseccomp-golang v0.9.2-0.20220502022130-f33da4d89646 // indirect
	github.com/sirupsen/logrus v1.8.1 // indirect
	github.com/soheilhy/cmux v0.1.5 // indirect
	github.com/stoewer/go-strcase v1.2.0 // indirect
	github.com/stretchr/objx v0.2.0 // indirect
	github.com/syndtr/gocapability v0.0.0-20200815063812-42c35b437635 // indirect
	github.com/tmc/grpc-websocket-proxy v0.0.0-20201229170055-e5319fda7802 // indirect
	github.com/urfave/negroni v1.0.0 // indirect
	github.com/vishvananda/netns v0.0.0-20200728191858-db3c7e526aae // indirect
	github.com/xiang90/probing v0.0.0-20190116061207-43a291ad63a2 // indirect
	github.com/xlab/treeprint v1.1.0 // indirect
	go.etcd.io/bbolt v1.3.6 // indirect
	go.etcd.io/etcd/client/v2 v2.305.4 // indirect
	go.etcd.io/etcd/pkg/v3 v3.5.4 // indirect
	go.etcd.io/etcd/raft/v3 v3.5.4 // indirect
	go.etcd.io/etcd/server/v3 v3.5.4 // indirect
	go.opencensus.io v0.23.0 // indirect
	go.opentelemetry.io/contrib v0.20.0 // indirect
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.20.0 // indirect
	go.opentelemetry.io/otel/metric v0.20.0 // indirect
	go.opentelemetry.io/otel/sdk/export/metric v0.20.0 // indirect
	go.opentelemetry.io/otel/sdk/metric v0.20.0 // indirect
	go.starlark.net v0.0.0-20200306205701-8dd3e2ee1dd5 // indirect
	go.uber.org/atomic v1.7.0 // indirect
	go.uber.org/multierr v1.6.0 // indirect
	golang.org/x/exp v0.0.0-20210220032938-85be41e4509f // indirect
	golang.org/x/mod v0.6.0-dev.0.20220419223038-86c51ed26bb4 // indirect
	golang.org/x/text v0.3.7 // indirect
	golang.org/x/xerrors v0.0.0-20200804184101-5ec99f83aff1 // indirect
	gonum.org/v1/netlib v0.0.0-20190331212654-76723241ea4e // indirect
	google.golang.org/appengine v1.6.7 // indirect
	gopkg.in/inf.v0 v0.9.1 // indirect
	gopkg.in/natefinch/lumberjack.v2 v2.0.0 // indirect
	gopkg.in/warnings.v0 v0.1.1 // indirect
	sigs.k8s.io/apiserver-network-proxy/konnectivity-client v0.0.32 // indirect
	sigs.k8s.io/json v0.0.0-20220713155537-f223a00ba0e2 // indirect
	sigs.k8s.io/kustomize/api v0.12.1 // indirect
	sigs.k8s.io/kustomize/kustomize/v4 v4.5.7 // indirect
	sigs.k8s.io/kustomize/kyaml v0.13.9 // indirect
)

replace (
	bazil.org/fuse => bazil.org/fuse v0.0.0-20160811212531-371fbbdaa898
	bitbucket.org/bertimus9/systemstat => bitbucket.org/bertimus9/systemstat v0.5.0
	cloud.google.com/go => cloud.google.com/go v0.97.0
	cloud.google.com/go/bigquery => cloud.google.com/go/bigquery v1.8.0
	cloud.google.com/go/storage => cloud.google.com/go/storage v1.10.0
	dmitri.shuralyov.com/gpu/mtl => dmitri.shuralyov.com/gpu/mtl v0.0.0-20201218220906-28db891af037
	github.com/Azure/azure-sdk-for-go => github.com/Azure/azure-sdk-for-go v55.0.0+incompatible
	github.com/Azure/go-ansiterm => github.com/Azure/go-ansiterm v0.0.0-20210617225240-d185dfc1b5a1
	github.com/Azure/go-autorest => github.com/Azure/go-autorest v14.2.0+incompatible
	github.com/Azure/go-autorest/autorest => github.com/Azure/go-autorest/autorest v0.11.27
	github.com/Azure/go-autorest/autorest/adal => github.com/Azure/go-autorest/autorest/adal v0.9.20
	github.com/Azure/go-autorest/autorest/date => github.com/Azure/go-autorest/autorest/date v0.3.0
	github.com/Azure/go-autorest/autorest/mocks => github.com/Azure/go-autorest/autorest/mocks v0.4.2
	github.com/Azure/go-autorest/autorest/to => github.com/Azure/go-autorest/autorest/to v0.4.0
	github.com/Azure/go-autorest/autorest/validation => github.com/Azure/go-autorest/autorest/validation v0.1.0
	github.com/Azure/go-autorest/logger => github.com/Azure/go-autorest/logger v0.2.1
	github.com/Azure/go-autorest/tracing => github.com/Azure/go-autorest/tracing v0.6.0
	github.com/BurntSushi/toml => github.com/BurntSushi/toml v0.3.1
	github.com/BurntSushi/xgb => github.com/BurntSushi/xgb v0.0.0-20160522181843-27f122750802
	github.com/GoogleCloudPlatform/k8s-cloud-provider => github.com/GoogleCloudPlatform/k8s-cloud-provider v1.18.1-0.20220218231025-f11817397a1b
	github.com/JeffAshton/win_pdh => github.com/JeffAshton/win_pdh v0.0.0-20161109143554-76bb4ee9f0ab
	github.com/MakeNowJust/heredoc => github.com/MakeNowJust/heredoc v1.0.0
	github.com/Microsoft/go-winio => github.com/Microsoft/go-winio v0.4.17
	github.com/Microsoft/hcsshim => github.com/Microsoft/hcsshim v0.8.22
	github.com/NYTimes/gziphandler => github.com/NYTimes/gziphandler v1.1.1
	github.com/PuerkitoBio/purell => github.com/PuerkitoBio/purell v1.1.1
	github.com/PuerkitoBio/urlesc => github.com/PuerkitoBio/urlesc v0.0.0-20170810143723-de5bf2ad4578
	github.com/ajstarks/svgo => github.com/ajstarks/svgo v0.0.0-20180226025133-644b8db467af
	github.com/antihax/optional => github.com/antihax/optional v1.0.0
	github.com/antlr/antlr4/runtime/Go/antlr => github.com/antlr/antlr4/runtime/Go/antlr v0.0.0-20220418222510-f25a4f6275ed
	github.com/armon/circbuf => github.com/armon/circbuf v0.0.0-20150827004946-bbbad097214e
	github.com/armon/go-socks5 => github.com/armon/go-socks5 v0.0.0-20160902184237-e75332964ef5
	github.com/asaskevich/govalidator => github.com/asaskevich/govalidator v0.0.0-20190424111038-f61b66f89f4a
	github.com/auth0/go-jwt-middleware => github.com/auth0/go-jwt-middleware v1.0.1
	github.com/aws/aws-sdk-go => github.com/aws/aws-sdk-go v1.38.49
	github.com/benbjohnson/clock => github.com/benbjohnson/clock v1.1.0
	github.com/beorn7/perks => github.com/beorn7/perks v1.0.1
	github.com/blang/semver => github.com/blang/semver v3.5.1+incompatible
	github.com/blang/semver/v4 => github.com/blang/semver/v4 v4.0.0
	github.com/boltdb/bolt => github.com/boltdb/bolt v1.3.1
	github.com/census-instrumentation/opencensus-proto => github.com/census-instrumentation/opencensus-proto v0.2.1
	github.com/certifi/gocertifi => github.com/certifi/gocertifi v0.0.0-20200922220541-2c3bb06c6054
	github.com/cespare/xxhash/v2 => github.com/cespare/xxhash/v2 v2.1.2
	github.com/chai2010/gettext-go => github.com/chai2010/gettext-go v1.0.2
	github.com/checkpoint-restore/go-criu/v5 => github.com/checkpoint-restore/go-criu/v5 v5.3.0
	github.com/chzyer/logex => github.com/chzyer/logex v1.1.10
	github.com/chzyer/readline => github.com/chzyer/readline v0.0.0-20180603132655-2972be24d48e
	github.com/chzyer/test => github.com/chzyer/test v0.0.0-20180213035817-a1ea475d72b1
	github.com/cilium/ebpf => github.com/cilium/ebpf v0.7.0
	github.com/cncf/udpa/go => github.com/cncf/udpa/go v0.0.0-20210930031921-04548b0d99d4
	github.com/cncf/xds/go => github.com/cncf/xds/go v0.0.0-20211011173535-cb28da3451f1
	github.com/cockroachdb/datadriven => github.com/cockroachdb/datadriven v0.0.0-20200714090401-bf6692d28da5
	github.com/cockroachdb/errors => github.com/cockroachdb/errors v1.2.4
	github.com/cockroachdb/logtags => github.com/cockroachdb/logtags v0.0.0-20190617123548-eb05cc24525f
	github.com/container-storage-interface/spec => github.com/container-storage-interface/spec v1.6.0
	github.com/containerd/cgroups => github.com/containerd/cgroups v1.0.1
	github.com/containerd/console => github.com/containerd/console v1.0.3
	github.com/containerd/containerd => github.com/containerd/containerd v1.4.9
	github.com/containerd/continuity => github.com/containerd/continuity v0.1.0
	github.com/containerd/fifo => github.com/containerd/fifo v1.0.0
	github.com/containerd/go-runc => github.com/containerd/go-runc v1.0.0
	github.com/containerd/ttrpc => github.com/containerd/ttrpc v1.0.2
	github.com/containerd/typeurl => github.com/containerd/typeurl v1.0.2
	github.com/coredns/caddy => github.com/coredns/caddy v1.1.0
	github.com/coredns/corefile-migration => github.com/coredns/corefile-migration v1.0.17
	github.com/coreos/go-oidc => github.com/coreos/go-oidc v2.1.0+incompatible
	github.com/coreos/go-semver => github.com/coreos/go-semver v0.3.0
	github.com/coreos/go-systemd/v22 => github.com/coreos/go-systemd/v22 v22.3.2
	github.com/cpuguy83/go-md2man/v2 => github.com/cpuguy83/go-md2man/v2 v2.0.1
	github.com/creack/pty => github.com/creack/pty v1.1.11
	github.com/cyphar/filepath-securejoin => github.com/cyphar/filepath-securejoin v0.2.3
	github.com/davecgh/go-spew => github.com/davecgh/go-spew v1.1.1
	github.com/daviddengcn/go-colortext => github.com/daviddengcn/go-colortext v1.0.0
	github.com/dnaeon/go-vcr => github.com/dnaeon/go-vcr v1.0.1
	github.com/docker/distribution => github.com/docker/distribution v2.8.1+incompatible
	github.com/docker/docker => github.com/docker/docker v20.10.17+incompatible
	github.com/docker/go-connections => github.com/docker/go-connections v0.4.0
	github.com/docker/go-units => github.com/docker/go-units v0.4.0
	github.com/docopt/docopt-go => github.com/docopt/docopt-go v0.0.0-20180111231733-ee0de3bc6815
	github.com/dustin/go-humanize => github.com/dustin/go-humanize v1.0.0
	github.com/elazarl/goproxy => github.com/elazarl/goproxy v0.0.0-20180725130230-947c36da3153 // 947c36da3153 is the SHA for git tag v1.11
	github.com/emicklei/go-restful/v3 => github.com/emicklei/go-restful/v3 v3.8.0
	github.com/envoyproxy/go-control-plane => github.com/envoyproxy/go-control-plane v0.10.2-0.20220325020618-49ff273808a1
	github.com/envoyproxy/protoc-gen-validate => github.com/envoyproxy/protoc-gen-validate v0.1.0
	github.com/euank/go-kmsg-parser => github.com/euank/go-kmsg-parser v2.0.0+incompatible
	github.com/evanphx/json-patch => github.com/evanphx/json-patch v4.12.0+incompatible
	github.com/exponent-io/jsonpath => github.com/exponent-io/jsonpath v0.0.0-20151013193312-d6023ce2651d
	github.com/fatih/camelcase => github.com/fatih/camelcase v1.0.0
	github.com/felixge/httpsnoop => github.com/felixge/httpsnoop v1.0.1
	github.com/flynn/go-shlex => github.com/flynn/go-shlex v0.0.0-20150515145356-3f9db97f8568
	github.com/fogleman/gg => github.com/fogleman/gg v1.2.1-0.20190220221249-0403632d5b90
	github.com/form3tech-oss/jwt-go => github.com/form3tech-oss/jwt-go v3.2.3+incompatible
	github.com/frankban/quicktest => github.com/frankban/quicktest v1.11.3
	github.com/fsnotify/fsnotify => github.com/fsnotify/fsnotify v1.4.9
	github.com/fvbommel/sortorder => github.com/fvbommel/sortorder v1.0.1
	github.com/getkin/kin-openapi => github.com/getkin/kin-openapi v0.76.0
	github.com/getsentry/raven-go => github.com/getsentry/raven-go v0.2.0
	github.com/ghodss/yaml => github.com/ghodss/yaml v1.0.0
	github.com/go-errors/errors => github.com/go-errors/errors v1.0.1
	github.com/go-gl/glfw/v3.3/glfw => github.com/go-gl/glfw/v3.3/glfw v0.0.0-20200222043503-6f7a984d4dc4
	github.com/go-kit/kit => github.com/go-kit/kit v0.9.0
	github.com/go-kit/log => github.com/go-kit/log v0.1.0
	github.com/go-logfmt/logfmt => github.com/go-logfmt/logfmt v0.5.0
	github.com/go-logr/logr => github.com/go-logr/logr v1.2.3
	github.com/go-logr/zapr => github.com/go-logr/zapr v1.2.3
	github.com/go-openapi/jsonpointer => github.com/go-openapi/jsonpointer v0.19.5
	github.com/go-openapi/jsonreference => github.com/go-openapi/jsonreference v0.19.5
	github.com/go-openapi/swag => github.com/go-openapi/swag v0.19.14
	github.com/go-ozzo/ozzo-validation => github.com/go-ozzo/ozzo-validation v3.5.0+incompatible
	github.com/go-stack/stack => github.com/go-stack/stack v1.8.0
	github.com/go-task/slim-sprig => github.com/go-task/slim-sprig v0.0.0-20210107165309-348f09dbbbc0
	github.com/godbus/dbus/v5 => github.com/godbus/dbus/v5 v5.0.6
	github.com/gofrs/uuid => github.com/gofrs/uuid v4.0.0+incompatible
	github.com/gogo/googleapis => github.com/gogo/googleapis v1.4.1
	github.com/gogo/protobuf => github.com/gogo/protobuf v1.3.2
	github.com/golang-jwt/jwt/v4 => github.com/golang-jwt/jwt/v4 v4.2.0
	github.com/golang/freetype => github.com/golang/freetype v0.0.0-20170609003504-e2365dfdc4a0
	github.com/golang/glog => github.com/golang/glog v1.0.0
	github.com/golang/groupcache => github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da
	github.com/golang/mock => github.com/golang/mock v1.6.0
	github.com/golang/protobuf => github.com/golang/protobuf v1.5.2
	github.com/golang/snappy => github.com/golang/snappy v0.0.3
	github.com/golangplus/bytes => github.com/golangplus/bytes v1.0.0
	github.com/golangplus/fmt => github.com/golangplus/fmt v1.0.0
	github.com/golangplus/testing => github.com/golangplus/testing v1.0.0
	github.com/google/btree => github.com/google/btree v1.0.1
	github.com/google/cadvisor => github.com/google/cadvisor v0.45.0
	github.com/google/cel-go => github.com/google/cel-go v0.12.4
	github.com/google/gnostic => github.com/google/gnostic v0.5.7-v3refs
	github.com/google/go-cmp => github.com/google/go-cmp v0.5.6
	github.com/google/gofuzz => github.com/google/gofuzz v1.1.0
	github.com/google/martian/v3 => github.com/google/martian/v3 v3.2.1
	github.com/google/pprof => github.com/google/pprof v0.0.0-20210407192527-94a9f03dee38
	github.com/google/renameio => github.com/google/renameio v0.1.0
	github.com/google/shlex => github.com/google/shlex v0.0.0-20191202100458-e7afc7fbc510
	github.com/google/uuid => github.com/google/uuid v1.1.2
	github.com/googleapis/gax-go/v2 => github.com/googleapis/gax-go/v2 v2.1.1
	github.com/gopherjs/gopherjs => github.com/gopherjs/gopherjs v0.0.0-20200217142428-fce0ec30dd00
	github.com/gorilla/mux => github.com/gorilla/mux v1.8.0
	github.com/gorilla/websocket => github.com/gorilla/websocket v1.4.2
	github.com/gregjones/httpcache => github.com/gregjones/httpcache v0.0.0-20180305231024-9cad4c3443a7
	github.com/grpc-ecosystem/go-grpc-middleware => github.com/grpc-ecosystem/go-grpc-middleware v1.3.0
	github.com/grpc-ecosystem/go-grpc-prometheus => github.com/grpc-ecosystem/go-grpc-prometheus v1.2.0
	github.com/grpc-ecosystem/grpc-gateway => github.com/grpc-ecosystem/grpc-gateway v1.16.0
	github.com/heketi/heketi => github.com/heketi/heketi v10.3.0+incompatible
	github.com/heketi/tests => github.com/heketi/tests v0.0.0-20151005000721-f3775cbcefd6
	github.com/ianlancetaylor/demangle => github.com/ianlancetaylor/demangle v0.0.0-20200824232613-28f6c0f3b639
	github.com/imdario/mergo => github.com/imdario/mergo v0.3.6
	github.com/inconshreveable/mousetrap => github.com/inconshreveable/mousetrap v1.0.0
	github.com/ishidawataru/sctp => github.com/ishidawataru/sctp v0.0.0-20190723014705-7c296d48a2b5
	github.com/jmespath/go-jmespath => github.com/jmespath/go-jmespath v0.4.0
	github.com/jmespath/go-jmespath/internal/testify => github.com/jmespath/go-jmespath/internal/testify v1.5.1
	github.com/jonboulle/clockwork => github.com/jonboulle/clockwork v0.2.2
	github.com/josharian/intern => github.com/josharian/intern v1.0.0
	github.com/json-iterator/go => github.com/json-iterator/go v1.1.12
	github.com/jtolds/gls => github.com/jtolds/gls v4.20.0+incompatible
	github.com/julienschmidt/httprouter => github.com/julienschmidt/httprouter v1.3.0
	github.com/jung-kurt/gofpdf => github.com/jung-kurt/gofpdf v1.0.3-0.20190309125859-24315acbbda5
	github.com/karrick/godirwalk => github.com/karrick/godirwalk v1.16.1
	github.com/kisielk/errcheck => github.com/kisielk/errcheck v1.5.0
	github.com/kisielk/gotool => github.com/kisielk/gotool v1.0.0
	github.com/konsorten/go-windows-terminal-sequences => github.com/konsorten/go-windows-terminal-sequences v1.0.2
	github.com/kr/fs => github.com/kr/fs v0.1.0
	github.com/kr/pretty => github.com/kr/pretty v0.2.1
	github.com/kr/text => github.com/kr/text v0.2.0
	github.com/libopenstorage/openstorage => github.com/libopenstorage/openstorage v1.0.0
	github.com/liggitt/tabwriter => github.com/liggitt/tabwriter v0.0.0-20181228230101-89fcab3d43de
	github.com/lithammer/dedent => github.com/lithammer/dedent v1.1.0
	github.com/lpabon/godbc => github.com/lpabon/godbc v0.1.1
	github.com/mailru/easyjson => github.com/mailru/easyjson v0.7.6
	github.com/mattn/go-runewidth => github.com/mattn/go-runewidth v0.0.7
	github.com/matttproud/golang_protobuf_extensions => github.com/matttproud/golang_protobuf_extensions v1.0.1
	github.com/mindprince/gonvml => github.com/mindprince/gonvml v0.0.0-20190828220739-9ebdce4bb989
	github.com/mistifyio/go-zfs => github.com/mistifyio/go-zfs v2.1.2-0.20190413222219-f784269be439+incompatible
	github.com/mitchellh/go-wordwrap => github.com/mitchellh/go-wordwrap v1.0.0
	github.com/mitchellh/mapstructure => github.com/mitchellh/mapstructure v1.4.1
	github.com/moby/ipvs => github.com/moby/ipvs v1.0.1
	github.com/moby/spdystream => github.com/moby/spdystream v0.2.0
	github.com/moby/sys/mountinfo => github.com/moby/sys/mountinfo v0.6.0
	github.com/moby/term => github.com/moby/term v0.0.0-20210619224110-3f7ff695adc6
	github.com/modern-go/concurrent => github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd
	github.com/modern-go/reflect2 => github.com/modern-go/reflect2 v1.0.2
	github.com/mohae/deepcopy => github.com/mohae/deepcopy v0.0.0-20170603005431-491d3605edfb
	github.com/monochromegane/go-gitignore => github.com/monochromegane/go-gitignore v0.0.0-20200626010858-205db1a8cc00
	github.com/morikuni/aec => github.com/morikuni/aec v1.0.0
	github.com/mrunalp/fileutils => github.com/mrunalp/fileutils v0.5.0
	github.com/munnerz/goautoneg => github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822
	github.com/mvdan/xurls => github.com/mvdan/xurls v1.1.0
	github.com/mwitkow/go-conntrack => github.com/mwitkow/go-conntrack v0.0.0-20190716064945-2f068394615f
	github.com/mxk/go-flowrate => github.com/mxk/go-flowrate v0.0.0-20140419014527-cca7078d478f
	github.com/niemeyer/pretty => github.com/niemeyer/pretty v0.0.0-20200227124842-a10e7caefd8e
	github.com/olekukonko/tablewriter => github.com/olekukonko/tablewriter v0.0.4
	github.com/onsi/ginkgo/v2 => github.com/onsi/ginkgo/v2 v2.1.4
	github.com/onsi/gomega => github.com/onsi/gomega v1.19.0
	github.com/opencontainers/go-digest => github.com/opencontainers/go-digest v1.0.0
	github.com/opencontainers/image-spec => github.com/opencontainers/image-spec v1.0.2
	github.com/opencontainers/runc => github.com/opencontainers/runc v1.1.3
	github.com/opencontainers/runtime-spec => github.com/opencontainers/runtime-spec v1.0.3-0.20210326190908-1c3f411f0417
	github.com/opencontainers/selinux => github.com/opencontainers/selinux v1.10.0
	github.com/opentracing/opentracing-go => github.com/opentracing/opentracing-go v1.1.0
	github.com/peterbourgon/diskv => github.com/peterbourgon/diskv v2.0.1+incompatible
	github.com/pkg/errors => github.com/pkg/errors v0.9.1
	github.com/pkg/sftp => github.com/pkg/sftp v1.10.1
	github.com/pmezard/go-difflib => github.com/pmezard/go-difflib v1.0.0
	github.com/pquerna/cachecontrol => github.com/pquerna/cachecontrol v0.1.0
	github.com/prometheus/client_golang => github.com/prometheus/client_golang v1.12.1
	github.com/prometheus/client_model => github.com/prometheus/client_model v0.2.0
	github.com/prometheus/common => github.com/prometheus/common v0.32.1
	github.com/prometheus/procfs => github.com/prometheus/procfs v0.7.3
	github.com/remyoudompheng/bigfft => github.com/remyoudompheng/bigfft v0.0.0-20170806203942-52369c62f446
	github.com/robfig/cron/v3 => github.com/robfig/cron/v3 v3.0.1
	github.com/rogpeppe/fastuuid => github.com/rogpeppe/fastuuid v1.2.0
	github.com/rogpeppe/go-internal => github.com/rogpeppe/go-internal v1.3.0
	github.com/rubiojr/go-vhd => github.com/rubiojr/go-vhd v0.0.0-20200706105327-02e210299021
	github.com/russross/blackfriday => github.com/russross/blackfriday v1.5.2
	github.com/russross/blackfriday/v2 => github.com/russross/blackfriday/v2 v2.1.0
	github.com/seccomp/libseccomp-golang => github.com/seccomp/libseccomp-golang v0.9.2-0.20220502022130-f33da4d89646
	github.com/sergi/go-diff => github.com/sergi/go-diff v1.1.0
	github.com/sirupsen/logrus => github.com/sirupsen/logrus v1.8.1
	github.com/smartystreets/assertions => github.com/smartystreets/assertions v1.1.0
	github.com/smartystreets/goconvey => github.com/smartystreets/goconvey v1.6.4
	github.com/soheilhy/cmux => github.com/soheilhy/cmux v0.1.5
	github.com/spf13/afero => github.com/spf13/afero v1.6.0
	github.com/spf13/cobra => github.com/spf13/cobra v1.4.0
	github.com/spf13/pflag => github.com/spf13/pflag v1.0.5
	github.com/stoewer/go-strcase => github.com/stoewer/go-strcase v1.2.0
	github.com/stretchr/objx => github.com/stretchr/objx v0.2.0
	github.com/stretchr/testify => github.com/stretchr/testify v1.7.0
	github.com/syndtr/gocapability => github.com/syndtr/gocapability v0.0.0-20200815063812-42c35b437635
	github.com/tmc/grpc-websocket-proxy => github.com/tmc/grpc-websocket-proxy v0.0.0-20201229170055-e5319fda7802
	github.com/urfave/cli => github.com/urfave/cli v1.22.2
	github.com/urfave/negroni => github.com/urfave/negroni v1.0.0
	github.com/vishvananda/netlink => github.com/vishvananda/netlink v1.1.0
	github.com/vishvananda/netns => github.com/vishvananda/netns v0.0.0-20200728191858-db3c7e526aae
	github.com/vmware/govmomi => github.com/vmware/govmomi v0.20.3
	github.com/xiang90/probing => github.com/xiang90/probing v0.0.0-20190116061207-43a291ad63a2
	github.com/xlab/treeprint => github.com/xlab/treeprint v1.1.0
	github.com/yuin/goldmark => github.com/yuin/goldmark v1.4.13
	go.etcd.io/bbolt => go.etcd.io/bbolt v1.3.6
	go.etcd.io/etcd/api/v3 => go.etcd.io/etcd/api/v3 v3.5.4
	go.etcd.io/etcd/client/pkg/v3 => go.etcd.io/etcd/client/pkg/v3 v3.5.4
	go.etcd.io/etcd/client/v2 => go.etcd.io/etcd/client/v2 v2.305.4
	go.etcd.io/etcd/client/v3 => go.etcd.io/etcd/client/v3 v3.5.4
	go.etcd.io/etcd/pkg/v3 => go.etcd.io/etcd/pkg/v3 v3.5.4
	go.etcd.io/etcd/raft/v3 => go.etcd.io/etcd/raft/v3 v3.5.4
	go.etcd.io/etcd/server/v3 => go.etcd.io/etcd/server/v3 v3.5.4
	go.opencensus.io => go.opencensus.io v0.23.0
	go.opentelemetry.io/contrib => go.opentelemetry.io/contrib v0.20.0
	go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful => go.opentelemetry.io/contrib/instrumentation/github.com/emicklei/go-restful/otelrestful v0.20.0
	go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc => go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc v0.20.0
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp => go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.20.0
	go.opentelemetry.io/contrib/propagators => go.opentelemetry.io/contrib/propagators v0.20.0
	go.opentelemetry.io/otel => go.opentelemetry.io/otel v0.20.0
	go.opentelemetry.io/otel/exporters/otlp => go.opentelemetry.io/otel/exporters/otlp v0.20.0
	go.opentelemetry.io/otel/metric => go.opentelemetry.io/otel/metric v0.20.0
	go.opentelemetry.io/otel/oteltest => go.opentelemetry.io/otel/oteltest v0.20.0
	go.opentelemetry.io/otel/sdk => go.opentelemetry.io/otel/sdk v0.20.0
	go.opentelemetry.io/otel/sdk/export/metric => go.opentelemetry.io/otel/sdk/export/metric v0.20.0
	go.opentelemetry.io/otel/sdk/metric => go.opentelemetry.io/otel/sdk/metric v0.20.0
	go.opentelemetry.io/otel/trace => go.opentelemetry.io/otel/trace v0.20.0
	go.opentelemetry.io/proto/otlp => go.opentelemetry.io/proto/otlp v0.7.0
	go.starlark.net => go.starlark.net v0.0.0-20200306205701-8dd3e2ee1dd5
	go.uber.org/atomic => go.uber.org/atomic v1.7.0
	go.uber.org/goleak => go.uber.org/goleak v1.1.10
	go.uber.org/multierr => go.uber.org/multierr v1.6.0
	go.uber.org/zap => go.uber.org/zap v1.19.0
	golang.org/x/crypto => golang.org/x/crypto v0.0.0-20220315160706-3147a52a75dd
	golang.org/x/exp => golang.org/x/exp v0.0.0-20210220032938-85be41e4509f
	golang.org/x/image => golang.org/x/image v0.0.0-20190802002840-cff245a6509b
	golang.org/x/lint => golang.org/x/lint v0.0.0-20210508222113-6edffad5e616
	golang.org/x/mobile => golang.org/x/mobile v0.0.0-20201217150744-e6ae53a27f4f
	golang.org/x/mod => golang.org/x/mod v0.6.0-dev.0.20220419223038-86c51ed26bb4
	golang.org/x/net => golang.org/x/net v0.0.0-20220722155237-a158d28d115b
	golang.org/x/oauth2 => golang.org/x/oauth2 v0.0.0-20211104180415-d3ed0bb246c8
	golang.org/x/sync => golang.org/x/sync v0.0.0-20220722155255-886fb9371eb4
	golang.org/x/sys => golang.org/x/sys v0.0.0-20220722155257-8c9f86f7a55f
	golang.org/x/term => golang.org/x/term v0.0.0-20210927222741-03fcf44c2211
	golang.org/x/text => golang.org/x/text v0.3.7
	golang.org/x/time => golang.org/x/time v0.0.0-20220210224613-90d013bbcef8
	golang.org/x/tools => golang.org/x/tools v0.1.12
	golang.org/x/xerrors => golang.org/x/xerrors v0.0.0-20200804184101-5ec99f83aff1
	gonum.org/v1/gonum => gonum.org/v1/gonum v0.6.2
	gonum.org/v1/netlib => gonum.org/v1/netlib v0.0.0-20190331212654-76723241ea4e
	gonum.org/v1/plot => gonum.org/v1/plot v0.0.0-20190515093506-e2840ee46a6b
	google.golang.org/api => google.golang.org/api v0.60.0
	google.golang.org/appengine => google.golang.org/appengine v1.6.7
	google.golang.org/genproto => google.golang.org/genproto v0.0.0-20220502173005-c8bf987b8c21
	google.golang.org/grpc => google.golang.org/grpc v1.47.0
	google.golang.org/grpc/cmd/protoc-gen-go-grpc => google.golang.org/grpc/cmd/protoc-gen-go-grpc v1.1.0
	google.golang.org/protobuf => google.golang.org/protobuf v1.28.0
	gopkg.in/alecthomas/kingpin.v2 => gopkg.in/alecthomas/kingpin.v2 v2.2.6
	gopkg.in/check.v1 => gopkg.in/check.v1 v1.0.0-20200227125254-8fa46927fb4f
	gopkg.in/errgo.v2 => gopkg.in/errgo.v2 v2.1.0
	gopkg.in/gcfg.v1 => gopkg.in/gcfg.v1 v1.2.0
	gopkg.in/inf.v0 => gopkg.in/inf.v0 v0.9.1
	gopkg.in/natefinch/lumberjack.v2 => gopkg.in/natefinch/lumberjack.v2 v2.0.0
	gopkg.in/square/go-jose.v2 => gopkg.in/square/go-jose.v2 v2.2.2
	gopkg.in/warnings.v0 => gopkg.in/warnings.v0 v0.1.1
	gopkg.in/yaml.v2 => gopkg.in/yaml.v2 v2.4.0
	gopkg.in/yaml.v3 => gopkg.in/yaml.v3 v3.0.1
	gotest.tools/v3 => gotest.tools/v3 v3.0.3
	honnef.co/go/tools => honnef.co/go/tools v0.0.1-2020.1.4
	k8s.io/api => ./staging/src/k8s.io/api
	k8s.io/apiextensions-apiserver => ./staging/src/k8s.io/apiextensions-apiserver
	k8s.io/apimachinery => ./staging/src/k8s.io/apimachinery
	k8s.io/apiserver => ./staging/src/k8s.io/apiserver
	k8s.io/cli-runtime => ./staging/src/k8s.io/cli-runtime
	k8s.io/client-go => ./staging/src/k8s.io/client-go
	k8s.io/cloud-provider => ./staging/src/k8s.io/cloud-provider
	k8s.io/cluster-bootstrap => ./staging/src/k8s.io/cluster-bootstrap
	k8s.io/code-generator => ./staging/src/k8s.io/code-generator
	k8s.io/component-base => ./staging/src/k8s.io/component-base
	k8s.io/component-helpers => ./staging/src/k8s.io/component-helpers
	k8s.io/controller-manager => ./staging/src/k8s.io/controller-manager
	k8s.io/cri-api => ./staging/src/k8s.io/cri-api
	k8s.io/csi-translation-lib => ./staging/src/k8s.io/csi-translation-lib
	k8s.io/gengo => k8s.io/gengo v0.0.0-20211129171323-c02415ce4185
	k8s.io/klog/v2 => k8s.io/klog/v2 v2.70.1
	k8s.io/kube-aggregator => ./staging/src/k8s.io/kube-aggregator
	k8s.io/kube-controller-manager => ./staging/src/k8s.io/kube-controller-manager
	k8s.io/kube-openapi => k8s.io/kube-openapi v0.0.0-20220803162953-67bda5d908f1
	k8s.io/kube-proxy => ./staging/src/k8s.io/kube-proxy
	k8s.io/kube-scheduler => ./staging/src/k8s.io/kube-scheduler
	k8s.io/kubectl => ./staging/src/k8s.io/kubectl
	k8s.io/kubelet => ./staging/src/k8s.io/kubelet
	k8s.io/legacy-cloud-providers => ./staging/src/k8s.io/legacy-cloud-providers
	k8s.io/metrics => ./staging/src/k8s.io/metrics
	k8s.io/mount-utils => ./staging/src/k8s.io/mount-utils
	k8s.io/pod-security-admission => ./staging/src/k8s.io/pod-security-admission
	k8s.io/sample-apiserver => ./staging/src/k8s.io/sample-apiserver
	k8s.io/sample-cli-plugin => ./staging/src/k8s.io/sample-cli-plugin
	k8s.io/sample-controller => ./staging/src/k8s.io/sample-controller
	k8s.io/system-validators => k8s.io/system-validators v1.7.0
	k8s.io/utils => k8s.io/utils v0.0.0-20220728103510-ee6ede2d64ed
	modernc.org/cc => modernc.org/cc v1.0.0
	modernc.org/golex => modernc.org/golex v1.0.0
	modernc.org/mathutil => modernc.org/mathutil v1.0.0
	modernc.org/strutil => modernc.org/strutil v1.0.0
	modernc.org/xc => modernc.org/xc v1.0.0
	rsc.io/pdf => rsc.io/pdf v0.1.1
	sigs.k8s.io/apiserver-network-proxy/konnectivity-client => sigs.k8s.io/apiserver-network-proxy/konnectivity-client v0.0.32
	sigs.k8s.io/json => sigs.k8s.io/json v0.0.0-20220713155537-f223a00ba0e2
	sigs.k8s.io/kustomize/api => sigs.k8s.io/kustomize/api v0.12.1
	sigs.k8s.io/kustomize/cmd/config => sigs.k8s.io/kustomize/cmd/config v0.10.9
	sigs.k8s.io/kustomize/kustomize/v4 => sigs.k8s.io/kustomize/kustomize/v4 v4.5.7
	sigs.k8s.io/kustomize/kyaml => sigs.k8s.io/kustomize/kyaml v0.13.9
	sigs.k8s.io/structured-merge-diff/v4 => sigs.k8s.io/structured-merge-diff/v4 v4.2.3
	sigs.k8s.io/yaml => sigs.k8s.io/yaml v1.2.0
)
