// This is a generated file. Do not edit directly.
// Ensure you've carefully read
// https://git.k8s.io/community/contributors/devel/sig-architecture/vendor.md
// Run hack/pin-dependency.sh to change pinned dependency versions.
// Run hack/update-vendor.sh to update go.mod files and the vendor directory.

module k8s.io/kubernetes

go 1.16

require (
	bitbucket.org/bertimus9/systemstat v0.0.0-20180207000608-0eeff89b0690
	github.com/Azure/azure-sdk-for-go v55.0.0+incompatible
	github.com/Azure/go-autorest/autorest v0.11.18
	github.com/Azure/go-autorest/autorest/adal v0.9.13
	github.com/Azure/go-autorest/autorest/to v0.4.0
	github.com/GoogleCloudPlatform/k8s-cloud-provider v0.0.0-20200415212048-7901bc822317
	github.com/JeffAshton/win_pdh v0.0.0-20161109143554-76bb4ee9f0ab
	github.com/Microsoft/go-winio v0.4.15
	github.com/Microsoft/hcsshim v0.8.10-0.20200715222032-5eafd1556990
	github.com/PuerkitoBio/purell v1.1.1
	github.com/armon/circbuf v0.0.0-20150827004946-bbbad097214e
	github.com/auth0/go-jwt-middleware v1.0.1 // indirect
	github.com/aws/aws-sdk-go v1.38.49
	github.com/blang/semver v3.5.1+incompatible
	github.com/boltdb/bolt v1.3.1 // indirect
	github.com/clusterhq/flocker-go v0.0.0-20160920122132-2b8b7259d313
	github.com/container-storage-interface/spec v1.5.0
	github.com/containernetworking/cni v0.8.1
	github.com/coredns/corefile-migration v1.0.12
	github.com/coreos/go-oidc v2.1.0+incompatible
	github.com/coreos/go-systemd/v22 v22.3.2
	github.com/cpuguy83/go-md2man/v2 v2.0.0
	github.com/davecgh/go-spew v1.1.1
	github.com/docker/distribution v2.7.1+incompatible
	github.com/docker/docker v20.10.2+incompatible
	github.com/docker/go-connections v0.4.0
	github.com/docker/go-units v0.4.0
	github.com/elazarl/goproxy v0.0.0-20180725130230-947c36da3153
	github.com/emicklei/go-restful v2.9.5+incompatible
	github.com/evanphx/json-patch v4.11.0+incompatible
	github.com/fsnotify/fsnotify v1.4.9
	github.com/go-ozzo/ozzo-validation v3.5.0+incompatible // indirect
	github.com/godbus/dbus/v5 v5.0.4
	github.com/gogo/protobuf v1.3.2
	github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da
	github.com/golang/mock v1.4.4
	github.com/golang/protobuf v1.5.2
	github.com/google/cadvisor v0.39.2
	github.com/google/go-cmp v0.5.5
	github.com/google/gofuzz v1.1.0
	github.com/google/uuid v1.1.2
	github.com/googleapis/gnostic v0.5.5
	github.com/heketi/heketi v10.3.0+incompatible
	github.com/heketi/tests v0.0.0-20151005000721-f3775cbcefd6 // indirect
	github.com/ishidawataru/sctp v0.0.0-20190723014705-7c296d48a2b5
	github.com/json-iterator/go v1.1.11
	github.com/libopenstorage/openstorage v1.0.0
	github.com/lithammer/dedent v1.1.0
	github.com/lpabon/godbc v0.1.1 // indirect
	github.com/moby/ipvs v1.0.1
	github.com/mohae/deepcopy v0.0.0-20170603005431-491d3605edfb // indirect
	github.com/mrunalp/fileutils v0.5.0
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822
	github.com/mvdan/xurls v1.1.0
	github.com/onsi/ginkgo v1.14.0
	github.com/onsi/gomega v1.10.1
	github.com/opencontainers/go-digest v1.0.0
	github.com/opencontainers/runc v1.0.0-rc95
	github.com/opencontainers/selinux v1.8.0
	github.com/pkg/errors v0.9.1
	github.com/pmezard/go-difflib v1.0.0
	github.com/prometheus/client_golang v1.11.0
	github.com/prometheus/client_model v0.2.0
	github.com/prometheus/common v0.26.0
	github.com/quobyte/api v0.1.8
	github.com/robfig/cron/v3 v3.0.1
	github.com/spf13/cobra v1.1.3
	github.com/spf13/pflag v1.0.5
	github.com/storageos/go-api v2.2.0+incompatible
	github.com/stretchr/testify v1.7.0
	github.com/vishvananda/netlink v1.1.0
	github.com/vmware/govmomi v0.20.3
	go.etcd.io/etcd/client/pkg/v3 v3.5.0
	go.etcd.io/etcd/client/v3 v3.5.0
	go.opentelemetry.io/otel/trace v0.20.0
	golang.org/x/crypto v0.0.0-20210220033148-5ea612d1eb83
	golang.org/x/exp v0.0.0-20210220032938-85be41e4509f // indirect
	golang.org/x/net v0.0.0-20210520170846-37e1c6afe023
	golang.org/x/oauth2 v0.0.0-20200107190931-bf48bf16ab8d
	golang.org/x/sync v0.0.0-20210220032951-036812b2e83c
	golang.org/x/sys v0.0.0-20210616094352-59db8d763f22
	golang.org/x/term v0.0.0-20210220032956-6a3ed077a48d
	golang.org/x/time v0.0.0-20210220033141-f8bda1e9f3ba
	golang.org/x/tools v0.1.2
	gonum.org/v1/gonum v0.6.2
	gonum.org/v1/netlib v0.0.0-20190331212654-76723241ea4e // indirect
	google.golang.org/api v0.20.0
	google.golang.org/appengine v1.6.5 // indirect
	google.golang.org/genproto v0.0.0-20210602131652-f16073e35f0c
	google.golang.org/grpc v1.38.0
	google.golang.org/protobuf v1.26.0
	gopkg.in/gcfg.v1 v1.2.0
	gopkg.in/square/go-jose.v2 v2.2.2
	gopkg.in/yaml.v2 v2.4.0
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
	k8s.io/gengo v0.0.0-20201214224949-b6c5ce23f027
	k8s.io/klog/v2 v2.9.0
	k8s.io/kube-aggregator v0.0.0
	k8s.io/kube-controller-manager v0.0.0
	k8s.io/kube-openapi v0.0.0-20210421082810-95288971da7e
	k8s.io/kube-proxy v0.0.0
	k8s.io/kube-scheduler v0.0.0
	k8s.io/kubectl v0.0.0
	k8s.io/kubelet v0.0.0
	k8s.io/legacy-cloud-providers v0.0.0
	k8s.io/metrics v0.0.0
	k8s.io/mount-utils v0.0.0
	k8s.io/pod-security-admission v0.0.0
	k8s.io/sample-apiserver v0.0.0
	k8s.io/system-validators v1.5.0
	k8s.io/utils v0.0.0-20210707171843-4b05e18ac7d9
	sigs.k8s.io/structured-merge-diff/v4 v4.1.2
	sigs.k8s.io/yaml v1.2.0
)

replace (
	bitbucket.org/bertimus9/systemstat => bitbucket.org/bertimus9/systemstat v0.0.0-20180207000608-0eeff89b0690
	cloud.google.com/go => cloud.google.com/go v0.54.0
	cloud.google.com/go/bigquery => cloud.google.com/go/bigquery v1.4.0
	cloud.google.com/go/datastore => cloud.google.com/go/datastore v1.1.0
	cloud.google.com/go/firestore => cloud.google.com/go/firestore v1.1.0
	cloud.google.com/go/pubsub => cloud.google.com/go/pubsub v1.2.0
	cloud.google.com/go/storage => cloud.google.com/go/storage v1.6.0
	dmitri.shuralyov.com/gpu/mtl => dmitri.shuralyov.com/gpu/mtl v0.0.0-20201218220906-28db891af037
	github.com/Azure/azure-sdk-for-go => github.com/Azure/azure-sdk-for-go v55.0.0+incompatible
	github.com/Azure/go-ansiterm => github.com/Azure/go-ansiterm v0.0.0-20210617225240-d185dfc1b5a1
	github.com/Azure/go-autorest => github.com/Azure/go-autorest v14.2.0+incompatible
	github.com/Azure/go-autorest/autorest => github.com/Azure/go-autorest/autorest v0.11.18
	github.com/Azure/go-autorest/autorest/adal => github.com/Azure/go-autorest/autorest/adal v0.9.13
	github.com/Azure/go-autorest/autorest/date => github.com/Azure/go-autorest/autorest/date v0.3.0
	github.com/Azure/go-autorest/autorest/mocks => github.com/Azure/go-autorest/autorest/mocks v0.4.1
	github.com/Azure/go-autorest/autorest/to => github.com/Azure/go-autorest/autorest/to v0.4.0
	github.com/Azure/go-autorest/autorest/validation => github.com/Azure/go-autorest/autorest/validation v0.1.0
	github.com/Azure/go-autorest/logger => github.com/Azure/go-autorest/logger v0.2.1
	github.com/Azure/go-autorest/tracing => github.com/Azure/go-autorest/tracing v0.6.0
	github.com/BurntSushi/toml => github.com/BurntSushi/toml v0.3.1
	github.com/BurntSushi/xgb => github.com/BurntSushi/xgb v0.0.0-20160522181843-27f122750802
	github.com/GoogleCloudPlatform/k8s-cloud-provider => github.com/GoogleCloudPlatform/k8s-cloud-provider v0.0.0-20200415212048-7901bc822317
	github.com/JeffAshton/win_pdh => github.com/JeffAshton/win_pdh v0.0.0-20161109143554-76bb4ee9f0ab
	github.com/MakeNowJust/heredoc => github.com/MakeNowJust/heredoc v0.0.0-20170808103936-bb23615498cd
	github.com/Microsoft/go-winio => github.com/Microsoft/go-winio v0.4.15
	github.com/Microsoft/hcsshim => github.com/Microsoft/hcsshim v0.8.10-0.20200715222032-5eafd1556990
	github.com/NYTimes/gziphandler => github.com/NYTimes/gziphandler v1.1.1
	github.com/PuerkitoBio/purell => github.com/PuerkitoBio/purell v1.1.1
	github.com/PuerkitoBio/urlesc => github.com/PuerkitoBio/urlesc v0.0.0-20170810143723-de5bf2ad4578
	github.com/ajstarks/svgo => github.com/ajstarks/svgo v0.0.0-20180226025133-644b8db467af
	github.com/alecthomas/units => github.com/alecthomas/units v0.0.0-20190924025748-f65c72e2690d
	github.com/antihax/optional => github.com/antihax/optional v1.0.0
	github.com/armon/circbuf => github.com/armon/circbuf v0.0.0-20150827004946-bbbad097214e
	github.com/armon/go-metrics => github.com/armon/go-metrics v0.0.0-20180917152333-f0300d1749da
	github.com/armon/go-radix => github.com/armon/go-radix v0.0.0-20180808171621-7fddfc383310
	github.com/asaskevich/govalidator => github.com/asaskevich/govalidator v0.0.0-20190424111038-f61b66f89f4a
	github.com/auth0/go-jwt-middleware => github.com/auth0/go-jwt-middleware v1.0.1
	github.com/aws/aws-sdk-go => github.com/aws/aws-sdk-go v1.38.49
	github.com/benbjohnson/clock => github.com/benbjohnson/clock v1.0.3
	github.com/beorn7/perks => github.com/beorn7/perks v1.0.1
	github.com/bgentry/speakeasy => github.com/bgentry/speakeasy v0.1.0
	github.com/bketelsen/crypt => github.com/bketelsen/crypt v0.0.3-0.20200106085610-5cbc8cc4026c
	github.com/blang/semver => github.com/blang/semver v3.5.1+incompatible
	github.com/boltdb/bolt => github.com/boltdb/bolt v1.3.1
	github.com/census-instrumentation/opencensus-proto => github.com/census-instrumentation/opencensus-proto v0.2.1
	github.com/certifi/gocertifi => github.com/certifi/gocertifi v0.0.0-20200922220541-2c3bb06c6054
	github.com/cespare/xxhash/v2 => github.com/cespare/xxhash/v2 v2.1.1
	github.com/chai2010/gettext-go => github.com/chai2010/gettext-go v0.0.0-20160711120539-c6fed771bfd5
	github.com/checkpoint-restore/go-criu/v5 => github.com/checkpoint-restore/go-criu/v5 v5.0.0
	github.com/chzyer/logex => github.com/chzyer/logex v1.1.10
	github.com/chzyer/readline => github.com/chzyer/readline v0.0.0-20180603132655-2972be24d48e
	github.com/chzyer/test => github.com/chzyer/test v0.0.0-20180213035817-a1ea475d72b1
	github.com/cilium/ebpf => github.com/cilium/ebpf v0.5.0
	github.com/clusterhq/flocker-go => github.com/clusterhq/flocker-go v0.0.0-20160920122132-2b8b7259d313
	github.com/cncf/udpa/go => github.com/cncf/udpa/go v0.0.0-20201120205902-5459f2c99403
	github.com/cockroachdb/datadriven => github.com/cockroachdb/datadriven v0.0.0-20200714090401-bf6692d28da5
	github.com/cockroachdb/errors => github.com/cockroachdb/errors v1.2.4
	github.com/cockroachdb/logtags => github.com/cockroachdb/logtags v0.0.0-20190617123548-eb05cc24525f
	github.com/container-storage-interface/spec => github.com/container-storage-interface/spec v1.5.0
	github.com/containerd/cgroups => github.com/containerd/cgroups v0.0.0-20200531161412-0dbf7f05ba59
	github.com/containerd/console => github.com/containerd/console v1.0.2
	github.com/containerd/containerd => github.com/containerd/containerd v1.4.4
	github.com/containerd/continuity => github.com/containerd/continuity v0.0.0-20190426062206-aaeac12a7ffc
	github.com/containerd/fifo => github.com/containerd/fifo v0.0.0-20190226154929-a9fb20d87448
	github.com/containerd/go-runc => github.com/containerd/go-runc v0.0.0-20180907222934-5a6d9f37cfa3
	github.com/containerd/ttrpc => github.com/containerd/ttrpc v1.0.2
	github.com/containerd/typeurl => github.com/containerd/typeurl v1.0.1
	github.com/containernetworking/cni => github.com/containernetworking/cni v0.8.1
	github.com/coredns/caddy => github.com/coredns/caddy v1.1.0
	github.com/coredns/corefile-migration => github.com/coredns/corefile-migration v1.0.12
	github.com/coreos/bbolt => github.com/coreos/bbolt v1.3.2
	github.com/coreos/etcd => github.com/coreos/etcd v3.3.13+incompatible
	github.com/coreos/go-oidc => github.com/coreos/go-oidc v2.1.0+incompatible
	github.com/coreos/go-semver => github.com/coreos/go-semver v0.3.0
	github.com/coreos/go-systemd => github.com/coreos/go-systemd v0.0.0-20190321100706-95778dfbb74e
	github.com/coreos/go-systemd/v22 => github.com/coreos/go-systemd/v22 v22.3.2
	github.com/coreos/pkg => github.com/coreos/pkg v0.0.0-20180928190104-399ea9e2e55f
	github.com/cpuguy83/go-md2man/v2 => github.com/cpuguy83/go-md2man/v2 v2.0.0
	github.com/creack/pty => github.com/creack/pty v1.1.11
	github.com/cyphar/filepath-securejoin => github.com/cyphar/filepath-securejoin v0.2.2
	github.com/davecgh/go-spew => github.com/davecgh/go-spew v1.1.1
	github.com/daviddengcn/go-colortext => github.com/daviddengcn/go-colortext v0.0.0-20160507010035-511bcaf42ccd
	github.com/dgrijalva/jwt-go => github.com/dgrijalva/jwt-go v3.2.0+incompatible
	github.com/dnaeon/go-vcr => github.com/dnaeon/go-vcr v1.0.1
	github.com/docker/distribution => github.com/docker/distribution v2.7.1+incompatible
	github.com/docker/docker => github.com/docker/docker v20.10.2+incompatible
	github.com/docker/go-connections => github.com/docker/go-connections v0.4.0
	github.com/docker/go-units => github.com/docker/go-units v0.4.0
	github.com/docopt/docopt-go => github.com/docopt/docopt-go v0.0.0-20180111231733-ee0de3bc6815
	github.com/dustin/go-humanize => github.com/dustin/go-humanize v1.0.0
	github.com/elazarl/goproxy => github.com/elazarl/goproxy v0.0.0-20180725130230-947c36da3153 // 947c36da3153 is the SHA for git tag v1.11
	github.com/emicklei/go-restful => github.com/emicklei/go-restful v2.9.5+incompatible
	github.com/envoyproxy/go-control-plane => github.com/envoyproxy/go-control-plane v0.9.9-0.20210217033140-668b12f5399d
	github.com/envoyproxy/protoc-gen-validate => github.com/envoyproxy/protoc-gen-validate v0.1.0
	github.com/euank/go-kmsg-parser => github.com/euank/go-kmsg-parser v2.0.0+incompatible
	github.com/evanphx/json-patch => github.com/evanphx/json-patch v4.11.0+incompatible
	github.com/exponent-io/jsonpath => github.com/exponent-io/jsonpath v0.0.0-20151013193312-d6023ce2651d
	github.com/fatih/camelcase => github.com/fatih/camelcase v1.0.0
	github.com/fatih/color => github.com/fatih/color v1.7.0
	github.com/felixge/httpsnoop => github.com/felixge/httpsnoop v1.0.1
	github.com/flynn/go-shlex => github.com/flynn/go-shlex v0.0.0-20150515145356-3f9db97f8568
	github.com/fogleman/gg => github.com/fogleman/gg v1.2.1-0.20190220221249-0403632d5b90
	github.com/form3tech-oss/jwt-go => github.com/form3tech-oss/jwt-go v3.2.3+incompatible
	github.com/frankban/quicktest => github.com/frankban/quicktest v1.11.3
	github.com/fsnotify/fsnotify => github.com/fsnotify/fsnotify v1.4.9
	github.com/fvbommel/sortorder => github.com/fvbommel/sortorder v1.0.1
	github.com/getsentry/raven-go => github.com/getsentry/raven-go v0.2.0
	github.com/ghodss/yaml => github.com/ghodss/yaml v1.0.0
	github.com/go-errors/errors => github.com/go-errors/errors v1.0.1
	github.com/go-gl/glfw/v3.3/glfw => github.com/go-gl/glfw/v3.3/glfw v0.0.0-20200222043503-6f7a984d4dc4
	github.com/go-kit/kit => github.com/go-kit/kit v0.9.0
	github.com/go-kit/log => github.com/go-kit/log v0.1.0
	github.com/go-logfmt/logfmt => github.com/go-logfmt/logfmt v0.5.0
	github.com/go-logr/logr => github.com/go-logr/logr v0.4.0
	github.com/go-openapi/jsonpointer => github.com/go-openapi/jsonpointer v0.19.5
	github.com/go-openapi/jsonreference => github.com/go-openapi/jsonreference v0.19.5
	github.com/go-openapi/swag => github.com/go-openapi/swag v0.19.14
	github.com/go-ozzo/ozzo-validation => github.com/go-ozzo/ozzo-validation v3.5.0+incompatible
	github.com/go-stack/stack => github.com/go-stack/stack v1.8.0
	github.com/godbus/dbus/v5 => github.com/godbus/dbus/v5 v5.0.4
	github.com/gofrs/uuid => github.com/gofrs/uuid v4.0.0+incompatible
	github.com/gogo/protobuf => github.com/gogo/protobuf v1.3.2
	github.com/golang/freetype => github.com/golang/freetype v0.0.0-20170609003504-e2365dfdc4a0
	github.com/golang/glog => github.com/golang/glog v0.0.0-20160126235308-23def4e6c14b
	github.com/golang/groupcache => github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da
	github.com/golang/mock => github.com/golang/mock v1.4.4
	github.com/golang/protobuf => github.com/golang/protobuf v1.5.2
	github.com/golangplus/testing => github.com/golangplus/testing v0.0.0-20180327235837-af21d9c3145e
	github.com/google/btree => github.com/google/btree v1.0.1
	github.com/google/cadvisor => github.com/google/cadvisor v0.39.2
	github.com/google/go-cmp => github.com/google/go-cmp v0.5.5
	github.com/google/gofuzz => github.com/google/gofuzz v1.1.0
	github.com/google/martian => github.com/google/martian v2.1.0+incompatible
	github.com/google/pprof => github.com/google/pprof v0.0.0-20200229191704-1ebb73c60ed3
	github.com/google/renameio => github.com/google/renameio v0.1.0
	github.com/google/shlex => github.com/google/shlex v0.0.0-20191202100458-e7afc7fbc510
	github.com/google/uuid => github.com/google/uuid v1.1.2
	github.com/googleapis/gax-go/v2 => github.com/googleapis/gax-go/v2 v2.0.5
	github.com/googleapis/gnostic => github.com/googleapis/gnostic v0.5.5
	github.com/gophercloud/gophercloud => github.com/gophercloud/gophercloud v0.1.0
	github.com/gopherjs/gopherjs => github.com/gopherjs/gopherjs v0.0.0-20200217142428-fce0ec30dd00
	github.com/gorilla/mux => github.com/gorilla/mux v1.8.0
	github.com/gorilla/websocket => github.com/gorilla/websocket v1.4.2
	github.com/gregjones/httpcache => github.com/gregjones/httpcache v0.0.0-20180305231024-9cad4c3443a7
	github.com/grpc-ecosystem/go-grpc-middleware => github.com/grpc-ecosystem/go-grpc-middleware v1.3.0
	github.com/grpc-ecosystem/go-grpc-prometheus => github.com/grpc-ecosystem/go-grpc-prometheus v1.2.0
	github.com/grpc-ecosystem/grpc-gateway => github.com/grpc-ecosystem/grpc-gateway v1.16.0
	github.com/hashicorp/consul/api => github.com/hashicorp/consul/api v1.1.0
	github.com/hashicorp/consul/sdk => github.com/hashicorp/consul/sdk v0.1.1
	github.com/hashicorp/errwrap => github.com/hashicorp/errwrap v1.0.0
	github.com/hashicorp/go-cleanhttp => github.com/hashicorp/go-cleanhttp v0.5.1
	github.com/hashicorp/go-immutable-radix => github.com/hashicorp/go-immutable-radix v1.0.0
	github.com/hashicorp/go-msgpack => github.com/hashicorp/go-msgpack v0.5.3
	github.com/hashicorp/go-multierror => github.com/hashicorp/go-multierror v1.0.0
	github.com/hashicorp/go-rootcerts => github.com/hashicorp/go-rootcerts v1.0.0
	github.com/hashicorp/go-sockaddr => github.com/hashicorp/go-sockaddr v1.0.0
	github.com/hashicorp/go-syslog => github.com/hashicorp/go-syslog v1.0.0
	github.com/hashicorp/go-uuid => github.com/hashicorp/go-uuid v1.0.1
	github.com/hashicorp/go.net => github.com/hashicorp/go.net v0.0.1
	github.com/hashicorp/golang-lru => github.com/hashicorp/golang-lru v0.5.1
	github.com/hashicorp/hcl => github.com/hashicorp/hcl v1.0.0
	github.com/hashicorp/logutils => github.com/hashicorp/logutils v1.0.0
	github.com/hashicorp/mdns => github.com/hashicorp/mdns v1.0.0
	github.com/hashicorp/memberlist => github.com/hashicorp/memberlist v0.1.3
	github.com/hashicorp/serf => github.com/hashicorp/serf v0.8.2
	github.com/heketi/heketi => github.com/heketi/heketi v10.3.0+incompatible
	github.com/heketi/tests => github.com/heketi/tests v0.0.0-20151005000721-f3775cbcefd6
	github.com/ianlancetaylor/demangle => github.com/ianlancetaylor/demangle v0.0.0-20181102032728-5e5cf60278f6
	github.com/imdario/mergo => github.com/imdario/mergo v0.3.5
	github.com/inconshreveable/mousetrap => github.com/inconshreveable/mousetrap v1.0.0
	github.com/ishidawataru/sctp => github.com/ishidawataru/sctp v0.0.0-20190723014705-7c296d48a2b5
	github.com/jmespath/go-jmespath => github.com/jmespath/go-jmespath v0.4.0
	github.com/jmespath/go-jmespath/internal/testify => github.com/jmespath/go-jmespath/internal/testify v1.5.1
	github.com/jonboulle/clockwork => github.com/jonboulle/clockwork v0.2.2
	github.com/josharian/intern => github.com/josharian/intern v1.0.0
	github.com/jpillora/backoff => github.com/jpillora/backoff v1.0.0
	github.com/json-iterator/go => github.com/json-iterator/go v1.1.11
	github.com/jstemmer/go-junit-report => github.com/jstemmer/go-junit-report v0.9.1
	github.com/jtolds/gls => github.com/jtolds/gls v4.20.0+incompatible
	github.com/julienschmidt/httprouter => github.com/julienschmidt/httprouter v1.3.0
	github.com/jung-kurt/gofpdf => github.com/jung-kurt/gofpdf v1.0.3-0.20190309125859-24315acbbda5
	github.com/karrick/godirwalk => github.com/karrick/godirwalk v1.16.1
	github.com/kisielk/errcheck => github.com/kisielk/errcheck v1.5.0
	github.com/kisielk/gotool => github.com/kisielk/gotool v1.0.0
	github.com/konsorten/go-windows-terminal-sequences => github.com/konsorten/go-windows-terminal-sequences v1.0.2
	github.com/kr/pretty => github.com/kr/pretty v0.2.1
	github.com/kr/text => github.com/kr/text v0.2.0
	github.com/libopenstorage/openstorage => github.com/libopenstorage/openstorage v1.0.0
	github.com/liggitt/tabwriter => github.com/liggitt/tabwriter v0.0.0-20181228230101-89fcab3d43de
	github.com/lithammer/dedent => github.com/lithammer/dedent v1.1.0
	github.com/lpabon/godbc => github.com/lpabon/godbc v0.1.1
	github.com/magiconair/properties => github.com/magiconair/properties v1.8.1
	github.com/mailru/easyjson => github.com/mailru/easyjson v0.7.6
	github.com/mattn/go-colorable => github.com/mattn/go-colorable v0.0.9
	github.com/mattn/go-isatty => github.com/mattn/go-isatty v0.0.3
	github.com/mattn/go-runewidth => github.com/mattn/go-runewidth v0.0.7
	github.com/matttproud/golang_protobuf_extensions => github.com/matttproud/golang_protobuf_extensions v1.0.2-0.20181231171920-c182affec369
	github.com/miekg/dns => github.com/miekg/dns v1.0.14
	github.com/mindprince/gonvml => github.com/mindprince/gonvml v0.0.0-20190828220739-9ebdce4bb989
	github.com/mistifyio/go-zfs => github.com/mistifyio/go-zfs v2.1.2-0.20190413222219-f784269be439+incompatible
	github.com/mitchellh/cli => github.com/mitchellh/cli v1.0.0
	github.com/mitchellh/go-homedir => github.com/mitchellh/go-homedir v1.1.0
	github.com/mitchellh/go-testing-interface => github.com/mitchellh/go-testing-interface v1.0.0
	github.com/mitchellh/go-wordwrap => github.com/mitchellh/go-wordwrap v1.0.0
	github.com/mitchellh/gox => github.com/mitchellh/gox v0.4.0
	github.com/mitchellh/iochan => github.com/mitchellh/iochan v1.0.0
	github.com/mitchellh/mapstructure => github.com/mitchellh/mapstructure v1.1.2
	github.com/moby/ipvs => github.com/moby/ipvs v1.0.1
	github.com/moby/spdystream => github.com/moby/spdystream v0.2.0
	github.com/moby/sys/mountinfo => github.com/moby/sys/mountinfo v0.4.1
	github.com/moby/term => github.com/moby/term v0.0.0-20210610120745-9d4ed1856297
	github.com/modern-go/concurrent => github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd
	github.com/modern-go/reflect2 => github.com/modern-go/reflect2 v1.0.1
	github.com/mohae/deepcopy => github.com/mohae/deepcopy v0.0.0-20170603005431-491d3605edfb
	github.com/monochromegane/go-gitignore => github.com/monochromegane/go-gitignore v0.0.0-20200626010858-205db1a8cc00
	github.com/morikuni/aec => github.com/morikuni/aec v1.0.0
	github.com/mrunalp/fileutils => github.com/mrunalp/fileutils v0.5.0
	github.com/munnerz/goautoneg => github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822
	github.com/mvdan/xurls => github.com/mvdan/xurls v1.1.0
	github.com/mwitkow/go-conntrack => github.com/mwitkow/go-conntrack v0.0.0-20190716064945-2f068394615f
	github.com/mxk/go-flowrate => github.com/mxk/go-flowrate v0.0.0-20140419014527-cca7078d478f
	github.com/niemeyer/pretty => github.com/niemeyer/pretty v0.0.0-20200227124842-a10e7caefd8e
	github.com/nxadm/tail => github.com/nxadm/tail v1.4.4
	github.com/olekukonko/tablewriter => github.com/olekukonko/tablewriter v0.0.4
	github.com/onsi/ginkgo => github.com/onsi/ginkgo v1.14.0
	github.com/onsi/gomega => github.com/onsi/gomega v1.10.1
	github.com/opencontainers/go-digest => github.com/opencontainers/go-digest v1.0.0
	github.com/opencontainers/image-spec => github.com/opencontainers/image-spec v1.0.1
	github.com/opencontainers/runc => github.com/opencontainers/runc v1.0.0-rc95
	github.com/opencontainers/runtime-spec => github.com/opencontainers/runtime-spec v1.0.3-0.20210326190908-1c3f411f0417
	github.com/opencontainers/selinux => github.com/opencontainers/selinux v1.8.0
	github.com/opentracing/opentracing-go => github.com/opentracing/opentracing-go v1.1.0
	github.com/pascaldekloe/goe => github.com/pascaldekloe/goe v0.0.0-20180627143212-57f6aae5913c
	github.com/pelletier/go-toml => github.com/pelletier/go-toml v1.2.0
	github.com/peterbourgon/diskv => github.com/peterbourgon/diskv v2.0.1+incompatible
	github.com/pkg/errors => github.com/pkg/errors v0.9.1
	github.com/pmezard/go-difflib => github.com/pmezard/go-difflib v1.0.0
	github.com/posener/complete => github.com/posener/complete v1.1.1
	github.com/pquerna/cachecontrol => github.com/pquerna/cachecontrol v0.0.0-20171018203845-0dec1b30a021
	github.com/prometheus/client_golang => github.com/prometheus/client_golang v1.11.0
	github.com/prometheus/client_model => github.com/prometheus/client_model v0.2.0
	github.com/prometheus/common => github.com/prometheus/common v0.26.0
	github.com/prometheus/procfs => github.com/prometheus/procfs v0.6.0
	github.com/quobyte/api => github.com/quobyte/api v0.1.8
	github.com/remyoudompheng/bigfft => github.com/remyoudompheng/bigfft v0.0.0-20170806203942-52369c62f446
	github.com/robfig/cron/v3 => github.com/robfig/cron/v3 v3.0.1
	github.com/rogpeppe/fastuuid => github.com/rogpeppe/fastuuid v1.2.0
	github.com/rogpeppe/go-internal => github.com/rogpeppe/go-internal v1.3.0
	github.com/rubiojr/go-vhd => github.com/rubiojr/go-vhd v0.0.0-20200706105327-02e210299021
	github.com/russross/blackfriday => github.com/russross/blackfriday v1.5.2
	github.com/russross/blackfriday/v2 => github.com/russross/blackfriday/v2 v2.0.1
	github.com/ryanuber/columnize => github.com/ryanuber/columnize v0.0.0-20160712163229-9b3edd62028f
	github.com/sean-/seed => github.com/sean-/seed v0.0.0-20170313163322-e2103e2c3529
	github.com/seccomp/libseccomp-golang => github.com/seccomp/libseccomp-golang v0.9.1
	github.com/sergi/go-diff => github.com/sergi/go-diff v1.1.0
	github.com/shurcooL/sanitized_anchor_name => github.com/shurcooL/sanitized_anchor_name v1.0.0
	github.com/sirupsen/logrus => github.com/sirupsen/logrus v1.7.0
	github.com/smartystreets/assertions => github.com/smartystreets/assertions v1.1.0
	github.com/smartystreets/goconvey => github.com/smartystreets/goconvey v1.6.4
	github.com/soheilhy/cmux => github.com/soheilhy/cmux v0.1.5
	github.com/spf13/afero => github.com/spf13/afero v1.2.2
	github.com/spf13/cast => github.com/spf13/cast v1.3.0
	github.com/spf13/cobra => github.com/spf13/cobra v1.1.3
	github.com/spf13/jwalterweatherman => github.com/spf13/jwalterweatherman v1.0.0
	github.com/spf13/pflag => github.com/spf13/pflag v1.0.5
	github.com/spf13/viper => github.com/spf13/viper v1.7.0
	github.com/stoewer/go-strcase => github.com/stoewer/go-strcase v1.2.0
	github.com/storageos/go-api => github.com/storageos/go-api v2.2.0+incompatible
	github.com/stretchr/objx => github.com/stretchr/objx v0.2.0
	github.com/stretchr/testify => github.com/stretchr/testify v1.7.0
	github.com/subosito/gotenv => github.com/subosito/gotenv v1.2.0
	github.com/syndtr/gocapability => github.com/syndtr/gocapability v0.0.0-20200815063812-42c35b437635
	github.com/tmc/grpc-websocket-proxy => github.com/tmc/grpc-websocket-proxy v0.0.0-20201229170055-e5319fda7802
	github.com/urfave/cli => github.com/urfave/cli v1.22.2
	github.com/urfave/negroni => github.com/urfave/negroni v1.0.0
	github.com/vishvananda/netlink => github.com/vishvananda/netlink v1.1.0
	github.com/vishvananda/netns => github.com/vishvananda/netns v0.0.0-20200728191858-db3c7e526aae
	github.com/vmware/govmomi => github.com/vmware/govmomi v0.20.3
	github.com/willf/bitset => github.com/willf/bitset v1.1.11
	github.com/xiang90/probing => github.com/xiang90/probing v0.0.0-20190116061207-43a291ad63a2
	github.com/xlab/treeprint => github.com/xlab/treeprint v0.0.0-20181112141820-a009c3971eca
	github.com/yuin/goldmark => github.com/yuin/goldmark v1.3.5
	go.etcd.io/bbolt => go.etcd.io/bbolt v1.3.6
	go.etcd.io/etcd/api/v3 => go.etcd.io/etcd/api/v3 v3.5.0
	go.etcd.io/etcd/client/pkg/v3 => go.etcd.io/etcd/client/pkg/v3 v3.5.0
	go.etcd.io/etcd/client/v2 => go.etcd.io/etcd/client/v2 v2.305.0
	go.etcd.io/etcd/client/v3 => go.etcd.io/etcd/client/v3 v3.5.0
	go.etcd.io/etcd/pkg/v3 => go.etcd.io/etcd/pkg/v3 v3.5.0
	go.etcd.io/etcd/raft/v3 => go.etcd.io/etcd/raft/v3 v3.5.0
	go.etcd.io/etcd/server/v3 => go.etcd.io/etcd/server/v3 v3.5.0
	go.opencensus.io => go.opencensus.io v0.22.3
	go.opentelemetry.io/contrib => go.opentelemetry.io/contrib v0.20.0
	go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc => go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc v0.20.0
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp => go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.20.0
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
	go.uber.org/zap => go.uber.org/zap v1.17.0
	golang.org/x/crypto => golang.org/x/crypto v0.0.0-20210220033148-5ea612d1eb83
	golang.org/x/exp => golang.org/x/exp v0.0.0-20210220032938-85be41e4509f
	golang.org/x/image => golang.org/x/image v0.0.0-20190802002840-cff245a6509b
	golang.org/x/lint => golang.org/x/lint v0.0.0-20210508222113-6edffad5e616
	golang.org/x/mobile => golang.org/x/mobile v0.0.0-20201217150744-e6ae53a27f4f
	golang.org/x/mod => golang.org/x/mod v0.4.2
	golang.org/x/net => golang.org/x/net v0.0.0-20210520170846-37e1c6afe023
	golang.org/x/oauth2 => golang.org/x/oauth2 v0.0.0-20200107190931-bf48bf16ab8d
	golang.org/x/sync => golang.org/x/sync v0.0.0-20210220032951-036812b2e83c
	golang.org/x/sys => golang.org/x/sys v0.0.0-20210616094352-59db8d763f22
	golang.org/x/term => golang.org/x/term v0.0.0-20210220032956-6a3ed077a48d
	golang.org/x/text => golang.org/x/text v0.3.6
	golang.org/x/time => golang.org/x/time v0.0.0-20210220033141-f8bda1e9f3ba
	golang.org/x/tools => golang.org/x/tools v0.1.2
	golang.org/x/xerrors => golang.org/x/xerrors v0.0.0-20200804184101-5ec99f83aff1
	gonum.org/v1/gonum => gonum.org/v1/gonum v0.6.2
	gonum.org/v1/netlib => gonum.org/v1/netlib v0.0.0-20190331212654-76723241ea4e
	gonum.org/v1/plot => gonum.org/v1/plot v0.0.0-20190515093506-e2840ee46a6b
	google.golang.org/api => google.golang.org/api v0.20.0
	google.golang.org/appengine => google.golang.org/appengine v1.6.5
	google.golang.org/genproto => google.golang.org/genproto v0.0.0-20210602131652-f16073e35f0c
	google.golang.org/grpc => google.golang.org/grpc v1.38.0
	google.golang.org/protobuf => google.golang.org/protobuf v1.26.0
	gopkg.in/alecthomas/kingpin.v2 => gopkg.in/alecthomas/kingpin.v2 v2.2.6
	gopkg.in/check.v1 => gopkg.in/check.v1 v1.0.0-20200227125254-8fa46927fb4f
	gopkg.in/errgo.v2 => gopkg.in/errgo.v2 v2.1.0
	gopkg.in/gcfg.v1 => gopkg.in/gcfg.v1 v1.2.0
	gopkg.in/inf.v0 => gopkg.in/inf.v0 v0.9.1
	gopkg.in/ini.v1 => gopkg.in/ini.v1 v1.51.0
	gopkg.in/natefinch/lumberjack.v2 => gopkg.in/natefinch/lumberjack.v2 v2.0.0
	gopkg.in/square/go-jose.v2 => gopkg.in/square/go-jose.v2 v2.2.2
	gopkg.in/tomb.v1 => gopkg.in/tomb.v1 v1.0.0-20141024135613-dd632973f1e7
	gopkg.in/warnings.v0 => gopkg.in/warnings.v0 v0.1.1
	gopkg.in/yaml.v2 => gopkg.in/yaml.v2 v2.4.0
	gopkg.in/yaml.v3 => gopkg.in/yaml.v3 v3.0.0-20210107192922-496545a6307b
	gotest.tools => gotest.tools v2.2.0+incompatible
	gotest.tools/v3 => gotest.tools/v3 v3.0.3
	honnef.co/go/tools => honnef.co/go/tools v0.0.1-2020.1.3
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
	k8s.io/gengo => k8s.io/gengo v0.0.0-20201214224949-b6c5ce23f027
	k8s.io/klog/v2 => k8s.io/klog/v2 v2.9.0
	k8s.io/kube-aggregator => ./staging/src/k8s.io/kube-aggregator
	k8s.io/kube-controller-manager => ./staging/src/k8s.io/kube-controller-manager
	k8s.io/kube-openapi => k8s.io/kube-openapi v0.0.0-20210421082810-95288971da7e
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
	k8s.io/system-validators => k8s.io/system-validators v1.5.0
	k8s.io/utils => k8s.io/utils v0.0.0-20210707171843-4b05e18ac7d9
	modernc.org/cc => modernc.org/cc v1.0.0
	modernc.org/golex => modernc.org/golex v1.0.0
	modernc.org/mathutil => modernc.org/mathutil v1.0.0
	modernc.org/strutil => modernc.org/strutil v1.0.0
	modernc.org/xc => modernc.org/xc v1.0.0
	rsc.io/pdf => rsc.io/pdf v0.1.1
	sigs.k8s.io/apiserver-network-proxy/konnectivity-client => sigs.k8s.io/apiserver-network-proxy/konnectivity-client v0.0.21
	sigs.k8s.io/kustomize/api => sigs.k8s.io/kustomize/api v0.8.11
	sigs.k8s.io/kustomize/cmd/config => sigs.k8s.io/kustomize/cmd/config v0.9.13
	sigs.k8s.io/kustomize/kustomize/v4 => sigs.k8s.io/kustomize/kustomize/v4 v4.2.0
	sigs.k8s.io/kustomize/kyaml => sigs.k8s.io/kustomize/kyaml v0.11.0
	sigs.k8s.io/structured-merge-diff/v4 => sigs.k8s.io/structured-merge-diff/v4 v4.1.2
	sigs.k8s.io/yaml => sigs.k8s.io/yaml v1.2.0
)
