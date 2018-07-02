module k8s.io/kubernetes

// These k8s.io modules live in staging/src/k8s.io subdirectories.
// In this PR, they are versioned using the same version as kubernetes,
// establishing the convention that k8s.io/kubernetes v1.N matches
// the submodules v1.N too.
//
// PR NOTE: k8s.io/client-go is already on v8.0.0, so something
// more will have to be done for that. (This is just a demo PR.)

require (
	k8s.io/api v1.12.0
	k8s.io/apiextensions-apiserver v1.12.0
	k8s.io/apimachinery v1.12.0
	k8s.io/apiserver v1.12.0
	k8s.io/client-go v1.12.0
	k8s.io/code-generator v1.12.0
	k8s.io/kube-aggregator v1.12.0
	k8s.io/metrics v1.12.0
	k8s.io/sample-apiserver v1.12.0
)

replace (
	k8s.io/api v1.12.0 => ./staging/src/k8s.io/api
	k8s.io/apiextensions-apiserver v1.12.0 => ./staging/src/k8s.io/apiextensions-apiserver
	k8s.io/apimachinery v1.12.0 => ./staging/src/k8s.io/apimachinery
	k8s.io/apiserver v1.12.0 => ./staging/src/k8s.io/apiserver
	k8s.io/client-go v1.12.0 => ./staging/src/k8s.io/client-go
	k8s.io/code-generator v1.12.0 => ./staging/src/k8s.io/code-generator
	k8s.io/kube-aggregator v1.12.0 => ./staging/src/k8s.io/kube-aggregator
	k8s.io/metrics v1.12.0 => ./staging/src/k8s.io/metrics
	k8s.io/sample-apiserver v1.12.0 => ./staging/src/k8s.io/sample-apiserver
	k8s.io/sample-controller v1.12.0 => ./staging/src/k8s.io/sample-controller
)

// k8s.io/kubernetes/pkg/volume/glusterfs imports
// github.com/heketi/heketi/client/api/go-client, and
// github.com/heketi/heketi/client/api/go-client's test imports
// github.com/heketi/heketi/apps/glusterfs, which imports
// github.com/heketi/heketi/executors/kubeexec, which imports
//
//   - k8s.io/kubernetes/pkg/client/clientset_generated/clientset
//   - k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1
//   - k8s.io/kubernetes/pkg/client/unversioned/remotecommand
//   - k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake
//
// None of these packages exist anymore.
//
// rsc/heketi@18ce0f5326d5 is a fork of heketi/heketi@aaf40619d85f
// that simply deletes the offending test.
//
// If anything in Kubernetes started importing the heketi glusterfs or kubeexec
// packages directly, this problem would come back.
// The solution is probably to get heketi fixed and then update to a newer version.
// This replacement statement only applies to heketi/heketi@aaf40619d85f:
// that is, if any newer version starts being used, this "replace" line becomes a no-op.
replace github.com/heketi/heketi v0.0.0-20170623005005-aaf40619d85f => github.com/rsc/heketi v0.0.0-20180629190409-18ce0f5326d5

require (
	bitbucket.org/bertimus9/systemstat v0.0.0-20180207000608-0eeff89b0690
	cloud.google.com/go v0.0.0-20160913182117-3b1ae45394a2
	github.com/Azure/azure-sdk-for-go v0.0.0-20180321163057-56a0b1d2af3b
	github.com/Azure/go-ansiterm v0.0.0-20170629204627-19f72df4d05d // indirect
	github.com/Azure/go-autorest v0.0.0-20180418234723-1ff28809256a
	github.com/JeffAshton/win_pdh v0.0.0-20161109143554-76bb4ee9f0ab
	github.com/MakeNowJust/heredoc v0.0.0-20170808103936-bb23615498cd
	github.com/Microsoft/go-winio v0.4.5 // indirect
	github.com/Microsoft/hcsshim v0.6.11
	github.com/Nvveen/Gotty v0.0.0-20120604004816-cd527374f1e5 // indirect
	github.com/PuerkitoBio/purell v1.0.0
	github.com/abbot/go-http-auth v0.0.0-20140618235127-c0ef4539dfab // indirect
	github.com/armon/circbuf v0.0.0-20150827004946-bbbad097214e
	github.com/aws/aws-sdk-go v1.12.7
	github.com/bazelbuild/bazel-gazelle v0.0.0-20180508174903-7f30ba724af9
	github.com/bazelbuild/buildtools v0.0.0-20171220125010-1a9c38e0df93 // indirect
	github.com/blang/semver v0.0.0-20170130170546-b38d23b8782a
	github.com/chai2010/gettext-go v0.0.0-20160711120539-c6fed771bfd5
	github.com/client9/misspell v0.0.0-20170928000206-9ce5d979ffda
	github.com/cloudflare/cfssl v0.0.0-20180223231731-4e2dcbde5004
	github.com/clusterhq/flocker-go v0.0.0-20160920122132-2b8b7259d313
	github.com/codedellemc/goscaleio v0.0.0-20170830184815-20e2ce2cf885
	github.com/container-storage-interface/spec v0.3.0
	github.com/containerd/console v0.0.0-20170925154832-84eeaae905fa // indirect
	github.com/containerd/containerd v1.0.2 // indirect
	github.com/containerd/typeurl v0.0.0-20180627222232-a93fcdb778cd // indirect
	github.com/containernetworking/cni v0.6.0
	github.com/coreos/etcd v0.0.0-20180102212956-95a726a27e09
	github.com/coreos/go-semver v0.0.0-20150304020126-568e959cd898
	github.com/coreos/go-systemd v0.0.0-20161114122254-48702e0da86b
	github.com/coreos/pkg v0.0.0-20160620232715-fa29b1d70f0b
	github.com/coreos/rkt v1.25.0 // indirect
	github.com/cpuguy83/go-md2man v1.0.4
	github.com/cyphar/filepath-securejoin v0.0.0-20170720062807-ae69057f2299 // indirect
	github.com/d2g/dhcp4 v0.0.0-20170904100407-a1d1b6c41b1c
	github.com/d2g/dhcp4client v0.0.0-20170829104524-6e570ed0a266
	github.com/davecgh/go-spew v0.0.0-20170626231645-782f4967f2dc
	github.com/daviddengcn/go-colortext v0.0.0-20160507010035-511bcaf42ccd
	github.com/dgrijalva/jwt-go v0.0.0-20160705203006-01aeca54ebda
	github.com/dnaeon/go-vcr v0.0.0-20180607100630-8b144be0744f // indirect
	github.com/docker/distribution v0.0.0-20170726174610-edc3ab29cdff
	github.com/docker/docker v0.0.0-20170731201938-4f3616fb1c11
	github.com/docker/go-connections v0.3.0
	github.com/docker/go-units v0.0.0-20170127094116-9e638d38cf69
	github.com/docker/libnetwork v0.0.0-20170905174201-ba46b9284449
	github.com/docker/libtrust v0.0.0-20150526203908-9cbd2a1374f4 // indirect
	github.com/elazarl/goproxy v0.0.0-20170405201442-c4fc26588b6e
	github.com/emicklei/go-restful v0.0.0-20170410110728-ff4f55a20633
	github.com/euank/go-kmsg-parser v0.0.0-20161120035913-5ba4d492e455 // indirect
	github.com/evanphx/json-patch v0.0.0-20180525161421-94e38aa1586e
	github.com/exponent-io/jsonpath v0.0.0-20151013193312-d6023ce2651d
	github.com/fatih/camelcase v0.0.0-20160318181535-f6a740d52f96
	github.com/fsnotify/fsnotify v0.0.0-20160816051541-f12c6236fe7b
	github.com/ghodss/yaml v0.0.0-20150909031657-73d445a93680
	github.com/go-ini/ini v1.25.4 // indirect
	github.com/go-openapi/loads v0.0.0-20170520182102-a80dea3052f0
	github.com/go-openapi/spec v0.0.0-20180213232550-1de3e0542de6
	github.com/go-openapi/strfmt v0.0.0-20160812050534-d65c7fdb29ec
	github.com/go-openapi/validate v0.0.0-20171117174350-d509235108fc
	github.com/godbus/dbus v0.0.0-20151105175453-c7fdd8b5cd55
	github.com/gogo/protobuf v0.0.0-20170330071051-c0656edd0d9e
	github.com/golang/glog v0.0.0-20141105023935-44145f04b68c
	github.com/golang/groupcache v0.0.0-20160516000752-02826c3e7903
	github.com/golang/mock v0.0.0-20160127222235-bd3c8e81be01
	github.com/golang/protobuf v1.1.0
	github.com/golangplus/bytes v0.0.0-20160111154220-45c989fe5450 // indirect
	github.com/golangplus/fmt v0.0.0-20150411045040-2a5d6d7d2995 // indirect
	github.com/golangplus/testing v0.0.0-20180327235837-af21d9c3145e // indirect
	github.com/google/cadvisor v0.30.2
	github.com/google/certificate-transparency-go v1.0.10 // indirect
	github.com/google/gofuzz v0.0.0-20161122191042-44d81051d367
	github.com/google/uuid v0.0.0-20171113160352-8c31c18f31ed // indirect
	github.com/googleapis/gnostic v0.0.0-20170729233727-0c5108395e2d
	github.com/gophercloud/gophercloud v0.0.0-20180330165814-781450b3c4fc
	github.com/gopherjs/gopherjs v0.0.0-20180628210949-0892b62f0d9f // indirect
	github.com/gorilla/websocket v0.0.0-20150714140627-6eb6ad425a89 // indirect
	github.com/hashicorp/golang-lru v0.0.0-20160207214719-a0d98a5f2880
	github.com/hashicorp/hcl v0.0.0-20160711231752-d8c773c4cba1 // indirect
	github.com/heketi/heketi v0.0.0-20170623005005-aaf40619d85f
	github.com/heketi/tests v0.0.0-20151005000721-f3775cbcefd6 // indirect
	github.com/influxdata/influxdb v1.1.1
	github.com/jmespath/go-jmespath v0.0.0-20160202185014-0b12d6b521d8 // indirect
	github.com/jonboulle/clockwork v0.0.0-20141017032234-72f9bd7c4e0c
	github.com/json-iterator/go v0.0.0-20180612202835-f2b4162afba3
	github.com/jteeuwen/go-bindata v0.0.0-20151023091102-a0ff2567cfb7
	github.com/jtolds/gls v0.0.0-20170503224851-77f18212c9c7 // indirect
	github.com/kardianos/osext v0.0.0-20150410034420-8fef92e41e22
	github.com/kr/fs v0.0.0-20131111012553-2788f0dbd169 // indirect
	github.com/kubernetes/repo-infra v0.0.0-20180411215455-d9bb9fdc9076
	github.com/libopenstorage/openstorage v0.0.0-20170906232338-093a0c388875
	github.com/lpabon/godbc v0.0.0-20140613165803-9577782540c1 // indirect
	github.com/magiconair/properties v0.0.0-20160816085511-61b492c03cf4 // indirect
	github.com/marstr/guid v0.0.0-20170427235115-8bdf7d1a087c // indirect
	github.com/mattn/go-shellwords v0.0.0-20180605041737-f8471b0a71de // indirect
	github.com/mholt/caddy v0.0.0-20180213163048-2de495001514
	github.com/miekg/dns v0.0.0-20160614162101-5d001d020961
	github.com/mindprince/gonvml v0.0.0-20171110221305-fee913ce8fb2 // indirect
	github.com/mistifyio/go-zfs v0.0.0-20151009155749-1b4ae6fb4e77 // indirect
	github.com/mitchellh/go-wordwrap v0.0.0-20150314170334-ad45545899c7
	github.com/mitchellh/mapstructure v0.0.0-20170307201123-53818660ed49
	github.com/mohae/deepcopy v0.0.0-20170603005431-491d3605edfb // indirect
	github.com/mrunalp/fileutils v0.0.0-20160930181131-4ee1cc9a8058 // indirect
	github.com/mvdan/xurls v0.0.0-20160110113200-1b768d7c393a
	github.com/onsi/ginkgo v0.0.0-20170318221715-67b9df7f55fe
	github.com/onsi/gomega v0.0.0-20160911051023-d59fa0ac68bb
	github.com/opencontainers/go-digest v0.0.0-20170106003457-a6d0ee40d420
	github.com/opencontainers/image-spec v0.0.0-20170604055404-372ad780f634 // indirect
	github.com/opencontainers/runc v0.0.0-20180424185634-871ba2e58e24
	github.com/opencontainers/runtime-spec v1.0.0 // indirect
	github.com/opencontainers/selinux v0.0.0-20170621221121-4a2974bf1ee9
	github.com/pborman/uuid v0.0.0-20150603214016-ca53cad383ca
	github.com/pelletier/go-toml v1.2.0 // indirect
	github.com/pkg/errors v0.8.0
	github.com/pkg/sftp v0.0.0-20160930220758-4d0e916071f6 // indirect
	github.com/pmezard/go-difflib v0.0.0-20151028094244-d8ed2627bdf0
	github.com/prometheus/client_golang v0.0.0-20170531130054-e7e903064f5e
	github.com/prometheus/client_model v0.0.0-20150212101744-fa8ad6fec335
	github.com/prometheus/common v0.0.0-20170427095455-13ba4ddd0caa
	github.com/quobyte/api v0.0.0-20171020135407-f2b94aa4aa4f
	github.com/rancher/go-rancher v0.0.0-20160922212217-09693a8743ba
	github.com/renstrom/dedent v0.0.0-20150819195903-020d11c3b9c0
	github.com/robfig/cron v0.0.0-20170309132418-df38d32658d8
	github.com/rubiojr/go-vhd v0.0.0-20160810183302-0bfd3b39853c
	github.com/russross/blackfriday v0.0.0-20151117072312-300106c228d5
	github.com/satori/go.uuid v1.2.0 // indirect
	github.com/seccomp/libseccomp-golang v0.0.0-20150813023252-1b506fc7c24e // indirect
	github.com/shurcooL/sanitized_anchor_name v0.0.0-20151028001915-10ef21a441db // indirect
	github.com/sirupsen/logrus v0.0.0-20170822132746-89742aefa4b2 // indirect
	github.com/smartystreets/assertions v0.0.0-20180607162144-eb5b59917fa2 // indirect
	github.com/smartystreets/goconvey v0.0.0-20180222194500-ef6db91d284a // indirect
	github.com/smartystreets/gunit v0.0.0-20180314194857-6f0d6275bdcd // indirect
	github.com/spf13/afero v0.0.0-20160816080757-b28a7effac97
	github.com/spf13/cast v0.0.0-20160730092037-e31f36ffc91a // indirect
	github.com/spf13/cobra v0.0.0-20180319062004-c439c4fa0937
	github.com/spf13/jwalterweatherman v0.0.0-20160311093646-33c24e77fb80 // indirect
	github.com/spf13/pflag v1.0.1
	github.com/spf13/viper v0.0.0-20160820190039-7fb2782df3d8
	github.com/storageos/go-api v0.0.0-20180126153955-3a4032328d99
	github.com/stretchr/testify v0.0.0-20180319223459-c679ae2cc0cb
	github.com/syndtr/gocapability v0.0.0-20160928074757-e7cb7fa329f4 // indirect
	github.com/tools/godep v0.0.0-20180126220526-ce0bfadeb516
	github.com/ugorji/go v0.0.0-20170107133203-ded73eae5db7
	github.com/vishvananda/netlink v0.0.0-20171128170821-f67b75edbf5e
	github.com/vishvananda/netns v0.0.0-20171111001504-be1fbeda1936 // indirect
	github.com/vmware/govmomi v0.0.0-20180508155031-e70dd44f80ba
	github.com/vmware/photon-controller-go-sdk v0.0.0-20170310013346-4a435daef6cc
	github.com/xanzy/go-cloudstack v0.0.0-20160728180336-1e2cbf647e57
	github.com/xlab/handysort v0.0.0-20150421192137-fb3537ed64a1 // indirect
	golang.org/x/crypto v0.0.0-20180222182404-49796115aa4b
	golang.org/x/exp v0.0.0-20160623011055-292a51b8d262
	golang.org/x/net v0.0.0-20170809000501-1c05540f6879
	golang.org/x/oauth2 v0.0.0-20170412232759-a6bd8cefa181
	golang.org/x/sys v0.0.0-20171031081856-95c657629925
	golang.org/x/text v0.0.0-20170810154203-b19bf474d317
	golang.org/x/time v0.0.0-20161028155119-f51c12702a4d
	golang.org/x/tools v0.0.0-20170428054726-2382e3994d48
	google.golang.org/api v0.0.0-20180603000442-8e296ef26005
	google.golang.org/grpc v1.7.5
	gopkg.in/airbrake/gobrake.v2 v2.0.9 // indirect
	gopkg.in/gcfg.v1 v1.2.0
	gopkg.in/gemnasium/logrus-airbrake-hook.v2 v2.1.2 // indirect
	gopkg.in/square/go-jose.v2 v2.1.3
	gopkg.in/warnings.v0 v0.1.1 // indirect
	gopkg.in/yaml.v2 v2.0.0-20170721113624-670d4cfef054
	k8s.io/gengo v0.0.0-20180612161529-dcbe4570f0cf
	k8s.io/heapster v1.2.0-beta.1
	k8s.io/kube-openapi v0.0.0-20180620173706-91cfa479c814
	k8s.io/utils v0.0.0-20180208044234-258e2a2fa645
	vbom.ml/util v0.0.0-20160121211510-db5cfe13f5cc
)
