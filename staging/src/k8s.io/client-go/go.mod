// This is a generated file. Do not edit directly.

module k8s.io/client-go

go 1.17

require (
	github.com/Azure/go-autorest/autorest v0.11.18
	github.com/Azure/go-autorest/autorest/adal v0.9.13
	github.com/davecgh/go-spew v1.1.1
	github.com/evanphx/json-patch v4.11.0+incompatible
	github.com/gogo/protobuf v1.3.2
	github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da
	github.com/golang/protobuf v1.5.2
	github.com/google/go-cmp v0.5.5
	github.com/google/gofuzz v1.1.0
	github.com/google/uuid v1.1.2
	github.com/googleapis/gnostic v0.5.5
	github.com/spf13/pflag v1.0.5
	github.com/stretchr/testify v1.7.0
	golang.org/x/net v0.0.0-20210813160813-60bc85c4be6d
	golang.org/x/oauth2 v0.0.0-20210819190943-2bc19b11175f
	golang.org/x/term v0.0.0-20210615171337-6886f2dfbf5b
	golang.org/x/time v0.0.0-20210723032227-1f47c861a9ac
	google.golang.org/protobuf v1.26.0
	k8s.io/klog/v2 v2.9.0
	k8s.io/kube-openapi v0.0.0-20210817084001-7fbd8d59e5b8
	k8s.io/utils v0.0.0-20210819203725-bdf08cb9a70a
	sigs.k8s.io/structured-merge-diff/v4 v4.1.2
	sigs.k8s.io/yaml v1.2.0
)

require (
	cloud.google.com/go v0.81.0 // indirect
	github.com/Azure/go-autorest v14.2.0+incompatible // indirect
	github.com/Azure/go-autorest/autorest/date v0.3.0 // indirect
	github.com/Azure/go-autorest/logger v0.2.1 // indirect
	github.com/Azure/go-autorest/tracing v0.6.0 // indirect
	github.com/form3tech-oss/jwt-go v3.2.3+incompatible // indirect
	github.com/go-logr/logr v0.4.0 // indirect
	github.com/google/btree v1.0.1 // indirect
	github.com/gregjones/httpcache v0.0.0-20180305231024-9cad4c3443a7
	github.com/imdario/mergo v0.3.5
	github.com/moby/spdystream v0.2.0 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.1 // indirect
	github.com/nxadm/tail v1.4.4 // indirect
	github.com/peterbourgon/diskv v2.0.1+incompatible
	golang.org/x/text v0.3.6 // indirect
	google.golang.org/appengine v1.6.7 // indirect
	gopkg.in/inf.v0 v0.9.1 // indirect
	gopkg.in/yaml.v3 v3.0.0-20210107192922-496545a6307b // indirect
	k8s.io/api v0.0.0
	k8s.io/apimachinery v0.0.0
)

require (
	github.com/fsnotify/fsnotify v1.4.9 // indirect
	github.com/json-iterator/go v1.1.11 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	golang.org/x/crypto v0.0.0-20210817164053-32db794688a5 // indirect
	golang.org/x/sys v0.0.0-20210820121016-41cdb8703e55 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
)

replace (
	k8s.io/api => ../api
	k8s.io/apimachinery => ../apimachinery
	k8s.io/client-go => ../client-go
)
