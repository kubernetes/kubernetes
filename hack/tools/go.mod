module k8s.io/kubernetes/hack/tools

go 1.25.0

godebug default=go1.25

require (
	golang.org/x/mod v0.27.0
	k8s.io/publishing-bot v0.5.0
)

require (
	github.com/BurntSushi/toml v1.5.0 // indirect
	github.com/aojea/sloppy-netparser v0.0.0-20210819225411-1b3bd8b3b975 // indirect
	github.com/bitfield/gotestdox v0.2.2 // indirect
	github.com/brunoga/deep v1.2.4 // indirect
	github.com/cespare/prettybench v0.0.0-20150116022406-03b8cfe5406c // indirect
	github.com/dnephin/pflag v1.0.7 // indirect
	github.com/fatih/color v1.18.0 // indirect
	github.com/fatih/structs v1.1.0 // indirect
	github.com/fsnotify/fsnotify v1.8.0 // indirect
	github.com/go-viper/mapstructure/v2 v2.3.0 // indirect
	github.com/golang/glog v1.2.2 // indirect
	github.com/golangci/misspell v0.6.0 // indirect
	github.com/google/go-cmp v0.7.0 // indirect
	github.com/google/shlex v0.0.0-20191202100458-e7afc7fbc510 // indirect
	github.com/huandu/xstrings v1.5.0 // indirect
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/jcchavezs/porto v0.6.0 // indirect
	github.com/jedib0t/go-pretty/v6 v6.6.7 // indirect
	github.com/knadh/koanf/maps v0.1.2 // indirect
	github.com/knadh/koanf/parsers/yaml v0.1.0 // indirect
	github.com/knadh/koanf/providers/env v1.0.0 // indirect
	github.com/knadh/koanf/providers/file v1.1.2 // indirect
	github.com/knadh/koanf/providers/posflag v0.1.0 // indirect
	github.com/knadh/koanf/providers/structs v0.1.0 // indirect
	github.com/knadh/koanf/v2 v2.2.1 // indirect
	github.com/mattn/go-colorable v0.1.14 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	github.com/mattn/go-runewidth v0.0.16 // indirect
	github.com/mitchellh/copystructure v1.2.0 // indirect
	github.com/mitchellh/reflectwalk v1.0.2 // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/rogpeppe/go-internal v1.14.1 // indirect
	github.com/rs/zerolog v1.33.0 // indirect
	github.com/spf13/cobra v1.9.1 // indirect
	github.com/spf13/pflag v1.0.6 // indirect
	github.com/vektra/mockery/v3 v3.5.4 // indirect
	github.com/xeipuuv/gojsonpointer v0.0.0-20180127040702-4e3ac2762d5f // indirect
	github.com/xeipuuv/gojsonreference v0.0.0-20180127040603-bd5ef7bd5415 // indirect
	github.com/xeipuuv/gojsonschema v1.2.0 // indirect
	go.yaml.in/yaml/v3 v3.0.3 // indirect
	golang.org/x/exp v0.0.0-20240719175910-8a7402abbf56 // indirect
	golang.org/x/exp/typeparams v0.0.0-20250210185358-939b2ce775ac // indirect
	golang.org/x/sync v0.16.0 // indirect
	golang.org/x/sys v0.35.0 // indirect
	golang.org/x/term v0.32.0 // indirect
	golang.org/x/text v0.25.0 // indirect
	golang.org/x/tools v0.36.0 // indirect
	golang.org/x/tools/go/expect v0.1.1-deprecated // indirect
	google.golang.org/grpc/cmd/protoc-gen-go-grpc v1.5.1 // indirect
	google.golang.org/protobuf v1.36.4 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	gotest.tools/gotestsum v1.12.0 // indirect
	honnef.co/go/tools v0.6.1 // indirect
	sigs.k8s.io/yaml v1.6.0 // indirect
)

tool (
	github.com/aojea/sloppy-netparser
	github.com/cespare/prettybench
	github.com/golangci/misspell
	github.com/jcchavezs/porto/cmd/porto
	github.com/vektra/mockery/v3
	golang.org/x/mod/modfile
	golang.org/x/tools/cmd/goimports
	google.golang.org/grpc/cmd/protoc-gen-go-grpc
	google.golang.org/protobuf/cmd/protoc-gen-go
	gotest.tools/gotestsum
	honnef.co/go/tools/cmd/staticcheck
	k8s.io/publishing-bot/cmd/publishing-bot/config
	sigs.k8s.io/yaml/yamlfmt
)
