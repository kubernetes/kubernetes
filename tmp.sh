#!/bin/bash

# run this from k/k root directory, make sure you download ripgrep "rg" from https://github.com/BurntSushi/ripgrep

# pick up newer dependency
hack/pin-dependency.sh k8s.io/klog/v2 v2.0.0-rc.1

# start with the main code base modify imports and references
find . -name "*.go" | grep -v vendor | xargs sed -i 's/\"k8s.io\/klog\"/\"k8s.io\/klog\/v2\"/'
rg "if\sklog\.V\(([0-9]+)\)\s\\{" | cut -f 1 -d ':' | sort | uniq | grep -v vendor | xargs sed -i -r "s/if\sklog\.V\(([0-9]+)\)\s\\{/if klog.V(\1).Enabled() {/"
sed -i -r "s/klog\.V\(([0-9]+)\)\s\\{/klog.V(\1).Enabled() {/" pkg/kubelet/eviction/helpers.go
hack/update-vendor.sh
git add .
git commit -a -m 'Update kubernetes code to klog v2'

# Now work on the vendor/ directory and modify usages and import references
sed -i -r "s/klog\.V\(([0-9]+)\)\s\\{/klog.V(\1).Enabled() {/" vendor/github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud/gen.go
sed -i -r "s/bool\(klog\.V\(([0-9]+)\)/bool(klog.V(\1).Enabled()/" vendor/k8s.io/client-go/rest/request.go vendor/k8s.io/client-go/transport/round_trippers.go
sed -i "s/bool(klog.V(klog.Level(l)))/bool(klog.V(klog.Level(l)).Enabled())/" vendor/k8s.io/apiserver/pkg/storage/etcd3/logger.go
sed -i "s/if bool(klog.V(setting.logLevel))/if bool(klog.V(setting.logLevel).Enabled())/" vendor/k8s.io/client-go/rest/request.go 
sed -i -r "s/klog\.V\(([0-9]+)\)\s\\{/klog.V(\1).Enabled() {/" vendor/k8s.io/utils/trace/trace.go
rg '"k8s.io/klog/v2"' | grep "\.go" | cut -f 1 -d ':' | sort | uniq | xargs sed -i 's/\"k8s.io\/klog\"/\"k8s.io\/klog\/v2\"/'
git add .
git commit -a -m 'hack vendored code to use klog v2'

# cleanup the older klog
rm vendor/k8s.io/klog/*

# ensure bazel references are updated
hack/update-bazel.sh

git add .
git commit -a -m 'remove old klog'

# Ensure that we have all the ducks in a row (we are using newer klog everywhere and we have fixed all the modified apis)
#hack/verify-typecheck.sh
