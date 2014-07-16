TOP_PACKAGES="
  github.com/coreos/go-etcd/etcd
  github.com/fsouza/go-dockerclient
  github.com/golang/glog
  code.google.com/p/goauth2/compute/serviceaccount
  code.google.com/p/goauth2/oauth
  code.google.com/p/google-api-go-client/compute/v1
  github.com/google/cadvisor/info
  github.com/google/cadvisor/client
"

DEP_PACKAGES="
  gopkg.in/v1/yaml
  bitbucket.org/kardianos/osext
  code.google.com/p/google-api-go-client/googleapi
  github.com/coreos/go-log/log
  github.com/coreos/go-systemd/journal
"

PACKAGES="$TOP_PACKAGES $DEP_PACKAGES"
