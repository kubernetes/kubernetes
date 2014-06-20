TOP_PACKAGES="
  github.com/coreos/go-etcd/etcd
  github.com/fsouza/go-dockerclient
  code.google.com/p/goauth2/compute/serviceaccount
  code.google.com/p/goauth2/oauth
  code.google.com/p/google-api-go-client/compute/v1
"

DEP_PACKAGES="
  gopkg.in/v1/yaml
  bitbucket.org/kardianos/osext
  code.google.com/p/google-api-go-client/googleapi
  github.com/coreos/go-log/log
  github.com/coreos/go-systemd/journal
  github.com/google/cadvisor/info
"

PACKAGES="$TOP_PACKAGES $DEP_PACKAGES"
