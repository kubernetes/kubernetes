base:
  '*':
    - base

  'roles:kubernetes-pool':
    - match: grain
    - golang
    - docker
    - kubelet
    - kube-proxy

  'roles:kubernetes-master':
    - match: grain
    - golang
    - apiserver
    - controller-manager
    - etcd
    - nginx
