base:
  '*':
    - base

  'roles:kubernetes-pool':
    - match: grain
    - golang
    - docker
    - kubelet
    - kube-proxy
    - cadvisor
    - nsinit

  'roles:kubernetes-master':
    - match: grain
    - golang
    - etcd
    - apiserver
    - controller-manager
    - nginx
