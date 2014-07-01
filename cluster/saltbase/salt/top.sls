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

  'roles:kubernetes-master':
    - match: grain
    - golang
    - apiserver
    - controller-manager
    - etcd
    - nginx
