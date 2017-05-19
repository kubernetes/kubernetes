# Deploy kubernetes on centos

1. run build.sh to download dependences

  Before installing, you can define the path or version of dependences in **cluster/centos/config-build.sh**

  ```
  cd cluster/centos && ./build.sh all
  ```

2. Just up your kubernetes

  Before running, you can define your machines' IP or other args  in **cluster/centos/config-default.sh**

  ```
  export KUBERNETES_PROVIDER=centos && cluster/kube-up.sh
  ```
