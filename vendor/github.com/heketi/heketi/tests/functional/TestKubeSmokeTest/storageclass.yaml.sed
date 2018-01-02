apiVersion: storage.k8s.io/v1beta1
kind: StorageClass
metadata:
  name: slow
provisioner: kubernetes.io/glusterfs
parameters:
  endpoint: "glusterfs-cluster"
  resturl: "%%URL%%"
