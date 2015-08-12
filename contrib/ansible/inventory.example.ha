
[masters]
kube-master-test-[1:3].example.com

[etcd:children]
masters

[nodes]
kube-minion-test-[1:2].example.com
