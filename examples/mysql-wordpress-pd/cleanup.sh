kubectl delete deployment wordpress
kubectl delete deployment wordpress-mysql

kubectl delete service wordpress
kubectl delete service wordpress-mysql

kubectl delete pv local-pv-1
kubectl delete pv local-pv-2

kubectl delete pvc mysql-pv-claim
kubectl delete pvc wp-pv-claim

kubectl delete secret mysql-pass
