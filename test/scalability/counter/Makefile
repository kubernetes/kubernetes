counter:
	kubectl create --validate -f counter-rc.yaml

counter1:
	kubectl scale rc counter --replicas=1

counter10:
	kubectl scale rc counter --replicas=10

counter50:
	kubectl scale rc counter --replicas=50

counter100:
	kubectl scale rc counter --replicas=100

counter200:
	kubectl scale rc counter --replicas=200

counter500:
	kubectl scale rc counter --replicas=500


counter5000:
	kubectl scale rc counter --replicas=5000

stop:
	kubectl delete rc counter
