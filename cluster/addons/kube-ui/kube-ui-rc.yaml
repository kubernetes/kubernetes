apiVersion: v1
kind: ReplicationController
metadata:
  name: kube-ui-v4
  namespace: kube-system
  labels:
    k8s-app: kube-ui
    version: v4
    kubernetes.io/cluster-service: "true"
spec:
  replicas: 1
  selector:
    k8s-app: kube-ui
    version: v4
  template:
    metadata:
      labels:
        k8s-app: kube-ui
        version: v4
        kubernetes.io/cluster-service: "true"
    spec:
      containers:
      - name: kube-ui
        image: gcr.io/google_containers/kube-ui:v4
        resources:
          limits:
            cpu: 100m
            memory: 50Mi
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /
            port: 8080
          initialDelaySeconds: 30
          timeoutSeconds: 5
