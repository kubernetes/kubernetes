# Test Metrics Server

A Kubernetes external metrics API server for testing and development purposes. This server implements the `external.metrics.k8s.io` API and provides endpoints to dynamically create, update, and configure metrics for testing HPA (Horizontal Pod Autoscaler) and other metric-based scenarios.

## API Endpoints

### Health Checks

#### Check Server Health
```bash
curl -k https://localhost:6443/healthz
curl -k https://localhost:6443/readyz
```

### Kubernetes External Metrics API

#### List API Groups
```bash
curl -k https://localhost:6443/apis/external.metrics.k8s.io
```

#### List Available Resources
```bash
curl -k https://localhost:6443/apis/external.metrics.k8s.io/v1beta1
```

#### Get Metric Value
```bash
# Query a specific metric
curl -k https://localhost:6443/apis/external.metrics.k8s.io/v1beta1/namespaces/default/queue_messages_ready

# Query a metric with label selector
curl -k "https://localhost:6443/apis/external.metrics.k8s.io/v1beta1/namespaces/production/error_rate?labelSelector=env=prod,region=us-west"

# Or via kubectl
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/default/queue_messages_ready"
kubectl get --raw "/apis/external.metrics.k8s.io/v1beta1/namespaces/default/error_rate?labelSelector=env=prod"
```

**Error Response (Metric Not Found):**
```bash
curl -k https://localhost:6443/apis/external.metrics.k8s.io/v1beta1/namespaces/default/nonexistent_metric
# HTTP 404: no metric name called nonexistent_metric
```

**Error Response (Metric Configured to Fail):**
```bash
# HTTP 500: metric queue_messages_ready is configured to fail
```

---

### Metric Management

#### Create a New Metric
```bash
# Create a metric with value 250
curl -k "https://localhost:6443/create/my_custom_metric?value=250"

# Create a metric with labels
curl -k "https://localhost:6443/create/error_rate?value=25&labels=env=prod,region=us-west"

# Create a metric that will fail
curl -k "https://localhost:6443/create/failing_metric?value=100&fail=true"

# Create a metric with labels that will fail
curl -k "https://localhost:6443/create/error_rate?value=50&labels=env=staging&fail=true"
```

#### Update Metric Value
```bash
# Update the value of an existing metric
curl -k "https://localhost:6443/set/queue_messages_ready?value=500"
```

```bash
# Update the value of an existing labels metric
curl -k "https://localhost:6443/set/queue_messages_ready?value=500&labels=env=staging"
```

**Error Response (Metric Not Found):**
```bash
curl -k "https://localhost:6443/set/nonexistent_metric?value=100"
# HTTP 404: metric nonexistent_metric not found
```

#### Configure Metric Failure State
```bash
# Make a metric fail (return HTTP 500)
curl -k "https://localhost:6443/fail/queue_messages_ready?fail=true"

# Make a metric succeed again
curl -k "https://localhost:6443/fail/queue_messages_ready?fail=false"

# Make a metric with specific labels fail
curl -k "https://localhost:6443/fail/error_rate?fail=true&labels=env=prod,region=us-west"

# Make a labeled metric succeed again
curl -k "https://localhost:6443/fail/error_rate?fail=false&labels=env=prod,region=us-west"
```
### Querying Metrics with Labels
```bash
# Get all response_time metrics (returns all label variants)
curl -k "https://localhost:6443/apis/external.metrics.k8s.io/v1beta1/namespaces/default/response_time"

# Get only prod environment metrics
curl -k "https://localhost:6443/apis/external.metrics.k8s.io/v1beta1/namespaces/default/response_time?labelSelector=env=prod"

# Get specific metric with exact label match
curl -k "https://localhost:6443/apis/external.metrics.k8s.io/v1beta1/namespaces/default/response_time?labelSelector=env=prod,service=api"
```

## Default Metrics

The server starts with two pre-configured metrics:

1. **queue_messages_ready** - Initial value: 100
2. **http_requests_total** - Initial value: 500