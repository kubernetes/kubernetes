# Kubernetes External Admission Webhook Test Image

The image tests CustomResourceConversionWebhook. After deploying it to kubernetes cluster,
administrator needs to create a CustomResourceConversion.Webhook
in kubernetes cluster to use remote webhook for conversions.

## Build the code

```bash
make build
```
