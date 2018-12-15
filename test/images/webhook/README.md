# Kubernetes External Admission Webhook Test Image

The image tests MutatingAdmissionWebhook and ValidatingAdmissionWebhook. After deploying
it to kubernetes cluster, administrator needs to create a ValidatingWebhookConfiguration
in kubernetes cluster to register remote webhook admission controllers.

TODO: add the reference when the document for admission webhook v1beta1 API is done.

## Build the code

The binary can be built using the following command :

```bash
make bin
```

The images can be built using the Makefile in the parent directory (`test/images`) :

```bash
cd ..
make all WHAT=webhook
```
