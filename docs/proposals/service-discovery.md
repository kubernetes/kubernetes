# Service Discovery Proposal

## Goal of this document

To consume a service, a developer needs to know the full URL and a description of the API. Kubernetes contains the host and port information of a service, but it lacks the scheme and the path information needed if the service is not bound at the root. In this document we propose some standard kubernetes service annotations to fix these gaps. It is important that these annotations are a standard to allow for standard service discovery across Kubernetes implementations. Note that the example largely speaks to consuming WebServices but that the same concepts apply to other types of services.

## Endpoint URL, Service Type

A URL can accurately describe the location of a Service. A generic URL is of the following form

    scheme:[//[user:password@]host[:port]][/]path[?query][#fragment]

however for the purpose of service discovery we can simplify this to the following form

    scheme:[//host[:port]][/]path

If a user and/or password is required then this information can be passed using Kubernetes Secrets. Kubernetes contains the host and port of each service but it lacks the scheme and path.

`Service Path` - Every Service has one (or more) endpoint. As a rule the endpoint should be located at the root "/" of the location URL, i.e. `http://172.100.1.52/`. There are cases where this is not possible and the actual service endpoint could be located at `http://172.100.1.52/cxfcdi`. The Kubernetes metadata for a service does not capture the path part, making it hard to consume this service.

`Service Scheme` - Services can be deployed using different schemes. Some popular schemes include `http`,`https`,`file`,`ftp` and `jdbc`.

`Service Protocol` - Services use different protocols that clients need to speak in order to communicate with the service, some examples of service level protocols are SOAP, REST (Yes, technically REST isn’t a protocol but an architectural style). For service consumers it can be hard to tell what protocol is expected.

## Service Description

The API of a service is the point of interaction with a service consumer. The description of the API is an essential piece of information at creation time of the service consumer. It has become common to publish a service definition document on a know location on the service itself. This 'well known' place it not very standard, so it is proposed the service developer provides the service description path and the type of Definition Language (DL) used.

`Service Description Path` - To facilitate the consumption of the service by client, the location this document would be greatly helpful to the service consumer. In some cases the client side code can be generated from such a document. It is assumed that the service description document is published somewhere on the service endpoint itself.

`Service Description Language` - A number of Definition Languages (DL) have been developed to describe the service. Some of examples are `WSDL`, `WADL` and `Swagger`. In order to consume a description document it is good to know the type of DL used.

## Standard Service Annotations

Kubernetes allows the creation of Service Annotations. Here we propose the use of the following standard annotations

* `api.service.kubernetes.io/path` - the path part of the service endpoint url. An example value could be `cxfcdi`,
* `api.service.kubernetes.io/scheme` - the scheme part of the service endpoint url. Some values could be `http` or `https`.
* `api.service.kubernetes.io/protocol` - the protocol of the service. Known values are `SOAP`, `XML-RPC` and `REST`,
* `api.service.kubernetes.io/description-path` - the path part of the service description document’s endpoint. It is a pretty safe assumption that the service self-documents. An example value for a swagger 2.0 document can be `cxfcdi/swagger.json`,
* `api.kubernetes.io/description-language` - the type of Description Language used. Known values are `WSDL`, `WADL`, `SwaggerJSON`, `SwaggerYAML`.

The fragment below is taken from the service section of the kubernetes.json were these annotations are used

    ...
    "objects" : [ {
      "apiVersion" : "v1",
      "kind" : "Service",
      "metadata" : {
        "annotations" : {
          "api.service.kubernetes.io/protocol" : "REST",
          "api.service.kubernetes.io/scheme" "http",
          "api.service.kubernetes.io/path" : "cxfcdi",
          "api.service.kubernetes.io/description-path" : "cxfcdi/swagger.json",
          "api.service.kubernetes.io/description-language" : "SwaggerJSON"
        },
    ...

## Conclusion

Five service annotations are proposed as a standard way to describe a service endpoint. These five annotation are promoted as a Kubernetes standard, so that services can be discovered and a service catalog can be build to facilitate service consumers.





<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/service-discovery.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
