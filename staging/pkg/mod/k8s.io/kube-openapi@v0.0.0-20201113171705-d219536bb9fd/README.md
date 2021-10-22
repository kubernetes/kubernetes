# Kube OpenAPI

This repo is the home for Kubernetes OpenAPI discovery spec generation. The goal 
is to support a subset of OpenAPI features to satisfy kubernetes use-cases but 
implement that subset with little to no assumption about the structure of the 
code or routes. Thus, there should be no kubernetes specific code in this repo. 


There are two main parts: 
 - A model generator that goes through .go files, find and generate model 
definitions. 
 - The spec generator that is responsible for dynamically generate 
the final OpenAPI spec using web service routes or combining other 
OpenAPI/Json specs.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.
