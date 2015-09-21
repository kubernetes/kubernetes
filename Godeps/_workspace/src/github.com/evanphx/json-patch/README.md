## JSON-Patch

Provides the abiilty to modify and test a JSON according to a
[RFC6902 JSON patch](http://tools.ietf.org/html/rfc6902) and [RFC7386 JSON Merge Patch](https://tools.ietf.org/html/rfc7386).

*Version*: **1.0**

[![GoDoc](https://godoc.org/github.com/evanphx/json-patch?status.svg)](http://godoc.org/github.com/evanphx/json-patch)

[![Build Status](https://travis-ci.org/evanphx/json-patch.svg?branch=RFC7386)](https://travis-ci.org/evanphx/json-patch)

### API Usage

* Given a `[]byte`, obtain a Patch object

  `obj, err := jsonpatch.DecodePatch(patch)`

* Apply the patch and get a new document back

  `out, err := obj.Apply(doc)`

* Create a JSON Merge Patch document based on two json documents (a to b):

  `mergeDoc, err := jsonpatch.CreateMergePatch(a, b)`
 
* Bonus API: compare documents for structural equality

  `jsonpatch.Equal(doca, docb)`

