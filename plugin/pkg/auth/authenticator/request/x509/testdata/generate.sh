#!/usr/bin/env bash

cfssl gencert -initca root.csr.json | cfssljson -bare root

cfssl gencert -initca intermediate.csr.json | cfssljson -bare intermediate
cfssl sign -ca root.pem -ca-key root-key.pem -config intermediate.config.json intermediate.csr | cfssljson -bare intermediate

cfssl gencert -ca intermediate.pem -ca-key intermediate-key.pem -config client.config.json --profile=valid   client.csr.json | cfssljson -bare client-valid
cfssl gencert -ca intermediate.pem -ca-key intermediate-key.pem -config client.config.json --profile=expired client.csr.json | cfssljson -bare client-expired
 
